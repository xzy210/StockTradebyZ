"""
ETF rotation runtime orchestration service.

This module owns the live runtime workflow: signal-check orchestration, data
update orchestration, and automatic schedule triggering. RotationEngine keeps Qt
signals and compatibility wrappers while delegating the run loop here.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from common.events import BacktestEvent, EventBus
from common.execution_contract import OrderIntent, PortfolioPlanner, RebalanceIntent, StrategySignal, TargetPortfolio

from .config import ConfigManager, RotationConfig
from .holiday_calendar import is_trading_day
from .rotation_data_service import RotationDataService
from .rotation_guard_service import RotationGuardService
from .rotation_ledger_service import RotationLedgerService
from .rotation_signal_service import RotationDecisionService, RotationSignalService
from .state_manager import RotationState, StateManager
from .trade_executor import TradeExecutor
from trading_app.services.market_data.market_data_status_service import get_market_data_status_service
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service

logger = logging.getLogger(__name__)


class RotationRuntimeService:
    """Coordinate signal checks, data updates, and scheduled execution."""

    def __init__(
        self,
        *,
        config: RotationConfig,
        state: RotationState,
        state_mgr: StateManager,
        config_mgr: ConfigManager,
        executor: TradeExecutor,
        data_dir: Path,
        signal_service: RotationSignalService,
        decision_service: RotationDecisionService,
        guard_service: RotationGuardService,
        ledger_service: RotationLedgerService,
        auto_timer=None,
        update_parent=None,
        logger_fn: Optional[Callable[[str], None]] = None,
        status_fn: Optional[Callable[[str], None]] = None,
        signal_fn: Optional[Callable[[str, dict], None]] = None,
        scores_fn: Optional[Callable[[dict], None]] = None,
        notify_signal_fn: Optional[Callable[[str, dict, Optional[str], Optional[str], str], None]] = None,
        execute_rebalance_fn: Optional[Callable[[RebalanceIntent], list[Any]]] = None,
        pending_order_sync_fn: Optional[Callable[[], int]] = None,
        code_name_fn: Optional[Callable[[str], str]] = None,
        data_service: Optional[RotationDataService] = None,
        event_bus: Optional[EventBus] = None,
        now_fn: Callable[[], datetime] = datetime.now,
        trading_day_fn: Callable[[object], bool] = is_trading_day,
    ) -> None:
        self.config = config
        self.state = state
        self.state_mgr = state_mgr
        self.config_mgr = config_mgr
        self.executor = executor
        self.data_dir = Path(data_dir)
        self.data_service = data_service or RotationDataService(self.data_dir)
        self.signal_service = signal_service
        self.decision_service = decision_service
        self.guard_service = guard_service
        self.ledger_service = ledger_service
        self.auto_timer = auto_timer
        self.update_parent = update_parent
        self.logger_fn = logger_fn or (lambda message: None)
        self.status_fn = status_fn or (lambda message: None)
        self.signal_fn = signal_fn or (lambda signal, result: None)
        self.scores_fn = scores_fn or (lambda scores: None)
        self.notify_signal_fn = notify_signal_fn or (
            lambda signal, scores, current, target, reason: None
        )
        self.execute_rebalance_fn = execute_rebalance_fn
        self.pending_order_sync_fn = pending_order_sync_fn
        self.code_name_fn = code_name_fn or (lambda code: code)
        self.event_bus = event_bus
        self.now_fn = now_fn
        self.trading_day_fn = trading_day_fn

        self.auto_check_interval = 30_000
        self.auto_data_done_date = ""
        self.auto_signal_done_date = ""
        self._auto_timer_thread: Optional[threading.Timer] = None
        self._auto_running = False
        self._update_thread: Optional[threading.Thread] = None
        self._update_lock = threading.Lock()
        self.update_thread: Optional[object] = None
        self.update_pending_signal_check = False
        self.update_schedule_context: Optional[dict] = None

    def update_context(
        self,
        *,
        config: RotationConfig,
        state: RotationState,
        executor: TradeExecutor,
        data_dir: Optional[Path] = None,
    ) -> None:
        """Refresh mutable runtime dependencies."""
        self.config = config
        self.state = state
        self.executor = executor
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            self.data_service.update_context(data_dir=self.data_dir)

    def set_event_bus(self, event_bus: Optional[EventBus]) -> None:
        """Attach or replace the shell event bus used by live runtime events."""
        self.event_bus = event_bus

    def _publish_event(
        self,
        event_type: str,
        message: str,
        *,
        run_id: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(
                BacktestEvent(
                    date=None,
                    bars={},
                    history={},
                    prices={},
                    valid_symbols=list(self.config.etf_pool or []),
                    event_type=event_type,
                    message=message,
                    mode="live",
                    run_id=run_id,
                    payload={"strategy_id": "etf_rotation", **dict(payload or {})},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive event bridge
            logger.debug("发布 ETF 轮动事件失败: %s", exc)

    def get_data_version_audit(self) -> dict[str, Any]:
        """Return a serializable data-version audit for the current ETF pool."""
        try:
            return self.data_service.get_data_version(self.config.etf_pool).to_dict()
        except Exception as exc:
            logger.debug("读取 ETF 轮动 data_version 失败: %s", exc)
            return {"data_version": "", "error": str(exc)}

    def check_live_market_data_ready(
        self,
        *,
        require_minute_freshness: bool = False,
    ) -> Tuple[bool, str]:
        """Check whether market data is ready for a live strategy run."""
        etf_codes = list(
            dict.fromkeys(
                str(code or "").strip()
                for code in self.config.etf_pool
                if str(code or "").strip()
            )
        )
        status = get_market_data_status_service().check_status(
            stock_codes=[],
            etf_codes=etf_codes,
            index_codes=[],
            realtime_probe_codes=etf_codes[:3] if etf_codes else None,
            require_minute_freshness=require_minute_freshness,
            etf_data_dir=self.data_dir,
        )
        if status.can_run_live_strategy:
            return True, status.summary
        return False, status.summary

    def run_signal_check(
        self,
        *,
        schedule_context: Optional[dict] = None,
    ) -> dict:
        """Run one full signal check and return pure strategy signals."""
        started_at = self.now_fn()
        run_id = f"etf_rotation_live_{started_at.strftime('%Y%m%d%H%M%S')}"
        data_audit = self.get_data_version_audit()
        data_version = str(data_audit.get("data_version", "") or "")
        self.logger_fn("=" * 50)
        self.logger_fn(f"开始信号检查 [{started_at.strftime('%Y-%m-%d %H:%M:%S')}]")
        self.logger_fn(f"数据版本: {data_version or 'unknown'}")
        self.status_fn("正在计算信号...")
        self._publish_event(
            "run_started",
            "ETF rotation live signal check started",
            run_id=run_id,
            payload={
                "schedule_context": dict(schedule_context or {}),
                "etf_pool": list(self.config.etf_pool or []),
                "data_version": data_version,
                "data_audit": data_audit,
            },
        )

        result = {
            "run_id": run_id,
            "strategy_id": str(getattr(self.config, "strategy_id", "") or "etf_rotation"),
            "mode": "live_dry_run",
            "data_version": data_version,
            "data_audit": data_audit,
            "signal": "ERROR",
            "scores": {},
            "target": None,
            "reason": "",
            "executed": False,
        }

        schedule_done = False

        def finalize_schedule(status: str, error: str = "") -> None:
            nonlocal schedule_done
            if schedule_done or not schedule_context:
                return
            self.state_mgr.mark_auto_signal_task(
                status=status,
                schedule_time=str(schedule_context.get("schedule_time", "") or ""),
                trigger=str(schedule_context.get("trigger", "") or ""),
                task_date=str(schedule_context.get("task_date", "") or ""),
                error=error,
            )
            schedule_done = True

        if schedule_context:
            self.state_mgr.mark_auto_signal_task(
                status="running",
                schedule_time=str(schedule_context.get("schedule_time", "") or ""),
                trigger=str(schedule_context.get("trigger", "") or ""),
                task_date=str(schedule_context.get("task_date", "") or ""),
            )

        ready, reason = self.check_live_market_data_ready()
        if not ready:
            result["signal"] = "BLOCKED"
            result["reason"] = f"行情数据未就绪: {reason}"
            self.logger_fn(f"⛔ {result['reason']}")
            self.signal_fn(result["signal"], result)
            self.state_mgr.update_check_result(result["signal"], {})
            self.status_fn(result["reason"])
            finalize_schedule("failed", result["reason"])
            self._publish_event("rebalance_failed", result["reason"], run_id=run_id, payload={"result": result})
            self._publish_event("run_completed", "ETF rotation live signal check blocked", run_id=run_id, payload={"result": result})
            self.logger_fn("=" * 50)
            return result

        try:
            if self.pending_order_sync_fn is not None:
                synced_orders = int(self.pending_order_sync_fn() or 0)
                if synced_orders > 0:
                    self.logger_fn(f"🔄 已同步 {synced_orders} 条未完成 ETF 委托状态")

            if self.guard_service.in_drawdown_cooldown():
                result["signal"] = "COOLDOWN"
                result["reason"] = f"回撤保护冷却期（剩余{self.state.cooldown_remaining}天）"
                self.logger_fn(f"⏸ {result['reason']}")
                self.signal_fn(result["signal"], result)
                self.state_mgr.update_check_result(result["signal"], {})
                self.status_fn(result["reason"])
                finalize_schedule("completed")
                self._publish_event("guard_triggered", result["reason"], run_id=run_id, payload={"result": result})
                self._publish_event("run_completed", "ETF rotation live signal check completed", run_id=run_id, payload={"result": result})
                self.logger_fn("=" * 50)
                return result

            dd_triggered, dd_result = self.guard_service.check_drawdown_protection()
            if dd_triggered:
                result.update(dd_result)
                rebalance_intent = self._attach_rebalance_plan(result, result["signal"], None, result["reason"])
                self._emit_guard_signal(result)
                self.state_mgr.update_check_result(result["signal"], {})
                execution_error = self._maybe_auto_execute(result, rebalance_intent, schedule_context=schedule_context)
                finalize_schedule("failed" if execution_error else "completed", execution_error)
                self._publish_event("guard_triggered", result["reason"], run_id=run_id, payload={"result": result})
                self._publish_event("run_completed", "ETF rotation live signal check completed", run_id=run_id, payload={"result": result})
                self.logger_fn("=" * 50)
                return result

            ts_triggered, ts_result = self.guard_service.check_trailing_stop()
            if ts_triggered:
                result.update(ts_result)
                rebalance_intent = self._attach_rebalance_plan(result, result["signal"], None, result["reason"])
                self._emit_guard_signal(result)
                self.state_mgr.update_check_result(result["signal"], {})
                execution_error = self._maybe_auto_execute(result, rebalance_intent, schedule_context=schedule_context)
                finalize_schedule("failed" if execution_error else "completed", execution_error)
                self._publish_event("guard_triggered", result["reason"], run_id=run_id, payload={"result": result})
                self._publish_event("run_completed", "ETF rotation live signal check completed", run_id=run_id, payload={"result": result})
                self.logger_fn("=" * 50)
                return result

            self.guard_service.update_check_count()

            self.signal_service.update_context(config=self.config, data_dir=self.data_dir)
            scores = self.signal_service.calculate_scores()
            if not scores:
                result["reason"] = "因子得分计算失败（数据不足或加载失败）"
                self.logger_fn(f"❌ {result['reason']}")
                self.status_fn("信号检查失败")
                finalize_schedule("failed", result["reason"])
                self._publish_event("rebalance_failed", result["reason"], run_id=run_id, payload={"result": result})
                self.logger_fn("=" * 50)
                return result

            result["scores"] = scores
            self.scores_fn(scores)
            self._publish_event("signal_calculated", "ETF rotation scores calculated", run_id=run_id, payload={"scores": scores})

            self.decision_service.update_context(config=self.config, state=self.state)
            signal, target, reason = self.decision_service.make_decision(scores)
            signal, target, reason, filtered = self.guard_service.filter_rebalance_signal(
                signal,
                target,
                reason,
            )
            if filtered:
                self.logger_fn(f"📅 {reason}")

            if signal in {"BUY", "SWITCH"} and self.ledger_service.has_active_order_records():
                active_orders = self.ledger_service.active_order_records()
                first_order = active_orders[0] if active_orders else {}
                result["signal"] = "BLOCKED"
                result["target"] = target
                result["reason"] = (
                    "存在未完成 ETF 委托，禁止生成新的买入信号: "
                    f"#{first_order.get('order_id', '')} "
                    f"{first_order.get('action', '')} "
                    f"{first_order.get('code', '')} "
                    f"status={first_order.get('status', '')}"
                )
                result["scores"] = scores
                self.logger_fn(f"⛔ {result['reason']}")
                self.signal_fn(result["signal"], result)
                self.state_mgr.update_check_result(result["signal"], scores)
                self.status_fn(result["reason"])
                finalize_schedule("completed")
                self._publish_event("rebalance_blocked", result["reason"], run_id=run_id, payload={"result": result})
                self._publish_event("run_completed", "ETF rotation live signal check blocked", run_id=run_id, payload={"result": result})
                self.logger_fn("=" * 50)
                return result

            result["signal"] = signal
            result["target"] = target
            result["reason"] = reason
            rebalance_intent = self._attach_rebalance_plan(result, signal, target, reason)
            self._publish_event(
                "decision_made",
                f"ETF rotation decision: {signal}",
                run_id=run_id,
                payload={"signal": signal, "target": target, "reason": reason, "scores": scores},
            )
            if result.get("strategy_signals"):
                self._publish_event(
                    "rebalance_submitted",
                    f"ETF rotation generated {len(result.get('strategy_signals') or [])} strategy signals",
                    run_id=run_id,
                    payload={"strategy_signals": result.get("strategy_signals"), "rebalance_intent": result.get("rebalance_intent")},
                )

            self.logger_fn(f"📊 信号: {signal} | 目标: {target} | 原因: {reason}")
            self.signal_fn(signal, result)
            self.state_mgr.update_check_result(signal, scores)

            if self.config.notify_on_signal:
                self.notify_signal_fn(
                    signal,
                    scores,
                    self.state.current_holding,
                    target,
                    reason,
                )

            self.status_fn(
                f"信号: {signal} "
                f"{'| 已生成执行信号' if result.get('strategy_signals') else '| 无需执行'}"
            )
            execution_error = self._maybe_auto_execute(result, rebalance_intent, schedule_context=schedule_context)
            finalize_schedule("failed" if execution_error else "completed", execution_error)
            self._publish_event("run_completed", "ETF rotation live signal check completed", run_id=run_id, payload={"result": result})

        except Exception as e:
            logger.exception("信号检查异常")
            result["reason"] = f"异常: {e}"
            self.logger_fn(f"❌ 信号检查异常: {e}")
            self.status_fn("信号检查异常")
            finalize_schedule("failed", result["reason"])
            self._publish_event("rebalance_failed", result["reason"], run_id=run_id, payload={"result": result})

        self.ledger_service.record_daily_equity()
        self.logger_fn("=" * 50)
        return result

    def _emit_guard_signal(self, result: dict) -> None:
        """Emit status/notification for guard-generated signal results."""
        signal = str(result.get("signal") or "")
        reason = str(result.get("reason") or "")
        self.signal_fn(signal, result)
        self.status_fn(reason)
        if self.config.notify_on_signal:
            self.notify_signal_fn(signal, {}, self.state.current_holding, None, reason)

    def _maybe_auto_execute(
        self,
        result: dict,
        rebalance_intent: RebalanceIntent,
        *,
        schedule_context: Optional[dict] = None,
    ) -> str:
        if not self._should_auto_execute(schedule_context):
            return ""
        if not rebalance_intent.order_intents:
            self.logger_fn("🤖 自动执行已开启，但当前信号无可提交委托")
            self.status_fn("自动执行已开启，当前信号无需下单")
            return ""
        if self.execute_rebalance_fn is None:
            message = "自动执行已开启，但未配置统一执行入口"
            self.logger_fn(f"⛔ {message}")
            self.status_fn(message)
            result["auto_execute_error"] = message
            return message
        try:
            self.logger_fn(f"🤖 自动执行已开启，准备提交 {len(rebalance_intent.order_intents)} 笔委托")
            reports = list(self.execute_rebalance_fn(rebalance_intent) or [])
            result["execution_reports"] = [
                report.to_dict() if hasattr(report, "to_dict") else dict(report or {})
                for report in reports
            ]
            submitted = any(
                bool(getattr(report, "submitted", False) or getattr(report, "accepted", False) or getattr(report, "filled", False))
                for report in reports
            )
            result["executed"] = bool(submitted)
            if reports and submitted:
                self.logger_fn(f"✅ 自动执行已提交 {len(reports)} 笔委托")
                self.status_fn(f"自动执行已提交 {len(reports)} 笔委托")
                return ""
            message = "自动执行未生成有效委托回报"
            if reports:
                messages = [str(getattr(report, "message", "") or "") for report in reports]
                message = "；".join([item for item in messages if item]) or message
            self.logger_fn(f"⛔ {message}")
            self.status_fn(message)
            result["auto_execute_error"] = message
            return message
        except Exception as exc:
            logger.exception("ETF 自动执行异常")
            message = f"自动执行异常: {exc}"
            self.logger_fn(f"❌ {message}")
            self.status_fn(message)
            result["auto_execute_error"] = message
            return message

    def _should_auto_execute(self, schedule_context: Optional[dict]) -> bool:
        if not bool(getattr(self.config, "auto_execute_enabled", False)):
            return False
        if not schedule_context:
            return False
        trigger = str(schedule_context.get("trigger", "") or "").strip().lower()
        return trigger in {"scheduled", "manual"}

    def build_strategy_signals(
        self,
        signal: str,
        target: Optional[str],
        reason: str,
        *,
        price: Optional[float] = None,
    ) -> list[StrategySignal]:
        """Build gateway-compatible StrategySignal objects from a portfolio rebalance plan."""
        rebalance_intent = self.build_rebalance_intent(signal, target, reason, price=price)
        return self._signals_from_rebalance_intent(rebalance_intent)

    def build_rebalance_intent(
        self,
        signal: str,
        target: Optional[str],
        reason: str,
        *,
        price: Optional[float] = None,
    ) -> RebalanceIntent:
        """Build a portfolio-level rebalance intent from one ETF rotation decision."""
        target_portfolio = self.build_target_portfolio(signal, target, reason)
        prices = self._collect_rebalance_prices(target_portfolio, price=price)
        total_asset = self._safe_total_asset(prices)
        available_cash = self._safe_available_cash()
        metadata = {
            "rotation_signal": signal,
            "virtual_account_id": target_portfolio.metadata.get("virtual_account_id", ""),
        }
        if signal not in {"BUY", "SWITCH", "SELL_ALL", "DRAWDOWN_STOP", "TRAILING_STOP"}:
            return RebalanceIntent(
                target_portfolio=target_portfolio,
                order_intents=(),
                current_positions=self._current_positions(),
                prices=prices,
                total_asset=total_asset,
                available_cash=available_cash,
                reason=reason,
                metadata={**dict(target_portfolio.metadata), **metadata},
            )
        planner = PortfolioPlanner(min_trade_amount=float(getattr(self.config, "min_trade_amount", 0.0) or 0.0))
        return planner.plan(
            target_portfolio,
            current_positions=self._current_positions(),
            prices=prices,
            total_asset=total_asset,
            available_cash=available_cash,
            reason=reason,
            source="live_strategy_center",
            trigger="strategy_center",
            metadata=metadata,
        )

    def build_target_portfolio(self, signal: str, target: Optional[str], reason: str) -> TargetPortfolio:
        """Translate rotation decision into a target portfolio instead of imperative buy/sell steps."""
        strategy_id = str(getattr(self.config, "strategy_id", "") or "etf_rotation").strip() or "etf_rotation"
        spec = get_strategy_spec_service().get(strategy_id, fallback_name="ETF轮动实盘")
        metadata = {
            "virtual_account_id": spec.virtual_account_id,
            "source": "live_strategy_center",
            "trigger": "strategy_center",
            "rotation_signal": signal,
        }
        if signal in {"BUY", "SWITCH"} and target:
            return TargetPortfolio.single_asset(
                symbol=target,
                weight=float(getattr(self.config, "cash_ratio", 1.0) or 1.0),
                strategy_id=strategy_id,
                strategy_name=spec.strategy_name,
                reason=reason,
                metadata=metadata,
            )
        if signal in {"SELL_ALL", "DRAWDOWN_STOP", "TRAILING_STOP"}:
            return TargetPortfolio.cash_only(
                strategy_id=strategy_id,
                strategy_name=spec.strategy_name,
                reason=reason,
                metadata=metadata,
            )
        weights = {}
        holding = str(self.state.current_holding or "").strip()
        if holding:
            weights[holding] = float(getattr(self.config, "cash_ratio", 1.0) or 1.0)
        return TargetPortfolio(
            weights=weights,
            cash_weight=max(0.0, 1.0 - sum(weights.values())) if weights else 1.0,
            strategy_id=strategy_id,
            strategy_name=spec.strategy_name,
            reason=reason,
            metadata=metadata,
        )

    def _attach_rebalance_plan(self, result: dict, signal: str, target: Optional[str], reason: str) -> RebalanceIntent:
        rebalance_intent = self.build_rebalance_intent(signal, target, reason)
        result["target_portfolio"] = rebalance_intent.target_portfolio.to_dict()
        result["rebalance_intent"] = rebalance_intent.to_dict()
        result["order_intents"] = [intent.to_dict() for intent in rebalance_intent.order_intents]
        result["strategy_signals"] = [item.to_dict() for item in self._signals_from_rebalance_intent(rebalance_intent)]
        return rebalance_intent

    def _signals_from_rebalance_intent(self, rebalance_intent: RebalanceIntent) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        for intent in rebalance_intent.order_intents:
            signals.append(self._signal_from_order_intent(intent, rebalance_intent))
        return signals

    @staticmethod
    def _signal_from_order_intent(intent: OrderIntent, rebalance_intent: RebalanceIntent) -> StrategySignal:
        metadata = {
            **dict(intent.metadata),
            "virtual_account_id": intent.virtual_account_id,
            "source": intent.source,
            "trigger": intent.trigger,
            "rebalance_intent_id": rebalance_intent.intent_id,
            "order_intent_id": intent.intent_id,
            "quantity_mode": "delta",
            "quantity": intent.quantity,
        }
        return StrategySignal(
            symbol=intent.symbol,
            action=intent.side,
            strategy_id=intent.strategy_id,
            strategy_name=intent.strategy_name,
            target_quantity=int(intent.quantity or 0),
            price=intent.price,
            reason=intent.reason,
            metadata=metadata,
        )

    def _collect_rebalance_prices(self, target_portfolio: TargetPortfolio, *, price: Optional[float] = None) -> dict[str, float]:
        symbols = set(target_portfolio.weights.keys())
        if self.state.current_holding:
            symbols.add(str(self.state.current_holding or ""))
        prices: dict[str, float] = {}
        for symbol in symbols:
            if not symbol:
                continue
            resolved = float(price or self.executor.get_current_price(symbol) or 0.0)
            if resolved <= 0 and symbol == self.state.current_holding:
                resolved = float(self.state.buy_price or 0.0)
            prices[symbol] = resolved
        return prices

    def _current_positions(self) -> dict[str, int]:
        holding = str(self.state.current_holding or "").strip()
        if not holding:
            return {}
        quantity = int(self.state.buy_quantity or 0)
        return {holding: quantity} if quantity > 0 else {}

    def _safe_total_asset(self, prices: dict[str, float]) -> float:
        try:
            total_asset = float(self.ledger_service.total_asset() or 0.0)
        except Exception:
            total_asset = 0.0
        if total_asset > 0:
            return total_asset
        cash = self._safe_available_cash()
        holding = str(self.state.current_holding or "").strip()
        if holding and int(self.state.buy_quantity or 0) > 0:
            price = float(prices.get(holding, 0.0) or self.state.buy_price or 0.0)
            if price > 0:
                cash += price * int(self.state.buy_quantity or 0)
        return cash

    def _safe_available_cash(self) -> float:
        try:
            return max(float(self.ledger_service.available_cash() or 0.0), 0.0)
        except Exception:
            return 0.0

    def start_auto(self) -> None:
        """Start automatic schedule polling without requiring a GUI event loop."""
        self.config.auto_enabled = True
        self.config_mgr.save(self.config)
        self._auto_running = True
        if self.auto_timer is not None:
            self.auto_timer.start(self.auto_check_interval)
        else:
            self._schedule_next_auto_tick()
        self.logger_fn("✅ 自动调度已启动")
        self.status_fn("自动模式运行中")

    def stop_auto(self) -> None:
        """Stop automatic schedule polling."""
        self.config.auto_enabled = False
        self.config_mgr.save(self.config)
        self._auto_running = False
        if self.auto_timer is not None:
            self.auto_timer.stop()
        if self._auto_timer_thread is not None:
            self._auto_timer_thread.cancel()
            self._auto_timer_thread = None
        self.logger_fn("⏹ 自动调度已停止")
        self.status_fn("自动模式已停止")

    def _schedule_next_auto_tick(self) -> None:
        if not self._auto_running:
            return
        interval_seconds = max(float(self.auto_check_interval or 0) / 1000.0, 1.0)
        timer = threading.Timer(interval_seconds, self._auto_timer_tick)
        timer.daemon = True
        self._auto_timer_thread = timer
        timer.start()

    def _auto_timer_tick(self) -> None:
        if not self._auto_running:
            return
        try:
            self.on_auto_timer()
        finally:
            self._schedule_next_auto_tick()

    def update_data(
        self,
        *,
        run_signal_check_after: bool = False,
        schedule_context: Optional[dict] = None,
    ) -> None:
        """Start asynchronous ETF data update and optionally run a signal check after it."""
        if self.is_update_running():
            self.logger_fn("⚠ 数据更新正在进行中，请稍候")
            return

        self.update_pending_signal_check = bool(run_signal_check_after)
        self.update_schedule_context = dict(schedule_context or {}) if schedule_context else None
        if self.update_schedule_context:
            self.state_mgr.mark_auto_data_task(
                status="running",
                schedule_time=str(self.update_schedule_context.get("schedule_time", "") or ""),
                trigger=str(self.update_schedule_context.get("trigger", "") or ""),
                task_date=str(self.update_schedule_context.get("task_date", "") or ""),
            )
        self.logger_fn(f"🔄 开始更新 {len(self.config.etf_pool)} 只ETF数据...")
        self.status_fn("正在更新ETF数据...")
        self._publish_event(
            "data_update_started",
            "ETF rotation data update started",
            payload={"etf_pool": list(self.config.etf_pool or [])},
        )

        worker = threading.Thread(target=self._run_data_update_worker, daemon=True)
        with self._update_lock:
            self._update_thread = worker
            self.update_thread = worker
        worker.start()

    def is_update_running(self) -> bool:
        """Return whether an asynchronous data update is currently running."""
        thread = self._update_thread
        if thread is not None and thread.is_alive():
            return True
        legacy_thread = self.update_thread
        if legacy_thread is not None and hasattr(legacy_thread, "isRunning"):
            try:
                return bool(legacy_thread.isRunning())
            except Exception:
                return False
        return False

    def _run_data_update_worker(self) -> None:
        try:
            success, total, errors = self.data_service.update_pool(
                list(self.config.etf_pool),
                progress_cb=self.on_update_progress,
            )
        except Exception as exc:
            success, total, errors = 0, len(self.config.etf_pool), [str(exc)]
        finally:
            with self._update_lock:
                self._update_thread = None
                self.update_thread = None
        self.on_update_finished(success, total, errors)

    def update_data_sync(self) -> Tuple[int, int, list[str]]:
        """Synchronously update ETF data."""
        self.logger_fn(f"🔄 同步更新 {len(self.config.etf_pool)} 只ETF数据...")
        success, total, errors = self.data_service.update_pool(self.config.etf_pool)
        if errors:
            for error in errors:
                self.logger_fn(f"  ✗ {error}")
        data_version = self.get_data_version_audit().get("data_version", "")
        self.logger_fn(f"✅ 数据更新完成 ({success}/{total}) | data_version={data_version or 'unknown'}")
        return success, total, errors

    def is_data_fresh(self) -> bool:
        """Check whether all ETF pool data includes today's bars."""
        return self.data_service.is_pool_fresh(self.config.etf_pool)

    def on_update_progress(self, current, total, code, message) -> None:
        """Log asynchronous data update progress."""
        self.logger_fn(f"  [{current}/{total}] {self.code_name_fn(code)}: {message}")

    def on_update_finished(self, success, total, errors) -> None:
        """Handle asynchronous data update completion."""
        if errors:
            for error in errors:
                self.logger_fn(f"  ✗ {error}")
        data_audit = self.get_data_version_audit()
        data_version = str(data_audit.get("data_version", "") or "")
        self.logger_fn(f"✅ ETF数据更新完成 ({success}/{total}) | data_version={data_version or 'unknown'}")
        self.status_fn(f"数据更新完成 ({success}/{total})")
        self._publish_event(
            "data_update_completed",
            "ETF rotation data update completed",
            payload={"success": success, "total": total, "errors": list(errors or []), "data_version": data_version, "data_audit": data_audit},
        )

        update_ok = not errors and int(success or 0) >= int(total or 0)

        if self.update_schedule_context:
            data_status = "completed" if update_ok else "failed"
            self.state_mgr.mark_auto_data_task(
                status=data_status,
                schedule_time=str(self.update_schedule_context.get("schedule_time", "") or ""),
                trigger=str(self.update_schedule_context.get("trigger", "") or ""),
                task_date=str(self.update_schedule_context.get("task_date", "") or ""),
                error="; ".join(str(error) for error in (errors or [])),
            )

        if not update_ok:
            error_msg = "; ".join(str(error) for error in (errors or [])) or "ETF数据更新失败"
            self.logger_fn(f"⛔ 数据未就绪，已停止本次信号检查: {error_msg}")
            self.status_fn("数据未就绪，已停止信号检查")
            self.update_pending_signal_check = False
            self.update_schedule_context = None
            return

        if self.update_pending_signal_check:
            self.update_pending_signal_check = False
            self.logger_fn("⏰ 数据已更新，开始信号检查...")
            signal_context = None
            if self.update_schedule_context:
                signal_context = {
                    "trigger": str(self.update_schedule_context.get("trigger", "") or ""),
                    "task_date": str(self.update_schedule_context.get("task_date", "") or ""),
                    "schedule_time": str(self.config.check_time or ""),
                }
            self.run_signal_check(schedule_context=signal_context)
        self.update_schedule_context = None

    @staticmethod
    def hm_to_minutes(hm: str) -> int:
        """Convert 'HH:MM' to minutes since midnight."""
        hour, minute = map(int, hm.split(":"))
        return hour * 60 + minute

    def on_auto_timer(self) -> None:
        """Poll automatic schedule and trigger data update or signal check once per day."""
        now = self.now_fn()
        if not self.trading_day_fn(now.date()):
            return

        try:
            trading_end_minutes = self.hm_to_minutes(self.config.trading_end)
        except Exception:
            trading_end_minutes = self.hm_to_minutes("14:57")
        now_minutes = now.hour * 60 + now.minute
        if now_minutes > trading_end_minutes:
            return

        today = now.strftime("%Y-%m-%d")
        data_completed_today = (
            self.auto_data_done_date == today
            or self.state_mgr.is_auto_data_task_completed(
                task_date=today,
                schedule_time=self.config.data_update_time,
                trigger="scheduled",
            )
        )
        signal_completed_today = (
            self.auto_signal_done_date == today
            or self.state_mgr.is_auto_signal_task_completed(
                task_date=today,
                schedule_time=self.config.check_time,
                trigger="scheduled",
            )
        )

        data_target = self.hm_to_minutes(self.config.data_update_time)
        if now_minutes >= data_target and not data_completed_today:
            update_running = self.is_update_running()
            if not update_running:
                self.auto_data_done_date = today
                self.logger_fn(f"⏰ 定时触发数据更新 ({self.config.data_update_time})")
                self.update_data(
                    run_signal_check_after=False,
                    schedule_context={
                        "trigger": "scheduled",
                        "task_date": today,
                        "schedule_time": self.config.data_update_time,
                    },
                )
                return
            self.logger_fn("⏰ 定时数据更新已到点，但已有更新任务正在运行")
            self.status_fn("ETF数据更新进行中，定时更新等待当前任务完成")

        signal_target = self.hm_to_minutes(self.config.check_time)
        if now_minutes >= signal_target and not signal_completed_today and bool(getattr(self.config, "auto_signal_enabled", True)):
            if self.state_mgr.is_auto_signal_task_completed(
                task_date=today,
                schedule_time=self.config.check_time,
                trigger="scheduled",
            ):
                self.auto_signal_done_date = today
                return

            self.auto_signal_done_date = today

            if not self.is_data_fresh():
                self.logger_fn("⏰ 数据尚未更新，先更新数据再检查信号...")
                self.update_data(
                    run_signal_check_after=True,
                    schedule_context={
                        "trigger": "scheduled",
                        "task_date": today,
                        "schedule_time": self.config.data_update_time,
                    },
                )
            else:
                self.logger_fn(f"⏰ 定时触发信号检查 ({self.config.check_time})")
                self.run_signal_check(
                    schedule_context={
                        "trigger": "scheduled",
                        "task_date": today,
                        "schedule_time": self.config.check_time,
                    },
                )
