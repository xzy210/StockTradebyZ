"""
ETF轮动实盘 - 核心轮动引擎

将策略信号计算、风控检查、交易执行、状态管理、通知推送串联起来。
支持手动触发和定时自动生成信号两种模式。
"""
import logging
import threading
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, List

from PyQt6.QtCore import QObject, pyqtSignal

from common.events import EventBus
from common.execution_contract import OrderIntent, RebalanceIntent, StrategySignal

from .config import RotationConfig, ConfigManager
from .state_manager import RotationState, StateManager
from .strategy_provider import DefaultStrategyProvider
from .reconciler import StartupReconciler
from .rotation_execution_service import RotationExecutionService
from .rotation_guard_service import RotationGuardService
from .rotation_ledger_service import RotationLedgerService
from .rotation_risk_policy import ETFRotationRiskPolicy
from .rotation_data_service import RotationDataService
from .rotation_runtime_service import RotationRuntimeService
from .rotation_signal_service import RotationDecisionService, RotationSignalService
from .rotation_status_service import RotationStatusService
from .trade_executor import BrokerReadOnlyExecutor, TradeExecutor, SimulatedExecutor
from .notifier import RotationNotifier
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service

logger = logging.getLogger(__name__)


class _RuntimeAutoTimerFacade:
    """Compatibility facade for old widgets that inspect engine._auto_timer."""

    def __init__(self, runtime_provider):
        self._runtime_provider = runtime_provider

    def isActive(self) -> bool:  # noqa: N802 - Qt compatibility
        runtime = self._runtime_provider()
        return bool(getattr(runtime, "_auto_running", False)) if runtime is not None else False

    def start(self, interval: int) -> None:
        runtime = self._runtime_provider()
        if runtime is not None:
            runtime.auto_check_interval = int(interval or runtime.auto_check_interval)
            runtime.start_auto()

    def stop(self) -> None:
        runtime = self._runtime_provider()
        if runtime is not None:
            runtime.stop_auto()


class RotationEngine(QObject):
    """
    ETF轮动实盘引擎

    信号:
        signal_generated: 产生信号 (signal_type, detail_dict)
        trade_executed: 交易已执行 (success, detail_dict)
        status_updated: 状态更新 (status_text)
        log_message: 日志消息 (message)
        scores_updated: 得分更新 (scores_dict)
    """

    signal_generated = pyqtSignal(str, dict)
    trade_executed = pyqtSignal(bool, dict)
    status_updated = pyqtSignal(str)
    log_message = pyqtSignal(str)
    scores_updated = pyqtSignal(dict)

    def __init__(self, config: Optional[RotationConfig] = None,
                 executor: Optional[TradeExecutor] = None,
                 strategy_provider=None,
                 event_bus: Optional[EventBus] = None,
                 parent=None):
        super().__init__(parent)
        self.event_bus = event_bus

        # 配置与状态
        self.config_mgr = ConfigManager()
        self.config = config or self.config_mgr.load()

        strategy_spec = get_strategy_spec_service().etf_rotation()
        strategy_id = (self.config.strategy_id or strategy_spec.strategy_id or "etf_rotation").strip() or "etf_rotation"
        self.state_mgr = StateManager(
            strategy_id=strategy_id,
            strategy_name=strategy_spec.strategy_name,
            virtual_account_id=strategy_spec.virtual_account_id if strategy_id == strategy_spec.strategy_id else f"va_{strategy_id}",
        )
        self.state = self.state_mgr.state

        # 组件
        # 注意：策略级风控统一由 ETFRotationRiskPolicy + StrategyRiskRegistry 承担，
        #   实盘委托必须走 TradeExecutionService 统一网关；executor 仅提供只读券商/行情上下文。
        #   显式注入 SimulatedExecutor 时，仅用于本地调试或测试。
        self.executor: TradeExecutor = executor or BrokerReadOnlyExecutor()
        if executor is None and hasattr(self.executor, "set_broker_session_service"):
            self.executor.set_broker_session_service()
        self.strategy_provider = strategy_provider or DefaultStrategyProvider()
        self.reconciler = StartupReconciler()
        self.notifier = RotationNotifier()

        # ETF名称映射
        self._etf_name_map: Dict[str, str] = {}
        self._load_etf_names()

        # 自动调度由 RotationRuntimeService 使用 threading.Timer 管理。

        # 共享 DataPortal 数据目录（显式传入旧目录时可作为 overlay）
        self.data_service = RotationDataService()
        self._data_dir = self.data_service.data_dir

        # 信号计算与调仓决策服务
        self.signal_service = RotationSignalService(
            config=self.config,
            data_dir=self._data_dir,
            strategy_provider=self.strategy_provider,
            data_service=self.data_service,
            logger_fn=self._log,
            code_name_fn=self._code_name,
        )
        self.decision_service = RotationDecisionService(
            config=self.config,
            state=self.state,
            code_name_fn=self._code_name,
        )
        self.guard_service = RotationGuardService(
            config=self.config,
            state=self.state,
            state_saver=self.state_mgr.save,
            total_asset_fn=self._get_total_asset,
            current_price_fn=lambda code: self.executor.get_current_price(code),
            logger_fn=self._log,
            code_name_fn=self._code_name,
        )
        self.ledger_service = RotationLedgerService(
            config=self.config,
            state=self.state,
            state_mgr=self.state_mgr,
            executor=self.executor,
            strategy_identity_fn=self._etf_strategy_identity,
            code_name_map_fn=lambda code: self._etf_name_map.get(code, ""),
            logger_fn=self._log,
        )
        self.execution_service = RotationExecutionService(
            config=self.config,
            state=self.state,
            state_mgr=self.state_mgr,
            executor=self.executor,
            ledger_service=self.ledger_service,
            trade_event_fn=self._on_execution_trade_event,
            logger_fn=self._log,
            code_name_fn=self._code_name,
            code_name_map_fn=lambda code: self._etf_name_map.get(code, ""),
        )
        self.runtime_service = RotationRuntimeService(
            config=self.config,
            state=self.state,
            state_mgr=self.state_mgr,
            config_mgr=self.config_mgr,
            executor=self.executor,
            data_dir=self._data_dir,
            data_service=self.data_service,
            signal_service=self.signal_service,
            decision_service=self.decision_service,
            guard_service=self.guard_service,
            ledger_service=self.ledger_service,
            update_parent=self,
            logger_fn=self._log,
            status_fn=self.status_updated.emit,
            signal_fn=self.signal_generated.emit,
            scores_fn=self.scores_updated.emit,
            notify_signal_fn=self.notifier.send_signal,
            execute_rebalance_fn=self.execute_live_rebalance_intent,
            code_name_fn=self._code_name,
            event_bus=self.event_bus,
        )
        self._auto_timer = _RuntimeAutoTimerFacade(lambda: getattr(self, "runtime_service", None))
        self.status_service = RotationStatusService(
            config=self.config,
            state=self.state,
            executor=self.executor,
            ledger_service=self.ledger_service,
            data_dir=self._data_dir,
            data_fresh_fn=self.is_data_fresh,
        )

        # 专用资金初始化（真实账户首次启动时写入账本）
        self._init_dedicated_capital()

        # 策略级风控 policy 注册到统一网关
        self._strategy_risk_policy: Optional[ETFRotationRiskPolicy] = None
        self._register_strategy_risk_policy()

    # ======================================================================
    #  公开 API
    # ======================================================================

    def update_config(self, config: RotationConfig):
        """更新配置"""
        self.config = config
        # policy 通过 lambda late-binding 读取 self.config，无需显式推送
        self.config_mgr.save(config)
        self._refresh_service_contexts(reset_strategy=True)
        self.notifier.etf_name_map = self._etf_name_map
        self._log("配置已更新")

    def set_executor(self, executor: TradeExecutor):
        """设置 ETF 轮动只读执行上下文。"""
        self.executor = executor
        self._refresh_service_contexts()
        self._log(f"ETF 轮动执行上下文已设置: {type(executor).__name__}")
        self._run_startup_reconcile()
        self._init_dedicated_capital()

    def set_event_bus(self, event_bus: Optional[EventBus]) -> None:
        """Attach or replace the shell EventBus used by runtime events."""
        self.event_bus = event_bus
        if hasattr(self, "runtime_service"):
            self.runtime_service.set_event_bus(event_bus)

    # ------------------------------------------------------------------
    #  策略级风控 Policy（注册到统一网关）
    # ------------------------------------------------------------------

    def _register_strategy_risk_policy(self) -> None:
        """Register this engine's risk rules with the unified gateway registry.

        使用 override=True 保证同一 strategy_id 多次初始化（比如测试或热重载）
        不会累积出多个同类 policy。provider 采用 late-binding，因此 update_config
        / state_mgr 重新加载后 policy 能自动读到最新对象。
        """
        try:
            from trading_app.services.strategy.strategy_risk import get_strategy_risk_registry

            strategy_id, _, _ = self._etf_strategy_identity()
            policy = ETFRotationRiskPolicy(
                strategy_id=strategy_id,
                config_provider=lambda: self.config,
                state_provider=lambda: self.state,
                config_saver=self._apply_risk_policy_values,
            )
            get_strategy_risk_registry().register(policy, override=True)
            self._strategy_risk_policy = policy
            logger.info("ETF 轮动实盘 policy 已注册到统一风控 registry: strategy_id=%s", strategy_id)
        except Exception as exc:
            logger.error("注册 ETF 轮动实盘 policy 失败: %s", exc, exc_info=True)
            self._strategy_risk_policy = None

    def _apply_risk_policy_values(self, values: Dict[str, object]) -> None:
        """策略风控面板保存回调：把 UI 提交的字段写回 RotationConfig 并落盘。

        面板负责把控件显示值还原到存储单位（例如 ``15.0%`` → ``15.0``），此处
        只管透传到 :class:`RotationConfig` 对应字段、触发一次
        ``update_config``（会 persist 到 ``rotation_config.json`` 且通知引擎
        重建策略 / 通知器）。
        """
        if not values:
            return
        try:
            cfg = self.config
            for key, value in values.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            self.update_config(cfg)
            self._log("📋 策略风控参数已更新并保存")
        except Exception as exc:
            logger.error("保存策略风控参数失败: %s", exc, exc_info=True)
            raise

    def unregister_strategy_risk_policy(self) -> None:
        """Remove the policy from the registry (call on shutdown / teardown)."""
        if self._strategy_risk_policy is None:
            return
        try:
            from trading_app.services.strategy.strategy_risk import get_strategy_risk_registry

            get_strategy_risk_registry().unregister(
                self._strategy_risk_policy.strategy_id,
                self._strategy_risk_policy,
            )
            logger.info(
                "ETF 轮动实盘 policy 已从风控 registry 卸载: strategy_id=%s",
                self._strategy_risk_policy.strategy_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("卸载 ETF 轮动实盘 policy 失败: %s", exc)
        self._strategy_risk_policy = None

    def _preflight_strategy_risk_policy(
        self,
        *,
        is_sell: bool,
        current_price: float = 0.0,
    ) -> Tuple[bool, str]:
        """下单前统一触发策略级风控 policy。

        - 实盘订单会走 :class:`TradeExecutionService` 统一网关，policy 在那边触发。
        - 显式注入 ``SimulatedExecutor`` 的本地调试/测试场景不经过统一网关，
          这里显式调用 ``StrategyRiskRegistry`` 作为兜底。

        ``warn`` 级 decision 按既有 "允许止损卖出" 语义放行。
        """
        if not isinstance(self.executor, SimulatedExecutor):
            return True, ""

        try:
            from trading_app.services.strategy.strategy_risk import (
                StrategyRiskContext,
                get_strategy_risk_registry,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("策略风控模块不可用，跳过预检: %s", exc)
            return True, ""

        strategy_id, _, _ = self._etf_strategy_identity()
        registry = get_strategy_risk_registry()
        if not registry.has(strategy_id):
            return True, ""

        fake_request = SimpleNamespace(
            strategy_id=strategy_id,
            order_type=24 if is_sell else 23,
            price=float(current_price or 0.0),
        )
        decision = registry.evaluate(fake_request, StrategyRiskContext())
        if not decision.passed:
            return False, decision.reason or "策略风控未通过"
        return True, decision.reason or "策略风控通过"

    def _run_startup_reconcile(self):
        if isinstance(self.executor, SimulatedExecutor):
            return
        if not self.executor.is_connected():
            return
        try:
            result = self.reconciler.reconcile(self)
            self._log(f"启动对账完成: {result}")
        except Exception as exc:
            logger.error(f"启动对账失败: {exc}")

    def _check_live_market_data_ready(self, *, require_minute_freshness: bool = False) -> tuple[bool, str]:
        """检查行情数据是否就绪（委托给运行编排服务）。"""
        return self.runtime_service.check_live_market_data_ready(
            require_minute_freshness=require_minute_freshness,
        )

    def run_signal_check(self, schedule_context: Optional[dict] = None) -> dict:
        """Run one signal check and return pure strategy signals."""
        self._refresh_service_contexts()
        return self.runtime_service.run_signal_check(schedule_context=schedule_context)

    def generate_live_signals(self, payload: Optional[dict] = None) -> list[StrategySignal]:
        """Generate unified live StrategySignal outputs without submitting orders."""
        result = self.run_signal_check(schedule_context=dict(payload or {}).get("schedule_context"))
        signals = []
        for item in list(result.get("strategy_signals", []) or []):
            if isinstance(item, StrategySignal):
                signals.append(item)
            elif isinstance(item, dict):
                signals.append(StrategySignal(**item))
        return signals

    def generate_live_rebalance_intent(self, payload: Optional[dict] = None) -> Optional[RebalanceIntent]:
        """Generate a native RebalanceIntent output without submitting orders."""
        result = self.run_signal_check(schedule_context=dict(payload or {}).get("schedule_context"))
        item = result.get("rebalance_intent") if isinstance(result, dict) else None
        if isinstance(item, RebalanceIntent):
            return item
        if not isinstance(item, dict):
            return None
        target_payload = item.get("target_portfolio")
        if not isinstance(target_payload, dict):
            return None
        from common.execution_contract import TargetPortfolio

        target_portfolio = TargetPortfolio(**target_payload)
        order_intents = []
        for raw_intent in list(item.get("order_intents", []) or []):
            if isinstance(raw_intent, OrderIntent):
                order_intents.append(raw_intent)
            elif isinstance(raw_intent, dict):
                order_intents.append(OrderIntent(**raw_intent))
        return RebalanceIntent(
            target_portfolio=target_portfolio,
            order_intents=tuple(order_intents),
            current_positions=dict(item.get("current_positions", {}) or {}),
            prices=dict(item.get("prices", {}) or {}),
            total_asset=float(item.get("total_asset", 0.0) or 0.0),
            available_cash=float(item.get("available_cash", 0.0) or 0.0),
            intent_id=str(item.get("intent_id", "") or ""),
            reason=str(item.get("reason", "") or ""),
            timestamp=item.get("timestamp"),
            metadata=dict(item.get("metadata", {}) or {}),
            schema_version=str(item.get("schema_version", "rebalance_intent.v1") or "rebalance_intent.v1"),
        )

    def generate_live_order_intents(self, payload: Optional[dict] = None) -> list[OrderIntent]:
        """Generate native OrderIntent outputs without submitting orders."""
        rebalance_intent = self.generate_live_rebalance_intent(payload)
        if rebalance_intent is None:
            return []
        return list(rebalance_intent.order_intents or ())

    def execute_live_rebalance_intent(self, rebalance_intent: RebalanceIntent, *, execution_service=None, stock_name_map: Optional[dict[str, str]] = None):
        """Execute ETF RebalanceIntent through the unified live gateway and apply reports."""
        if rebalance_intent is None or not rebalance_intent.order_intents:
            return []
        service = self._resolve_live_execution_service(execution_service)
        reports = list(service.execute_rebalance_intent(rebalance_intent, stock_name_map=stock_name_map or self._etf_name_map))
        reason = str(rebalance_intent.reason or rebalance_intent.target_portfolio.reason or "")
        scores = dict(getattr(self.state, "last_scores", {}) or {})
        self.execution_service.apply_execution_reports(reports, scores=scores, reason=reason)
        return reports

    def execute_live_order_intents(self, intents: list[OrderIntent], *, execution_service=None, stock_name_map: Optional[dict[str, str]] = None):
        """Execute ETF OrderIntent outputs through the unified live gateway and apply reports."""
        normalized = list(intents or [])
        if not normalized:
            return []
        service = self._resolve_live_execution_service(execution_service)
        reports = list(service.execute_order_intents(normalized, stock_name_map=stock_name_map or self._etf_name_map))
        reason = str(normalized[-1].reason or "")
        scores = dict(getattr(self.state, "last_scores", {}) or {})
        self.execution_service.apply_execution_reports(reports, scores=scores, reason=reason)
        return reports

    def execute_live_signals(self, signals: list[StrategySignal], *, execution_service=None, stock_name_map: Optional[dict[str, str]] = None):
        """Execute ETF StrategySignal outputs through the unified live gateway and apply reports."""
        normalized = list(signals or [])
        if not normalized:
            return []
        service = self._resolve_live_execution_service(execution_service)
        reports = list(service.execute_signals(normalized, stock_name_map=stock_name_map or self._etf_name_map))
        reason = str(normalized[-1].reason or "")
        scores = dict(getattr(self.state, "last_scores", {}) or {})
        self.execution_service.apply_execution_reports(reports, scores=scores, reason=reason)
        return reports

    def execute_manual(
        self,
        action: str,
        code: str,
        quantity: int = 0,
        amount: float = 0.0,
        price: Optional[float] = None,
    ) -> dict:
        """
        手动执行交易

        Args:
            action: "BUY" / "SELL"
            code: ETF代码
            quantity: 卖出数量（BUY时可为0，用amount计算）
            amount: 买入金额
        """
        result = {'success': False, 'message': ''}

        ready, reason = self._check_live_market_data_ready()
        if not ready:
            result['message'] = f"行情数据未就绪: {reason}"
            self._log(f"⛔ 手动委托已阻断: {result['message']}")
            self.status_updated.emit(result['message'])
            return result

        # 风控检查（仅模拟盘；真实盘由统一网关触发）
        ok, msg = self._preflight_strategy_risk_policy(
            is_sell=(action != "BUY"),
        )
        if not ok:
            result['message'] = f"风控拦截: {msg}"
            self._log(f"⚠ {result['message']}")
            return result

        if action == "BUY":
            return self._execute_manual_signal(
                action="BUY",
                code=code,
                quantity=quantity,
                amount=amount,
                reason="手动买入",
                price=price,
            )
        elif action == "SELL":
            return self._execute_manual_signal(
                action="SELL",
                code=code,
                quantity=quantity,
                amount=amount,
                reason="手动卖出",
                price=price,
            )
        else:
            result['message'] = f"未知操作: {action}"
            return result

    def start_auto(self):
        """启动自动调度（委托给运行编排服务）。"""
        return self.runtime_service.start_auto()

    def stop_auto(self):
        """停止自动调度（委托给运行编排服务）。"""
        return self.runtime_service.stop_auto()

    # ------------------------------------------------------------------
    #  数据更新
    # ------------------------------------------------------------------

    def update_data(self, run_signal_check_after: bool = False, schedule_context: Optional[dict] = None):
        """Start background ETF data update and optionally run a signal check afterward."""
        return self.runtime_service.update_data(
            run_signal_check_after=run_signal_check_after,
            schedule_context=schedule_context,
        )

    def update_data_sync(self) -> Tuple[int, int, List[str]]:
        """同步更新ETF数据（委托给运行编排服务）。"""
        return self.runtime_service.update_data_sync()

    def is_data_fresh(self) -> bool:
        """检查ETF池数据是否都已包含今天的K线（委托给运行编排服务）。"""
        return self.runtime_service.is_data_fresh()

    def _on_update_progress(self, current, total, code, message):
        return self.runtime_service.on_update_progress(current, total, code, message)

    def _on_update_finished(self, success, total, errors):
        return self.runtime_service.on_update_finished(success, total, errors)

    def get_status_summary(self) -> dict:
        """获取当前状态摘要（委托给状态摘要服务）。"""
        return self.status_service.get_status_summary()

    def get_statistics(self) -> dict:
        """计算实盘收益统计指标（委托给状态摘要服务）。"""
        return self.status_service.get_statistics()

    # ------------------------------------------------------------------
    #  分析数据记录辅助方法
    # ------------------------------------------------------------------

    def _add_capital_entry(self, action: str, code: str = "", name: str = "",
                           amount: float = 0.0, commission: float = 0.0,
                           fee_source: str = ""):
        """向资金流水账本追加一条记录（委托给账本服务）。"""
        return self.ledger_service.add_capital_entry(
            action, code, name,
            amount=amount,
            commission=commission,
            fee_source=fee_source,
        )

    def _record_daily_equity(self):
        """记录当日净值快照（委托给账本服务）。"""
        return self.ledger_service.record_daily_equity()

    def _add_order_record(self, order_id: int, action: str, code: str,
                          ordered_qty: int, ordered_price: float,
                          reason: str = ""):
        """创建并保存委托记录（委托给账本服务）。"""
        return self.ledger_service.add_order_record(
            order_id, action, code, ordered_qty, ordered_price, reason
        )

    def _update_order_record(self, order_id: int, fill: dict, pnl: float = 0.0):
        """根据成交结果更新委托记录（委托给账本服务）。"""
        return self.ledger_service.update_order_record(order_id, fill, pnl=pnl)

    def _resolve_trade_fees(
        self,
        *,
        direction: str,
        amount: float,
        stock_code: str,
        actual_commission: float = -1.0,
    ) -> dict:
        """统一读取手续费配置（委托给账本服务）。"""
        return self.ledger_service.resolve_trade_fees(
            direction=direction,
            amount=amount,
            stock_code=stock_code,
            actual_commission=actual_commission,
        )

    # ======================================================================
    #  策略计算
    # ======================================================================

    def _get_strategy(self):
        """延迟创建策略实例（委托给信号计算服务）。"""
        return self.signal_service.get_strategy()

    def _calculate_scores(self) -> Dict[str, float]:
        """加载数据并计算所有ETF的综合动量得分。"""
        self._refresh_service_contexts()
        return self.signal_service.calculate_scores()

    def _make_decision(self, scores: Dict[str, float]) -> Tuple[str, Optional[str], str]:
        """基于得分做出调仓决策。"""
        self._refresh_service_contexts()
        return self.decision_service.make_decision(scores)

    # ======================================================================
    #  交易执行
    # ======================================================================

    def _resolve_live_execution_service(self, execution_service=None):
        """Return the only live order gateway for ETF rotation."""
        if execution_service is not None:
            return execution_service
        if isinstance(self.executor, SimulatedExecutor):
            raise RuntimeError(
                "SimulatedExecutor 仅允许配合显式注入的测试执行服务；"
                "ETF 轮动实盘委托必须通过 TradeExecutionService 统一入口。"
            )
        from trading_app.services.execution.trade_execution_service import get_trade_execution_service
        return get_trade_execution_service()

    def _on_execution_trade_event(self, success: bool, result: dict) -> None:
        """Handle trade events emitted by RotationExecutionService."""
        self.trade_executed.emit(success, result)

        if not self.config.notify_on_trade:
            return

        action = str(result.get('action') or '')
        code = str(result.get('code') or '')
        quantity = int(result.get('quantity') or 0)
        price = float(result.get('price') or 0.0)
        message = str(result.get('message') or '')
        reason = str(result.get('reason') or '')
        action_name = "买入" if action == "BUY" else "卖出" if action == "SELL" else action
        self.notifier.send_trade_result(
            action_name, code, quantity, price, success, message, reason
        )

    def _on_partial_switch_stop(
        self,
        sell_result: dict,
        remaining: int,
        message: str,
        reason: str,
    ) -> None:
        """Handle partial sell during SWITCH and keep old UI/notification behavior."""
        self.status_updated.emit(message)
        if self.config.notify_on_trade:
            self.notifier.send_trade_result(
                "卖出(部分成交-切换中止)",
                self.state.current_holding or "",
                remaining,
                sell_result.get('price', 0),
                False,
                message,
                reason,
            )

    def _ensure_sim_price(self, code: str) -> float:
        """
        确保模拟执行器持有最新价格。
        - 若已有价格（>0），直接返回。
        - 否则从本地 parquet 读最新收盘价并注入执行器。
        对真实执行器直接返回其报价（不做额外操作）。
        """
        if not isinstance(self.executor, SimulatedExecutor):
            return self.executor.get_current_price(code)

        price = self.executor.get_current_price(code)
        if price > 0:
            return price

        try:
            last_close = self.data_service.latest_close(code)
            if last_close > 0:
                self.executor.set_prices({code: last_close})
                self._log(
                    f"[模拟] {self._code_name(code)} "
                    f"价格从数据服务读取: {last_close:.3f}"
                )
                return last_close
        except Exception as e:
            logger.warning(f"读取 {code} 价格失败: {e}")
        return 0.0

    def _confirm_fill(self, order_id: int,
                      expected_qty: int, expected_price: float,
                      timeout_secs: float = 5.0) -> dict:
        """
        在后台 daemon 线程轮询 miniQMT，避免阻塞调用线程。
        模拟器或不支持查询时直接返回 commission=-1（调用方按配置估算）。

        Returns: 与 TradeExecutor.query_order_fill 相同的 dict
        """
        if isinstance(self.executor, SimulatedExecutor):
            return {
                'filled': True,
                'filled_qty': expected_qty,
                'filled_price': expected_price,
                'commission': -1.0,
                'timed_out': False,
            }

        self._log(f"⏳ 查询委托 #{order_id} 成交情况（最长 {timeout_secs:.0f} 秒）...")

        fill_result: list = [None]

        def _poll():
            fill_result[0] = self.executor.query_order_fill(
                order_id, timeout_secs
            )

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        t.join(timeout_secs + 1)

        info = fill_result[0]
        if info is None:
            self._log("⚠ 成交查询超时，回退到估算值")
            return {
                'filled': True,
                'filled_qty': expected_qty,
                'filled_price': expected_price,
                'commission': -1.0,
                'timed_out': True,
            }

        if info.get('timed_out'):
            self._log(
                f"⚠ 委托 #{order_id} 查询超时，"
                f"已知成交量: {info.get('filled_qty', 0)} 股"
            )
        return info

    def _execute_manual_signal(
        self,
        *,
        action: str,
        code: str,
        quantity: int,
        amount: float,
        reason: str,
        price: Optional[float] = None,
    ) -> dict:
        """Execute a manual ETF order through the unified StrategySignal gateway."""
        self._refresh_service_contexts()
        strategy_id, strategy_name, virtual_account_id = self._etf_strategy_identity()
        resolved_price = float(price or self._ensure_sim_price(code) or 0.0)
        if resolved_price <= 0:
            return {
                "success": False,
                "action": action,
                "code": code,
                "message": f"无法获取 {self._code_name(code)} 有效价格",
            }

        metadata = {
            "virtual_account_id": virtual_account_id,
            "source": "etf_rotation",
            "trigger": "manual",
            "owner_type": "etf_rotation",
            "rotation_signal": f"MANUAL_{action}",
            "quantity_mode": "delta",
        }
        target_quantity = 0
        if action == "BUY":
            target_quantity = int(quantity or 0) // 100 * 100
            if target_quantity > 0:
                buy_amount = target_quantity * resolved_price
                if buy_amount < float(self.config.min_trade_amount or 0.0):
                    return {
                        "success": False,
                        "action": "BUY",
                        "code": code,
                        "message": f"金额过小 ({buy_amount:.2f})",
                    }
            else:
                buy_amount = float(amount or 0.0) * float(self.config.cash_ratio or 1.0)
                if buy_amount < float(self.config.min_trade_amount or 0.0):
                    return {
                        "success": False,
                        "action": "BUY",
                        "code": code,
                        "message": f"金额过小 ({buy_amount:.2f})",
                    }
                target_quantity = int(buy_amount / resolved_price / 100) * 100
                if target_quantity <= 0:
                    return {
                        "success": False,
                        "action": "BUY",
                        "code": code,
                        "message": f"资金不足，最低需要 {resolved_price * 100:.2f} 元",
                    }
            signal_action = "buy"
        else:
            target_quantity = int(quantity or 0) // 100 * 100
            if target_quantity <= 0:
                return {
                    "success": False,
                    "action": "SELL",
                    "code": code,
                    "message": "卖出数量必须为100股的整数倍",
                }
            signal_action = "sell"

        signal = StrategySignal(
            symbol=code,
            action=signal_action,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            target_quantity=target_quantity,
            price=resolved_price,
            reason=reason,
            metadata=metadata,
        )
        reports = self.execute_live_signals([signal], stock_name_map=self._etf_name_map)
        if not reports:
            return {
                "success": False,
                "action": action,
                "code": code,
                "message": "统一执行网关未生成有效委托",
            }
        report = reports[-1]
        return {
            "success": bool(report.accepted),
            "action": action,
            "code": code,
            "message": report.message,
            "order_id": int(report.order_id or 0) if str(report.order_id or "").isdigit() else -1,
            "price": resolved_price,
            "quantity": target_quantity,
            "reason": reason,
            "submitted": bool(report.submitted or report.accepted),
        }

    def _etf_strategy_identity(self) -> Tuple[str, str, str]:
        spec = get_strategy_spec_service().etf_rotation()
        strategy_id = (self.config.strategy_id or spec.strategy_id or "etf_rotation").strip() or "etf_rotation"
        virtual_account_id = spec.virtual_account_id if strategy_id == spec.strategy_id else f"va_{strategy_id}"
        return strategy_id, spec.strategy_name, virtual_account_id

    def _sync_unified_ledger_on_buy(
        self,
        *,
        code: str,
        name: str,
        price: float,
        volume: int,
        commission: float,
        stamp_tax: float,
        transfer_fee: float,
        broker_order_id: int,
        reason: str,
    ) -> None:
        """同步买入成交到统一账本（委托给账本服务）。"""
        return self.ledger_service.sync_unified_ledger_on_buy(
            code=code,
            name=name,
            price=price,
            volume=volume,
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            broker_order_id=broker_order_id,
            reason=reason,
        )

    def _sync_unified_ledger_on_sell(
        self,
        *,
        code: str,
        name: str,
        price: float,
        volume: int,
        commission: float,
        stamp_tax: float,
        transfer_fee: float,
        broker_order_id: int,
        reason: str,
    ) -> None:
        """同步卖出成交到统一账本（委托给账本服务）。"""
        return self.ledger_service.sync_unified_ledger_on_sell(
            code=code,
            name=name,
            price=price,
            volume=volume,
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            broker_order_id=broker_order_id,
            reason=reason,
        )

    def _get_available_cash(self) -> float:
        """获取策略可用资金（委托给账本服务）。"""
        return self.ledger_service.available_cash()

    def _ledger_available_cash(self) -> float:
        """从主账本读取本策略当前可用现金（委托给账本服务）。"""
        return self.ledger_service.ledger_available_cash()

    def _init_dedicated_capital(self):
        """初始化专用资金主账本（委托给账本服务）。"""
        return self.ledger_service.init_dedicated_capital()

    def reset_dedicated_capital(self, new_capital: Optional[float] = None):
        """重置本策略启动资金（委托给账本服务）。"""
        return self.ledger_service.reset_dedicated_capital(new_capital)

    def clear_analytics_data(self):
        """清空历史分析数据（委托给账本服务）。"""
        return self.ledger_service.clear_analytics_data()

    # ======================================================================
    #  风控检查（调仓周期 / 移动止盈 / 账户回撤保护）
    # ======================================================================

    def _in_drawdown_cooldown(self) -> bool:
        """检查是否处于账户回撤保护冷却期（委托给 guard 服务）。"""
        self._refresh_service_contexts()
        return self.guard_service.in_drawdown_cooldown()

    def _check_drawdown_protection(self) -> tuple:
        """检查账户最大回撤保护（委托给 guard 服务）。"""
        self._refresh_service_contexts()
        triggered, result = self.guard_service.check_drawdown_protection()
        if not triggered:
            return triggered, result

        signal = str(result.get('signal') or 'DRAWDOWN_STOP')
        reason = str(result.get('reason') or '')
        self.signal_generated.emit(signal, result)
        self.status_updated.emit(reason)

        if self.config.notify_on_signal:
            self.notifier.send_signal(
                signal, {}, self.state.current_holding, None, reason
            )

        return triggered, result

    def _check_trailing_stop(self) -> tuple:
        """检查移动止盈（委托给 guard 服务）。"""
        self._refresh_service_contexts()
        triggered, result = self.guard_service.check_trailing_stop()
        if not triggered:
            return triggered, result

        signal = str(result.get('signal') or 'TRAILING_STOP')
        reason = str(result.get('reason') or '')
        self.signal_generated.emit(signal, result)
        self.status_updated.emit(reason)

        if self.config.notify_on_signal:
            self.notifier.send_signal(
                signal, {}, self.state.current_holding, None, reason
            )

        return triggered, result

    def _is_rebalance_day(self) -> bool:
        """检查今天是否为调仓日（委托给 guard 服务）。"""
        self._refresh_service_contexts()
        return self.guard_service.is_rebalance_day()

    def _update_check_count(self):
        """更新信号检查计数（委托给 guard 服务）。"""
        self._refresh_service_contexts()
        self.guard_service.update_check_count()

    def _get_total_asset(self) -> float:
        """计算策略总资产（委托给账本服务）。"""
        return self.ledger_service.total_asset()

    # ======================================================================
    #  自动调度
    # ======================================================================

    @staticmethod
    def _hm_to_minutes(hm: str) -> int:
        """将 'HH:MM' 转为当日分钟数（委托给运行编排服务实现）。"""
        return RotationRuntimeService.hm_to_minutes(hm)

    def _on_auto_timer(self):
        """自动调度定时器回调（委托给运行编排服务）。"""
        return self.runtime_service.on_auto_timer()

    # ======================================================================
    #  辅助方法
    # ======================================================================

    def _load_etf_names(self):
        """加载ETF名称映射"""
        try:
            from common.data_portal import get_data_portal
            self._etf_name_map = get_data_portal().get_name_map(asset_type="etf")
        except Exception:
            self._etf_name_map = {}
        self.notifier.etf_name_map = self._etf_name_map

    def _refresh_service_contexts(self, *, reset_strategy: bool = False) -> None:
        """Synchronize mutable config/state/executor/data dependencies across services."""
        self.data_service.update_context(data_dir=self._data_dir)
        self.signal_service.update_context(
            config=self.config,
            data_dir=self._data_dir,
            strategy_provider=self.strategy_provider,
            reset_strategy=reset_strategy,
        )
        self.decision_service.update_context(config=self.config, state=self.state)
        self.guard_service.update_context(config=self.config, state=self.state)
        self.ledger_service.update_context(
            config=self.config,
            state=self.state,
            executor=self.executor,
        )
        self.execution_service.update_context(
            config=self.config,
            state=self.state,
            executor=self.executor,
        )
        self.runtime_service.update_context(
            config=self.config,
            state=self.state,
            executor=self.executor,
            data_dir=self._data_dir,
        )
        self.status_service.update_context(
            config=self.config,
            state=self.state,
            executor=self.executor,
            data_dir=self._data_dir,
        )

    def _code_name(self, code: str) -> str:
        name = self._etf_name_map.get(code, "")
        return f"{code}({name})" if name else code

    def _log(self, msg: str):
        logger.info(msg)
        self.log_message.emit(msg)
