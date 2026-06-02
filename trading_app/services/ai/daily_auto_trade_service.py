from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QCoreApplication, QObject, QTimer, pyqtSignal

from common.broker_session_service import BrokerSessionService, get_broker_session_service

from trading_app.services.agent_context_service import BrokerContext
from trading_app.services.auto_trade_config_service import AutoTradeConfig, get_auto_trade_config_service
from trading_app.services.trade_decision_models import DecisionOutcome, TradeAction
from trading_app.services.trade_execution_service import ExecutionRequest, ExecutionResult, TradeExecutionService, get_trade_execution_service
from trading_app.services.trade_record_service import TradeRecordService, get_trade_record_service
from trading_app.services.decision_tracker_service import DecisionTrackerService
from trading_app.services.strategy_budget_service import get_strategy_budget_service
from trading_app.services.strategy_constants import AI_STOCK_STRATEGY_ID, AI_STOCK_STRATEGY_NAME, AI_STOCK_VIRTUAL_ACCOUNT_ID
from trading_app.services.strategy_registry_service import get_strategy_registry_service

from live_rotation.holiday_calendar import is_trading_day

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "daily_auto_trade_state.json"
_RUNNING_STATE_STALE_AFTER = timedelta(minutes=30)


@dataclass
class PlannedOrder:
    symbol_code: str
    symbol_name: str
    action: str
    priority: float
    planned_volume: int
    price: float
    decision_record_id: str = ""
    reason: str = ""
    decision_payload: Dict[str, Any] = field(default_factory=dict)
    risk_payload: Dict[str, Any] = field(default_factory=dict)


class DailyAutoTradeService(QObject):
    status_changed = pyqtSignal(str)
    cycle_finished = pyqtSignal(str, bool, str, dict)
    reconcile_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        broker_service: Optional[BrokerSessionService] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.broker_service = broker_service or get_broker_session_service()
        self.execution_service: TradeExecutionService = get_trade_execution_service()
        self.trade_service: TradeRecordService = get_trade_record_service()
        self.decision_tracker = DecisionTrackerService()
        self.config_service = get_auto_trade_config_service()
        self.strategy_registry = get_strategy_registry_service()
        self.strategy_budget = get_strategy_budget_service()
        self._state_path = _STATE_PATH
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._reconcile_timer = QTimer(self)
        self._reconcile_timer.setSingleShot(True)
        self._reconcile_timer.timeout.connect(self._on_reconcile_timer)
        self._pending_reconcile_slot = "primary"
        self._guard_log_context: Optional[Dict[str, Any]] = None
        self._schedule_next_reconcile()

    def begin_task(self, task_id: str, task_config: dict) -> tuple[bool, str]:
        state = self._get_task_state(task_id)
        if state.get("status") == "completed":
            return False, "今日该自动任务已完成，跳过重复执行"
        if state.get("status") == "running":
            if self._is_running_state_stale(state):
                logger.warning("检测到陈旧自动任务状态，自动回收: %s", task_id)
                self._update_task_state(
                    task_id,
                    status="failed",
                    completed_at=self._now(),
                    error="检测到陈旧运行状态，已自动回收",
                )
            else:
                return False, "今日该自动任务正在执行中"
        snapshot_guard_error = self._check_previous_snapshot_guard()
        if snapshot_guard_error:
            return False, snapshot_guard_error
        self._update_task_state(task_id, status="running", task_name=task_config.get("name", task_id), started_at=self._now())
        self.status_changed.emit(f"自动任务开始: {task_config.get('name', task_id)}")
        return True, "自动任务开始"

    def finish_task(
        self,
        task_id: str,
        success: bool,
        message: str,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        summary = summary or {
            "planned": [],
            "executed": [],
            "skipped": True,
            "reason": message,
        }
        self._update_task_state(
            task_id,
            status="completed" if success else "failed",
            completed_at=self._now(),
            summary=summary,
            error="" if success else message,
        )

    def handle_scan_results(
        self,
        task_id: str,
        task_config: dict,
        scan_results: List[Dict[str, Any]],
        broker_context: BrokerContext,
    ) -> None:
        try:
            self._guard_log_context = {"stale_snapshot_logged": False}
            cfg = self.config_service.get_config()
            stats = self._summarize_scan_results(scan_results)
            logger.info(
                "自动任务 %s 巡检完成: 总计 %d, 可执行 %d, hold/观望 %d, 风控拦截 %d",
                task_id,
                stats["total"],
                stats["actionable"],
                stats["hold_like"],
                stats["risk_blocked"],
            )
            if not bool(task_config.get("auto_execute", False)):
                summary = {
                    "planned": [],
                    "executed": [],
                    "skipped": True,
                    "reason": "当前任务未启用自动执行",
                }
                self._update_task_state(task_id, status="completed", completed_at=self._now(), summary=summary)
                self.cycle_finished.emit(task_id, True, "任务分析完成，未启用自动执行", summary)
                return

            guard_error = self._check_daily_guard(cfg, broker_context, log_stale_snapshot_skip=True)
            if guard_error:
                message = self._format_guard_skip_message(guard_error)
                summary = {"planned": [], "executed": [], "skipped": True, "reason": guard_error}
                self._update_task_state(task_id, status="completed", completed_at=self._now(), summary=summary)
                logger.info("自动任务 %s 因风控规则跳过执行: %s", task_id, guard_error)
                self.cycle_finished.emit(task_id, True, message, summary)
                return

            plan = self._build_daily_plan(scan_results, broker_context, cfg)
            if not plan:
                summary = {"planned": [], "executed": [], "skipped": True, "reason": "没有满足条件的自动执行候选"}
                self._update_task_state(task_id, status="completed", completed_at=self._now(), summary=summary)
                self.cycle_finished.emit(task_id, True, "没有满足条件的自动执行候选", summary)
                return

            logger.info(
                "自动任务 %s 完成交易规划: 卖出 %d 笔, 买入 %d 笔",
                task_id,
                sum(1 for item in plan if item.action in (TradeAction.SELL.value, TradeAction.REDUCE.value)),
                sum(1 for item in plan if item.action not in (TradeAction.SELL.value, TradeAction.REDUCE.value)),
            )
            self.status_changed.emit(f"自动交易规划完成: {len(plan)} 笔待执行")
            executed: List[Dict[str, Any]] = []
            failures = 0
            for item in plan:
                self.status_changed.emit(f"自动执行: {item.symbol_name} {item.action} {item.planned_volume}股")
                latest_broker = self._load_broker_context()
                guard_error = self._check_daily_guard(cfg, latest_broker, log_stale_snapshot_skip=False)
                if guard_error:
                    message = self._format_guard_skip_message(guard_error)
                    executed.append({
                        "symbol_code": item.symbol_code,
                        "symbol_name": item.symbol_name,
                        "action": item.action,
                        "success": True,
                        "message": message,
                        "execution_mode": "skipped",
                    })
                    logger.info("自动任务 %s 在执行前被风控拦截: %s", task_id, guard_error)
                    break

                result = self._execute_planned_order(item)
                executed.append({
                    "symbol_code": item.symbol_code,
                    "symbol_name": item.symbol_name,
                    "action": item.action,
                    "success": result.success,
                    "message": result.message,
                    "broker_order_id": result.broker_order_id,
                    "execution_mode": result.execution_mode,
                    "filled_confirmed": result.filled_confirmed,
                    "submitted_only": result.submitted_only,
                })
                self._record_execution_result(task_id, result)
                if result.success and item.decision_record_id:
                    if result.broker_order_id > 0:
                        self.decision_tracker.update_outcome(
                            item.decision_record_id,
                            broker_order_id=result.broker_order_id,
                        )
                if result.success and result.filled_confirmed and item.decision_record_id:
                    self.decision_tracker.update_outcome(
                        item.decision_record_id,
                        outcome=DecisionOutcome.EXECUTED.value,
                        broker_order_id=result.broker_order_id,
                    )
                    if item.action in (TradeAction.SELL.value, TradeAction.REDUCE.value):
                        self.decision_tracker.auto_close_by_symbol(
                            item.symbol_code,
                            item.price,
                            broker_order_id=result.broker_order_id,
                        )
                if not result.success:
                    failures += 1
                    if failures >= cfg.max_intraday_failures:
                        break

            summary = {
                "planned": [asdict(item) for item in plan],
                "executed": executed,
                "skipped": bool(executed) and all(item.get("execution_mode") == "skipped" for item in executed),
                "reason": "",
            }
            success = bool(executed) or not plan
            self._update_task_state(
                task_id,
                status="completed" if success else "failed",
                completed_at=self._now(),
                summary=summary,
            )
            filled_count = sum(1 for item in executed if item.get("filled_confirmed"))
            submitted_count = sum(1 for item in executed if item.get("submitted_only"))
            if summary["skipped"]:
                message = f"自动执行已跳过: 计划 {len(plan)} 笔，因风控未实际下单"
            elif submitted_count > 0:
                message = (
                    f"自动执行已提交: 计划 {len(plan)} 笔，"
                    f"已确认成交 {filled_count} 笔，待确认 {submitted_count} 笔"
                )
            else:
                message = f"自动执行完成: 计划 {len(plan)} 笔，结果 {len(executed)} 笔"
            self.cycle_finished.emit(task_id, success, message, summary)
        except Exception as exc:
            logger.exception("Daily auto trade handling failed")
            self._update_task_state(task_id, status="failed", completed_at=self._now(), error=str(exc))
            self.cycle_finished.emit(task_id, False, f"自动执行异常: {exc}", {})
        finally:
            self._guard_log_context = None

    def run_end_of_day_reconcile(self, *, slot: str = "manual", snapshot_date: Optional[str] = None) -> tuple[bool, str]:
        if not self.config_service.get_config().auto_reconcile_enabled:
            return False, "日终对账未启用"
        snapshot_date = str(snapshot_date or self._today())
        started_at = self._now()
        reconcile_state = self._get_reconcile_state(snapshot_date)
        attempt_count = int(reconcile_state.get("attempt_count", 0) or 0) + 1
        slot_field_map = {
            "primary": "primary_attempted_at",
            "retry": "retry_attempted_at",
            "catchup": "catchup_attempted_at",
        }
        slot_field = slot_field_map.get(slot, "manual_attempted_at")
        self._update_reconcile_state(
            snapshot_date,
            status="running",
            slot=slot,
            started_at=started_at,
            last_attempt_at=started_at,
            attempt_count=attempt_count,
            **{slot_field: started_at},
        )
        if not self.broker_service.is_connected:
            self._update_reconcile_state(
                snapshot_date,
                status="failed",
                completed_at=self._now(),
                pnl_snapshot_saved=False,
                error="券商未连接，无法执行日终对账",
                summary_message="券商未连接，无法执行日终对账",
            )
            logger.warning("日终对账未执行: slot=%s reason=broker_disconnected", slot)
            return False, "券商未连接，无法执行日终对账"
        try:
            logger.info(
                "开始日终对账: slot=%s date=%s attempt=%d started_at=%s",
                slot,
                snapshot_date,
                attempt_count,
                started_at,
            )
            self.status_changed.emit("开始日终对账")
            orders = self.broker_service.query_stock_orders_safe(timeout_seconds=6.0) or []
            broker_trades = self.broker_service.query_stock_trades_safe(timeout_seconds=6.0) or []
            if not broker_trades:
                broker_trades = self.broker_service.query_stock_deals_safe(timeout_seconds=6.0) or []
            logger.info(
                "日终对账已拉取券商数据: slot=%s orders=%d trades=%d",
                slot,
                len(orders),
                len(broker_trades),
            )
            name_map = {}
            order_records_synced = self.trade_service.sync_order_records_from_orders(orders)
            inferred_trades_synced = self.trade_service.sync_from_orders(orders, source="broker_sync", name_map=name_map)
            local_order_record_trades_synced = self.trade_service.sync_from_order_records(
                self.trade_service.get_order_records(
                    start_time=f"{snapshot_date} 00:00:00",
                    end_time=f"{snapshot_date} 23:59:59",
                    limit=5000,
                ),
                source="broker_sync",
            )
            broker_trades_synced = self.trade_service.sync_broker_trades(broker_trades, source="broker_sync")
            logger.info(
                "日终对账同步结果: slot=%s synced_orders=%d synced_trades=%d inferred_trades=%d local_order_record_trades=%d",
                slot,
                order_records_synced,
                broker_trades_synced,
                inferred_trades_synced,
                local_order_record_trades_synced,
            )
            asset = self.broker_service.query_stock_asset()
            positions = self.broker_service.query_stock_positions() or []
            logger.info(
                "日终对账已拉取资产持仓: slot=%s total_asset=%.2f cash=%.2f market_value=%.2f positions=%d",
                slot,
                float(getattr(asset, "total_asset", 0) or 0),
                float(getattr(asset, "cash", 0) or 0),
                float(getattr(asset, "market_value", 0) or 0),
                len([p for p in positions if int(getattr(p, "volume", 0) or 0) > 0]),
            )
            snapshot = self.trade_service.save_daily_pnl(
                snapshot_date=snapshot_date,
                total_asset=float(getattr(asset, "total_asset", 0) or 0),
                cash=float(getattr(asset, "cash", 0) or 0),
                market_value=float(getattr(asset, "market_value", 0) or 0),
                position_count=len([p for p in positions if int(getattr(p, "volume", 0) or 0) > 0]),
                remark="AI交易中心日终自动对账",
            )
            if snapshot is None:
                self._update_reconcile_state(
                    snapshot_date,
                    status="failed",
                    completed_at=self._now(),
                    pnl_snapshot_saved=False,
                    diff_summary={"broker_orders_total": len(orders), "broker_trades_total": len(broker_trades)},
                    error="保存日终快照失败",
                    summary_message="保存日终快照失败",
                )
                logger.error(
                    "日终对账失败: slot=%s reason=save_daily_pnl_failed orders=%d trades=%d",
                    slot,
                    len(orders),
                    len(broker_trades),
                )
                return False, "保存日终快照失败"
            position_snapshot_count = self.trade_service.save_daily_position_snapshots(snapshot_date, positions)
            strategy_positions = self._build_strategy_position_payloads(positions)
            strategy_position_snapshot_count = 0
            logger.info(
                "日终对账快照保存完成: slot=%s position_snapshots=%d strategy_position_snapshot_rows=%d",
                slot,
                position_snapshot_count,
                strategy_position_snapshot_count,
            )
            strategy_trade_rows = {
                row.get("strategy_id", ""): row
                for row in self.trade_service.summarize_trades_by_strategy(snapshot_date)
                if row.get("strategy_id")
            }
            strategy_ids = set(strategy_positions.keys()) | set(strategy_trade_rows.keys())
            for snapshot_item in self.strategy_budget.list_strategy_snapshots():
                strategy_id = str(snapshot_item.get("strategy_id", "") or "")
                if strategy_id:
                    strategy_ids.add(strategy_id)
            has_new_trade_records = (
                inferred_trades_synced > 0
                or local_order_record_trades_synced > 0
                or broker_trades_synced > 0
            )
            if has_new_trade_records or AI_STOCK_STRATEGY_ID in strategy_ids:
                for strategy_id in sorted(strategy_ids):
                    if not strategy_id:
                        continue
                    if not has_new_trade_records and strategy_id != AI_STOCK_STRATEGY_ID:
                        continue
                    position_payload = strategy_positions.get(strategy_id, {})
                    trade_row = strategy_trade_rows.get(strategy_id, {})
                    strategy_name = str(position_payload.get("strategy_name", "") or "")
                    virtual_account_id = str(
                        position_payload.get("virtual_account_id", "")
                        or trade_row.get("virtual_account_id", "")
                        or ""
                    )
                    try:
                        self.strategy_budget.rebuild_strategy_state_from_trade_records(
                            strategy_id,
                            strategy_name=strategy_name,
                            virtual_account_id=virtual_account_id,
                            real_total_asset=float(getattr(asset, "total_asset", 0) or 0),
                        )
                    except Exception as exc:
                        logger.warning("日终对账重建策略账本失败: strategy=%s error=%s", strategy_id, exc)
            strategy_daily_pnl_count = 0
            strategy_trade_summary_count = 0
            strategy_position_snapshot_payloads: Dict[str, Dict[str, Any]] = {}
            logger.info(
                "日终对账开始生成策略快照: slot=%s strategy_count=%d",
                slot,
                len(strategy_ids),
            )
            for strategy_id in sorted(strategy_ids):
                position_payload = strategy_positions.get(
                    strategy_id,
                    {"strategy_name": "", "virtual_account_id": "", "positions": []},
                )
                strategy_name = str(position_payload.get("strategy_name", "") or "")
                virtual_account_id = str(position_payload.get("virtual_account_id", "") or "")
                self.strategy_budget.sync_strategy_positions(
                    strategy_id=strategy_id,
                    positions=list(position_payload.get("positions", []) or []),
                    strategy_name=strategy_name,
                    virtual_account_id=virtual_account_id,
                    real_total_asset=float(getattr(asset, "total_asset", 0) or 0),
                    clear_reservations=True,
                )
                state = self.strategy_budget.get_strategy_state_record(
                    strategy_id,
                    strategy_name=strategy_name,
                    virtual_account_id=virtual_account_id,
                    real_total_asset=float(getattr(asset, "total_asset", 0) or 0),
                )
                budget_snapshot = self.strategy_budget.get_strategy_snapshot(
                    strategy_id,
                    strategy_name=strategy_name,
                    virtual_account_id=virtual_account_id,
                    real_total_asset=float(getattr(asset, "total_asset", 0) or 0),
                )
                market_value = round(
                    sum(float(item.get("market_value", 0) or 0.0) for item in position_payload.get("positions", []) or []),
                    2,
                )
                cash_balance = float(budget_snapshot.get("cash_balance", 0.0) or 0.0)
                if strategy_id == AI_STOCK_STRATEGY_ID:
                    invested_cost = round(
                        sum(
                            float(getattr(pos, "avg_cost", 0.0) or 0.0) * int(getattr(pos, "quantity", 0) or 0)
                            for pos in state.get_positions().values()
                        ),
                        2,
                    )
                    realized_pnl = float(getattr(state, "realized_pnl", 0.0) or 0.0)
                    cash_balance = round(
                        max(float(budget_snapshot.get("capital_limit", 0.0) or 0.0) + realized_pnl - invested_cost, 0.0),
                        2,
                    )
                strategy_pnl = self.trade_service.save_strategy_daily_pnl_snapshot(
                    snapshot_date=snapshot_date,
                    strategy_id=strategy_id,
                    strategy_name=str(budget_snapshot.get("strategy_name", "") or strategy_name),
                    virtual_account_id=str(budget_snapshot.get("virtual_account_id", "") or virtual_account_id),
                    total_asset=round(cash_balance + market_value, 2),
                    cash=cash_balance,
                    market_value=market_value,
                    position_count=len(position_payload.get("positions", []) or []),
                    remark="按策略归因的日终快照",
                )
                if strategy_pnl is not None:
                    strategy_daily_pnl_count += 1
                trade_row = strategy_trade_rows.get(strategy_id, {})
                trade_summary = self.trade_service.save_strategy_daily_trade_summary(
                    snapshot_date=snapshot_date,
                    strategy_id=strategy_id,
                    strategy_name=str(budget_snapshot.get("strategy_name", "") or strategy_name),
                    virtual_account_id=str(budget_snapshot.get("virtual_account_id", "") or virtual_account_id),
                    trade_count=int(trade_row.get("trade_count", 0) or 0),
                    buy_count=int(trade_row.get("buy_count", 0) or 0),
                    sell_count=int(trade_row.get("sell_count", 0) or 0),
                    total_buy_amount=float(trade_row.get("total_buy_amount", 0) or 0.0),
                    total_sell_amount=float(trade_row.get("total_sell_amount", 0) or 0.0),
                    total_commission=float(trade_row.get("total_commission", 0) or 0.0),
                    remark="按策略归因的日成交汇总",
                )
                if trade_summary is not None:
                    strategy_trade_summary_count += 1
                live_position_map = {
                    self._plain_code(str(item.get("stock_code", "") or "")): item
                    for item in list(position_payload.get("positions", []) or [])
                    if self._plain_code(str(item.get("stock_code", "") or ""))
                }
                strategy_position_snapshot_payloads[strategy_id] = {
                    "strategy_name": str(budget_snapshot.get("strategy_name", "") or strategy_name),
                    "virtual_account_id": str(budget_snapshot.get("virtual_account_id", "") or virtual_account_id),
                    "positions": [
                        {
                            "stock_code": code,
                            "stock_name": str(live_position_map.get(code, {}).get("stock_name", "") or code),
                            "volume": int(pos.quantity or 0),
                            "can_use_volume": int(live_position_map.get(code, {}).get("can_use_volume", 0) or 0),
                            "open_price": float(pos.avg_cost or 0.0),
                            "cost_price": float(pos.avg_cost or 0.0),
                            "cost_amount": round(float(pos.avg_cost or 0.0) * int(pos.quantity or 0), 2),
                            "market_value": float(live_position_map.get(code, {}).get("market_value", 0.0) or 0.0),
                        }
                        for code, pos in state.get_positions().items()
                        if int(pos.quantity or 0) > 0
                    ],
                }
            if strategy_position_snapshot_payloads:
                strategy_position_snapshot_count = self.trade_service.save_strategy_position_snapshots(
                    snapshot_date,
                    strategy_position_snapshot_payloads,
                )
            broker_order_ids = [int(getattr(order, "order_id", 0) or 0) for order in orders if int(getattr(order, "order_id", 0) or 0) > 0]
            broker_trade_ids = [int(getattr(trade, "traded_id", 0) or 0) for trade in broker_trades if int(getattr(trade, "traded_id", 0) or 0) > 0]
            matched_order_records = self.trade_service.count_order_records_by_broker_ids(broker_order_ids)
            matched_trade_records = self.trade_service.count_trade_records_by_trade_ids(broker_trade_ids, snapshot_date)
            diff_summary = {
                "broker_orders_total": len(broker_order_ids),
                "broker_trades_total": len(broker_trade_ids),
                "matched_order_records": matched_order_records,
                "matched_trade_records": matched_trade_records,
                "missing_order_records": max(len(broker_order_ids) - matched_order_records, 0),
                "missing_trade_records": max(len(broker_trade_ids) - matched_trade_records, 0),
                "inferred_trades_synced": inferred_trades_synced,
                "local_order_record_trades_synced": local_order_record_trades_synced,
                "broker_trades_synced": broker_trades_synced,
                "strategy_position_snapshots": strategy_position_snapshot_count,
                "strategy_daily_pnl_saved": strategy_daily_pnl_count,
                "strategy_trade_summary_saved": strategy_trade_summary_count,
            }
            summary_message = (
                f"日终对账完成，总资产 {snapshot.total_asset:,.2f}，"
                f"委托 {len(broker_order_ids)}，成交 {len(broker_trade_ids)}，持仓快照 {position_snapshot_count} 条，"
                f"策略快照 {strategy_daily_pnl_count} 组"
            )
            self._update_reconcile_state(
                snapshot_date,
                status="completed",
                completed_at=self._now(),
                pnl_snapshot_saved=True,
                orders_synced=order_records_synced,
                trades_synced=broker_trades_synced,
                inferred_trades_synced=inferred_trades_synced,
                local_order_record_trades_synced=local_order_record_trades_synced,
                position_snapshot_count=position_snapshot_count,
                diff_summary=diff_summary,
                error="",
                summary_message=summary_message,
            )
            self._mark_reconciled_day(snapshot_date)
            self.status_changed.emit("日终对账完成")
            logger.info(
                "日终对账汇总: slot=%s total_asset=%.2f orders=%d trades=%d positions=%d synced_orders=%d synced_trades=%d inferred_trades=%d local_order_record_trades=%d",
                slot,
                snapshot.total_asset,
                len(broker_order_ids),
                len(broker_trade_ids),
                position_snapshot_count,
                order_records_synced,
                broker_trades_synced,
                inferred_trades_synced,
                local_order_record_trades_synced,
            )
            self.reconcile_finished.emit(True, summary_message)
            return True, summary_message
        except Exception as exc:
            logger.exception("End-of-day reconcile failed")
            failure_message = f"日终对账异常: {exc}"
            self._update_reconcile_state(
                snapshot_date,
                status="failed",
                completed_at=self._now(),
                pnl_snapshot_saved=False,
                error=str(exc),
                summary_message=failure_message,
            )
            logger.error("日终对账异常结束: slot=%s error=%s", slot, exc)
            self.reconcile_finished.emit(False, failure_message)
            return False, failure_message

    def should_run_reconcile_catchup(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        target_day, reason = self._resolve_reconcile_catchup_target(now)
        return (target_day is not None), reason

    def _resolve_reconcile_catchup_target(self, now: Optional[datetime] = None) -> tuple[Optional[str], str]:
        cfg = self.config_service.get_config()
        if not cfg.auto_reconcile_enabled:
            return None, "日终对账未启用"
        if not self.broker_service.is_connected:
            return None, "券商未连接"
        now = now or datetime.now()
        expected_snapshot_date = self._latest_expected_snapshot_date()
        if expected_snapshot_date != self._today():
            previous_state = self._get_reconcile_state(expected_snapshot_date)
            previous_status = str(previous_state.get("status", "") or "")
            if previous_status != "completed" and previous_status != "skipped":
                if previous_status == "running":
                    if self._is_inconsistent_running_state(previous_state):
                        logger.warning("检测到上一交易日日终对账运行状态不一致，自动回收: date=%s", expected_snapshot_date)
                        self._update_reconcile_state(
                            expected_snapshot_date,
                            status="failed",
                            completed_at=self._now(),
                            error="检测到不一致的运行状态，已自动回收",
                        )
                    elif self._is_running_state_stale(previous_state):
                        logger.warning("检测到上一交易日日终对账运行状态陈旧，自动回收: date=%s", expected_snapshot_date)
                        self._update_reconcile_state(
                            expected_snapshot_date,
                            status="failed",
                            completed_at=self._now(),
                            error="检测到陈旧日终对账运行状态，已自动回收",
                        )
                    else:
                        return None, f"上一交易日日终对账正在执行中（{expected_snapshot_date}）"
                logger.warning(
                    "跳过上一交易日(%s)日终对账补跑: 当前数据源为今日，用当日数据写历史快照会导致数据失真",
                    expected_snapshot_date,
                )
                self._update_reconcile_state(
                    expected_snapshot_date,
                    status="skipped",
                    completed_at=self._now(),
                    error=f"已跳过: 非当日数据无法准确补跑({self._today()}检测)",
                )
                # 跳过历史日期后继续判断今日是否需要对账
        if not is_trading_day(now.date()):
            return None, "今日非交易日"
        primary = self._build_schedule_time(now, cfg.reconcile_time, "15:10")
        if now < primary:
            return None, "尚未到日终对账时间"
        reconcile_state = self._get_reconcile_state(self._today())
        status = str(reconcile_state.get("status", "") or "")
        if status == "completed":
            return None, "今日日终对账已完成"
        if status == "running":
            if self._is_inconsistent_running_state(reconcile_state):
                logger.warning("检测到不一致的日终对账运行状态，自动回收")
                self._update_reconcile_state(
                    self._today(),
                    status="failed",
                    completed_at=self._now(),
                    error="检测到不一致的运行状态，已自动回收",
                )
            elif self._is_running_state_stale(reconcile_state):
                logger.warning("检测到陈旧日终对账运行状态，自动回收")
                self._update_reconcile_state(
                    self._today(),
                    status="failed",
                    completed_at=self._now(),
                    error="检测到陈旧日终对账运行状态，已自动回收",
                )
            else:
                return None, "今日日终对账正在执行中"
        return self._today(), "需要补跑今日日终对账"

    def run_reconcile_catchup_if_needed(self) -> tuple[bool, str]:
        target_day, reason = self._resolve_reconcile_catchup_target()
        if not target_day:
            logger.info("补跑日终对账跳过: %s", reason)
            return False, reason
        logger.info("检测到需要补跑日终对账: target_date=%s reason=%s", target_day, reason)
        return self.run_end_of_day_reconcile(slot="catchup", snapshot_date=target_day)

    def _execute_planned_order(self, item: PlannedOrder) -> ExecutionResult:
        decision = item.decision_payload
        risk = item.risk_payload
        from .trade_decision_models import TradeDecision, RiskCheckResult, RiskCheckItem

        decision_obj = TradeDecision.from_dict(decision)
        risk_obj = None
        if risk:
            risk_obj = RiskCheckResult(
                passed=bool(risk.get("passed", False)),
                overall_risk_level=str(risk.get("overall_risk_level", "low")),
                warnings=list(risk.get("warnings", []) or []),
                blocked_reasons=list(risk.get("blocked_reasons", []) or []),
                checks=[RiskCheckItem(**c) for c in list(risk.get("checks", []) or [])],
            )
        request = ExecutionRequest(
            stock_code=item.symbol_code,
            stock_name=item.symbol_name,
            order_type=23 if item.action in (TradeAction.BUY.value, TradeAction.ADD.value) else 24,
            order_volume=item.planned_volume,
            price_type=5,
            price=item.price,
            source="ai_agent",
            trigger="auto",
            strategy_name=AI_STOCK_STRATEGY_NAME,
            strategy_id=AI_STOCK_STRATEGY_ID,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            intent_id=item.decision_record_id or f"auto_{item.symbol_code}_{datetime.now().strftime('%H%M%S')}",
            remark=item.reason or "每日自动交易",
            decision=decision_obj,
            risk_result=risk_obj,
            decision_record_id=item.decision_record_id,
            require_approval=False,
            approved=True,
        )
        return self.execution_service.execute(request)

    def _build_daily_plan(
        self,
        scan_results: List[Dict[str, Any]],
        broker_context: BrokerContext,
        cfg: AutoTradeConfig,
    ) -> List[PlannedOrder]:
        position_map = {
            self._plain_code(item.get("code", "")): dict(item)
            for item in broker_context.top_positions
            if self._plain_code(item.get("code", ""))
        }
        positions = set(position_map.keys())
        sell_candidates: List[PlannedOrder] = []
        buy_candidates: List[PlannedOrder] = []
        tradable_cash = self._calc_tradable_cash(broker_context, cfg)

        for result in scan_results:
            decision = result.get("decision")
            risk_result = result.get("risk_result")
            if decision is None or risk_result is None or not getattr(risk_result, "passed", False):
                continue
            if not getattr(decision, "is_actionable", False):
                continue
            code = self._plain_code(getattr(decision, "symbol_code", "") or result.get("symbol_code", ""))
            name = getattr(decision, "symbol_name", "") or result.get("symbol_name", code)
            price = float(getattr(decision, "current_price", 0) or 0)
            if not code or price <= 0:
                continue
            owner = self.strategy_registry.get_owner(code)
            if owner and owner.enabled and owner.strategy_id != AI_STOCK_STRATEGY_ID:
                logger.info(
                    "自动任务跳过跨策略标的: %s 当前归属于 %s",
                    code,
                    owner.strategy_name or owner.strategy_id,
                )
                continue

            decision_record_id = str(result.get("decision_record_id", "") or "")
            priority = float(getattr(decision, "confidence", 0) or 0) - float(getattr(decision, "risk_score", 0) or 0) * 0.2
            action = getattr(decision, "action", "")
            if action in (TradeAction.SELL.value, TradeAction.REDUCE.value):
                volume = self._resolve_sell_volume(cfg, decision, position_map.get(code, {}))
                if volume <= 0:
                    continue
                sell_candidates.append(
                    PlannedOrder(
                        symbol_code=code,
                        symbol_name=name,
                        action=action,
                        priority=priority + 1.0,
                        planned_volume=volume,
                        price=price,
                        decision_record_id=decision_record_id,
                        reason="卖出类信号优先执行",
                        decision_payload=decision.to_dict(),
                        risk_payload=risk_result.to_dict(),
                    )
                )
                continue

            is_new = code not in positions
            if is_new and not bool(cfg.allow_open_new_position):
                continue
            if (not is_new) and not bool(cfg.allow_add_to_existing):
                continue
            buy_candidates.append(
                PlannedOrder(
                    symbol_code=code,
                    symbol_name=name,
                    action=action,
                    priority=priority,
                    planned_volume=0,
                    price=price,
                    decision_record_id=decision_record_id,
                    reason="买入候选",
                    decision_payload=decision.to_dict(),
                    risk_payload=risk_result.to_dict(),
                )
            )

        sell_candidates.sort(key=lambda item: item.priority, reverse=True)
        buy_candidates.sort(key=lambda item: item.priority, reverse=True)

        if cfg.execution_sequence == "buy_only":
            sell_candidates = []
        elif cfg.execution_sequence == "sell_only":
            buy_candidates = []

        planned: List[PlannedOrder] = list(sell_candidates[: cfg.max_sell_orders_per_day])
        if tradable_cash <= 0 or cfg.max_buy_orders_per_day <= 0:
            return self._sort_planned_orders(planned, cfg)

        chosen_buys: List[PlannedOrder] = []
        new_position_count = 0
        for item in buy_candidates:
            is_new = item.symbol_code not in positions
            if is_new and new_position_count >= cfg.max_new_positions_per_day:
                continue
            if len(chosen_buys) >= cfg.max_buy_orders_per_day:
                break
            chosen_buys.append(item)
            if is_new:
                new_position_count += 1

        remaining_cash = tradable_cash
        remaining_slots = len(chosen_buys)
        for item in chosen_buys:
            if remaining_slots <= 0:
                break
            target_cash = self._resolve_buy_target_cash(
                cfg,
                item,
                broker_context=broker_context,
                remaining_cash=remaining_cash,
                remaining_slots=remaining_slots,
            )
            volume = int(target_cash / max(item.price, 0.01) / 100) * 100
            if volume <= 0:
                remaining_slots -= 1
                continue
            item.planned_volume = volume
            item.reason = self._build_buy_reason(cfg, item, volume)
            planned.append(item)
            remaining_cash -= volume * item.price
            remaining_slots -= 1

        return self._sort_planned_orders(planned, cfg)

    def _resolve_sell_volume(self, cfg: AutoTradeConfig, decision: Any, position_info: Optional[Dict[str, Any]] = None) -> int:
        position_info = dict(position_info or {})
        position_volume = int(position_info.get("volume", 0) or 0)
        if cfg.sell_sizing_mode == "full_exit":
            return self._round_down_lot(position_volume)
        if cfg.sell_sizing_mode == "half_exit":
            half_volume = position_volume if position_volume < 200 else int(position_volume * 0.5)
            return self._round_down_lot(half_volume)
        return self.execution_service.estimate_volume_for_decision(decision)

    def _resolve_buy_target_cash(
        self,
        cfg: AutoTradeConfig,
        item: PlannedOrder,
        *,
        broker_context: BrokerContext,
        remaining_cash: float,
        remaining_slots: int,
    ) -> float:
        if cfg.buy_sizing_mode == "fixed_amount":
            return min(remaining_cash, max(float(cfg.buy_value_per_order or 0.0), 0.0))
        if cfg.buy_sizing_mode == "fixed_pct":
            target_cash = broker_context.total_asset * max(float(cfg.buy_position_pct or 0.0), 0.0)
            return min(remaining_cash, target_cash)
        decision_position_pct = float(item.decision_payload.get("position_pct", 0.1) or 0.1)
        return min(
            remaining_cash / max(remaining_slots, 1),
            max(0.0, broker_context.total_asset * decision_position_pct),
        )

    def _build_buy_reason(self, cfg: AutoTradeConfig, item: PlannedOrder, volume: int) -> str:
        if cfg.buy_sizing_mode == "fixed_amount":
            return f"固定金额买入 {volume} 股"
        if cfg.buy_sizing_mode == "fixed_pct":
            pct = max(float(cfg.buy_position_pct or 0.0), 0.0) * 100
            return f"按固定仓位 {pct:.1f}% 买入 {volume} 股"
        return f"按剩余槽位均分买入 {volume} 股"

    def _sort_planned_orders(self, planned: List[PlannedOrder], cfg: AutoTradeConfig) -> List[PlannedOrder]:
        if cfg.execution_sequence == "buy_first":
            planned.sort(key=lambda item: (0 if item.action not in (TradeAction.SELL.value, TradeAction.REDUCE.value) else 1, -item.priority))
            return planned
        planned.sort(key=lambda item: (0 if item.action in (TradeAction.SELL.value, TradeAction.REDUCE.value) else 1, -item.priority))
        return planned

    @staticmethod
    def _round_down_lot(volume: int) -> int:
        return max(int(int(volume or 0) / 100) * 100, 0)

    def _build_strategy_position_payloads(self, positions: List[Any]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for pos in positions or []:
            volume = int(getattr(pos, "volume", 0) or 0)
            if volume <= 0:
                continue
            code = self._plain_code(getattr(pos, "stock_code", "") or "")
            if not code:
                continue
            owner = self.strategy_registry.get_owner(code)
            if owner is None or not owner.enabled:
                continue
            payload = grouped.setdefault(
                owner.strategy_id,
                {
                    "strategy_name": owner.strategy_name,
                    "virtual_account_id": owner.virtual_account_id,
                    "positions": [],
                },
            )
            payload["positions"].append(
                {
                    "stock_code": code,
                    "stock_name": getattr(pos, "stock_name", "") or code,
                    "volume": volume,
                    "can_use_volume": int(getattr(pos, "can_use_volume", 0) or 0),
                    "open_price": round(float(getattr(pos, "open_price", 0) or 0), 4),
                    "market_value": round(float(getattr(pos, "market_value", 0) or 0), 2),
                }
            )
        return grouped

    def _check_daily_guard(
        self,
        cfg: AutoTradeConfig,
        broker_context: BrokerContext,
        *,
        log_stale_snapshot_skip: bool = True,
    ) -> str:
        if not is_trading_day(date.today()):
            return "今日非交易日，停止自动交易"
        if not self.broker_service.is_connected:
            return "券商未连接，停止自动交易"
        state = self._get_today_state()
        if int(state.get("failed_orders", 0) or 0) >= cfg.max_intraday_failures:
            return "今日失败次数已达上限，停止自动交易"
        if broker_context.total_asset > 0:
            today_str = datetime.now().strftime("%Y-%m-%d")
            prev = self.trade_service.get_previous_pnl_snapshot(today_str)
            if prev and prev.total_asset > 0:
                expected_snapshot_date = self._latest_expected_snapshot_date()
                if prev.snapshot_date != expected_snapshot_date:
                    if log_stale_snapshot_skip and not self._has_logged_stale_snapshot_skip():
                        logger.warning(
                            "跳过单日熔断检查: 基准快照过旧，期望=%s，实际=%s",
                            expected_snapshot_date,
                            prev.snapshot_date,
                        )
                        self._mark_stale_snapshot_skip_logged()
                    return ""
                pnl_pct = (broker_context.total_asset - prev.total_asset) / prev.total_asset
                if pnl_pct <= -abs(cfg.max_daily_loss_pct):
                    return f"触发单日熔断，当前收益率 {pnl_pct:.2%}"
        return ""

    def _has_logged_stale_snapshot_skip(self) -> bool:
        ctx = getattr(self, "_guard_log_context", None)
        if not isinstance(ctx, dict):
            return False
        return bool(ctx.get("stale_snapshot_logged", False))

    def _mark_stale_snapshot_skip_logged(self) -> None:
        ctx = getattr(self, "_guard_log_context", None)
        if isinstance(ctx, dict):
            ctx["stale_snapshot_logged"] = True

    @staticmethod
    def _latest_expected_snapshot_date() -> str:
        expected = date.today() - timedelta(days=1)
        while not is_trading_day(expected):
            expected -= timedelta(days=1)
        return expected.strftime("%Y-%m-%d")

    def _format_guard_skip_message(self, guard_error: str) -> str:
        if "熔断" in guard_error:
            return f"因风控熔断跳过自动执行: {guard_error}"
        return f"因风控规则跳过自动执行: {guard_error}"

    def _check_previous_snapshot_guard(self) -> str:
        expected_snapshot_date = self._latest_expected_snapshot_date()
        snapshot = self.trade_service.get_pnl_snapshot(expected_snapshot_date)
        if snapshot is None:
            return f"上一交易日日终快照缺失（{expected_snapshot_date}），停止自动交易"
        position_snapshot_count = self.trade_service.get_daily_position_snapshot_count(expected_snapshot_date)
        if int(getattr(snapshot, "position_count", 0) or 0) > 0 and position_snapshot_count <= 0:
            return f"上一交易日持仓快照缺失（{expected_snapshot_date}），停止自动交易"
        reconcile_state = self._get_reconcile_state(expected_snapshot_date)
        status = str(reconcile_state.get("status", "") or "")
        if status and status != "completed":
            logger.warning("上一交易日日终对账状态异常: date=%s status=%s", expected_snapshot_date, status)
        return ""

    def _summarize_scan_results(self, scan_results: List[Dict[str, Any]]) -> Dict[str, int]:
        total = len(scan_results)
        actionable = 0
        hold_like = 0
        risk_blocked = 0
        for result in scan_results:
            decision = result.get("decision")
            risk_result = result.get("risk_result")
            if risk_result is not None and not getattr(risk_result, "passed", False):
                risk_blocked += 1
            action = str(getattr(decision, "action", "") or "").lower()
            is_actionable = bool(getattr(decision, "is_actionable", False)) if decision is not None else False
            if is_actionable:
                actionable += 1
            elif action in {"hold", "watch", "observe", "观望", "持有", ""}:
                hold_like += 1
        return {
            "total": total,
            "actionable": actionable,
            "hold_like": hold_like,
            "risk_blocked": risk_blocked,
        }

    def _record_execution_result(self, task_id: str, result: ExecutionResult) -> None:
        state = self._get_task_state(task_id)
        executions = list(state.get("executions", []) or [])
        executions.append({
            "time": self._now(),
            "success": result.success,
            "message": result.message,
            "broker_order_id": result.broker_order_id,
            "mode": result.execution_mode,
            "filled_confirmed": result.filled_confirmed,
            "submitted_only": result.submitted_only,
        })
        failed_orders = int(state.get("failed_orders", 0) or 0) + (0 if result.success else 1)
        success_orders = int(state.get("success_orders", 0) or 0) + (1 if result.success else 0)
        self._update_task_state(
            task_id,
            executions=executions,
            failed_orders=failed_orders,
            success_orders=success_orders,
        )

    def _calc_tradable_cash(self, broker_context: BrokerContext, cfg: AutoTradeConfig) -> float:
        """下单前可用现金（只守账本层）。

        统一走 strategy_budget 主账本口径：
          available = min(capital_limit + realized_pnl - invested_cost - reserved_cash,
                          capital_limit * (1 - reserve_cash_pct))
        不再查询券商 available_cash/total_asset，保证 AI / ETF 之间基于
        各自虚拟账户的启动资金严格隔离。
        """
        budget = self.strategy_budget.get_available_budget(
            AI_STOCK_STRATEGY_ID,
            reserve_pct=cfg.reserve_cash_pct,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            real_total_asset=broker_context.total_asset,
        )
        return float(budget.get("available", 0.0) or 0.0)

    def _load_broker_context(self) -> BrokerContext:
        try:
            asset = self.broker_service.query_stock_asset()
            positions = self.broker_service.query_stock_positions() or []
            top = [
                {"code": getattr(p, "stock_code", ""), "volume": int(getattr(p, "volume", 0) or 0)}
                for p in positions
                if int(getattr(p, "volume", 0) or 0) > 0
            ]
            return BrokerContext(
                connected=True,
                total_asset=float(getattr(asset, "total_asset", 0) or 0),
                available_cash=float(getattr(asset, "cash", 0) or 0),
                position_count=len(top),
                top_positions=top,
            )
        except Exception:
            return BrokerContext(connected=self.broker_service.is_connected)

    def _schedule_next_reconcile(self) -> None:
        cfg = self.config_service.get_config()
        if QCoreApplication.instance() is None:
            logger.debug("未检测到 Qt 应用实例，跳过自动日终对账定时器")
            return
        if not cfg.auto_reconcile_enabled:
            self._reconcile_timer.stop()
            return
        now = datetime.now()
        target, slot = self._resolve_next_reconcile_target(now, cfg)
        self._pending_reconcile_slot = slot
        self._reconcile_timer.start(max(int((target - now).total_seconds() * 1000), 1000))
        logger.info("已计划%s日终任务: %s", "补偿" if slot == "retry" else "主", target.strftime("%Y-%m-%d %H:%M:%S"))

    def _on_reconcile_timer(self) -> None:
        if not is_trading_day(date.today()):
            logger.info("今日非交易日，跳过定时日终对账")
            self._schedule_next_reconcile()
            return
        self.run_end_of_day_reconcile(slot=self._pending_reconcile_slot or "primary")
        self._schedule_next_reconcile()

    def _mark_reconciled_day(self, day: str) -> None:
        state = self._load_state()
        target_day = str(day or self._today())
        state.setdefault(target_day, {})
        state[target_day]["reconciled_at"] = self._now()
        self._save_state(state)

    def _resolve_next_reconcile_target(self, now: datetime, cfg: AutoTradeConfig) -> tuple[datetime, str]:
        primary = self._build_schedule_time(now, cfg.reconcile_time, "15:10")
        retry = self._build_schedule_time(now, getattr(cfg, "reconcile_retry_time", "15:20"), "15:20")
        if retry <= primary:
            retry = primary + timedelta(minutes=10)
        today_state = self._get_reconcile_state(self._today())
        if str(today_state.get("status", "") or "") == "completed":
            if primary <= now:
                primary += timedelta(days=1)
            return primary, "primary"
        if not str(today_state.get("primary_attempted_at", "") or ""):
            if now < primary:
                return primary, "primary"
            if now < retry:
                return retry, "retry"
        if not str(today_state.get("retry_attempted_at", "") or "") and now < retry:
            return retry, "retry"
        primary += timedelta(days=1)
        return primary, "primary"

    @staticmethod
    def _build_schedule_time(now: datetime, time_value: str, default_value: str) -> datetime:
        try:
            hour, minute = map(int, str(time_value or default_value).split(":"))
        except Exception:
            hour, minute = map(int, default_value.split(":"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text("utf-8"))
        except Exception:
            return {}

    def _save_state(self, data: Dict[str, Any]) -> None:
        self._state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_today_state(self) -> Dict[str, Any]:
        data = self._load_state()
        return dict(data.get(self._today(), {}) or {})

    def _get_day_state(self, day: str) -> Dict[str, Any]:
        data = self._load_state()
        return dict(data.get(day, {}) or {})

    def _get_task_state(self, task_id: str) -> Dict[str, Any]:
        state = self._get_today_state()
        return dict(state.get(task_id, {}) or {})

    def get_task_state_for_day(self, task_id: str, day: Optional[str] = None) -> Dict[str, Any]:
        state = self._get_day_state(day or self._today())
        return dict(state.get(task_id, {}) or {})

    def get_latest_task_state(self, task_id: str) -> Dict[str, Any]:
        data = self._load_state()
        latest_day = ""
        latest_state: Dict[str, Any] = {}
        for day in sorted(data.keys(), reverse=True):
            day_state = dict(data.get(day, {}) or {})
            task_state = dict(day_state.get(task_id, {}) or {})
            if not task_state:
                continue
            latest_day = str(day or "")
            latest_state = task_state
            break
        if not latest_state:
            return {}
        latest_state["_day"] = latest_day
        return latest_state

    def _get_reconcile_state(self, day: Optional[str] = None) -> Dict[str, Any]:
        state = self._get_day_state(day or self._today())
        return dict(state.get("reconcile", {}) or {})

    def get_day_state_section(self, section: str, day: Optional[str] = None) -> Dict[str, Any]:
        state = self._get_day_state(day or self._today())
        return dict(state.get(str(section or "").strip(), {}) or {})

    def _update_task_state(self, task_id: str, **fields) -> None:
        data = self._load_state()
        today = self._today()
        day = dict(data.get(today, {}) or {})
        task_state = dict(day.get(task_id, {}) or {})
        task_state.update(fields)
        day[task_id] = task_state
        data[today] = day
        self._save_state(data)

    def update_task_state_for_day(self, task_id: str, day: Optional[str] = None, **fields) -> None:
        target_day = str(day or self._today())
        data = self._load_state()
        day_state = dict(data.get(target_day, {}) or {})
        task_state = dict(day_state.get(task_id, {}) or {})
        task_state.update(fields)
        day_state[task_id] = task_state
        data[target_day] = day_state
        self._save_state(data)

    def _update_reconcile_state(self, day: str, **fields) -> None:
        data = self._load_state()
        day_state = dict(data.get(day, {}) or {})
        reconcile_state = dict(day_state.get("reconcile", {}) or {})
        reconcile_state.update(fields)
        day_state["reconcile"] = reconcile_state
        data[day] = day_state
        self._save_state(data)

    def update_day_state_section(self, section: str, day: Optional[str] = None, **fields) -> None:
        key = str(section or "").strip()
        if not key:
            return
        target_day = str(day or self._today())
        data = self._load_state()
        day_state = dict(data.get(target_day, {}) or {})
        section_state = dict(day_state.get(key, {}) or {})
        section_state.update(fields)
        day_state[key] = section_state
        data[target_day] = day_state
        self._save_state(data)

    @staticmethod
    def _plain_code(code: str) -> str:
        value = (code or "").strip().upper()
        return value.split(".")[0] if "." in value else value

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _is_running_state_stale(state: Dict[str, Any]) -> bool:
        started_at = str(state.get("started_at", "") or "").strip()
        if not started_at:
            return True
        try:
            started_dt = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return True
        return datetime.now() - started_dt >= _RUNNING_STATE_STALE_AFTER

    @staticmethod
    def _is_inconsistent_running_state(state: Dict[str, Any]) -> bool:
        if str(state.get("status", "") or "") != "running":
            return False
        if str(state.get("completed_at", "") or "").strip():
            return True
        if str(state.get("error", "") or "").strip():
            return True
        return False


_daily_auto_trade_service: Optional[DailyAutoTradeService] = None


def get_daily_auto_trade_service() -> DailyAutoTradeService:
    global _daily_auto_trade_service
    if _daily_auto_trade_service is None:
        _daily_auto_trade_service = DailyAutoTradeService()
    return _daily_auto_trade_service
