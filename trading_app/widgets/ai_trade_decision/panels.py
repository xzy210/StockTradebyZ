"""AI 实盘决策 — 独立窗口

将 AI 实盘决策分析、统一执行、账户信息三大功能聚合在同一面板中，
使用户无需在多个窗口间切换即可完成「分析 → 决策 → 执行 → 追踪」的完整流程。
"""
from __future__ import annotations

import json
import logging
import math
import os
from uuid import uuid4
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, QTime, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QFontMetrics, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTimeEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from common.execution_contract import OrderExecutionReport, StrategySignal
from common.live_strategy_shell import LiveStrategyShell
from common.market_data_policy import is_etf_like_code
from common.scheduler_dialog_base import BaseSchedulerSettingsDialog
from common.strategy_config_dialog_base import BaseStrategyConfigDialog
from common.strategy_panel_context import StrategyPanelContext

try:
    from common.agent.agent_context_service import (
        AgentContextService,
        AgentRuntimeContext,
        BrokerContext,
        SymbolContext,
        TASK_MODE_TRADE_DECISION,
    )
    from trading_app.services.ai.agent_prompt_builder import AgentPromptBuilder
    from trading_app.services.ai.agent_runtime import StockAgentRuntime
    from trading_app.services.ai.trade_decision_extractor import TradeDecisionExtractor
    from trading_app.services.ai.trade_decision_models import (
        DecisionOutcome,
        RiskCheckItem,
        RiskCheckResult,
        TRADE_ACTION_LABELS,
        TradeAction,
        TradeDecision,
    )
    from trading_app.services.execution.risk_guard_service import RiskGuardService
    from trading_app.services.strategy.strategy_risk import get_strategy_risk_registry, is_configurable
    from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context
    from trading_app.services.ai.daily_auto_trade_service import get_daily_auto_trade_service
    from trading_app.services.ai.auto_trade_config_service import get_auto_trade_config_service
    from trading_app.services.ai.stock_pool_service import get_stock_pool_service
    from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service
    from trading_app.services.strategy.strategy_constants import (
        AI_STOCK_STRATEGY_ID,
        AI_STOCK_STRATEGY_NAME,
        AI_STOCK_VIRTUAL_ACCOUNT_ID,
        UNMANAGED_STRATEGY_ID,
        UNMANAGED_STRATEGY_NAME,
        UNMANAGED_VIRTUAL_ACCOUNT_ID,
    )
    from trading_app.services.strategy.strategy_registry_service import get_strategy_registry_service
    from trading_app.services.execution.trade_execution_service import ExecutionRequest, get_trade_execution_service
    from trading_app.services.execution.trade_record_service import TradeSource
    from trading_app.services.ops.live_strategy_end_of_day_service import StrategyEndOfDayResult
    from trading_app.services.market_data.data_update_result import DataUpdateResult
    from trading_app.services.market_data.market_data_status_service import get_market_data_status_service
    from common.broker_session_service import get_broker_session_service
    from trading_app.watchlist_manager import WatchlistManager
except ImportError:
    from common.agent.agent_context_service import (
        AgentContextService,
        AgentRuntimeContext,
        BrokerContext,
        SymbolContext,
        TASK_MODE_TRADE_DECISION,
    )
    from trading_app.services.ai.agent_prompt_builder import AgentPromptBuilder
    from trading_app.services.ai.agent_runtime import StockAgentRuntime
    from trading_app.services.ai.trade_decision_extractor import TradeDecisionExtractor
    from trading_app.services.ai.trade_decision_models import (
        DecisionOutcome,
        RiskCheckItem,
        RiskCheckResult,
        TRADE_ACTION_LABELS,
        TradeAction,
        TradeDecision,
    )
    from trading_app.services.execution.risk_guard_service import RiskGuardService
    from trading_app.services.strategy.strategy_risk import get_strategy_risk_registry, is_configurable
    from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context
    from trading_app.services.ai.daily_auto_trade_service import get_daily_auto_trade_service
    from trading_app.services.ai.auto_trade_config_service import get_auto_trade_config_service
    from trading_app.services.ai.stock_pool_service import get_stock_pool_service
    from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service
    from trading_app.services.strategy.strategy_constants import (
        AI_STOCK_STRATEGY_ID,
        AI_STOCK_STRATEGY_NAME,
        AI_STOCK_VIRTUAL_ACCOUNT_ID,
        UNMANAGED_STRATEGY_ID,
        UNMANAGED_STRATEGY_NAME,
        UNMANAGED_VIRTUAL_ACCOUNT_ID,
    )
    from trading_app.services.strategy.strategy_registry_service import get_strategy_registry_service
    from trading_app.services.execution.trade_execution_service import ExecutionRequest, get_trade_execution_service
    from trading_app.services.execution.trade_record_service import TradeSource
    from trading_app.services.ops.live_strategy_end_of_day_service import StrategyEndOfDayResult
    from trading_app.services.market_data.data_update_result import DataUpdateResult
    from trading_app.services.market_data.market_data_status_service import get_market_data_status_service
    from common.broker_session_service import get_broker_session_service
    from trading_app.watchlist_manager import WatchlistManager

from trading_app.widgets.strategy_risk_settings_panel import StrategyRiskSettingsPanel
from trading_app.widgets.ai_trade_decision.collapsible_step_card import CollapsibleStepCard
from trading_app.widgets.ai_trade_decision.constants import (
    DECISION_MODE_CANDIDATE_POOL_SCAN,
    DECISION_MODE_POSITION_SCAN,
    SCAN_SCOPE_AI_MANAGED,
    SCAN_SCOPE_LABELS,
    SCAN_SCOPE_UNMANAGED,
    SCAN_SUBAGENT_CONCURRENCY,
    SCAN_SUBAGENT_REQUEST_TIMEOUT_SECONDS,
    TASK_TYPE_AI_STRATEGY_CYCLE,
    TASK_TYPE_CANDIDATE_POOL_SCAN,
    TASK_TYPE_POSITION_SCAN,
    TASK_TYPE_UNMANAGED_POSITION_SCAN,
)
from trading_app.widgets.ai_trade_decision.helpers import (
    _StatusMessageProxy,
    _build_scheduled_scan_batch_record,
    _build_scan_status_text,
    _check_ai_live_market_data_ready,
    _get_chat_thread_class,
    _make_json_safe,
    _serialize_scan_result_for_record,
)
from trading_app.widgets.ai_trade_decision.order_execution_panel import OrderExecutionPanel
from trading_app.widgets.ai_trade_decision.workers import (
    _AccountRefreshWorker,
    _ClientActionWorker,
    _ClientStatusWorker,
    _ReconcileCatchupWorker,
)

logger = logging.getLogger(__name__)

from trading_app.widgets.ai_trade_decision.account_panel import AccountPanel
from trading_app.widgets.ai_trade_decision.decision_panel import DecisionPanel
from trading_app.widgets.ai_trade_decision.dialogs import SchedulerSettingsDialog

class AITradeDecisionPanel(QWidget):
    """Embeddable AI live strategy panel."""

    market_view_requested = pyqtSignal(str, str)

    def __init__(
        self,
        context_provider=None,
        parent=None,
        *,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
        name_map: Optional[Dict[str, str]] = None,
        etf_name_map: Optional[Dict[str, str]] = None,
        shared_broker_panel=None,
        manage_startup: bool = True,
    ):
        super().__init__(parent)
        self.context_provider = context_provider
        self.symbol_name_resolver = symbol_name_resolver
        self.name_map = dict(name_map or {})
        self.etf_name_map = dict(etf_name_map or {})
        self.shared_broker_panel = shared_broker_panel
        self.manage_startup = bool(manage_startup)
        self._status_proxy = _StatusMessageProxy(self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Left: Account panel
        self.account_panel = AccountPanel(
            show_connection_panel=self.shared_broker_panel is None,
            shared_broker_panel=self.shared_broker_panel,
        )
        self.account_panel.setMinimumWidth(260)
        self.account_panel.setMaximumWidth(360)
        self.account_panel.configure_primary_actions(
            show_controls=True,
            show_candidate_pool=True,
            position_text="持仓巡检",
            candidate_text="候选池巡检",
        )
        self.account_panel.scheduler_settings_requested.connect(self._open_scheduler_settings)
        self.account_panel.primary_action_requested.connect(self._on_account_primary_action_requested)
        self.account_panel.model_select_requested.connect(self._on_account_model_select_requested)
        self.account_panel.manual_order_requested.connect(self._open_order_dialog)

        # Center: Decision panel
        self.decision_panel = DecisionPanel(context_provider=context_provider)
        self.decision_panel.setMinimumWidth(500)
        self.decision_panel.set_top_controls_visible(False)
        self.decision_panel.model_combo.currentTextChanged.connect(self.account_panel.set_current_model_display)
        self.decision_panel.market_view_requested.connect(self.market_view_requested)
        self.account_panel.set_current_model_display(self.decision_panel.get_current_model_name())

        # Detached: Order execution dialog panel
        self.order_panel = OrderExecutionPanel(
            strategy_context=self._build_strategy_context(),
            symbol_name_resolver=self.lookup_symbol_name,
        )
        self.order_dialog = QDialog(self)
        self.order_dialog.setWindowTitle("手动委托")
        self.order_dialog.resize(560, 680)
        order_dialog_layout = QVBoxLayout(self.order_dialog)
        order_dialog_layout.setContentsMargins(8, 8, 8, 8)
        order_dialog_layout.addWidget(self.order_panel)

        bottom_controls = QWidget(self)
        bottom_bar = QHBoxLayout(bottom_controls)
        bottom_bar.setContentsMargins(4, 2, 4, 2)

        self.shell = LiveStrategyShell(
            self._build_strategy_context(),
            self.account_panel,
            self.decision_panel,
            footer_panel=bottom_controls,
            parent=self,
        )
        self.shell.horizontal_splitter.setSizes([320, 980])
        self.strategy_trade_panel = self.shell.strategy_trade_panel
        main_layout.addWidget(self.shell)

        # ── Scheduler / Freshness ──
        try:
            from trading_app.services.ai.ai_decision_scheduler import AIDecisionScheduler
            from trading_app.services.market_data.data_freshness_service import DataFreshnessGuard
            from trading_app.services.ops.qmt_startup_orchestrator import QmtStartupOrchestrator
        except ImportError:
            from trading_app.services.ai.ai_decision_scheduler import AIDecisionScheduler
            from trading_app.services.market_data.data_freshness_service import DataFreshnessGuard
            from trading_app.services.ops.qmt_startup_orchestrator import QmtStartupOrchestrator

        self.scheduler = AIDecisionScheduler(self)
        self.scheduler.ensure_defaults()
        self.scheduler.task_triggered.connect(self._on_scheduled_task)
        self.scheduler.task_log.connect(lambda msg: self.statusBar().showMessage(msg))

        self.freshness_guard = DataFreshnessGuard(self)
        self.freshness_guard.update_needed.connect(
            lambda cnt, msg: self.statusBar().showMessage(f"📡 {msg}，正在更新...")
        )
        self.freshness_guard.update_progress.connect(
            lambda c, t, m: self.statusBar().showMessage(f"📡 数据更新 {c}/{t}: {m}")
        )
        self.freshness_guard.update_finished.connect(self._show_freshness_status)
        self.freshness_guard.result_signal.connect(self._on_freshness_result)
        self.freshness_guard.status_notice.connect(self._on_freshness_notice)
        self.freshness_guard.update_finished.connect(self._on_freshness_finished)
        self.freshness_guard.xtquant_failed.connect(self._on_xtquant_failed)
        self.startup_orchestrator = None
        if self.manage_startup:
            self.startup_orchestrator = QmtStartupOrchestrator(self.account_panel.broker, self)
            self.startup_orchestrator.status_changed.connect(self._on_startup_status)
            self.startup_orchestrator.finished.connect(self._on_startup_finished)
        self.daily_auto_trade = get_daily_auto_trade_service()
        self.daily_auto_trade.status_changed.connect(self.statusBar().showMessage)
        self.daily_auto_trade.cycle_finished.connect(self._on_daily_auto_trade_finished)
        self.daily_auto_trade.reconcile_finished.connect(self._on_daily_reconcile_finished)
        self._pending_scheduled_auto_task: Optional[dict] = None
        self._reconcile_catchup_worker: Optional[_ReconcileCatchupWorker] = None
        self._paused_scheduler_task_ids: list[str] = []

        self._broker_svc = get_broker_session_service()
        self._broker_svc.trade_occurred.connect(self._on_broker_trade_callback)
        self._broker_svc.order_changed.connect(self._on_broker_order_callback)

        bottom_bar.addStretch()

        # Status bar
        self.statusBar().showMessage("就绪")

        # Wiring
        self.decision_panel.decision_ready.connect(self._on_decision_ready)
        self.decision_panel.scan_completed.connect(self._on_scan_completed)
        self.strategy_trade_panel.order_requested.connect(self._open_order_dialog_with_order)
        self.strategy_trade_panel.market_view_requested.connect(self.market_view_requested)
        self.order_panel.order_executed.connect(self._on_order_executed)
        self._refresh_scheduler_status()
        self._refresh_scheduled_scan_records()

        expired = self.decision_panel.decision_lifecycle.expire_stale_decisions()
        if expired > 0:
            self.statusBar().showMessage(f"已自动标记 {expired} 条过期决策")
            self.decision_panel._refresh_history()

        if self.manage_startup:
            QTimer.singleShot(600, self._start_startup_orchestration)

    def _build_strategy_context(self) -> StrategyPanelContext:
        return StrategyPanelContext(
            strategy_id=AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            owner_type="ai",
        )

    def statusBar(self):
        window = self.window()
        if isinstance(window, QMainWindow):
            return window.statusBar()
        return self._status_proxy

    def _on_decision_ready(self, payload: object):
        if isinstance(payload, dict):
            decision = payload.get("decision")
            risk_result = payload.get("risk_result")
            decision_record_id = str(payload.get("decision_record_id", "") or "")
        else:
            decision = payload
            risk_result = None
            decision_record_id = ""
        if decision is None:
            return
        self.order_panel.fill_from_decision(
            decision,
            risk_result=risk_result,
            decision_record_id=decision_record_id,
        )
        self.statusBar().showMessage(
            f"决策: {TRADE_ACTION_LABELS.get(decision.action, decision.action)} "
            f"{decision.symbol_name} | 置信度 {decision.confidence:.0%}"
        )

    def _on_account_primary_action_requested(self, action_key: str) -> None:
        if action_key == "candidate_pool_scan":
            codes = self.decision_panel._collect_scan_codes_for_freshness("candidate_pool")
            idx = self.decision_panel.mode_combo.findData(DECISION_MODE_CANDIDATE_POOL_SCAN)
        else:
            codes = self.decision_panel._collect_scan_codes_for_freshness("position")
            idx = self.decision_panel.mode_combo.findData(DECISION_MODE_POSITION_SCAN)
        ok, reason = _check_ai_live_market_data_ready(codes)
        if not ok:
            message = f"实盘策略已阻断: {reason}"
            self.statusBar().showMessage(f"⛔ {message}")
            QMessageBox.warning(self, "行情数据未就绪", message)
            return
        if idx >= 0:
            self.decision_panel.mode_combo.setCurrentIndex(idx)
        self.decision_panel._on_analyze_clicked()

    def _on_account_model_select_requested(self) -> None:
        selected = self.decision_panel.prompt_select_model(self)
        if selected:
            self.account_panel.set_current_model_display(selected)

    def _open_order_dialog(self):
        self.order_dialog.show()
        self.order_dialog.raise_()
        self.order_dialog.activateWindow()

    def _open_order_dialog_with_order(self, code: str, direction: str, price: float):
        self.order_panel.fill_order(code, direction, price)
        self._open_order_dialog()

    def _on_order_executed(
        self,
        success: bool,
        filled_confirmed: bool,
        message: str,
        order_id: int = -1,
        price: float = 0.0,
    ):
        self.strategy_trade_panel.refresh_all()
        if success:
            prefix = "✅" if filled_confirmed else "⏳"
            self.statusBar().showMessage(f"{prefix} {message}")
            QTimer.singleShot(2000, self.account_panel.refresh)
            decision_ctx = getattr(self.order_panel, "_decision_context", {}) or {}
            record_id = str(decision_ctx.get("decision_record_id", "") or "")
            lifecycle = self.decision_panel.decision_lifecycle
            decision = self.decision_panel._current_decision

            if record_id and order_id > 0:
                lifecycle.update_decision_outcome(record_id, broker_order_id=order_id)

            if record_id and filled_confirmed:
                lifecycle.update_decision_outcome(
                    record_id,
                    outcome=DecisionOutcome.EXECUTED.value,
                    broker_order_id=order_id,
                )
                if decision and decision.action in ("sell", "reduce"):
                    closed_ids = lifecycle.auto_close_by_symbol(
                        decision.symbol_code,
                        price or decision.current_price,
                        broker_order_id=order_id,
                    )
                    if closed_ids:
                        self.statusBar().showMessage(
                            f"✅ {message} | 已自动平仓 {len(closed_ids)} 条买入记录"
                        )

                self.decision_panel._refresh_history()
        else:
            self.statusBar().showMessage(f"❌ {message}")
        QMessageBox.information(self, "下单结果", message)

    # ── Broker trade/order callback integration ──

    def _on_broker_trade_callback(self, trade_data: dict) -> None:
        code = trade_data.get("stock_code", "")
        price = trade_data.get("traded_price", "")
        vol = trade_data.get("traded_volume", "")
        self.statusBar().showMessage(f"📬 成交回报: {code} 价格={price} 数量={vol}")
        QTimer.singleShot(1500, self.account_panel.refresh)

    def _on_broker_order_callback(self, order_data: dict) -> None:
        status = order_data.get("order_status")
        code = order_data.get("stock_code", "")
        _STATUS_LABELS = {48: "未报", 50: "已报", 51: "部撤", 52: "已报待撤", 54: "已撤", 55: "部成", 56: "已成", 57: "废单"}
        label = _STATUS_LABELS.get(status, str(status))
        self.statusBar().showMessage(f"📋 委托回报: {code} 状态={label}")

    # ── Scheduler integration ──

    def _refresh_scheduler_status(self):
        task = self.scheduler.get_tasks().get("daily_ai_strategy_cycle")
        if task and bool(getattr(task, "enabled", False)):
            mode_label = "自动执行" if bool(getattr(task, "auto_execute", False)) else "仅检查"
            time_text = str(getattr(task, "time", "") or "").strip()
            summary = f"定时任务: {time_text} {mode_label}".strip()
            self.account_panel.set_scheduler_status(summary, "#16A34A")
        else:
            self.account_panel.set_scheduler_status("定时任务: 未启用", "#6B7B8D")

    def _refresh_scheduled_scan_records(self, *, focus_latest: bool = False) -> None:
        records: list[dict] = []
        task_id = "daily_ai_strategy_cycle"
        task = self.scheduler.get_tasks().get(task_id)
        latest_state = self.daily_auto_trade.get_latest_task_state(task_id)
        for record in list(latest_state.get("scheduled_scan_batches", []) or []):
            if not isinstance(record, dict):
                continue
            item = dict(record)
            item.setdefault("task_id", task_id)
            item.setdefault("task_name", str(getattr(task, "name", task_id) or task_id))
            records.append(item)
        records.sort(key=lambda item: str(item.get("completed_at", "") or ""), reverse=True)
        self.decision_panel.set_scheduled_scan_records(records, focus_latest=focus_latest)

    def _notify_unmanaged_panel_scan_records_updated(self, *, focus_latest: bool = False) -> None:
        parent = self.parent()
        while parent is not None:
            candidate = getattr(parent, "unmanaged_panel", None)
            if candidate is not None and hasattr(candidate, "_refresh_scheduled_scan_records"):
                try:
                    candidate._refresh_scheduled_scan_records(focus_latest=focus_latest)
                except Exception:
                    logger.debug("刷新未管理持仓定时记录面板失败", exc_info=True)
                return
            parent = parent.parent() if hasattr(parent, "parent") and callable(parent.parent) else None

    def _resolve_unmanaged_panel(self) -> Optional[QWidget]:
        parent = self.parent()
        while parent is not None:
            candidate = getattr(parent, "unmanaged_panel", None)
            if candidate is not None and hasattr(candidate, "decision_panel"):
                return candidate
            parent = parent.parent() if hasattr(parent, "parent") and callable(parent.parent) else None
        return None

    def _get_scheduled_target_decision_panel(self, task_type: str):
        if str(task_type or "") == TASK_TYPE_UNMANAGED_POSITION_SCAN:
            unmanaged_panel = self._resolve_unmanaged_panel()
            if unmanaged_panel is not None:
                decision_panel = getattr(unmanaged_panel, "decision_panel", None)
                if decision_panel is not None:
                    return decision_panel
        return self.decision_panel

    def _clear_scheduled_target_run_context(self, task_type: str) -> None:
        target_panel = self._get_scheduled_target_decision_panel(task_type)
        if hasattr(target_panel, "_clear_run_context_override"):
            target_panel._clear_run_context_override()

    def _refresh_scheduled_target_account_panel(self, task_type: str) -> None:
        if str(task_type or "") == TASK_TYPE_UNMANAGED_POSITION_SCAN:
            unmanaged_panel = self._resolve_unmanaged_panel()
            target_account_panel = getattr(unmanaged_panel, "account_panel", None) if unmanaged_panel is not None else None
            if target_account_panel is not None and hasattr(target_account_panel, "refresh"):
                QTimer.singleShot(200, target_account_panel.refresh)
                return
        QTimer.singleShot(200, self.account_panel.refresh)

    def _ensure_scheduled_target_signal_bridge(self, task_type: str) -> None:
        target_panel = self._get_scheduled_target_decision_panel(task_type)
        if target_panel is self.decision_panel:
            return
        if getattr(target_panel, "_scheduled_scan_bridge_connected", False):
            return
        target_panel.scan_completed.connect(self._on_scan_completed)
        setattr(target_panel, "_scheduled_scan_bridge_connected", True)

    def _append_scheduled_scan_batch(
        self,
        task_id: str,
        task_config: dict,
        payload: Dict[str, Any],
        *,
        focus_unmanaged_panel: bool = False,
    ) -> None:
        if not task_id:
            return
        latest_state = self.daily_auto_trade.get_task_state_for_day(task_id)
        batches = [
            dict(item)
            for item in list(latest_state.get("scheduled_scan_batches", []) or [])
            if isinstance(item, dict)
        ]
        record = _build_scheduled_scan_batch_record(
            task_id,
            str(task_config.get("name", task_id) or task_id),
            payload,
        )
        record_key = (
            str(record.get("scan_run_id", "") or ""),
            str(record.get("scan_label", "") or ""),
        )
        filtered = []
        for item in batches:
            item_key = (
                str(item.get("scan_run_id", "") or ""),
                str(item.get("scan_label", "") or ""),
            )
            if item_key == record_key and any(item_key):
                continue
            filtered.append(item)
        filtered.append(record)
        filtered.sort(key=lambda item: str(item.get("completed_at", "") or ""), reverse=True)
        filtered = filtered[:12]
        self.daily_auto_trade.update_task_state_for_day(
            task_id,
            scheduled_scan_batches=filtered,
            latest_scan_batch=record,
        )
        self._refresh_scheduled_scan_records()
        if task_id == "daily_unmanaged_position_scan":
            self._notify_unmanaged_panel_scan_records_updated(focus_latest=focus_unmanaged_panel)

    def _open_scheduler_settings(self):
        dlg = SchedulerSettingsDialog(
            self.scheduler,
            parent=self,
            visible_task_ids=["daily_ai_strategy_cycle"],
            dialog_title="AI实盘决策定时任务设置",
        )
        dlg.exec()
        self._refresh_scheduler_status()

    def _build_unmanaged_scan_bundle(self) -> tuple[list[dict], BrokerContext]:
        positions = self.account_panel.get_unmanaged_live_positions()
        broker_context = self.account_panel.get_unmanaged_broker_context(positions)
        return positions, broker_context

    def run_unmanaged_position_scan_now(self) -> str:
        self.decision_panel.mode_combo.setCurrentIndex(
            self.decision_panel.mode_combo.findData(DECISION_MODE_POSITION_SCAN)
        )
        model_cfg = self.decision_panel._resolve_model_config(show_dialog=True)
        if not model_cfg:
            return "未配置可用的 AI 模型"
        positions, broker_context = self._build_unmanaged_scan_bundle()
        if not positions:
            QMessageBox.information(self, "提示", "未管理账户当前无可巡检持仓")
            return "未管理账户当前无可巡检持仓"
        codes = [str(item.get("code", "") or "") for item in positions if item.get("code")]
        ok, reason = _check_ai_live_market_data_ready(codes)
        if not ok:
            message = f"实盘策略已阻断: {reason}"
            self.statusBar().showMessage(f"⛔ {message}")
            QMessageBox.warning(self, "行情数据未就绪", message)
            return message
        self.decision_panel._run_with_freshness_check(
            codes,
            lambda: self.decision_panel._start_position_scan(
                model_cfg,
                scan_source="manual",
                scheduled_task_id="",
                items=positions,
                scan_scope=SCAN_SCOPE_UNMANAGED,
                scan_label="未管理持仓巡检",
                allow_auto_execute=False,
                broker_context=broker_context,
            ),
        )
        return "已触发未管理持仓巡检"

    def _on_scheduled_task(self, task_id: str, task_config: dict):
        task_type = task_config.get("task_type", TASK_TYPE_AI_STRATEGY_CYCLE)
        target_decision_panel = self._get_scheduled_target_decision_panel(task_type)
        self._ensure_scheduled_target_signal_bridge(task_type)
        scheduled_run_context = build_decision_run_context(prefer_realtime=True)
        logger.info("AI交易中心收到定时任务: %s (%s)", task_id, task_type)
        self.statusBar().showMessage(f"⏰ 定时任务触发: {task_config.get('name', task_id)}，正在检查数据新鲜度...")
        target_decision_panel._set_run_context_override(scheduled_run_context)
        self._pending_scheduled_auto_task = {
            "task_id": task_id,
            "task_config": dict(task_config or {}),
            "expected_scan_run_id": "",
            "expected_scan_mode": "",
            "run_context": scheduled_run_context.to_dict(),
        }
        started, begin_msg = self.daily_auto_trade.begin_task(task_id, task_config)
        if not started:
            logger.info("定时任务 %s 未启动: %s", task_id, begin_msg)
            self.statusBar().showMessage(f"⏰ {begin_msg}")
            self.scheduler.mark_task_result(task_id, begin_msg, dispatch_status="skipped")
            self._pending_scheduled_auto_task = None
            self._clear_scheduled_target_run_context(task_type)
            return

        logger.info("定时任务 %s 已进入自动任务编排", task_id)
        self.scheduler.mark_task_dispatch(task_id, "accepted", "定时任务已进入自动任务编排")
        if task_type == TASK_TYPE_AI_STRATEGY_CYCLE:
            target_decision_panel.mode_combo.setCurrentIndex(
                target_decision_panel.mode_combo.findData(DECISION_MODE_POSITION_SCAN)
            )
        elif task_type in (TASK_TYPE_POSITION_SCAN, TASK_TYPE_UNMANAGED_POSITION_SCAN):
            target_decision_panel.mode_combo.setCurrentIndex(
                target_decision_panel.mode_combo.findData(DECISION_MODE_POSITION_SCAN)
            )
        elif task_type == TASK_TYPE_CANDIDATE_POOL_SCAN:
            target_decision_panel.mode_combo.setCurrentIndex(
                target_decision_panel.mode_combo.findData(DECISION_MODE_CANDIDATE_POOL_SCAN)
            )

        model_name = task_config.get("model_name", "")
        if model_name:
            idx = target_decision_panel.model_combo.findText(model_name)
            if idx >= 0:
                target_decision_panel.model_combo.setCurrentIndex(idx)

        current_model = target_decision_panel.model_combo.currentText()
        logger.info("定时任务 %s 使用模型: %s", task_id, current_model)
        model_cfg = target_decision_panel._resolve_model_config(show_dialog=False)
        if not model_cfg:
            self._finish_pending_scheduled_task(task_id, False, f"未配置可用的 AI 模型: {current_model}")
            return

        cycle_plan = None
        if task_type == TASK_TYPE_AI_STRATEGY_CYCLE:
            cycle_plan = self._build_ai_strategy_cycle_plan()
            self._pending_scheduled_auto_task["cycle_plan"] = cycle_plan
            self._pending_scheduled_auto_task["cycle_results"] = []
            self._pending_scheduled_auto_task["cycle_index"] = 0
            self._pending_scheduled_auto_task["model_cfg"] = dict(model_cfg)

        codes = self._collect_codes_for_task(task_type, task_config, cycle_plan=cycle_plan)
        if not codes:
            self._finish_pending_scheduled_task(task_id, False, "当前任务没有可分析标的")
            return
        ok, reason = _check_ai_live_market_data_ready(codes)
        if not ok:
            self._finish_pending_scheduled_task(task_id, False, f"行情数据未就绪: {reason}")
            self.statusBar().showMessage(f"⛔ 实盘策略已阻断: {reason}")
            return
        logger.info("定时任务 %s 准备校验数据: %d 只标的", task_id, len(codes))
        self.freshness_guard.ensure_fresh_then_run(
            codes,
            lambda task_type=task_type, model_cfg=model_cfg, task_id=task_id: self._run_scheduled_analysis(task_id, task_type, model_cfg),
            include_indices=True,
            prefer_realtime=True,
            require_minute_freshness=False,
        )

    def _on_scan_completed(self, payload: object):
        if not isinstance(payload, dict):
            return
        if not self._pending_scheduled_auto_task:
            return
        pending = dict(self._pending_scheduled_auto_task)
        task_id = str(pending.get("task_id", "") or "")
        task_config = dict(pending.get("task_config", {}) or {})
        task_type = str(task_config.get("task_type", "") or "")
        expected_run_id = str(pending.get("expected_scan_run_id", "") or "")
        expected_mode = str(pending.get("expected_scan_mode", "") or "")
        payload_run_id = str(payload.get("scan_run_id", "") or "")
        payload_mode = str(payload.get("mode", "") or "")
        if not expected_run_id:
            logger.warning(
                "定时任务 %s 忽略未登记轮次的扫描完成事件: actual_run=%s mode=%s",
                task_id,
                payload_run_id,
                payload_mode,
            )
            return
        if expected_run_id and payload_run_id != expected_run_id:
            logger.warning(
                "定时任务 %s 忽略非当前轮次的扫描完成事件: expected_run=%s actual_run=%s mode=%s",
                task_id,
                expected_run_id,
                payload_run_id,
                payload_mode,
            )
            return
        if expected_mode and payload_mode != expected_mode:
            logger.warning(
                "定时任务 %s 忽略非当前阶段的扫描完成事件: expected_mode=%s actual_mode=%s run=%s",
                task_id,
                expected_mode,
                payload_mode,
                payload_run_id,
            )
            return
        self._append_scheduled_scan_batch(
            task_id,
            task_config,
            payload,
            focus_unmanaged_panel=(task_type == TASK_TYPE_UNMANAGED_POSITION_SCAN),
        )
        if task_type == TASK_TYPE_AI_STRATEGY_CYCLE:
            all_results = list(self._pending_scheduled_auto_task.get("cycle_results", []) or [])
            all_results.extend(list(payload.get("results", []) or []))
            self._pending_scheduled_auto_task["cycle_results"] = all_results
            self._pending_scheduled_auto_task["cycle_index"] = int(self._pending_scheduled_auto_task.get("cycle_index", 0)) + 1
            plan = dict(self._pending_scheduled_auto_task.get("cycle_plan", {}) or {})
            phases = list(plan.get("phases", []) or [])
            next_index = int(self._pending_scheduled_auto_task.get("cycle_index", 0))
            if next_index < len(phases):
                logger.info("定时任务 %s 进入下一阶段: %s", task_id, phases[next_index].get("label", ""))
                if self._start_next_ai_strategy_cycle_phase(task_id):
                    return
                return
            self._pending_scheduled_auto_task = None
            self.decision_panel._clear_run_context_override()
            logger.info("定时任务 %s 的总巡检已完成，准备进入自动执行编排", task_id)
            broker_context = self.account_panel.get_broker_context()
            self.daily_auto_trade.handle_scan_results(
                task_id,
                task_config,
                all_results,
                broker_context,
            )
            return

        if task_type == TASK_TYPE_UNMANAGED_POSITION_SCAN:
            self._finish_scan_only_task(
                task_id,
                task_config,
                list(payload.get("results", []) or []),
                reason="未管理持仓巡检仅生成建议，不进入自动交易编排",
            )
            return

        self._pending_scheduled_auto_task = None
        self._clear_scheduled_target_run_context(task_type)
        logger.info("定时任务 %s 的巡检已完成，准备进入自动执行编排", task_id)
        broker_context = self.account_panel.get_broker_context()
        self.daily_auto_trade.handle_scan_results(
            task_id,
            task_config,
            list(payload.get("results", []) or []),
            broker_context,
        )

    def _on_daily_auto_trade_finished(self, task_id: str, success: bool, message: str, summary: dict):
        logger.info("定时任务 %s 自动交易结束: success=%s message=%s", task_id, success, message)
        self.statusBar().showMessage(f"{'✅' if success else '❌'} {message}")
        planned = int(len(summary.get("planned", []) or []))
        executed = int(len(summary.get("executed", []) or []))
        if summary.get("skipped"):
            result_text = f"{message}（跳过）"
        else:
            result_text = f"{message}（计划 {planned} / 执行 {executed}）"
        if task_id:
            self.scheduler.mark_task_result(task_id, result_text, dispatch_status="completed" if success else "failed")
        QTimer.singleShot(200, self.account_panel.refresh)

    def _on_daily_reconcile_finished(self, success: bool, message: str):
        self.statusBar().showMessage(f"{'✅' if success else '❌'} {message}")
        QTimer.singleShot(200, self.account_panel.refresh)
        QTimer.singleShot(300, self.strategy_trade_panel.refresh_all)

    def _finish_pending_scheduled_task(self, task_id: str, success: bool, message: str):
        pending = dict(self._pending_scheduled_auto_task or {})
        task_type = str(dict(pending.get("task_config", {}) or {}).get("task_type", "") or "")
        logger.info("定时任务 %s 结束: %s", task_id, message)
        self.daily_auto_trade.finish_task(task_id, success, message)
        self.scheduler.mark_task_result(task_id, message, dispatch_status="completed" if success else "failed")
        self.statusBar().showMessage(f"{'✅' if success else '❌'} {message}")
        self._pending_scheduled_auto_task = None
        self._clear_scheduled_target_run_context(task_type)

    def _finish_scan_only_task(
        self,
        task_id: str,
        task_config: dict,
        scan_results: list[dict],
        *,
        reason: str,
    ) -> None:
        task_type = str(task_config.get("task_type", "") or "")
        actionable = 0
        blocked = 0
        for item in scan_results:
            decision = item.get("decision")
            risk_result = item.get("risk_result")
            if decision is not None and getattr(decision, "is_actionable", False):
                actionable += 1
            if risk_result is not None and not getattr(risk_result, "passed", True):
                blocked += 1
        message = (
            f"{task_config.get('name', task_id)}完成，共 {len(scan_results)} 只，"
            f"可操作建议 {actionable} 只，风控拦截 {blocked} 只"
        )
        summary = {
            "planned": [],
            "executed": [],
            "skipped": True,
            "reason": reason,
            "scan_total": len(scan_results),
            "actionable": actionable,
            "risk_blocked": blocked,
        }
        logger.info("定时任务 %s 以 scan-only 方式结束: %s", task_id, message)
        self.daily_auto_trade.finish_task(task_id, True, message, summary=summary)
        self.scheduler.mark_task_result(task_id, message, dispatch_status="completed")
        self.statusBar().showMessage(f"✅ {message}")
        self._pending_scheduled_auto_task = None
        self._clear_scheduled_target_run_context(task_type)
        self._refresh_scheduled_target_account_panel(task_type)

    def _run_scheduled_analysis(self, task_id: str, task_type: str, model_cfg: dict):
        logger.info("定时任务 %s 通过数据校验，开始执行巡检", task_id)
        try:
            pending = dict(self._pending_scheduled_auto_task or {})
            target_decision_panel = self._get_scheduled_target_decision_panel(task_type)
            self._ensure_scheduled_target_signal_bridge(task_type)
            target_decision_panel._set_run_context_override(pending.get("run_context"))
            if task_type == TASK_TYPE_AI_STRATEGY_CYCLE:
                if self._pending_scheduled_auto_task is not None:
                    self._pending_scheduled_auto_task["model_cfg"] = dict(model_cfg)
                if self._start_next_ai_strategy_cycle_phase(task_id):
                    return
                self._finish_pending_scheduled_task(task_id, False, "每日AI实盘决策任务没有可执行的巡检阶段")
                return
            if task_type == TASK_TYPE_POSITION_SCAN:
                run_id = target_decision_panel._start_position_scan(
                    model_cfg,
                    scan_source="scheduled",
                    scheduled_task_id=task_id,
                )
                if not run_id:
                    self._finish_pending_scheduled_task(task_id, False, "持仓巡检未能启动，可能仍有其他扫描在运行")
                    return
                if self._pending_scheduled_auto_task is not None:
                    self._pending_scheduled_auto_task["expected_scan_run_id"] = run_id
                    self._pending_scheduled_auto_task["expected_scan_mode"] = DECISION_MODE_POSITION_SCAN
                return
            if task_type == TASK_TYPE_UNMANAGED_POSITION_SCAN:
                positions, broker_context = self._build_unmanaged_scan_bundle()
                run_id = target_decision_panel._start_position_scan(
                    model_cfg,
                    scan_source="scheduled",
                    scheduled_task_id=task_id,
                    items=positions,
                    scan_scope=SCAN_SCOPE_UNMANAGED,
                    scan_label="未管理持仓巡检",
                    allow_auto_execute=False,
                    broker_context=broker_context,
                )
                if not run_id:
                    self._finish_pending_scheduled_task(task_id, False, "未管理持仓巡检未能启动，可能仍有其他扫描在运行")
                    return
                if self._pending_scheduled_auto_task is not None:
                    self._pending_scheduled_auto_task["expected_scan_run_id"] = run_id
                    self._pending_scheduled_auto_task["expected_scan_mode"] = DECISION_MODE_POSITION_SCAN
                return
            if task_type == TASK_TYPE_CANDIDATE_POOL_SCAN:
                items = target_decision_panel._load_candidate_pool_items(refresh=True)
                run_id = target_decision_panel._start_candidate_pool_scan(
                    model_cfg,
                    items,
                    scan_source="scheduled",
                    scheduled_task_id=task_id,
                )
                if not run_id:
                    self._finish_pending_scheduled_task(task_id, False, "候选池巡检未能启动，可能仍有其他扫描在运行")
                    return
                if self._pending_scheduled_auto_task is not None:
                    self._pending_scheduled_auto_task["expected_scan_run_id"] = run_id
                    self._pending_scheduled_auto_task["expected_scan_mode"] = DECISION_MODE_CANDIDATE_POOL_SCAN
                return
            self._finish_pending_scheduled_task(task_id, False, f"不支持的任务类型: {task_type}")
        except Exception as exc:
            logger.exception("定时任务 %s 启动巡检失败", task_id)
            self._finish_pending_scheduled_task(task_id, False, f"启动巡检失败: {exc}")

    def _build_ai_strategy_cycle_plan(self) -> dict:
        phases: list[dict] = []
        position_codes: list[str] = []
        held_codes: set[str] = set()
        try:
            positions = self.account_panel.get_live_positions()
            for position in positions:
                code = str(position.get("code", "") or "").strip()
                if not code:
                    continue
                position_codes.append(code)
                held_codes.add(code[-6:])
            if position_codes:
                phases.append({
                    "type": "position_scan",
                    "label": "持仓巡检",
                    "count": len(position_codes),
                })
        except Exception:
            positions = []

        candidate_items: list[dict] = []
        candidate_codes: list[str] = []
        try:
            raw_items = self.decision_panel._load_candidate_pool_items(refresh=True)
            for item in raw_items:
                code = str(item.get("symbol_code", "") or item.get("code", "") or "").strip()
                if not code or code[-6:] in held_codes:
                    continue
                candidate_items.append(item)
                candidate_codes.append(code)
            if candidate_items:
                phases.append({
                    "type": "candidate_pool_scan",
                    "label": "候选池巡检",
                    "count": len(candidate_items),
                    "items": candidate_items,
                })
        except Exception:
            pass

        return {
            "phases": phases,
            "codes": list(dict.fromkeys(position_codes + candidate_codes)),
        }

    def _start_next_ai_strategy_cycle_phase(self, task_id: str) -> bool:
        pending = self._pending_scheduled_auto_task
        if not pending:
            return False
        model_cfg = dict(pending.get("model_cfg", {}) or {})
        plan = dict(pending.get("cycle_plan", {}) or {})
        phases = list(plan.get("phases", []) or [])
        phase_index = int(pending.get("cycle_index", 0))
        if phase_index >= len(phases):
            return False
        phase = dict(phases[phase_index] or {})
        phase_type = str(phase.get("type", "") or "")
        phase_label = str(phase.get("label", phase_type) or phase_type)
        phase_count = int(phase.get("count", 0) or 0)
        self.statusBar().showMessage(f"⏰ 定时任务触发: {phase_label}，共 {phase_count} 只")
        logger.info("定时任务 %s 开始阶段 %s (%d 只)", task_id, phase_label, phase_count)
        if phase_type == "position_scan":
            self.decision_panel.mode_combo.setCurrentIndex(
                self.decision_panel.mode_combo.findData(DECISION_MODE_POSITION_SCAN)
            )
            run_id = self.decision_panel._start_position_scan(
                model_cfg,
                scan_source="scheduled",
                scheduled_task_id=task_id,
            )
            if not run_id:
                self._finish_pending_scheduled_task(task_id, False, f"{phase_label}未能启动，可能仍有其他扫描在运行")
                return False
            self._pending_scheduled_auto_task["expected_scan_run_id"] = run_id
            self._pending_scheduled_auto_task["expected_scan_mode"] = DECISION_MODE_POSITION_SCAN
            return True
        if phase_type == "candidate_pool_scan":
            self.decision_panel.mode_combo.setCurrentIndex(
                self.decision_panel.mode_combo.findData(DECISION_MODE_CANDIDATE_POOL_SCAN)
            )
            run_id = self.decision_panel._start_candidate_pool_scan(
                model_cfg,
                list(phase.get("items", []) or []),
                scan_source="scheduled",
                scheduled_task_id=task_id,
            )
            if not run_id:
                self._finish_pending_scheduled_task(task_id, False, f"{phase_label}未能启动，可能仍有其他扫描在运行")
                return False
            self._pending_scheduled_auto_task["expected_scan_run_id"] = run_id
            self._pending_scheduled_auto_task["expected_scan_mode"] = DECISION_MODE_CANDIDATE_POOL_SCAN
            return True
        self._finish_pending_scheduled_task(task_id, False, f"不支持的巡检阶段: {phase_type}")
        return False

    def _collect_codes_for_task(self, task_type: str, task_config: dict, cycle_plan: Optional[dict] = None) -> list:
        """Gather stock codes that a scheduled task will need."""
        codes: list[str] = []
        if task_type == TASK_TYPE_AI_STRATEGY_CYCLE:
            codes = list((cycle_plan or {}).get("codes", []) or [])
        elif task_type == TASK_TYPE_POSITION_SCAN:
            try:
                positions = self.account_panel.get_live_positions()
                codes = [str(p.get("code", "")) for p in positions if p.get("code")]
            except Exception:
                pass
        elif task_type == TASK_TYPE_UNMANAGED_POSITION_SCAN:
            try:
                positions = self.account_panel.get_unmanaged_live_positions()
                codes = [str(p.get("code", "")) for p in positions if p.get("code")]
            except Exception:
                pass
        elif task_type == TASK_TYPE_CANDIDATE_POOL_SCAN:
            try:
                codes = self.decision_panel.stock_pool_service.get_candidate_codes(
                    refresh=True,
                    run_context=self.decision_panel._get_effective_run_context().to_dict(),
                )
            except Exception:
                pass
        return list(dict.fromkeys([c for c in codes if c]))

    def _on_xtquant_failed(self, message: str):
        if self._pending_scheduled_auto_task:
            pending = dict(self._pending_scheduled_auto_task)
            task_id = str(pending.get("task_id", "") or "")
            if task_id:
                self._finish_pending_scheduled_task(task_id, False, f"数据校验失败: {message}")
        QMessageBox.warning(
            self,
            "miniQMT 数据异常",
            f"数据更新前的新鲜度验证失败：\n\n{message}\n\n"
            "可能原因包括：实时行情未刷新、盘口不可用、日线拉取失败，或 miniQMT 会话异常。\n\n"
            "请执行以下操作：\n"
            "1. 先确认 miniQMT 已登录且行情在刷新\n"
            "2. 若实时行情/盘口长期无更新，再完全关闭并重启 miniQMT\n"
            "3. 等待行情连接就绪后重试\n\n"
            "本次定时任务将跳过，数据可能不是最新。",
        )

    def _show_freshness_status(self, ok: bool, message: str):
        prefix = "✅" if ok else "❌"
        if ok and ("告警" in message or "有告警" in message):
            prefix = "⚠"
        self.statusBar().showMessage(f"{prefix} {message}")

    def _on_freshness_result(self, result: object):
        if not isinstance(result, DataUpdateResult):
            return
        prefix = "✅" if result.ok else "❌"
        if result.ok and result.has_failures:
            prefix = "⚠"
        self.statusBar().showMessage(f"{prefix} {result.to_ui_message()}")

    def _on_freshness_notice(self, level: str, message: str):
        prefix_map = {
            "info": "📡",
            "success": "✅",
            "warning": "⚠",
            "error": "❌",
        }
        self.statusBar().showMessage(f"{prefix_map.get(level, '📡')} {message}")

    def _on_freshness_finished(self, ok: bool, message: str):
        if ok:
            return
        if self._pending_scheduled_auto_task:
            pending = dict(self._pending_scheduled_auto_task)
            task_id = str(pending.get("task_id", "") or "")
            if task_id:
                self._finish_pending_scheduled_task(task_id, False, f"数据更新失败: {message}")

    def _start_startup_orchestration(self):
        if self.startup_orchestrator is None:
            return
        if self.startup_orchestrator.is_running:
            return
        started = self.startup_orchestrator.start()
        if started:
            self.account_panel.show_client_workflow_status("启动自检中...", success=None)

    def _on_startup_status(self, message: str):
        self.statusBar().showMessage(message)
        self.account_panel.show_client_workflow_status(message, success=None)

    def _on_startup_finished(self, success: bool, message: str):
        self.statusBar().showMessage(f"{'✅' if success else '❌'} {message}")
        self.account_panel.show_client_workflow_status(message, success=success)
        self.account_panel._refresh_client_status_safe()
        if success:
            QTimer.singleShot(800, self._try_reconcile_catchup_after_startup)

    def _try_reconcile_catchup_after_startup(self):
        should_run, reason = self.daily_auto_trade.should_run_reconcile_catchup()
        if not should_run:
            logger.info("启动后日终对账补漏未触发: %s", reason)
            return
        logger.info("启动后触发今日日终对账补漏")
        self.statusBar().showMessage("检测到今日日终对账缺失，正在自动补跑...")
        self._start_reconcile_catchup_worker()

    def _start_reconcile_catchup_worker(self):
        if self._reconcile_catchup_worker is not None and self._reconcile_catchup_worker.isRunning():
            return
        self._reconcile_catchup_worker = _ReconcileCatchupWorker(self.daily_auto_trade, self)
        self._reconcile_catchup_worker.finished_reconcile.connect(self._on_reconcile_catchup_worker_finished)
        self._reconcile_catchup_worker.failed_reconcile.connect(self._on_reconcile_catchup_worker_failed)
        self._reconcile_catchup_worker.finished.connect(self._cleanup_reconcile_catchup_worker)
        self._reconcile_catchup_worker.start()

    def _on_reconcile_catchup_worker_finished(self, success: bool, message: str):
        self._on_daily_reconcile_finished(success, message)

    def _on_reconcile_catchup_worker_failed(self, message: str):
        self._on_daily_reconcile_finished(False, f"启动补漏对账异常: {message}")

    def _cleanup_reconcile_catchup_worker(self):
        worker = self._reconcile_catchup_worker
        if worker is None:
            return
        worker.deleteLater()
        self._reconcile_catchup_worker = None

    def closeEvent(self, event):
        if self.startup_orchestrator is not None:
            try:
                self.startup_orchestrator.cancel()
            except Exception:
                pass
        super().closeEvent(event)

    def set_symbol(self, code: str, name: str = ""):
        self.decision_panel.set_symbol(code, name)

    def lookup_symbol_name(self, code: str) -> str:
        if callable(self.symbol_name_resolver):
            try:
                resolved = self.symbol_name_resolver(code)
                if resolved:
                    return str(resolved)
            except Exception:
                pass
        candidates = [code]
        plain_code = code.split(".")[0] if "." in code else code
        if plain_code not in candidates:
            candidates.append(plain_code)
        if "." not in code and plain_code:
            if plain_code.startswith(("5", "6", "9")):
                candidates.append(f"{plain_code}.SH")
            elif plain_code.startswith(("0", "1", "2", "3")):
                candidates.append(f"{plain_code}.SZ")
        for name_map in (self.name_map, self.etf_name_map):
            for candidate in candidates:
                if candidate in name_map and name_map.get(candidate):
                    return str(name_map.get(candidate))
        parent = self.parent()
        if parent is None:
            return ""
        for attr_name in ("name_map", "etf_name_map"):
            inherited_map = getattr(parent, attr_name, None)
            if not isinstance(inherited_map, dict):
                continue
            for candidate in candidates:
                if candidate in inherited_map and inherited_map.get(candidate):
                    return str(inherited_map.get(candidate))
        return ""

    @staticmethod
    def _normalize_symbol_code(code: str) -> str:
        return str(code or "").split(".")[0].strip().upper()

    def get_center_status_summary(self) -> dict:
        runtime_display = self.scheduler.get_task_runtime_display("daily_ai_strategy_cycle")
        ai_task = self.scheduler.get_tasks().get("daily_ai_strategy_cycle")
        return {
            "strategy_id": AI_STOCK_STRATEGY_ID,
            "strategy_name": AI_STOCK_STRATEGY_NAME,
            "scheduler_enabled_count": 1 if bool(ai_task and getattr(ai_task, "enabled", False)) else 0,
            "scheduler_status_text": self.account_panel.lbl_scheduler_status.text(),
            "startup_running": bool(self.startup_orchestrator and self.startup_orchestrator.is_running),
            "last_run": str(runtime_display.get("last_run", "") or ""),
            "last_result": str(runtime_display.get("last_result", "") or ""),
            "positions_count": len(self.account_panel.get_live_positions()),
            "pending_task": bool(self._pending_scheduled_auto_task),
        }

    def get_center_task_summaries(self) -> list[dict]:
        results: list[dict] = []
        for task_id, task in self.scheduler.get_tasks().items():
            runtime_display = self.scheduler.get_task_runtime_display(task_id)
            results.append(
                {
                    "task_key": task_id,
                    "task_type": str(getattr(task, "task_type", "") or ""),
                    "title": str(getattr(task, "name", task_id) or task_id),
                    "status": "enabled" if bool(getattr(task, "enabled", False)) else "disabled",
                    "message": str(runtime_display.get("last_result", "") or ""),
                    "last_run": str(runtime_display.get("last_run", "") or ""),
                    "schedule_time": str(getattr(task, "time", "") or ""),
                    "next_mode": "auto_execute" if bool(getattr(task, "auto_execute", False)) else "scan_only",
                }
            )
        return results

    def get_center_task_summary(self, task_id: str) -> dict:
        for item in self.get_center_task_summaries():
            if str(item.get("task_key", "") or "") == str(task_id or ""):
                return item
        return {}

    def generate_live_signals(self, payload: Optional[dict] = None) -> list[StrategySignal]:
        """Expose latest AI stock decisions as unified live strategy signals."""
        payload = dict(payload or {})
        raw_results = payload.get("results")
        if raw_results is None:
            raw_results = list(getattr(self.decision_panel, "_scan_results", []) or [])
        signals: list[StrategySignal] = []
        seen_ids: set[str] = set()
        for result in list(raw_results or []):
            signal = self._build_signal_from_scan_result(dict(result or {}), payload=payload)
            if signal is None:
                continue
            if signal.signal_id in seen_ids:
                continue
            seen_ids.add(signal.signal_id)
            signals.append(signal)
        if signals:
            return signals
        decision = getattr(self.decision_panel, "_current_decision", None)
        risk_result = getattr(self.decision_panel, "_current_risk_result", None)
        decision_record_id = str(getattr(self.decision_panel, "_current_decision_record_id", "") or "")
        current_signal = self._build_signal_from_decision(
            decision,
            risk_result=risk_result,
            decision_record_id=decision_record_id,
            payload=payload,
        )
        return [current_signal] if current_signal is not None else []

    def execute_live_signals(
        self,
        signals: list[StrategySignal],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute AI stock StrategySignal outputs and update decision tracking."""
        service = execution_service or get_trade_execution_service()
        reports: list[OrderExecutionReport] = []
        names = dict(stock_name_map or {})
        for signal in list(signals or []):
            if signal is None or signal.action == "hold":
                continue
            quantity = int(signal.metadata.get("quantity", 0) or signal.target_quantity or 0)
            if quantity <= 0:
                continue
            price = float(signal.price or 0.0)
            if price <= 0:
                continue
            plain_code = self._normalize_symbol_code(signal.symbol)
            stock_name = names.get(plain_code) or names.get(signal.symbol) or self.lookup_symbol_name(plain_code) or plain_code
            decision = self._decision_from_signal(signal, stock_name=stock_name)
            risk_result = self._risk_from_signal(signal)
            decision_record_id = str(signal.metadata.get("decision_record_id", "") or "")
            result = service.execute(
                ExecutionRequest(
                    stock_code=plain_code,
                    stock_name=stock_name,
                    order_type=23 if signal.action == "buy" else 24,
                    order_volume=quantity,
                    price_type=int(signal.metadata.get("price_type", 5) or 5),
                    price=price,
                    source=TradeSource.AI_AGENT.value,
                    trigger=str(signal.metadata.get("trigger", "strategy_center") or "strategy_center"),
                    strategy_name=AI_STOCK_STRATEGY_NAME,
                    strategy_id=AI_STOCK_STRATEGY_ID,
                    virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
                    intent_id=signal.signal_id or decision_record_id,
                    remark=signal.reason or "AI股票策略输出",
                    decision=decision,
                    risk_result=risk_result,
                    decision_record_id=decision_record_id,
                    require_approval=False,
                    approved=True,
                    metadata=dict(signal.metadata or {}),
                )
            )
            report = OrderExecutionReport.from_live_execution_result(result, intent=None, fills=[])
            reports.append(report)
            self._sync_signal_execution_outcome(signal, result, decision)
        if reports:
            self.strategy_trade_panel.refresh_all()
            QTimer.singleShot(200, self.account_panel.refresh)
            self.decision_panel._refresh_history()
        return reports

    def _build_signal_from_scan_result(self, result: dict, *, payload: dict) -> Optional[StrategySignal]:
        return self._build_signal_from_decision(
            result.get("decision"),
            risk_result=result.get("risk_result"),
            decision_record_id=str(result.get("decision_record_id", "") or ""),
            payload=payload,
            scan_result=result,
        )

    def _build_signal_from_decision(
        self,
        decision: Optional[TradeDecision],
        *,
        risk_result: Optional[RiskCheckResult] = None,
        decision_record_id: str = "",
        payload: Optional[dict] = None,
        scan_result: Optional[dict] = None,
    ) -> Optional[StrategySignal]:
        if decision is None or not getattr(decision, "is_actionable", False):
            return None
        if risk_result is not None and not bool(getattr(risk_result, "passed", False)):
            return None
        code = self._normalize_symbol_code(getattr(decision, "symbol_code", "") or "")
        price = float(getattr(decision, "current_price", 0.0) or 0.0)
        if not code or price <= 0:
            return None
        action = str(getattr(decision, "action", "") or "").lower().strip()
        side = "buy" if action in (TradeAction.BUY.value, TradeAction.ADD.value) else "sell"
        quantity = get_trade_execution_service().estimate_volume_for_decision(decision)
        quantity = max(int(quantity or 0), 0)
        if quantity <= 0:
            return None
        signal_id = decision_record_id or f"ai_{code}_{uuid4().hex[:10]}"
        payload = dict(payload or {})
        scan_result = dict(scan_result or {})
        metadata = {
            "source": TradeSource.AI_AGENT.value,
            "trigger": str(payload.get("trigger", "strategy_center") or "strategy_center"),
            "virtual_account_id": AI_STOCK_VIRTUAL_ACCOUNT_ID,
            "quantity": quantity,
            "quantity_mode": "delta",
            "price_type": 5,
            "decision_record_id": decision_record_id,
            "decision": decision.to_dict(),
            "risk_result": risk_result.to_dict() if risk_result is not None else {},
            "scan_scope": str(scan_result.get("scan_scope", "") or payload.get("scan_scope", "") or ""),
            "scan_run_id": str(payload.get("scan_run_id", "") or scan_result.get("scan_run_id", "") or ""),
            "scheduled_task_id": str(payload.get("scheduled_task_id", "") or scan_result.get("scheduled_task_id", "") or ""),
        }
        return StrategySignal(
            symbol=code,
            action=side,
            signal_id=signal_id,
            strategy_id=AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            strength=float(getattr(decision, "confidence", 0.0) or 0.0),
            price=price,
            reason=getattr(decision, "reasoning", "") or getattr(decision, "invalidation", "") or "AI股票策略输出",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata,
        )

    @staticmethod
    def _decision_from_signal(signal: StrategySignal, *, stock_name: str = "") -> Optional[TradeDecision]:
        payload = dict(signal.metadata.get("decision", {}) or {})
        if payload:
            return TradeDecision.from_dict(payload)
        return TradeDecision(
            action=TradeAction.BUY.value if signal.action == "buy" else TradeAction.SELL.value,
            symbol_code=signal.symbol,
            symbol_name=stock_name or signal.symbol,
            confidence=float(signal.strength or 0.0),
            current_price=float(signal.price or 0.0),
            reasoning=signal.reason,
        )

    @staticmethod
    def _risk_from_signal(signal: StrategySignal) -> Optional[RiskCheckResult]:
        payload = dict(signal.metadata.get("risk_result", {}) or {})
        if not payload:
            return None
        try:
            return RiskCheckResult(
                passed=bool(payload.get("passed", False)),
                overall_risk_level=str(payload.get("overall_risk_level", "low") or "low"),
                warnings=list(payload.get("warnings", []) or []),
                blocked_reasons=list(payload.get("blocked_reasons", []) or []),
                checks=[RiskCheckItem(**dict(item or {})) for item in list(payload.get("checks", []) or [])],
            )
        except Exception:
            return None

    def _sync_signal_execution_outcome(self, signal: StrategySignal, result, decision: Optional[TradeDecision]) -> None:
        record_id = str(signal.metadata.get("decision_record_id", "") or "")
        if not record_id:
            return
        lifecycle = self.decision_panel.decision_lifecycle
        if result.success and result.broker_order_id > 0:
            lifecycle.update_decision_outcome(record_id, broker_order_id=result.broker_order_id)
        if result.success and result.filled_confirmed:
            lifecycle.update_decision_outcome(
                record_id,
                outcome=DecisionOutcome.EXECUTED.value,
                broker_order_id=result.broker_order_id,
            )
            if decision is not None and decision.action in (TradeAction.SELL.value, TradeAction.REDUCE.value):
                lifecycle.auto_close_by_symbol(
                    decision.symbol_code,
                    float(signal.price or getattr(decision, "current_price", 0.0) or 0.0),
                    broker_order_id=result.broker_order_id,
                )
        elif not result.success:
            lifecycle.update_decision_outcome(record_id, outcome=DecisionOutcome.EXECUTION_FAILED.value)

    def pause_center_automation(self) -> str:
        enabled_ids = [
            task_id for task_id, task in self.scheduler.get_tasks().items()
            if bool(getattr(task, "enabled", False))
        ]
        # 幂等：只有首次暂停时才记录需要恢复的任务，避免重复调用丢失原始状态。
        if not self._paused_scheduler_task_ids:
            self._paused_scheduler_task_ids = list(enabled_ids)
        else:
            existing = set(self._paused_scheduler_task_ids)
            for task_id in enabled_ids:
                if task_id not in existing:
                    self._paused_scheduler_task_ids.append(task_id)
        paused_count = 0
        for task_id in enabled_ids:
            self.scheduler.toggle_task(task_id, False)
            paused_count += 1
        self._refresh_scheduler_status()
        if paused_count == 0:
            return "AI 自动调度已处于暂停状态"
        return f"已暂停 AI 自动调度 {paused_count} 个任务"

    def resume_center_automation(self) -> str:
        restored = 0
        for task_id in list(self._paused_scheduler_task_ids or []):
            if task_id in self.scheduler.get_tasks():
                self.scheduler.toggle_task(task_id, True)
                restored += 1
        self._paused_scheduler_task_ids = []
        self._refresh_scheduler_status()
        return f"已恢复 AI 自动调度 {restored} 个任务"

    def run_end_of_day_tasks(self, snapshot_date: str) -> StrategyEndOfDayResult:
        lifecycle = self.decision_panel.decision_lifecycle
        expired_count = lifecycle.expire_stale_decisions()
        recent_records = lifecycle.query_recent_decisions(limit=200)
        today_records = [rec for rec in recent_records if str(rec.created_at or "").startswith(snapshot_date)]
        executed_count = sum(1 for rec in today_records if rec.outcome in (DecisionOutcome.EXECUTED.value, DecisionOutcome.APPROVED.value))
        closed_count = sum(1 for rec in today_records if bool(rec.closed_at))
        rejected_count = sum(
            1
            for rec in today_records
            if rec.outcome in (
                DecisionOutcome.REJECTED_BY_RISK.value,
                DecisionOutcome.REJECTED_BY_USER.value,
                DecisionOutcome.EXECUTION_FAILED.value,
            )
        )
        stats = lifecycle.get_decision_stats()
        message = f"决策复盘 {len(today_records)} 条，执行 {executed_count} 条，平仓 {closed_count} 条"
        if rejected_count:
            message += f"，未完成 {rejected_count} 条"
        if expired_count:
            message += f"，过期 {expired_count} 条"
        return StrategyEndOfDayResult(
            strategy_id=AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            success=True,
            message=message,
            details={
                "snapshot_date": snapshot_date,
                "today_records": len(today_records),
                "today_executed": executed_count,
                "today_closed": closed_count,
                "today_rejected": rejected_count,
                "expired_count": expired_count,
                "overall_stats": stats,
            },
        )

    def refresh_end_of_day_ui(self) -> None:
        """Refresh end-of-day related UI on the main thread only."""
        self.decision_panel._refresh_history()
        self.account_panel.refresh()
        self.strategy_trade_panel.refresh_all()


class UnmanagedPositionPanel(QWidget):
    """Embeddable panel for unmanaged holdings review."""

    market_view_requested = pyqtSignal(str, str)

    def __init__(
        self,
        context_provider=None,
        parent=None,
        *,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
        name_map: Optional[Dict[str, str]] = None,
        etf_name_map: Optional[Dict[str, str]] = None,
        shared_broker_panel=None,
        shared_ai_panel=None,
    ):
        super().__init__(parent)
        self.context_provider = context_provider
        self.symbol_name_resolver = symbol_name_resolver
        self.name_map = dict(name_map or {})
        self.etf_name_map = dict(etf_name_map or {})
        self.shared_broker_panel = shared_broker_panel
        self.shared_ai_panel = shared_ai_panel
        self._status_proxy = _StatusMessageProxy(self)
        self.order_panel = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.account_panel = AccountPanel(
            show_connection_panel=self.shared_broker_panel is None,
            shared_broker_panel=self.shared_broker_panel,
        )
        self.account_panel.configure_scope(
            position_scope=SCAN_SCOPE_UNMANAGED,
            asset_group_title="账户概览（未管理账户）",
            show_scheduler_controls=True,
            show_config_controls=False,
            show_manual_order_controls=True,
        )
        self.account_panel.setMinimumWidth(260)
        self.account_panel.setMaximumWidth(360)
        self.account_panel.configure_primary_actions(
            show_controls=True,
            show_candidate_pool=False,
            position_text="持仓巡检",
        )
        self.account_panel.scheduler_settings_requested.connect(self._open_scheduler_settings)
        self.account_panel.primary_action_requested.connect(self._on_account_primary_action_requested)
        self.account_panel.model_select_requested.connect(self._on_account_model_select_requested)
        self.account_panel.manual_order_requested.connect(self._open_order_dialog)

        self.decision_panel = DecisionPanel(
            context_provider=context_provider,
            allow_candidate_pool_scan=False,
            position_scan_label="未管理持仓巡检",
            position_scan_hint="未管理持仓巡检: 自动读取未管理账户当前持仓，逐只生成持有/加仓/减仓/卖出建议",
        )
        self.decision_panel.setMinimumWidth(500)
        self.decision_panel.set_top_controls_visible(False)
        self.decision_panel.model_combo.currentTextChanged.connect(self.account_panel.set_current_model_display)
        self.decision_panel.market_view_requested.connect(self.market_view_requested)
        self.account_panel.set_current_model_display(self.decision_panel.get_current_model_name())

        self.order_panel = OrderExecutionPanel(
            strategy_context=self._build_strategy_context(),
            symbol_name_resolver=self.lookup_symbol_name,
        )
        self.order_dialog = QDialog(self)
        self.order_dialog.setWindowTitle("手动委托")
        self.order_dialog.resize(560, 680)
        order_dialog_layout = QVBoxLayout(self.order_dialog)
        order_dialog_layout.setContentsMargins(8, 8, 8, 8)
        order_dialog_layout.addWidget(self.order_panel)

        self.shell = LiveStrategyShell(
            self._build_strategy_context(),
            self.account_panel,
            self.decision_panel,
            parent=self,
        )
        self.shell.horizontal_splitter.setSizes([320, 980])
        self.strategy_trade_panel = self.shell.strategy_trade_panel
        main_layout.addWidget(self.shell)
        self.decision_panel.decision_ready.connect(self._on_decision_ready)
        self.strategy_trade_panel.order_requested.connect(self._open_order_dialog_with_order)
        self.strategy_trade_panel.market_view_requested.connect(self.market_view_requested)
        self.order_panel.order_executed.connect(self._on_order_executed)
        self._refresh_scheduler_status()
        self._refresh_scheduled_scan_records(focus_latest=True)
        self.statusBar().showMessage("就绪")

    def _build_strategy_context(self) -> StrategyPanelContext:
        return StrategyPanelContext(
            strategy_id=UNMANAGED_STRATEGY_ID,
            strategy_name=UNMANAGED_STRATEGY_NAME,
            virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
            owner_type="unmanaged",
        )

    def statusBar(self):
        window = self.window()
        if isinstance(window, QMainWindow):
            return window.statusBar()
        return self._status_proxy

    def set_symbol(self, code: str, name: str = ""):
        self.decision_panel.set_symbol(code, name)

    def _resolve_shared_ai_panel(self) -> Optional[QWidget]:
        if self.shared_ai_panel is not None:
            return self.shared_ai_panel
        parent = self.parent()
        while parent is not None:
            candidate = getattr(parent, "ai_panel", None)
            if candidate is not None and hasattr(candidate, "scheduler"):
                return candidate
            parent = parent.parent() if hasattr(parent, "parent") and callable(parent.parent) else None
        return None

    @property
    def freshness_guard(self):
        ai_panel = self._resolve_shared_ai_panel()
        if ai_panel is not None:
            return getattr(ai_panel, "freshness_guard", None)
        return None

    def _refresh_scheduler_status(self) -> None:
        ai_panel = self._resolve_shared_ai_panel()
        if ai_panel is None:
            self.account_panel.set_scheduler_status("定时任务: 未接入", "#6B7B8D")
            return
        task = ai_panel.scheduler.get_tasks().get("daily_unmanaged_position_scan")
        if task and bool(getattr(task, "enabled", False)):
            time_text = str(getattr(task, "time", "") or "").strip()
            self.account_panel.set_scheduler_status(f"定时任务: {time_text} 仅检查", "#16A34A")
            return
        self.account_panel.set_scheduler_status("定时任务: 未启用", "#6B7B8D")

    def _refresh_scheduled_scan_records(self, *, focus_latest: bool = False) -> None:
        latest_state = get_daily_auto_trade_service().get_latest_task_state("daily_unmanaged_position_scan")
        records = []
        for record in list(latest_state.get("scheduled_scan_batches", []) or []):
            if not isinstance(record, dict):
                continue
            item = dict(record)
            item.setdefault("task_id", "daily_unmanaged_position_scan")
            item.setdefault("task_name", "未管理持仓AI巡检")
            records.append(item)
        records.sort(key=lambda item: str(item.get("completed_at", "") or ""), reverse=True)
        self.decision_panel.set_scheduled_scan_records(records, focus_latest=focus_latest)

    def _open_scheduler_settings(self) -> None:
        ai_panel = self._resolve_shared_ai_panel()
        if ai_panel is None:
            QMessageBox.information(self, "提示", "当前未接入 AI 调度中心，无法打开定时任务设置")
            return
        dlg = SchedulerSettingsDialog(
            ai_panel.scheduler,
            parent=self,
            visible_task_ids=["daily_unmanaged_position_scan"],
            dialog_title="未管理持仓定时任务设置",
        )
        dlg.exec()
        try:
            ai_panel._refresh_scheduler_status()
        except Exception:
            pass
        self._refresh_scheduler_status()

    def _on_decision_ready(self, payload: object):
        if isinstance(payload, dict):
            decision = payload.get("decision")
            risk_result = payload.get("risk_result")
            decision_record_id = str(payload.get("decision_record_id", "") or "")
        else:
            decision = payload
            risk_result = None
            decision_record_id = ""
        if decision is None:
            return
        self.order_panel.fill_from_decision(
            decision,
            risk_result=risk_result,
            decision_record_id=decision_record_id,
        )
        self.statusBar().showMessage(
            f"决策: {TRADE_ACTION_LABELS.get(decision.action, decision.action)} "
            f"{decision.symbol_name} | 置信度 {decision.confidence:.0%}"
        )

    def _on_account_primary_action_requested(self, action_key: str) -> None:
        codes = self.decision_panel._collect_scan_codes_for_freshness("position")
        ok, reason = _check_ai_live_market_data_ready(codes)
        if not ok:
            message = f"实盘策略已阻断: {reason}"
            self.statusBar().showMessage(f"⛔ {message}")
            QMessageBox.warning(self, "行情数据未就绪", message)
            return
        idx = self.decision_panel.mode_combo.findData(DECISION_MODE_POSITION_SCAN)
        if idx >= 0:
            self.decision_panel.mode_combo.setCurrentIndex(idx)
        self.decision_panel._on_analyze_clicked()

    def _on_account_model_select_requested(self) -> None:
        selected = self.decision_panel.prompt_select_model(self)
        if selected:
            self.account_panel.set_current_model_display(selected)

    def _open_order_dialog(self):
        self.order_dialog.show()
        self.order_dialog.raise_()
        self.order_dialog.activateWindow()

    def _open_order_dialog_with_order(self, code: str, direction: str, price: float):
        self.order_panel.fill_order(code, direction, price)
        self._open_order_dialog()

    def _on_order_executed(
        self,
        success: bool,
        filled_confirmed: bool,
        message: str,
        order_id: int = -1,
        price: float = 0.0,
    ):
        self.strategy_trade_panel.refresh_all()
        self.account_panel.refresh()
        if success:
            prefix = "✅" if filled_confirmed else "⏳"
            self.statusBar().showMessage(f"{prefix} {message}")
        else:
            self.statusBar().showMessage(f"❌ {message}")
        QMessageBox.information(self, "下单结果", message)

    def lookup_symbol_name(self, code: str) -> str:
        if callable(self.symbol_name_resolver):
            try:
                resolved = self.symbol_name_resolver(code)
                if resolved:
                    return str(resolved)
            except Exception:
                pass
        candidates = [code]
        plain_code = code.split(".")[0] if "." in code else code
        if plain_code not in candidates:
            candidates.append(plain_code)
        if "." not in code and plain_code:
            if plain_code.startswith(("5", "6", "9")):
                candidates.append(f"{plain_code}.SH")
            elif plain_code.startswith(("0", "1", "2", "3")):
                candidates.append(f"{plain_code}.SZ")
        for name_map in (self.name_map, self.etf_name_map):
            for candidate in candidates:
                if candidate in name_map and name_map.get(candidate):
                    return str(name_map.get(candidate))
        parent = self.parent()
        if parent is None:
            return ""
        for attr_name in ("name_map", "etf_name_map"):
            inherited_map = getattr(parent, attr_name, None)
            if not isinstance(inherited_map, dict):
                continue
            for candidate in candidates:
                if candidate in inherited_map and inherited_map.get(candidate):
                    return str(inherited_map.get(candidate))
        return ""


class AITradeDecisionWindow(QMainWindow):
    """Window wrapper for the embeddable AI live strategy panel."""

    def __init__(
        self,
        context_provider=None,
        parent=None,
        *,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
        name_map: Optional[Dict[str, str]] = None,
        etf_name_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("AI 交易决策中心")
        self.resize(1400, 850)
        self.name_map = dict(name_map or {})
        self.etf_name_map = dict(etf_name_map or {})

        self.panel = AITradeDecisionPanel(
            context_provider=context_provider,
            parent=self,
            symbol_name_resolver=symbol_name_resolver,
            name_map=self.name_map,
            etf_name_map=self.etf_name_map,
        )
        self.setCentralWidget(self.panel)

        # Preserve legacy attributes relied on by internal parent walking.
        self.account_panel = self.panel.account_panel
        self.decision_panel = self.panel.decision_panel
        self.order_panel = self.panel.order_panel
        self.strategy_trade_panel = self.panel.strategy_trade_panel

    def set_symbol(self, code: str, name: str = ""):
        self.panel.set_symbol(code, name)

    def lookup_symbol_name(self, code: str) -> str:
        return self.panel.lookup_symbol_name(code)
