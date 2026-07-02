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
    from trading_app.services.ai.decision_tracker_service import DecisionTrackerService
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
    from trading_app.services.ai.decision_tracker_service import DecisionTrackerService
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
    _ClientStatusWorker,
    _ReconcileCatchupWorker,
)

logger = logging.getLogger(__name__)

class SchedulerSettingsDialog(BaseSchedulerSettingsDialog):
    """Configure scheduled AI decision tasks."""

    def __init__(
        self,
        scheduler,
        parent=None,
        *,
        visible_task_ids: Optional[List[str]] = None,
        dialog_title: str = "定时任务设置",
    ):
        super().__init__(title=dialog_title, min_width=560, initial_height=500, parent=parent)
        self.scheduler = scheduler
        self.visible_task_ids = list(visible_task_ids or [])
        self._setup_ui()

    def _setup_ui(self):
        self.content_layout.addWidget(
            self.make_note_label(
                "说明：修改后点击底部“保存并关闭”生效。AI 任务配置写入实盘中枢任务配置，调度器仅负责到点触发执行。"
            )
        )

        tasks = self.scheduler.get_tasks()
        if self.visible_task_ids:
            tasks = {tid: task for tid, task in tasks.items() if tid in self.visible_task_ids}
        self._rows: Dict[str, Dict[str, Any]] = {}

        for tid, task in tasks.items():
            grp = QGroupBox(f"调度任务：{task.name}")
            grp_layout = QFormLayout(grp)
            grp_layout.setSpacing(6)
            runtime_display = self.scheduler.get_task_runtime_display(tid)
            scan_only_task = str(getattr(task, "task_type", "") or "") == TASK_TYPE_UNMANAGED_POSITION_SCAN

            enabled_cb = QCheckBox("启用任务")
            enabled_cb.setChecked(task.enabled)
            grp_layout.addRow("", enabled_cb)

            time_edit = QTimeEdit()
            try:
                h, m = map(int, task.time.split(":"))
                time_edit.setTime(QTime(h, m))
            except Exception:
                time_edit.setTime(QTime(9, 0))
            grp_layout.addRow("执行时间:", time_edit)

            notify_cb = QCheckBox("完成后发送通知")
            notify_cb.setChecked(task.notify_on_complete)
            grp_layout.addRow("", notify_cb)

            auto_execute_cb = QCheckBox("完成后自动执行交易")
            auto_execute_cb.setChecked(bool(getattr(task, "auto_execute", False)))
            if scan_only_task:
                auto_execute_cb.setChecked(False)
                auto_execute_cb.setEnabled(False)
                auto_execute_cb.setToolTip("未管理账户巡检仅生成建议，不支持自动执行")
            grp_layout.addRow("", auto_execute_cb)

            last_run = QLabel(runtime_display.get("last_run", "") or task.last_run or "从未执行")
            last_run.setStyleSheet("color:#888;")
            grp_layout.addRow("最近执行:", last_run)

            last_result = QLabel(runtime_display.get("last_result", "") or task.last_result or "-")
            last_result.setWordWrap(True)
            last_result.setStyleSheet("color:#888;")
            grp_layout.addRow("最近结果:", last_result)

            run_now_btn = self.make_action_button("立即执行一次")
            run_now_btn.setFixedWidth(96)
            run_now_btn.clicked.connect(lambda _, t=tid: self._run_now(t))
            grp_layout.addRow("", run_now_btn)

            self.content_layout.addWidget(grp)
            self._rows[tid] = {
                "enabled": enabled_cb,
                "time": time_edit,
                "notify": notify_cb,
                "auto_execute": auto_execute_cb,
            }

        self.btn_save, self.btn_cancel = self.setup_footer(
            primary_text="保存并关闭",
            primary_handler=self._save,
            secondary_text="取消",
            secondary_handler=self.reject,
        )

    def _save(self):
        try:
            from trading_app.services.ai.ai_decision_scheduler import ScheduledAITask
        except ImportError:
            from trading_app.services.ai.ai_decision_scheduler import ScheduledAITask

        for tid, widgets in self._rows.items():
            old = self.scheduler.get_tasks().get(tid)
            if not old:
                continue
            task = ScheduledAITask(
                task_id=tid,
                name=old.name,
                enabled=widgets["enabled"].isChecked(),
                time=widgets["time"].time().toString("HH:mm"),
                task_type=old.task_type or "ai_strategy_cycle",
                watchlist_group="",
                model_name=old.model_name,
                notify_on_complete=widgets["notify"].isChecked(),
                auto_execute=False if (old.task_type or "") == TASK_TYPE_UNMANAGED_POSITION_SCAN else widgets["auto_execute"].isChecked(),
                last_run=old.last_run,
                last_result=old.last_result,
            )
            self.scheduler.add_or_update_task(task)
        QMessageBox.information(self, "提示", "定时任务配置已保存。")
        self.accept()

    def _run_now(self, task_id: str):
        self.scheduler.run_now(task_id)
        QMessageBox.information(self, "提示", "任务已触发，请查看主面板")


class AIStrategyConfigDialog(BaseStrategyConfigDialog):
    """Standalone configuration dialog for AI live decisions."""

    def __init__(self, account_panel: AccountPanel, parent=None):
        super().__init__(title="AI 实盘决策配置", min_width=760, initial_height=660, parent=parent)
        self.account_panel = account_panel
        self.content_layout.addWidget(self.account_panel._config_container)
        self.btn_close, self.btn_unlock = self.setup_footer(
            close_text="关闭",
            close_handler=self.reject,
            unlock_text="🔓 解锁编辑",
            unlock_handler=self._unlock_for_edit,
        )

    def prepare_for_open(self) -> None:
        self.account_panel._config_container.setVisible(True)
        self.account_panel._lock_config_panels()
        self.btn_unlock.setText("🔓 解锁编辑")
        self.btn_unlock.setEnabled(True)

    def _unlock_for_edit(self) -> None:
        if self.account_panel.request_unlock_config():
            self.btn_unlock.setText("✏️ 编辑中…")
            self.btn_unlock.setEnabled(False)

    def reject(self) -> None:
        self.account_panel._lock_config_panels()
        self.btn_unlock.setText("🔓 解锁编辑")
        self.btn_unlock.setEnabled(True)
        super().reject()


# ───────────────────────────────────────────────────────────────────────────
#  Main Window: AI Trade Decision Center
# ───────────────────────────────────────────────────────────────────────────
