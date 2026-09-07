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

class AccountPanel(QWidget):
    """Compact account + position summary panel."""

    scheduler_settings_requested = pyqtSignal()
    manual_order_requested = pyqtSignal()
    primary_action_requested = pyqtSignal(str)
    model_select_requested = pyqtSignal()
    model_settings_requested = pyqtSignal()

    def __init__(self, parent=None, *, show_connection_panel: bool = True, shared_broker_panel=None):
        super().__init__(parent)
        self.broker = get_broker_session_service()
        self.strategy_registry = get_strategy_registry_service()
        self.strategy_budget = get_strategy_budget_service()
        self.auto_trade_config_service = get_auto_trade_config_service()
        self.show_connection_panel = bool(show_connection_panel)
        self.shared_broker_panel = shared_broker_panel
        self.position_scope = SCAN_SCOPE_AI_MANAGED
        self.asset_group_title = "账户概览（AI实盘决策虚拟账户）"
        self.show_scheduler_controls = True
        self.show_config_controls = True
        self.show_manual_order_controls = True
        self.action_note_text = ""
        self.show_primary_action_controls = False
        self.show_candidate_pool_action = False
        self.primary_position_action_text = "持仓巡检"
        self.primary_candidate_action_text = "候选池巡检"
        self.current_model_display = "-"
        self._status_worker = None
        self._refresh_worker = None
        self._trade_config_loading = False
        self._config_dialog = None
        self._setup_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(30_000)
        self.broker.client_state_changed.connect(self._on_client_state_changed)
        # 延后刷新状态，避免窗口构造阶段阻塞主界面显示。
        QTimer.singleShot(200, self._refresh_client_status_safe)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        section_title_style = "color:#94A3B8;font-size:11px;font-weight:bold;"

        # -- Connection status bar --
        self.connection_widget = QWidget(self)
        conn_group = QVBoxLayout(self.connection_widget)
        conn_group.setSpacing(6)
        conn_group.setContentsMargins(0, 0, 0, 0)

        conn_row = QHBoxLayout()
        self.status_icon = QLabel("🔴")
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("font-weight: bold;")
        conn_row.addWidget(self.status_icon)
        conn_row.addWidget(self.status_label)
        self.client_status_label = QLabel("客户端: 未检测")
        self.client_status_label.setStyleSheet("color: #888;")
        self.client_status_label.setWordWrap(True)
        conn_row.addWidget(self.client_status_label)
        conn_row.addStretch()
        conn_group.addLayout(conn_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.connect_btn = QPushButton("连接券商")
        self.connect_btn.setMinimumWidth(84)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        action_row.addWidget(self.connect_btn)
        action_row.addStretch()
        conn_group.addLayout(action_row)
        self.connection_widget.setVisible(self.show_connection_panel)
        layout.addWidget(self.connection_widget)

        # -- Asset summary --
        self.asset_group = QGroupBox(self.asset_group_title)
        asset_form = QFormLayout(self.asset_group)
        asset_form.setSpacing(4)
        self.lbl_total_asset = QLabel("-")
        self.lbl_available = QLabel("-")
        self.lbl_market_value = QLabel("-")
        self.lbl_profit = QLabel("-")
        for label in (self.lbl_total_asset, self.lbl_available, self.lbl_market_value, self.lbl_profit):
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setStyleSheet("font-weight: bold;")
        asset_form.addRow("总资产:", self.lbl_total_asset)
        asset_form.addRow("可用资金:", self.lbl_available)
        asset_form.addRow("持仓市值:", self.lbl_market_value)
        asset_form.addRow("总盈亏:", self.lbl_profit)
        layout.addWidget(self.asset_group)

        self.action_group = QGroupBox("操作")
        self.action_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        action_layout = QVBoxLayout(self.action_group)
        action_layout.setContentsMargins(8, 8, 8, 8)
        action_layout.setSpacing(6)
        action_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        utility_btn_min_width = 112
        utility_btn_height = 30
        manual_btn_height = 30

        self.settings_label = QLabel("设置")
        self.settings_label.setStyleSheet(section_title_style)
        action_layout.addWidget(self.settings_label)

        self.lbl_scheduler_status = QLabel("定时任务: 未启用")
        self.lbl_scheduler_status.setStyleSheet("color:#6B7B8D;font-size:11px;")
        action_layout.addWidget(self.lbl_scheduler_status)

        # 统一的配置入口：主面板只保留“查看配置”，详细参数迁入独立弹窗。
        config_ctl_row = QHBoxLayout()
        config_ctl_row.setSpacing(6)
        self.btn_open_schedule = QPushButton("⏰ 定时任务")
        self.btn_open_schedule.setToolTip("打开 AI 定时任务配置")
        self.btn_open_schedule.clicked.connect(lambda: self.scheduler_settings_requested.emit())
        self.btn_open_schedule.setMinimumWidth(utility_btn_min_width)
        self.btn_open_schedule.setMinimumHeight(utility_btn_height)
        self.btn_open_schedule.setStyleSheet(
            "QPushButton{background:#0EA5E9;color:white;padding:5px 10px;"
            "border-radius:4px;font-size:11px;}"
            "QPushButton:hover{background:#0284C7;}"
        )
        config_ctl_row.addWidget(self.btn_open_schedule)

        self.btn_toggle_config = QPushButton("⚙ 查看配置")
        self.btn_toggle_config.setToolTip("打开 AI 实盘决策配置弹窗（默认只读）")
        self.btn_toggle_config.clicked.connect(self._on_toggle_config)
        self.btn_toggle_config.setMinimumWidth(utility_btn_min_width)
        self.btn_toggle_config.setMinimumHeight(utility_btn_height)
        self.btn_toggle_config.setStyleSheet(
            "QPushButton{background:#6366F1;color:white;padding:5px 10px;"
            "border-radius:4px;font-size:11px;}"
            "QPushButton:hover{background:#4F46E5;}"
        )
        config_ctl_row.addWidget(self.btn_toggle_config)
        config_ctl_row.addStretch()
        action_layout.addLayout(config_ctl_row)
        self._config_locked = True

        # 配置容器：统一收纳“交易方式”和“策略风控（网关统一）”，由弹窗承载。
        self._config_container = QWidget()
        container_layout = QVBoxLayout(self._config_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(6)

        self.trade_group = QGroupBox("自动交易方式")
        trade_form = QFormLayout(self.trade_group)
        trade_form.setSpacing(4)

        self.execution_sequence_combo = QComboBox()
        self.execution_sequence_combo.addItem("先卖后买", "sell_first")
        self.execution_sequence_combo.addItem("先买后卖", "buy_first")
        self.execution_sequence_combo.addItem("只卖出", "sell_only")
        self.execution_sequence_combo.addItem("只买入", "buy_only")
        trade_form.addRow("执行顺序:", self.execution_sequence_combo)

        self.buy_sizing_combo = QComboBox()
        self.buy_sizing_combo.addItem("按剩余槽位均分", "equal_slots")
        self.buy_sizing_combo.addItem("每笔固定金额", "fixed_amount")
        self.buy_sizing_combo.addItem("每笔固定仓位%", "fixed_pct")
        self.buy_sizing_combo.currentIndexChanged.connect(self._update_trade_config_widget_state)
        trade_form.addRow("买入方式:", self.buy_sizing_combo)

        self.buy_amount_spin = QDoubleSpinBox()
        self.buy_amount_spin.setRange(0.0, 10_000_000.0)
        self.buy_amount_spin.setDecimals(0)
        self.buy_amount_spin.setSingleStep(1000.0)
        self.buy_amount_spin.setSuffix(" 元")
        trade_form.addRow("每笔金额:", self.buy_amount_spin)

        self.buy_pct_spin = QDoubleSpinBox()
        self.buy_pct_spin.setRange(0.0, 100.0)
        self.buy_pct_spin.setDecimals(1)
        self.buy_pct_spin.setSingleStep(1.0)
        self.buy_pct_spin.setSuffix(" %")
        trade_form.addRow("每笔仓位:", self.buy_pct_spin)

        self.sell_sizing_combo = QComboBox()
        self.sell_sizing_combo.addItem("按AI信号", "signal_driven")
        self.sell_sizing_combo.addItem("直接清仓", "full_exit")
        self.sell_sizing_combo.addItem("半仓卖出", "half_exit")
        trade_form.addRow("卖出方式:", self.sell_sizing_combo)

        self.max_buy_orders_spin = QSpinBox()
        self.max_buy_orders_spin.setRange(0, 20)
        trade_form.addRow("每日最多买单:", self.max_buy_orders_spin)

        self.max_sell_orders_spin = QSpinBox()
        self.max_sell_orders_spin.setRange(0, 20)
        trade_form.addRow("每日最多卖单:", self.max_sell_orders_spin)

        self.max_new_positions_spin = QSpinBox()
        self.max_new_positions_spin.setRange(0, 20)
        trade_form.addRow("每日最多新开仓:", self.max_new_positions_spin)

        self.allow_open_new_cb = QCheckBox("允许新开仓")
        self.allow_add_existing_cb = QCheckBox("允许已有持仓加仓")
        trade_form.addRow("", self.allow_open_new_cb)
        trade_form.addRow("", self.allow_add_existing_cb)

        trade_btn_widget = QWidget()
        trade_btn_row = QHBoxLayout(trade_btn_widget)
        trade_btn_row.setContentsMargins(0, 0, 0, 0)
        self.trade_config_status = QLabel("")
        self.trade_config_status.setStyleSheet("color:#888; font-size:12px;")
        self.trade_config_status.setWordWrap(True)
        trade_btn_row.addWidget(self.trade_config_status, stretch=1)
        self.trade_config_save_btn = QPushButton("保存交易方式")
        self.trade_config_save_btn.clicked.connect(self._save_trade_config)
        trade_btn_row.addWidget(self.trade_config_save_btn)
        trade_form.addRow("", trade_btn_widget)
        container_layout.addWidget(self.trade_group)

        # AI 实盘决策风控（声明式 schema 自动渲染；与 ETF Tab 共用 StrategyRiskSettingsPanel）
        # 触发一次 TradeExecutionService 初始化，确保 AIStockRiskPolicy 已注册到 registry
        configurable_policy = None
        try:
            get_trade_execution_service()
            registry = get_strategy_risk_registry()
            for candidate in registry.resolve(AI_STOCK_STRATEGY_ID):
                if is_configurable(candidate):
                    configurable_policy = candidate
                    break
        except Exception as exc:
            logger.error("初始化 AI 实盘决策风控面板失败: %s", exc, exc_info=True)

        self.risk_policy_panel: Optional[StrategyRiskSettingsPanel] = None
        if configurable_policy is not None:
            self.risk_policy_panel = StrategyRiskSettingsPanel(
                policy=configurable_policy,
                title="AI 实盘决策风控（网关统一）",
            )
            container_layout.addWidget(self.risk_policy_panel)

        self._lock_config_panels()

        self.sep_settings = QFrame()
        self.sep_settings.setFrameShape(QFrame.Shape.HLine)
        self.sep_settings.setStyleSheet("color:#3c3c3c;")
        action_layout.addWidget(self.sep_settings)

        self.primary_label = QLabel("主操作")
        self.primary_label.setStyleSheet(section_title_style)
        action_layout.addWidget(self.primary_label)

        self.lbl_current_model = QLabel("当前模型: -")
        self.lbl_current_model.setStyleSheet("color:#6B7280;font-size:11px;")
        self.lbl_current_model.setWordWrap(True)
        action_layout.addWidget(self.lbl_current_model)

        self.btn_select_model = QPushButton("切换模型")
        self.btn_select_model.setMinimumHeight(utility_btn_height)
        self.btn_select_model.setStyleSheet(
            "QPushButton{background:#334155;color:#ffffff;padding:6px 10px;"
            "border:1px solid #475569;border-radius:4px;font-size:11px;}"
            "QPushButton:hover{background:#475569;}"
        )
        self.btn_select_model.clicked.connect(lambda: self.model_select_requested.emit())
        action_layout.addWidget(self.btn_select_model)

        self.btn_primary_position_scan = QPushButton(self.primary_position_action_text)
        self.btn_primary_position_scan.setMinimumHeight(manual_btn_height)
        self.btn_primary_position_scan.setStyleSheet(
            "QPushButton{background:#2563EB;color:#ffffff;padding:6px 10px;"
            "border-radius:4px;font-size:11px;font-weight:bold;}"
            "QPushButton:hover{background:#1D4ED8;}"
        )
        self.btn_primary_position_scan.clicked.connect(
            lambda: self.primary_action_requested.emit("position_scan")
        )
        action_layout.addWidget(self.btn_primary_position_scan)

        self.btn_primary_candidate_pool_scan = QPushButton(self.primary_candidate_action_text)
        self.btn_primary_candidate_pool_scan.setMinimumHeight(manual_btn_height)
        self.btn_primary_candidate_pool_scan.setStyleSheet(
            "QPushButton{background:#7C3AED;color:#ffffff;padding:6px 10px;"
            "border-radius:4px;font-size:11px;font-weight:bold;}"
            "QPushButton:hover{background:#6D28D9;}"
        )
        self.btn_primary_candidate_pool_scan.clicked.connect(
            lambda: self.primary_action_requested.emit("candidate_pool_scan")
        )
        action_layout.addWidget(self.btn_primary_candidate_pool_scan)

        self.sep_primary = QFrame()
        self.sep_primary.setFrameShape(QFrame.Shape.HLine)
        self.sep_primary.setStyleSheet("color:#3c3c3c;")
        action_layout.addWidget(self.sep_primary)

        self.manual_label = QLabel("手动干预")
        self.manual_label.setStyleSheet(section_title_style)
        action_layout.addWidget(self.manual_label)

        self.btn_manual_order = QPushButton("手动委托")
        self.btn_manual_order.clicked.connect(lambda: self.manual_order_requested.emit())
        self.btn_manual_order.setMinimumHeight(manual_btn_height)
        self.btn_manual_order.setStyleSheet(
            "QPushButton{background:#2d2d2d;color:#ffffff;padding:6px 10px;"
            "border:1px solid #3c3c3c;border-radius:4px;font-size:11px;}"
            "QPushButton:hover{background:#3c3c3c;}"
        )
        action_layout.addWidget(self.btn_manual_order)

        self.action_note_label = QLabel(self.action_note_text)
        self.action_note_label.setWordWrap(True)
        self.action_note_label.setStyleSheet("color:#94A3B8;font-size:11px;")
        self.action_note_label.setVisible(bool(self.action_note_text))
        action_layout.addWidget(self.action_note_label)
        layout.addWidget(self.action_group)
        layout.addStretch()

        self.broker.connection_changed.connect(self._on_connection_changed)
        if self.broker.is_connected:
            self._on_connection_changed(True, "已连接")
        self._load_trade_config()
        self._apply_action_section_visibility()

    def configure_scope(
        self,
        *,
        position_scope: str,
        asset_group_title: str = "",
        show_scheduler_controls: bool = True,
        show_config_controls: bool = True,
        show_manual_order_controls: bool = True,
        action_note_text: str = "",
    ) -> None:
        self.position_scope = str(position_scope or SCAN_SCOPE_AI_MANAGED)
        self.asset_group_title = str(asset_group_title or self.asset_group_title)
        self.show_scheduler_controls = bool(show_scheduler_controls)
        self.show_config_controls = bool(show_config_controls)
        self.show_manual_order_controls = bool(show_manual_order_controls)
        self.action_note_text = str(action_note_text or "")
        if hasattr(self, "asset_group"):
            self.asset_group.setTitle(self.asset_group_title)
        if hasattr(self, "action_note_label"):
            self.action_note_label.setText(self.action_note_text)
            self.action_note_label.setVisible(bool(self.action_note_text))
        self._apply_action_section_visibility()

    def configure_primary_actions(
        self,
        *,
        show_controls: bool,
        show_candidate_pool: bool,
        position_text: str = "持仓巡检",
        candidate_text: str = "候选池巡检",
    ) -> None:
        self.show_primary_action_controls = bool(show_controls)
        self.show_candidate_pool_action = bool(show_candidate_pool)
        self.primary_position_action_text = str(position_text or "持仓巡检")
        self.primary_candidate_action_text = str(candidate_text or "候选池巡检")
        if hasattr(self, "btn_primary_position_scan"):
            self.btn_primary_position_scan.setText(self.primary_position_action_text)
        if hasattr(self, "btn_primary_candidate_pool_scan"):
            self.btn_primary_candidate_pool_scan.setText(self.primary_candidate_action_text)
        self._apply_action_section_visibility()

    def set_current_model_display(self, model_name: str) -> None:
        self.current_model_display = str(model_name or "-")
        if hasattr(self, "lbl_current_model"):
            self.lbl_current_model.setText(f"当前模型: {self.current_model_display}")

    def _apply_action_section_visibility(self) -> None:
        show_settings = bool(self.show_scheduler_controls or self.show_config_controls)
        show_primary = bool(self.show_primary_action_controls)
        show_manual = bool(self.show_manual_order_controls)
        self.settings_label.setVisible(show_settings)
        self.lbl_scheduler_status.setVisible(self.show_scheduler_controls)
        self.btn_open_schedule.setVisible(self.show_scheduler_controls)
        self.btn_toggle_config.setVisible(self.show_config_controls)
        self.sep_settings.setVisible(show_settings and (show_primary or show_manual))
        self.primary_label.setVisible(show_primary)
        self.lbl_current_model.setVisible(show_primary)
        self.btn_select_model.setVisible(show_primary)
        self.btn_primary_position_scan.setVisible(show_primary)
        self.btn_primary_candidate_pool_scan.setVisible(show_primary and self.show_candidate_pool_action)
        self.sep_primary.setVisible(show_primary and show_manual)
        self.manual_label.setVisible(show_manual)
        self.btn_manual_order.setVisible(show_manual)
        self.action_group.setVisible(show_settings or show_primary or show_manual or bool(self.action_note_text))

    def _on_connect_clicked(self):
        if self.broker.is_connected:
            self.broker.disconnect()
            return
        config = self.broker.get_config()
        qmt_path = config.get("qmt_path", "")
        account = config.get("account", "")
        if not qmt_path or not account:
            QMessageBox.warning(self, "提示", "请先在交易窗口中配置券商路径和账号")
            return
        self.broker.connect_async(qmt_path, account)
        self.connect_btn.setEnabled(False)
        self.status_label.setText("连接中...")
        self._refresh_client_status_safe()

    def _on_connection_changed(self, connected: bool, message: str):
        self.connect_btn.setEnabled(True)
        if connected:
            self.status_icon.setText("🟢")
            self.status_label.setText("已连接")
            self.connect_btn.setText("断开")
            self.refresh()
        else:
            self.status_icon.setText("🔴")
            self.status_label.setText("未连接")
            self.connect_btn.setText("连接券商")
            self._clear_display()
        self._refresh_client_status_safe()

    def _on_client_state_changed(self, _state: dict):
        self._refresh_client_status_safe()

    def _refresh_client_status_safe(self):
        if self.shared_broker_panel is not None and not self.show_connection_panel:
            self.shared_broker_panel.refresh_client_status()
            return
        if self._status_worker and self._status_worker.isRunning():
            return
        self._status_worker = _ClientStatusWorker(self.broker, parent=self)
        self._status_worker.finished_status.connect(self._apply_client_status)
        self._status_worker.failed_status.connect(self._apply_client_status_error)
        self._status_worker.start()

    def _apply_client_status(self, status: dict):
        text = status.get("message", "客户端: 未检测")
        self.client_status_label.setText(f"客户端: {text}")
        running = bool(status.get("running"))
        if running:
            self.client_status_label.setStyleSheet("color: #5cb85c;")
        else:
            self.client_status_label.setStyleSheet("color: #888;")
        self._status_worker = None

    def _apply_client_status_error(self, message: str):
        logger.warning("刷新 QMT 客户端状态失败: %s", message)
        self.client_status_label.setText("客户端: 状态检测失败")
        self.client_status_label.setStyleSheet("color: #d9534f;")
        self._status_worker = None

    def show_client_workflow_status(self, message: str, *, success: Optional[bool] = None):
        if self.shared_broker_panel is not None and not self.show_connection_panel:
            self.shared_broker_panel.show_client_workflow_status(message, success=success)
            return
        self.client_status_label.setText(f"客户端: {message}")
        if success is True:
            self.client_status_label.setStyleSheet("color: #5cb85c;")
        elif success is False:
            self.client_status_label.setStyleSheet("color: #d9534f;")
        else:
            self.client_status_label.setStyleSheet("color: #f0ad4e;")

    def set_scheduler_status(self, text: str, color: str = "#6B7B8D"):
        self.lbl_scheduler_status.setText(text)
        self.lbl_scheduler_status.setStyleSheet(f"color:{color};font-size:11px;")

    def refresh(self):
        if not self.broker.is_connected:
            return
        if self._refresh_worker and self._refresh_worker.isRunning():
            return
        self._refresh_worker = _AccountRefreshWorker(self.broker, parent=self)
        self._refresh_worker.refresh_ready.connect(self._apply_refresh_result)
        self._refresh_worker.refresh_failed.connect(self._on_refresh_failed)
        self._refresh_worker.start()

    def _apply_refresh_result(self, asset, positions):
        try:
            relevant_positions = positions
            if self.position_scope == SCAN_SCOPE_AI_MANAGED:
                relevant_positions = self._filter_ai_strategy_positions(positions)
            self._update_assets(asset, relevant_positions)
        except Exception as exc:
            logger.warning("AccountPanel refresh apply failed: %s", exc)
        finally:
            self._refresh_worker = None

    def _on_refresh_failed(self, message: str):
        logger.warning("AccountPanel refresh failed: %s", message)
        self._refresh_worker = None

    def _update_assets(self, asset=None, positions=None):
        try:
            if asset is None:
                asset = self.broker.query_stock_asset()
            if asset is None:
                return
            if self.position_scope == SCAN_SCOPE_UNMANAGED:
                live_positions = self._build_unmanaged_live_positions_from_records(asset, positions)
                account = self.strategy_budget.build_account_snapshot(
                    UNMANAGED_STRATEGY_ID,
                    strategy_name=UNMANAGED_STRATEGY_NAME,
                    virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
                    real_total_asset=float(self._record_value(asset, "total_asset", default=0.0) or 0.0),
                    live_positions=[
                        {
                            "stock_code": item.get("code", "") or "",
                            "name": item.get("name", "") or "",
                            "volume": int(item.get("volume", 0) or 0),
                            "market_value": float(item.get("market_value", 0.0) or 0.0),
                        }
                        for item in live_positions
                    ],
                )
            else:
                if positions is None:
                    positions = self._filter_ai_strategy_positions(self.broker.query_stock_positions() or [])
                live_positions = [
                    {"market_value": float(self._record_value(pos, "market_value", default=0.0) or 0.0)}
                    for pos in (positions or [])
                ]
                account = self.strategy_budget.build_account_snapshot(
                    AI_STOCK_STRATEGY_ID,
                    strategy_name=AI_STOCK_STRATEGY_NAME,
                    virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
                    real_total_asset=float(self._record_value(asset, "total_asset", default=0.0) or 0.0),
                    live_positions=live_positions,
                )
            total = float(account.get("total_asset", 0.0) or 0.0)
            cash = float(account.get("available_cash", 0.0) or 0.0)
            market = float(account.get("market_value", 0.0) or 0.0)
            profit = float(account.get("total_pnl", 0.0) or 0.0)
            self.lbl_total_asset.setText(f"¥{total:,.2f}")
            self.lbl_available.setText(f"¥{cash:,.2f}")
            self.lbl_market_value.setText(f"¥{market:,.2f}")
            color = "green" if profit >= 0 else "red"
            self.lbl_profit.setText(f"<span style='color:{color}'>¥{profit:,.2f}</span>")
        except Exception as exc:
            logger.warning("AccountPanel update assets failed: %s", exc)
            self._clear_display()

    def _clear_display(self):
        self.lbl_total_asset.setText("-")
        self.lbl_available.setText("-")
        self.lbl_market_value.setText("-")
        self.lbl_profit.setText("-")

    def _load_trade_config(self):
        self._trade_config_loading = True
        try:
            cfg = self.auto_trade_config_service.get_config()
            self._set_combo_data(self.execution_sequence_combo, cfg.execution_sequence)
            self._set_combo_data(self.buy_sizing_combo, cfg.buy_sizing_mode)
            self.buy_amount_spin.setValue(float(cfg.buy_value_per_order or 0.0))
            self.buy_pct_spin.setValue(float(cfg.buy_position_pct or 0.0) * 100.0)
            self._set_combo_data(self.sell_sizing_combo, cfg.sell_sizing_mode)
            self.max_buy_orders_spin.setValue(int(cfg.max_buy_orders_per_day or 0))
            self.max_sell_orders_spin.setValue(int(cfg.max_sell_orders_per_day or 0))
            self.max_new_positions_spin.setValue(int(cfg.max_new_positions_per_day or 0))
            self.allow_open_new_cb.setChecked(bool(cfg.allow_open_new_position))
            self.allow_add_existing_cb.setChecked(bool(cfg.allow_add_to_existing))
            self.trade_config_status.setText("已加载当前自动交易配置")
            self.trade_config_status.setStyleSheet("color:#888; font-size:12px;")
        finally:
            self._trade_config_loading = False
            self._update_trade_config_widget_state()

    # ------------------------------------------------------------------
    #  统一配置入口：⚙ 查看配置 -> 独立配置弹窗
    # ------------------------------------------------------------------

    def _get_config_dialog(self):
        from trading_app.widgets.ai_trade_decision.dialogs import AIStrategyConfigDialog

        if self._config_dialog is None:
            self._config_dialog = AIStrategyConfigDialog(self, parent=self.window())
        return self._config_dialog

    def _on_toggle_config(self):
        """打开独立的策略配置弹窗。"""
        self._load_trade_config()
        if self.risk_policy_panel is not None:
            try:
                self.risk_policy_panel.reload()
            except Exception as exc:
                logger.error("reload 策略风控面板失败: %s", exc, exc_info=True)
        self._lock_config_panels()
        dialog = self._get_config_dialog()
        dialog.prepare_for_open()
        dialog.exec()

    def refresh_shared_setting_hint(self) -> None:
        """保留空实现，供统一入口刷新时复用。"""
        return

    def request_unlock_config(self) -> bool:
        """二次确认后解锁配置容器。"""
        reply = QMessageBox.question(
            self,
            "解锁编辑",
            "确定要解锁配置面板进行编辑吗？\n修改后请点击对应模块的『保存』按钮。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._unlock_config_panels()
            return True
        return False

    def _lock_config_panels(self):
        """将配置容器内的输入控件设为只读（QLabel / QGroupBox 不动）。"""
        self._config_locked = True
        for w in self._config_container.findChildren(QWidget):
            if not isinstance(w, (QLabel, QGroupBox)):
                w.setEnabled(False)

    def _unlock_config_panels(self):
        """解锁配置容器，恢复所有控件的可编辑状态。"""
        self._config_locked = False
        for w in self._config_container.findChildren(QWidget):
            w.setEnabled(True)

    def _update_trade_config_widget_state(self):
        buy_mode = self.buy_sizing_combo.currentData() or "equal_slots"
        self.buy_amount_spin.setEnabled(buy_mode == "fixed_amount")
        self.buy_pct_spin.setEnabled(buy_mode == "fixed_pct")

    def _save_trade_config(self):
        if self._trade_config_loading:
            return
        try:
            config = self.auto_trade_config_service.update_config(
                execution_sequence=self.execution_sequence_combo.currentData() or "sell_first",
                buy_sizing_mode=self.buy_sizing_combo.currentData() or "equal_slots",
                buy_value_per_order=float(self.buy_amount_spin.value()),
                buy_position_pct=float(self.buy_pct_spin.value()) / 100.0,
                sell_sizing_mode=self.sell_sizing_combo.currentData() or "signal_driven",
                max_buy_orders_per_day=int(self.max_buy_orders_spin.value()),
                max_sell_orders_per_day=int(self.max_sell_orders_spin.value()),
                max_new_positions_per_day=int(self.max_new_positions_spin.value()),
                allow_open_new_position=self.allow_open_new_cb.isChecked(),
                allow_add_to_existing=self.allow_add_existing_cb.isChecked(),
            )
            self.trade_config_status.setText(
                f"已保存: {self._execution_sequence_label(config.execution_sequence)} / "
                f"{self._buy_mode_label(config.buy_sizing_mode)} / "
                f"{self._sell_mode_label(config.sell_sizing_mode)}"
            )
            self.trade_config_status.setStyleSheet("color:#4caf50; font-size:12px;")
        except Exception as exc:
            logger.exception("保存自动交易方式失败")
            self.trade_config_status.setText(f"保存失败: {exc}")
            self.trade_config_status.setStyleSheet("color:#d9534f; font-size:12px;")

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _execution_sequence_label(value: str) -> str:
        mapping = {
            "sell_first": "先卖后买",
            "buy_first": "先买后卖",
            "sell_only": "只卖出",
            "buy_only": "只买入",
        }
        return mapping.get(value, value or "-")

    @staticmethod
    def _buy_mode_label(value: str) -> str:
        mapping = {
            "equal_slots": "均分",
            "fixed_amount": "固定金额",
            "fixed_pct": "固定仓位",
        }
        return mapping.get(value, value or "-")

    @staticmethod
    def _sell_mode_label(value: str) -> str:
        mapping = {
            "signal_driven": "按AI信号",
            "full_exit": "清仓",
            "half_exit": "半仓",
        }
        return mapping.get(value, value or "-")

    def get_broker_context(self) -> BrokerContext:
        if self.position_scope == SCAN_SCOPE_UNMANAGED:
            return self.get_unmanaged_broker_context()
        if not self.broker.is_connected:
            return BrokerContext()
        try:
            asset = self.broker.query_stock_asset()
            positions = self.broker.query_stock_positions() or []
            positions = self._filter_ai_strategy_positions(positions)
            state_positions = self.strategy_budget.get_strategy_state_record(
                AI_STOCK_STRATEGY_ID,
                strategy_name=AI_STOCK_STRATEGY_NAME,
                virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            ).get_positions()
            top = []
            for p in positions:
                code = self._plain_code(getattr(p, "stock_code", "") or "")
                budget_pos = state_positions.get(code)
                top.append({
                    "code": code,
                    "volume": int(getattr(p, "volume", 0) or 0),
                    "cost_price": float(getattr(budget_pos, "avg_cost", 0.0) or 0.0),
                    "market_value": float(getattr(p, "market_value", 0) or 0),
                })
            return BrokerContext(
                connected=True,
                account_id=getattr(self.broker, "_last_config", {}).get("account", ""),
                total_asset=float(getattr(asset, "total_asset", 0) or 0),
                available_cash=float(getattr(asset, "cash", 0) or 0),
                position_count=len(positions),
                top_positions=top,
            )
        except Exception:
            return BrokerContext(connected=True)

    def get_unmanaged_broker_context(
        self,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> BrokerContext:
        if not self.broker.is_connected:
            return BrokerContext()
        positions = list(positions or self.get_unmanaged_live_positions())
        try:
            asset = self.broker.query_stock_asset()
        except Exception:
            return BrokerContext(connected=True)
        snapshot = self.strategy_budget.build_account_snapshot(
            UNMANAGED_STRATEGY_ID,
            strategy_name=UNMANAGED_STRATEGY_NAME,
            virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
            real_total_asset=float(self._record_value(asset, "total_asset", default=0.0) or 0.0),
            live_positions=[
                {
                    "stock_code": item.get("code", "") or "",
                    "name": item.get("name", "") or "",
                    "volume": int(item.get("volume", 0) or 0),
                    "market_value": float(item.get("market_value", 0.0) or 0.0),
                }
                for item in positions
            ],
        )
        top = [
            {
                "code": item.get("code", "") or "",
                "volume": int(item.get("volume", 0) or 0),
                "cost_price": float(item.get("cost_price", 0.0) or 0.0),
                "market_value": float(item.get("market_value", 0.0) or 0.0),
            }
            for item in positions
            if item.get("code")
        ]
        return BrokerContext(
            connected=True,
            account_id=getattr(self.broker, "_last_config", {}).get("account", ""),
            total_asset=float(snapshot.get("total_asset", 0.0) or 0.0),
            available_cash=float(snapshot.get("available_cash", 0.0) or 0.0),
            position_count=len(positions),
            top_positions=top,
        )

    def get_live_positions(self) -> List[Dict[str, Any]]:
        if self.position_scope == SCAN_SCOPE_UNMANAGED:
            return self.get_unmanaged_live_positions()
        if not self.broker.is_connected:
            return []
        try:
            positions = self.broker.query_stock_positions() or []
        except Exception:
            return []

        state_positions = self.strategy_budget.get_strategy_state_record(
            AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
        ).get_positions()
        results: List[Dict[str, Any]] = []
        for pos in self._filter_ai_strategy_positions(positions):
            volume = int(getattr(pos, "volume", 0) or 0)
            code = self._plain_code(getattr(pos, "stock_code", "") or "")
            budget_pos = state_positions.get(code)
            results.append({
                "code": code,
                "name": self._resolve_symbol_name(code, getattr(pos, "stock_name", "") or ""),
                "volume": volume,
                "can_use_volume": int(getattr(pos, "can_use_volume", 0) or 0),
                "cost_price": float(getattr(budget_pos, "avg_cost", 0.0) or 0.0),
                "market_value": float(getattr(pos, "market_value", 0) or 0),
                "profit_rate": float(getattr(pos, "profit_rate", 0) or 0),
            })
        return results

    def get_unmanaged_live_positions(self) -> List[Dict[str, Any]]:
        if not self.broker.is_connected:
            return []
        try:
            asset = self.broker.query_stock_asset()
            broker_positions = self.broker.query_stock_positions() or []
        except Exception:
            return []
        return self._build_unmanaged_live_positions_from_records(asset, broker_positions)

    def _build_unmanaged_live_positions_from_records(self, asset: Any, broker_positions: Any) -> List[Dict[str, Any]]:
        if not asset:
            return []

        broker_live_positions: List[Dict[str, Any]] = []
        broker_reconcile_positions: List[Dict[str, Any]] = []
        live_meta: Dict[str, Dict[str, Any]] = {}
        for pos in broker_positions:
            code = self._plain_code(self._record_value(pos, "stock_code", default="") or "")
            volume = int(self._record_value(pos, "volume", default=0) or 0)
            if not code or volume <= 0:
                continue
            item = {
                "stock_code": code,
                "market_value": float(self._record_value(pos, "market_value", default=0.0) or 0.0),
                "volume": volume,
                "name": self._resolve_symbol_name(code, self._record_value(pos, "stock_name", default="") or ""),
            }
            broker_live_positions.append(item)
            live_meta[code] = {
                "name": item["name"],
                "can_use_volume": int(self._record_value(pos, "can_use_volume", default=volume) or volume),
            }
            broker_reconcile_positions.append(
                {
                    "stock_code": code,
                    "volume": volume,
                    "open_price": float(self._record_value(pos, "open_price", default=0.0) or 0.0),
                }
            )

        try:
            self.strategy_budget.reconcile_unmanaged_with_broker(
                broker_cash=float(
                    self._record_value(
                        asset,
                        "cash",
                        default=self._record_value(asset, "available_cash", default=0.0),
                    )
                    or 0.0
                ),
                broker_positions=broker_reconcile_positions,
            )
        except Exception:
            logger.exception("对账未管理账户持仓失败")
            return []

        rows = self.strategy_budget.get_positions_view(
            UNMANAGED_STRATEGY_ID,
            strategy_name=UNMANAGED_STRATEGY_NAME,
            virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
            real_total_asset=float(self._record_value(asset, "total_asset", default=0.0) or 0.0),
            live_positions=broker_live_positions,
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            code = self._plain_code(str(row.get("stock_code", "") or ""))
            if not code:
                continue
            meta = live_meta.get(code, {})
            results.append(
                {
                    "code": code,
                    "name": meta.get("name") or row.get("stock_name") or self._resolve_symbol_name(code),
                    "volume": int(row.get("quantity", 0) or 0),
                    "can_use_volume": int(meta.get("can_use_volume", row.get("quantity", 0)) or 0),
                    "cost_price": float(row.get("avg_cost", 0.0) or 0.0),
                    "market_value": float(row.get("market_value", 0.0) or 0.0),
                    "profit_rate": float(row.get("unrealized_pnl_pct", 0.0) or 0.0) / 100.0,
                    "scan_scope": SCAN_SCOPE_UNMANAGED,
                }
            )
        return results

    def _filter_ai_strategy_positions(self, positions) -> List[Any]:
        filtered: List[Any] = []
        for pos in positions or []:
            volume = int(self._record_value(pos, "volume", default=0) or 0)
            if volume <= 0:
                continue
            code = self._plain_code(self._record_value(pos, "stock_code", default="") or "")
            if not code:
                continue
            owner = self.strategy_registry.get_owner(code)
            if owner is None or not owner.enabled:
                continue
            if owner.strategy_id != AI_STOCK_STRATEGY_ID:
                continue
            filtered.append(pos)
        return filtered

    @staticmethod
    def _plain_code(code: str) -> str:
        value = (code or "").strip().upper()
        return value.split(".")[0] if "." in value else value

    @staticmethod
    def _record_value(data: Any, key: str, default: Any = None) -> Any:
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)

    def _resolve_symbol_name(self, code: str, fallback_name: str = "") -> str:
        if fallback_name:
            return fallback_name
        trade_window = self._find_trade_window()
        if trade_window:
            looked_up = trade_window.lookup_symbol_name(code)
            if looked_up:
                return looked_up
        return code

    def _display_code(self, code: str) -> str:
        if "." in code:
            return code.split(".")[0]
        return code

    def _find_trade_window(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "lookup_symbol_name") and hasattr(parent, "order_panel"):
                return parent
            parent = parent.parent() if hasattr(parent, "parent") and callable(parent.parent) else None
        return None


# ───────────────────────────────────────────────────────────────────────────
#  Right panel: Order execution and details
# ───────────────────────────────────────────────────────────────────────────
