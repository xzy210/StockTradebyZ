"""AI 实盘决策 — 独立窗口

将 AI 实盘决策分析、统一执行、账户信息三大功能聚合在同一面板中，
使用户无需在多个窗口间切换即可完成「分析 → 决策 → 执行 → 追踪」的完整流程。
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
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
    from trading_app.services.ai.evidence_trace_store import (
        EvidenceStep,
        EvidenceTrace,
        ToolCallTrace,
    )
    from trading_app.services.ai.decision_session_store import (
        DecisionSession,
        DecisionSessionItem,
    )
    from trading_app.services.ai.decision_lifecycle_service import DecisionLifecycleService
    from trading_app.services.ai.ai_stock_strategy_params_service import get_ai_stock_strategy_params_service
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
    from trading_app.services.ai.evidence_trace_store import (
        EvidenceStep,
        EvidenceTrace,
        ToolCallTrace,
    )
    from trading_app.services.ai.decision_session_store import (
        DecisionSession,
        DecisionSessionItem,
    )
    from trading_app.services.ai.decision_lifecycle_service import DecisionLifecycleService
    from trading_app.services.ai.ai_stock_strategy_params_service import get_ai_stock_strategy_params_service
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


class _AgentPrepareRequestThread(QThread):
    """后台准备 Agent 请求，避免证据采集阻塞 UI 线程。"""

    prepared = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        base_system_prompt: str,
        context: AgentRuntimeContext,
        task_mode: str,
        chat_history: List[Dict[str, Any]],
        latest_user_content: Any,
        parent=None,
    ):
        super().__init__(parent)
        self.base_system_prompt = base_system_prompt
        self.context = context
        self.task_mode = task_mode
        self.chat_history = list(chat_history or [])
        self.latest_user_content = latest_user_content

    def run(self):
        try:
            runtime = StockAgentRuntime()
            prepared = runtime.prepare_request(
                base_system_prompt=self.base_system_prompt,
                context=self.context,
                task_mode=self.task_mode,
                chat_history=self.chat_history,
                latest_user_content=self.latest_user_content,
            )
        except Exception as exc:
            logger.exception("后台准备 AI 证据失败")
            self.failed.emit(str(exc))
            return
        self.prepared.emit(prepared)


class DecisionPanel(QWidget):
    """AI trade decision analysis and display panel."""

    decision_ready = pyqtSignal(object)  # TradeDecision
    scan_completed = pyqtSignal(object)
    market_view_requested = pyqtSignal(str, str)

    def __init__(
        self,
        context_provider=None,
        parent=None,
        *,
        allow_candidate_pool_scan: bool = True,
        position_scan_label: str = "持仓巡检",
        position_scan_hint: str = "持仓巡检: 自动读取当前券商持仓，逐只生成持有/加仓/减仓/卖出决策",
    ):
        super().__init__(parent)
        self.context_provider = context_provider
        self.allow_candidate_pool_scan = bool(allow_candidate_pool_scan)
        self.position_scan_label = str(position_scan_label or "持仓巡检")
        self.position_scan_hint = str(position_scan_hint or "")
        self.agent_runtime = StockAgentRuntime()
        self.risk_guard = RiskGuardService()
        self.decision_lifecycle = DecisionLifecycleService()
        self._current_decision: Optional[TradeDecision] = None
        self._current_risk_result = None
        self._current_decision_record_id = ""
        self._chat_thread = None
        self._ai_config = self._load_ai_config()
        self.strategy_params_service = get_ai_stock_strategy_params_service()
        self._strategy_params = self.strategy_params_service.load_params()
        self.stock_pool_service = get_stock_pool_service()
        self._full_response = ""
        self._context_for_decision = None
        self._current_prepared_request = None
        self._prepare_thread = None
        self._current_mode = DECISION_MODE_POSITION_SCAN
        self._scan_queue: List[Dict[str, Any]] = []
        self._scan_results: List[Dict[str, Any]] = []
        self._scheduled_scan_records: List[Dict[str, Any]] = []
        self._scheduled_scan_record_items: List[Dict[str, Any]] = []
        self._current_scan_item: Optional[Dict[str, Any]] = None
        self._current_scan_index = -1
        self._scan_in_progress = False
        self._scan_total_count = 0
        self._scan_completed_count = 0
        self._scan_active_workers: Dict[str, Any] = {}
        self._scan_worker_states: Dict[str, Dict[str, Any]] = {}
        self._scan_shared_evidence_cache: Optional[Dict[str, Any]] = None
        self._active_scan_run_id: str = ""
        self._active_scan_source: str = ""
        self._active_scan_task_id: str = ""
        self._active_scan_scope: str = SCAN_SCOPE_AI_MANAGED
        self._active_scan_label: str = ""
        self._active_scan_allow_auto_execute: bool = True
        self._active_scan_broker_context: Optional[BrokerContext] = None
        self._run_context_override: Optional[DecisionRunContext] = None
        self._stream_started = False
        self._progress_cards: List[CollapsibleStepCard] = []
        self._current_trace_path = ""
        self._trace_paths_by_key: Dict[str, str] = {}
        self._active_session_id = ""
        self._session_rows: List[DecisionSession] = []
        self._current_session_items: List[DecisionSessionItem] = []
        self._setup_ui()

    def _load_ai_config(self) -> dict:
        trading_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(
            trading_app_dir,
            "config", "ai_config.json",
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # -- Top: mode / symbol / model selector --
        self.top_controls_widget = QWidget(self)
        top_row = QHBoxLayout(self.top_controls_widget)
        top_row.setContentsMargins(0, 0, 0, 0)
        self.mode_label = QLabel("模式:")
        top_row.addWidget(self.mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(self.position_scan_label, DECISION_MODE_POSITION_SCAN)
        if self.allow_candidate_pool_scan:
            self.mode_combo.addItem("候选池巡检", DECISION_MODE_CANDIDATE_POOL_SCAN)
        self.mode_combo.setFixedWidth(120)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top_row.addWidget(self.mode_combo)

        self.symbol_label = QLabel("标的:")
        top_row.addWidget(self.symbol_label)
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("输入代码，如 000001.SZ（留空则用主窗口当前标的）")
        self.symbol_input.setFixedWidth(240)
        self.symbol_input.setVisible(False)
        top_row.addWidget(self.symbol_input)

        self.watchlist_group_combo = QComboBox()
        self.watchlist_group_combo.setFixedWidth(140)
        self.watchlist_group_combo.setVisible(False)
        top_row.addWidget(self.watchlist_group_combo)

        self.mode_hint_label = QLabel(self.position_scan_hint)
        self.mode_hint_label.setStyleSheet("color: #666;")
        top_row.addWidget(self.mode_hint_label)

        self.model_label = QLabel("模型:")
        top_row.addWidget(self.model_label)
        self.model_combo = QComboBox()
        model_configs = self._ai_config.get("model_configs", {})
        if model_configs:
            self.model_combo.addItems(list(model_configs.keys()))
        else:
            self.model_combo.addItems([
                "deepseek-chat",
                "gpt-4o",
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "kimi-k2.5",
            ])
        selected = self._ai_config.get("selected_model", "") or getattr(self._strategy_params, "model_name", "")
        if selected and self.model_combo.findText(selected) >= 0:
            self.model_combo.setCurrentText(selected)
        self.model_combo.setFixedWidth(180)
        top_row.addWidget(self.model_combo)

        top_row.addStretch()
        self.analyze_btn = QPushButton("🔍 生成交易决策")
        self.analyze_btn.setFixedHeight(36)
        self.analyze_btn.setStyleSheet(
            "QPushButton { background-color: #107c10; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { background-color: #0e6b0e; }"
            "QPushButton:disabled { background-color: #888888; }"
        )
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        top_row.addWidget(self.analyze_btn)
        layout.addWidget(self.top_controls_widget)

        # -- Stacked: placeholder vs result --
        self.stack = QStackedWidget()

        # Page 0: placeholder
        mode_placeholder = (
            "当前支持“持仓巡检”和“候选池巡检”两种模式。"
            if self.allow_candidate_pool_scan
            else "当前仅提供未管理持仓巡检模式。"
        )
        placeholder = QLabel(
            "点击「开始巡检」开始分析当前策略任务\n\n"
            + mode_placeholder
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self.stack.addWidget(placeholder)

        # Page 1: progress
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.setSpacing(10)
        self.progress_label = QLabel("正在分析...")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #0078d4; font-size: 14px; font-weight: bold;")
        progress_layout.addWidget(self.progress_label)

        self.progress_hint_label = QLabel("以下为本轮交易决策生成的中间步骤概要")
        self.progress_hint_label.setStyleSheet("color: #888;")
        self.progress_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_hint_label)

        self.progress_scroll = QScrollArea()
        self.progress_scroll.setWidgetResizable(True)
        self.progress_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.progress_scroll.setStyleSheet("background-color: transparent; border: none;")
        self.progress_cards_host = QWidget()
        self.progress_cards_layout = QVBoxLayout(self.progress_cards_host)
        self.progress_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_cards_layout.setSpacing(2)
        self.progress_cards_layout.addStretch()
        self.progress_scroll.setWidget(self.progress_cards_host)
        progress_layout.addWidget(self.progress_scroll, stretch=1)
        self.stack.addWidget(progress_widget)

        # Page 2: result area
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)

        # Tab: analysis text + decision card + history
        self.result_tabs = QTabWidget()

        # Tab 1: Decision sessions
        self.session_widget = QWidget()
        session_layout = QVBoxLayout(self.session_widget)
        session_layout.setContentsMargins(0, 0, 0, 0)
        session_layout.setSpacing(6)
        session_toolbar = QHBoxLayout()
        session_toolbar.setContentsMargins(0, 0, 0, 0)
        self.session_summary_label = QLabel("会话汇总：暂无记录")
        self.session_summary_label.setStyleSheet("color:#888;")
        session_toolbar.addWidget(self.session_summary_label, stretch=1)
        self.session_refresh_btn = QPushButton("刷新")
        self.session_refresh_btn.clicked.connect(self._refresh_session_list)
        session_toolbar.addWidget(self.session_refresh_btn)
        session_layout.addLayout(session_toolbar)

        session_splitter = QSplitter(Qt.Orientation.Horizontal)
        session_splitter.setChildrenCollapsible(False)
        self.session_table = QTableWidget(0, 5)
        self.session_table.setHorizontalHeaderLabels(["开始时间", "来源", "任务", "状态", "数量"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.session_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.session_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.session_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.setAlternatingRowColors(False)
        self.session_table.setStyleSheet(self._SESSION_TABLE_QSS)
        self.session_table.itemSelectionChanged.connect(self._on_session_selection_changed)
        session_splitter.addWidget(self.session_table)

        session_detail_host = QWidget()
        session_detail_layout = QVBoxLayout(session_detail_host)
        session_detail_layout.setContentsMargins(0, 0, 0, 0)
        session_detail_layout.setSpacing(4)
        self.session_detail_label = QLabel("选择左侧会话后查看本轮巡检/单票/调度明细")
        self.session_detail_label.setStyleSheet("color:#888;")
        session_detail_layout.addWidget(self.session_detail_label)
        self.session_item_table = QTableWidget(0, 7)
        self.session_item_table.setHorizontalHeaderLabels(["时间", "类型", "标的", "操作", "状态", "决策ID", "证据"])
        self.session_item_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.session_item_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.session_item_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.session_item_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.session_item_table.verticalHeader().setVisible(False)
        self.session_item_table.setAlternatingRowColors(False)
        self.session_item_table.setStyleSheet(self._SESSION_TABLE_QSS)
        self.session_item_table.itemSelectionChanged.connect(self._on_session_item_selection_changed)
        self.session_item_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_item_table.customContextMenuRequested.connect(self._on_session_item_context_menu)
        session_detail_layout.addWidget(self.session_item_table, stretch=1)
        session_splitter.addWidget(session_detail_host)
        session_splitter.setSizes([260, 520])
        session_layout.addWidget(session_splitter, stretch=1)
        self.result_tabs.addTab(self.session_widget, "决策会话")
        self._refresh_session_list()

        # Tab 2: Process review / evidence browser
        self.process_review_widget = QWidget()
        process_review_root = QVBoxLayout(self.process_review_widget)
        process_review_root.setContentsMargins(0, 0, 0, 0)
        process_review_root.setSpacing(6)
        process_toolbar = QHBoxLayout()
        process_toolbar.setContentsMargins(0, 0, 0, 0)
        process_toolbar.addWidget(QLabel("证据轨迹:"))
        self.process_trace_combo = QComboBox()
        self.process_trace_combo.setMinimumWidth(300)
        self.process_trace_combo.currentIndexChanged.connect(self._on_process_trace_selected)
        process_toolbar.addWidget(self.process_trace_combo, stretch=1)
        self.process_search_input = QLineEdit()
        self.process_search_input.setPlaceholderText("搜索步骤/工具/摘要")
        self.process_search_input.textChanged.connect(self._render_selected_trace)
        process_toolbar.addWidget(self.process_search_input)
        self.process_refresh_btn = QPushButton("刷新")
        self.process_refresh_btn.clicked.connect(self._refresh_process_trace_list)
        process_toolbar.addWidget(self.process_refresh_btn)
        process_review_root.addLayout(process_toolbar)
        self.process_review_summary_label = QLabel("当前实时过程预览")
        self.process_review_summary_label.setStyleSheet("color:#888;")
        process_review_root.addWidget(self.process_review_summary_label)
        self.process_review_scroll = QScrollArea()
        self.process_review_scroll.setWidgetResizable(True)
        self.process_review_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.process_review_scroll.setStyleSheet("background-color: transparent; border: none;")
        self.process_review_host = QWidget()
        self.process_review_layout = QVBoxLayout(self.process_review_host)
        self.process_review_layout.setContentsMargins(0, 0, 0, 0)
        self.process_review_layout.setSpacing(2)
        self.process_review_layout.addStretch()
        self.process_review_scroll.setWidget(self.process_review_host)
        process_review_root.addWidget(self.process_review_scroll, stretch=1)
        self.result_tabs.addTab(self.process_review_widget, "过程回看")
        self._refresh_process_trace_list()

        # Tab 3: Current batch summary
        self.scan_table = QTableWidget(0, 9)
        self.scan_table.setHorizontalHeaderLabels(
            ["序号", "代码", "名称", "操作", "置信度", "现价", "成本", "风控", "状态"]
        )
        self.scan_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scan_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scan_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scan_table.customContextMenuRequested.connect(
            lambda pos: self._on_scan_result_context_menu(self.scan_table, pos, self._scan_results)
        )
        self.scan_table.verticalHeader().setVisible(False)
        self.scan_table.setAlternatingRowColors(True)
        self.scan_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e2e;
                alternate-background-color: #2a2a3e;
                color: #d0d0d0;
                gridline-color: #3a3a4e;
                selection-background-color: #3a5fcd;
                selection-color: #ffffff;
            }
            QTableWidget::item { padding: 4px 6px; }
            QHeaderView::section {
                background-color: #16162a;
                color: #e0e0e0;
                padding: 5px 6px;
                border: 1px solid #3a3a4e;
                font-weight: bold;
            }
        """)
        self.scan_table.itemSelectionChanged.connect(self._on_scan_selection_changed)
        self.result_tabs.addTab(self.scan_table, "当前巡检")

        # Tab 4: Scheduled scan records
        self.scheduled_scan_records_widget = QWidget()
        scheduled_layout = QVBoxLayout(self.scheduled_scan_records_widget)
        scheduled_layout.setContentsMargins(0, 0, 0, 0)
        scheduled_layout.setSpacing(6)

        self.scheduled_scan_summary_label = QLabel("暂无定时巡检记录")
        self.scheduled_scan_summary_label.setStyleSheet("color: #888;")
        scheduled_layout.addWidget(self.scheduled_scan_summary_label)

        scheduled_splitter = QSplitter(Qt.Orientation.Vertical)
        scheduled_splitter.setChildrenCollapsible(False)
        scheduled_splitter.setHandleWidth(12)

        batch_host = QWidget()
        batch_layout = QVBoxLayout(batch_host)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(4)
        batch_layout.addWidget(QLabel("批次列表"))
        self.scheduled_scan_batches_table = QTableWidget(0, 7)
        self.scheduled_scan_batches_table.setHorizontalHeaderLabels(
            ["完成时间", "任务", "巡检", "总数", "可操作", "风控", "说明"]
        )
        self.scheduled_scan_batches_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scheduled_scan_batches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scheduled_scan_batches_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scheduled_scan_batches_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.scheduled_scan_batches_table.verticalHeader().setVisible(False)
        self.scheduled_scan_batches_table.setAlternatingRowColors(True)
        self.scheduled_scan_batches_table.setStyleSheet(self.scan_table.styleSheet())
        self.scheduled_scan_batches_table.itemSelectionChanged.connect(
            self._on_scheduled_scan_batch_selection_changed
        )
        batch_layout.addWidget(self.scheduled_scan_batches_table, stretch=1)
        scheduled_splitter.addWidget(batch_host)

        detail_host = QWidget()
        detail_layout = QVBoxLayout(detail_host)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(4)
        self.scheduled_scan_detail_label = QLabel("选择一组定时巡检记录后，可在下方查看明细")
        self.scheduled_scan_detail_label.setStyleSheet("color: #888;")
        detail_layout.addWidget(self.scheduled_scan_detail_label)
        self.scheduled_scan_detail_table = QTableWidget(0, 9)
        self.scheduled_scan_detail_table.setHorizontalHeaderLabels(
            ["序号", "代码", "名称", "操作", "置信度", "现价", "成本", "风控", "状态"]
        )
        self.scheduled_scan_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scheduled_scan_detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scheduled_scan_detail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scheduled_scan_detail_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.scheduled_scan_detail_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scheduled_scan_detail_table.customContextMenuRequested.connect(
            lambda pos: self._on_scan_result_context_menu(
                self.scheduled_scan_detail_table,
                pos,
                self._scheduled_scan_record_items,
            )
        )
        self.scheduled_scan_detail_table.verticalHeader().setVisible(False)
        self.scheduled_scan_detail_table.setAlternatingRowColors(True)
        self.scheduled_scan_detail_table.setStyleSheet(self.scan_table.styleSheet())
        self.scheduled_scan_detail_table.itemSelectionChanged.connect(
            self._on_scheduled_scan_detail_selection_changed
        )
        detail_layout.addWidget(self.scheduled_scan_detail_table, stretch=1)
        scheduled_splitter.addWidget(detail_host)
        scheduled_splitter.setSizes([180, 260])
        scheduled_layout.addWidget(scheduled_splitter, stretch=1)
        # Tab 4: Decision detail. Scheduled scan records are folded into
        # Decision Sessions, so this legacy widget is kept internal only.
        self.decision_card_widget = QWidget()
        decision_detail_layout = QVBoxLayout(self.decision_card_widget)
        decision_detail_layout.setContentsMargins(0, 0, 0, 0)
        decision_detail_layout.setSpacing(4)
        self.decision_detail_tabs = QTabWidget(self.decision_card_widget)

        self.decision_summary_widget = QWidget()
        self.decision_card_layout = QVBoxLayout(self.decision_summary_widget)
        self.decision_card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.decision_detail_tabs.addTab(self.decision_summary_widget, "结构化决策")

        self.analysis_display = QTextEdit()
        self.analysis_display.setReadOnly(True)
        self.decision_detail_tabs.addTab(self.analysis_display, "AI 原文")

        decision_detail_layout.addWidget(self.decision_detail_tabs)
        self.result_tabs.addTab(self.decision_card_widget, "决策详情")

        # Tab 5: Decision history + stats
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(4)

        self.stats_bar = QLabel("")
        self.stats_bar.setStyleSheet(
            "background-color: #1e1e2e; color: #d0d0d0; padding: 8px 12px; "
            "border-radius: 4px; font-size: 13px;"
        )
        self.stats_bar.setWordWrap(True)
        history_layout.addWidget(self.stats_bar)

        self.history_table = QTableWidget(0, 9)
        self.history_table.setHorizontalHeaderLabels(
            ["时间", "标的", "操作", "置信度", "入场价", "盈亏%", "盈亏额", "风控", "结果"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e2e;
                alternate-background-color: #2a2a3e;
                color: #d0d0d0;
                gridline-color: #3a3a4e;
                selection-background-color: #3a5fcd;
                selection-color: #ffffff;
            }
            QTableWidget::item { padding: 4px 6px; }
            QHeaderView::section {
                background-color: #16162a;
                color: #e0e0e0;
                padding: 5px 6px;
                border: 1px solid #3a3a4e;
                font-weight: bold;
            }
        """)
        self.history_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_table.customContextMenuRequested.connect(self._on_history_context_menu)
        history_layout.addWidget(self.history_table, stretch=1)

        export_bar = QHBoxLayout()
        export_bar.setContentsMargins(0, 2, 0, 0)
        export_bar.addStretch()
        export_csv_btn = QPushButton("📄 导出CSV")
        export_csv_btn.setFixedHeight(26)
        export_csv_btn.clicked.connect(self._export_csv)
        export_bar.addWidget(export_csv_btn)
        export_html_btn = QPushButton("📊 导出复盘报告")
        export_html_btn.setFixedHeight(26)
        export_html_btn.clicked.connect(self._export_html)
        export_bar.addWidget(export_html_btn)
        history_layout.addLayout(export_bar)

        self.result_tabs.addTab(history_widget, "决策记录")

        result_layout.addWidget(self.result_tabs)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self.decision_status_label = QLabel("")
        action_row.addWidget(self.decision_status_label)
        result_layout.addLayout(action_row)

        self.stack.addWidget(result_widget)
        layout.addWidget(self.stack, stretch=1)
        self._on_mode_changed()
        self._show_persisted_sessions_on_startup()

    _SESSION_TABLE_QSS = """
        QTableWidget {
            background-color: #111318;
            alternate-background-color: #111318;
            color: #F1F5F9;
            gridline-color: #2D3748;
            selection-background-color: #0B74D1;
            selection-color: #FFFFFF;
        }
        QTableWidget::item {
            background-color: #111318;
            color: #F1F5F9;
            padding: 4px 6px;
        }
        QTableWidget::item:selected {
            background-color: #0B74D1;
            color: #FFFFFF;
        }
        QHeaderView::section {
            background-color: #1A202C;
            color: #F8FAFC;
            padding: 5px 6px;
            border: 1px solid #2D3748;
            font-weight: bold;
        }
    """

    def _source_label(self, source: str) -> str:
        mapping = {
            "manual": "手动",
            "scheduled": "定时",
            "single": "单票",
        }
        return mapping.get(str(source or ""), str(source or "-"))

    def _show_persisted_sessions_on_startup(self) -> None:
        """Show persisted decision sessions after restart instead of the empty placeholder."""
        try:
            self._refresh_session_list()
        except Exception:
            logger.debug("启动时刷新 AI 决策会话失败", exc_info=True)
            return
        if not self._session_rows:
            return
        self.stack.setCurrentIndex(2)
        self.result_tabs.setCurrentWidget(self.session_widget)

    def _session_item_type_label(self, item_type: str) -> str:
        mapping = {
            "single_decision": "单票决策",
            "scan_result": "巡检明细",
            "scheduled_scan": "定时批次",
            "decision_record": "决策记录",
        }
        return mapping.get(str(item_type or ""), str(item_type or "-"))

    def _current_strategy_params_hash(self) -> str:
        try:
            self._strategy_params = self.strategy_params_service.load_params()
            return self._strategy_params.params_hash()
        except Exception:
            return ""

    def _system_prompt_seed(self) -> str:
        try:
            self._strategy_params = self.strategy_params_service.load_params()
            prompt = str(getattr(self._strategy_params, "system_prompt", "") or "").strip()
            if prompt:
                return prompt
        except Exception:
            pass
        return "你是一个专业的股票交易决策分析师。"

    def _upsert_decision_session(
        self,
        *,
        session_id: str,
        title: str,
        source: str,
        mode: str = "",
        scan_scope: str = "",
        task_id: str = "",
        status: str = "running",
        summary: str = "",
        started_at: str = "",
        completed_at: str = "",
        items: Optional[List[DecisionSessionItem]] = None,
    ) -> None:
        if not session_id:
            return
        session = DecisionSession(
            session_id=session_id,
            title=title or session_id,
            source=source or "manual",
            mode=mode or self._current_mode,
            scan_scope=scan_scope or self._active_scan_scope,
            task_id=task_id or self._active_scan_task_id,
            model_name=str((getattr(self, "_active_model_cfg", {}) or {}).get("model") or self.get_current_model_name()),
            params_hash=self._current_strategy_params_hash(),
            started_at=started_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            completed_at=completed_at,
            status=status,
            summary=summary,
            items=list(items or []),
        )
        try:
            self.decision_lifecycle.upsert_session(session)
            self._refresh_session_list(select_session_id=session_id)
        except Exception as exc:
            logger.warning("保存 AI 决策会话失败: %s", exc, exc_info=True)

    def _append_decision_session_item(
        self,
        *,
        session_id: str,
        result: Dict[str, Any],
        item_type: str,
    ) -> None:
        if not session_id:
            return
        decision = result.get("decision")
        action = str(getattr(decision, "action", "") or "")
        item = DecisionSessionItem(
            item_id=str(result.get("decision_record_id") or result.get("evidence_trace_path") or uuid4().hex),
            item_type=item_type,
            symbol_code=str(result.get("symbol_code", "") or ""),
            symbol_name=str(result.get("symbol_name", "") or ""),
            decision_record_id=str(result.get("decision_record_id", "") or ""),
            evidence_trace_path=str(result.get("evidence_trace_path", "") or ""),
            action=TRADE_ACTION_LABELS.get(action, action),
            status_text=_build_scan_status_text(decision, result.get("risk_result")),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            payload=_serialize_scan_result_for_record(result),
        )
        try:
            self.decision_lifecycle.append_session_item(session_id, item)
            self._refresh_session_list(select_session_id=session_id)
        except Exception as exc:
            logger.warning("追加 AI 决策会话明细失败: %s", exc, exc_info=True)

    def _complete_decision_session(self, session_id: str, *, summary: str, status: str = "done") -> None:
        if not session_id:
            return
        try:
            self.decision_lifecycle.complete_session(session_id, status=status, summary=summary)
            self._refresh_session_list(select_session_id=session_id)
        except Exception as exc:
            logger.warning("更新 AI 决策会话状态失败: %s", exc, exc_info=True)

    def _refresh_session_list(self, *, select_session_id: str = "") -> None:
        if not hasattr(self, "session_table"):
            return
        current_id = select_session_id
        if not current_id:
            row = self.session_table.currentRow()
            if 0 <= row < len(self._session_rows):
                current_id = self._session_rows[row].session_id
        self._session_rows = [session for _, session in self.decision_lifecycle.list_recent_sessions(limit=80)]
        self.session_table.blockSignals(True)
        self.session_table.setRowCount(0)
        for row, session in enumerate(self._session_rows):
            self.session_table.insertRow(row)
            values = [
                session.started_at or session.completed_at or "-",
                self._source_label(session.source),
                session.title or session.task_id or "-",
                session.status or "-",
                str(len(session.items)),
            ]
            for col, value in enumerate(values):
                self.session_table.setItem(row, col, QTableWidgetItem(value))
        self.session_table.blockSignals(False)
        self.session_summary_label.setText(f"会话汇总：最近 {len(self._session_rows)} 组")
        self.session_table.setToolTip("包含手动巡检、定时巡检和单票决策；原“定时记录”已并入此处。")
        target_row = 0
        if current_id:
            for idx, session in enumerate(self._session_rows):
                if session.session_id == current_id:
                    target_row = idx
                    break
        if self._session_rows:
            self.session_table.selectRow(target_row)
        else:
            self._current_session_items = []
            self.session_item_table.setRowCount(0)
            self.session_detail_label.setText("暂无决策会话")

    def _on_session_selection_changed(self) -> None:
        row = self.session_table.currentRow()
        if row < 0 or row >= len(self._session_rows):
            self._current_session_items = []
            self.session_item_table.setRowCount(0)
            return
        session = self._session_rows[row]
        self._current_session_items = list(session.items or [])
        self.session_detail_label.setText(
            f"{session.title or session.session_id} | {session.status} | "
            f"{len(self._current_session_items)} 条明细 | params={session.params_hash or '-'} | session_id={session.session_id}"
        )
        self.session_item_table.blockSignals(True)
        self.session_item_table.setRowCount(0)
        for item_row, item in enumerate(self._current_session_items):
            self.session_item_table.insertRow(item_row)
            values = [
                item.created_at or "-",
                self._session_item_type_label(item.item_type),
                f"{item.symbol_name}({item.symbol_code})" if item.symbol_name else item.symbol_code or "-",
                item.action or "-",
                item.status_text or "-",
                item.decision_record_id or "-",
                "有" if item.evidence_trace_path else "-",
            ]
            for col, value in enumerate(values):
                self.session_item_table.setItem(item_row, col, QTableWidgetItem(value))
        self.session_item_table.blockSignals(False)
        if self._current_session_items:
            self.session_item_table.selectRow(0)

    def _on_session_item_selection_changed(self) -> None:
        row = self.session_item_table.currentRow()
        if row < 0 or row >= len(self._current_session_items):
            return
        if not hasattr(self, "analysis_display"):
            return
        item = self._current_session_items[row]
        payload = dict(item.payload or {})
        if payload:
            result = self._build_runtime_scan_result_from_record(payload)
            self._display_result(result, switch_to_details=False, emit_decision=False)
        if item.evidence_trace_path:
            self._refresh_process_trace_list(select_path=item.evidence_trace_path)

    def _on_session_item_context_menu(self, pos) -> None:
        row = self.session_item_table.rowAt(pos.y())
        if row < 0 or row >= len(self._current_session_items):
            return
        item = self._current_session_items[row]
        menu = QMenu(self)
        view_action = None
        if item.symbol_code:
            label = f"查看K线 {item.symbol_name}({item.symbol_code})" if item.symbol_name else f"查看K线 {item.symbol_code}"
            view_action = menu.addAction(label)
        evidence_action = None
        if item.evidence_trace_path:
            evidence_action = menu.addAction("查看过程证据")
        chosen = menu.exec(self.session_item_table.viewport().mapToGlobal(pos))
        if view_action is not None and chosen == view_action:
            self.market_view_requested.emit(item.symbol_code, item.symbol_name)
        elif evidence_action is not None and chosen == evidence_action:
            self._refresh_process_trace_list(select_path=item.evidence_trace_path)
            self.result_tabs.setCurrentWidget(self.process_review_widget)

    def set_symbol(self, code: str, name: str = ""):
        self.symbol_input.setText(code)

    def set_top_controls_visible(self, visible: bool) -> None:
        if hasattr(self, "top_controls_widget"):
            self.top_controls_widget.setVisible(bool(visible))

    def get_current_model_name(self) -> str:
        return str(self.model_combo.currentText() or "")

    def prompt_select_model(self, parent: Optional[QWidget] = None) -> str:
        model_names = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        if not model_names:
            QMessageBox.warning(parent or self, "提示", "当前没有可用模型")
            return ""
        try:
            from PyQt6.QtWidgets import QInputDialog
        except ImportError:
            return ""
        current = self.get_current_model_name()
        current_index = max(self.model_combo.currentIndex(), 0)
        selected, ok = QInputDialog.getItem(
            parent or self,
            "选择模型",
            "请选择巡检使用的模型:",
            model_names,
            current_index,
            False,
        )
        if not ok or not str(selected or "").strip():
            return current
        idx = self.model_combo.findText(str(selected))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        return self.get_current_model_name()

    def _infer_asset_type_for_code(self, code: str, fallback: str = "") -> str:
        if fallback:
            return fallback
        plain_code = (code or "").split(".")[0]
        if plain_code.startswith(("51", "52", "56", "58", "15", "16", "18")):
            return "ETF"
        return "股票"

    def _resolve_runtime_symbol_name(self, code: str, fallback_name: str = "") -> str:
        if fallback_name:
            return fallback_name
        trade_window = self._find_trade_window()
        if trade_window:
            looked_up = trade_window.lookup_symbol_name(code)
            if looked_up:
                return looked_up
        return ""

    def _find_trade_window(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "lookup_symbol_name") and hasattr(parent, "order_panel"):
                return parent
            parent = parent.parent() if hasattr(parent, "parent") and callable(parent.parent) else None
        return None

    @staticmethod
    def _normalize_symbol_code(code: str) -> str:
        return str(code or "").split(".")[0].strip().upper()

    def _get_effective_run_context(self) -> DecisionRunContext:
        if isinstance(self._run_context_override, DecisionRunContext):
            return self._run_context_override
        return build_decision_run_context(prefer_realtime=True)

    def _set_run_context_override(
        self,
        run_context: DecisionRunContext | Dict[str, Any] | None,
    ) -> DecisionRunContext:
        if isinstance(run_context, DecisionRunContext):
            self._run_context_override = run_context
        elif isinstance(run_context, dict):
            self._run_context_override = DecisionRunContext.from_dict(run_context)
        else:
            self._run_context_override = build_decision_run_context(prefer_realtime=True)
        return self._run_context_override

    def _clear_run_context_override(self):
        self._run_context_override = None

    def _apply_symbol_override(
        self,
        raw_context: Dict[str, Any],
        code: str,
        fallback_name: str = "",
    ) -> Dict[str, Any]:
        symbol_raw = dict(raw_context.get("symbol", {}) or {})
        resolved_name = self._resolve_runtime_symbol_name(code, fallback_name)
        asset_type = self._infer_asset_type_for_code(code, str(symbol_raw.get("asset_type", "") or ""))
        current_view = "etf" if asset_type == "ETF" else "stock"
        indicators = list(symbol_raw.get("indicators", []) or [])
        same_symbol = self._normalize_symbol_code(symbol_raw.get("code", "")) == self._normalize_symbol_code(code)
        raw_context["symbol"] = {
            "code": code,
            "name": resolved_name or str(symbol_raw.get("name", "") or ""),
            "asset_type": asset_type,
            "current_view": current_view,
            "latest_close": float(symbol_raw.get("latest_close", 0.0) or 0.0) if same_symbol else 0.0,
            "latest_change_pct": float(symbol_raw.get("latest_change_pct", 0.0) or 0.0) if same_symbol else 0.0,
            "latest_volume": float(symbol_raw.get("latest_volume", 0.0) or 0.0) if same_symbol else 0.0,
            "data_points": int(symbol_raw.get("data_points", 0) or 0) if same_symbol else 0,
            "date_start": str(symbol_raw.get("date_start", "") or "") if same_symbol else "",
            "date_end": str(symbol_raw.get("date_end", "") or "") if same_symbol else "",
            "indicators": indicators,
        }
        return raw_context

    def _on_mode_changed(self, _index=None):
        self._current_mode = self.mode_combo.currentData() or DECISION_MODE_POSITION_SCAN
        self.symbol_input.setVisible(False)
        self.watchlist_group_combo.setVisible(False)
        hints = {
            DECISION_MODE_POSITION_SCAN: self.position_scan_hint,
            DECISION_MODE_CANDIDATE_POOL_SCAN: "候选池巡检: 先按量化规则生成今日候选池，再交给AI逐只评估买入机会",
        }
        self.mode_hint_label.setText(hints.get(self._current_mode, ""))
        btn_texts = {
            DECISION_MODE_POSITION_SCAN: f"🔎 开始{self.position_scan_label}",
            DECISION_MODE_CANDIDATE_POOL_SCAN: "🔎 开始候选池巡检",
        }
        self.analyze_btn.setText(btn_texts.get(self._current_mode, "🔍 开始"))

    def _refresh_watchlist_groups(self):
        try:
            wm = WatchlistManager()
            groups = wm.get_all_groups()
        except Exception:
            groups = []
        current = self.watchlist_group_combo.currentText()
        self.watchlist_group_combo.clear()
        for g in groups:
            self.watchlist_group_combo.addItem(g)
        if current and self.watchlist_group_combo.findText(current) >= 0:
            self.watchlist_group_combo.setCurrentText(current)

    def _request_runtime_raw_context(self, code: str = "", name: str = "") -> Dict[str, Any]:
        raw_context: Dict[str, Any] = {}
        if not self.context_provider:
            return raw_context
        run_context = self._get_effective_run_context()
        symbol_override = {
            "code": code,
            "name": name,
            "asset_type": self._infer_asset_type_for_code(code),
        }
        try:
            raw_context = self.context_provider(
                symbol_override=symbol_override,
                run_context=run_context.to_dict(),
            )
        except TypeError:
            try:
                raw_context = self.context_provider(symbol_override=symbol_override)
            except TypeError:
                try:
                    raw_context = self.context_provider()
                except Exception:
                    raw_context = {}
            except Exception:
                raw_context = {}
        except Exception:
            raw_context = {}
        raw_context = raw_context if isinstance(raw_context, dict) else {}
        raw_context["decision_run_context"] = run_context.to_dict()
        return raw_context

    def _build_runtime_context(self) -> AgentRuntimeContext:
        override_code = self.symbol_input.text().strip()
        raw_context = self._request_runtime_raw_context(override_code)
        if override_code:
            raw_context = self._apply_symbol_override(raw_context, override_code)

        context = AgentContextService.from_raw(raw_context)
        return context

    def _build_runtime_context_for_symbol(self, code: str, name: str = "") -> AgentRuntimeContext:
        raw_context = self._request_runtime_raw_context(code, name or "")
        raw_context = self._apply_symbol_override(raw_context, code, name or "")
        context = AgentContextService.from_raw(raw_context)
        return context

    def _resolve_model_config(self, *, show_dialog: bool = True) -> Optional[Dict[str, str]]:
        model = self.model_combo.currentText()
        model_configs = self._ai_config.get("model_configs", {})
        config = model_configs.get(model, {})
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "")
        if not api_key:
            logger.warning("AI 模型未配置可用 API Key: %s", model)
            if show_dialog:
                QMessageBox.warning(self, "提示", f"请先在智能体设置中配置模型 {model} 的 API Key")
            return None
        return {"model": model, "api_key": api_key, "base_url": base_url}

    def _clear_progress_cards(self):
        self._progress_cards = []
        while self.progress_cards_layout.count() > 1:
            item = self.progress_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._clear_review_cards()

    def _clear_review_cards(self):
        while self.process_review_layout.count() > 1:
            item = self.process_review_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _is_live_trace_selected(self) -> bool:
        if not hasattr(self, "process_trace_combo"):
            return True
        return str(self.process_trace_combo.currentData() or "") == "__live__"

    def _refresh_process_trace_list(self, *, select_path: str = "") -> None:
        if not hasattr(self, "process_trace_combo"):
            return
        selected = select_path or str(self.process_trace_combo.currentData() or "")
        self.process_trace_combo.blockSignals(True)
        self.process_trace_combo.clear()
        self.process_trace_combo.addItem("当前实时过程", "__live__")
        self._trace_paths_by_key = {}
        for path, trace in self.decision_lifecycle.list_recent_evidence_traces(limit=80):
            path_text = str(path)
            label = self._format_trace_label(trace)
            self.process_trace_combo.addItem(label, path_text)
            self._trace_paths_by_key[path_text] = path_text
        target = selected if selected in self._trace_paths_by_key else "__live__"
        if select_path and select_path in self._trace_paths_by_key:
            target = select_path
        idx = self.process_trace_combo.findData(target)
        self.process_trace_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.process_trace_combo.blockSignals(False)
        self._on_process_trace_selected()

    def _format_trace_label(self, trace: EvidenceTrace) -> str:
        when = trace.completed_at or trace.run_at or "-"
        symbol = f"{trace.symbol_name}({trace.symbol_code})" if trace.symbol_name else trace.symbol_code or "-"
        mode = trace.mode or trace.source or "-"
        status = trace.status or "-"
        return f"{when} | {symbol} | {mode} | {status}"

    def _on_process_trace_selected(self, _index: int | None = None) -> None:
        data = str(self.process_trace_combo.currentData() or "")
        if data and data != "__live__":
            self._render_selected_trace()
            return
        self._sync_progress_review_tab()

    def _render_selected_trace(self, *_args) -> None:
        if not hasattr(self, "process_trace_combo"):
            return
        path = str(self.process_trace_combo.currentData() or "")
        if not path or path == "__live__":
            self._sync_progress_review_tab()
            return
        trace = self.decision_lifecycle.load_evidence_trace(path)
        self._clear_review_cards()
        if trace is None:
            self.process_review_summary_label.setText("证据轨迹读取失败。")
            return
        keyword = self.process_search_input.text().strip().lower() if hasattr(self, "process_search_input") else ""
        summary = (
            f"{trace.symbol_name or '-'}({trace.symbol_code or '-'}) | "
            f"会话 {trace.session_id or '-'} | {trace.status} | "
            f"步骤 {len(trace.steps)} / 工具证据 {len(trace.tool_calls)}"
        )
        self.process_review_summary_label.setText(summary)
        rendered = 0
        for step in trace.steps:
            if keyword and not self._trace_step_matches(step, keyword):
                continue
            self.process_review_layout.insertWidget(
                self.process_review_layout.count() - 1,
                self._build_trace_step_card(step),
            )
            rendered += 1
        if trace.tool_calls:
            tools_step = EvidenceStep(
                title="工具证据明细",
                detail=f"共 {len(trace.tool_calls)} 条工具证据；展开查看输入/输出摘要和文件路径。",
                status="done",
                children=[self._tool_call_to_step(item, idx) for idx, item in enumerate(trace.tool_calls, start=1)],
            )
            if not keyword or self._trace_step_matches(tools_step, keyword):
                card = self._build_trace_step_card(tools_step)
                card.expand()
                self.process_review_layout.insertWidget(self.process_review_layout.count() - 1, card)
                rendered += 1
        if rendered == 0:
            self.process_review_summary_label.setText(summary + " | 无匹配步骤")

    def _trace_step_matches(self, step: EvidenceStep, keyword: str) -> bool:
        haystack = f"{step.title}\n{step.detail}".lower()
        if keyword in haystack:
            return True
        return any(self._trace_step_matches(child, keyword) for child in step.children)

    def _build_trace_step_card(self, step: EvidenceStep) -> CollapsibleStepCard:
        file_path = self._extract_path_from_detail(step.detail)
        preview_path = file_path if file_path and file_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else ""
        action_callback = (lambda p=file_path: self._open_local_evidence_path(p)) if file_path else None
        card = CollapsibleStepCard(
            title=step.title,
            detail=step.detail,
            status=step.status or "done",
            action_label="打开证据文件/图片" if file_path else "",
            action_callback=action_callback,
            preview_path=preview_path,
        )
        for child in step.children:
            card.add_child_card(self._build_trace_step_card(child))
        return card

    def _tool_call_to_step(self, item: ToolCallTrace, idx: int) -> EvidenceStep:
        detail_lines = [
            f"工具标识: {item.tool_name}",
            f"证据标题: {item.title}",
            f"摘要: {item.summary}",
        ]
        if item.content_preview:
            detail_lines.extend(["", "关键内容预览:", item.content_preview])
        path = item.image_path or item.file_path
        if path:
            detail_lines.extend(["", f"原始证据路径: {path}"])
        return EvidenceStep(
            title=f"子步骤 {idx}: {self._tool_display_name(item.tool_name)}",
            detail="\n".join(detail_lines),
            status="done",
        )

    @staticmethod
    def _extract_path_from_detail(detail: str) -> str:
        marker = "原始证据路径:"
        for line in str(detail or "").splitlines():
            if marker in line:
                return line.split(marker, 1)[1].strip()
        return ""

    def _clone_progress_card(self, card: CollapsibleStepCard) -> CollapsibleStepCard:
        cloned = CollapsibleStepCard(
            title=getattr(card, "title_text", ""),
            detail=getattr(card, "detail_text", ""),
            status=getattr(card, "status", "pending"),
            action_label=card.action_btn.text() if hasattr(card, "action_btn") else "",
            action_callback=getattr(card, "_action_callback", None),
            preview_path=getattr(card, "_preview_path", ""),
        )
        if card.header_btn.isChecked():
            cloned.expand()
        for idx in range(card.children_layout.count()):
            child_item = card.children_layout.itemAt(idx)
            child_widget = child_item.widget()
            if isinstance(child_widget, CollapsibleStepCard):
                cloned.add_child_card(self._clone_progress_card(child_widget))
        return cloned

    def _sync_progress_review_tab(self):
        if not self._is_live_trace_selected():
            return
        self._clear_review_cards()
        if hasattr(self, "process_review_summary_label"):
            self.process_review_summary_label.setText("当前实时过程预览（历史记录请选择上方证据轨迹）")
        for card in self._progress_cards:
            self.process_review_layout.insertWidget(
                self.process_review_layout.count() - 1,
                self._clone_progress_card(card),
            )

    def _last_progress_card(self) -> Optional[CollapsibleStepCard]:
        return self._progress_cards[-1] if self._progress_cards else None

    def _find_progress_card(self, contains_text: str) -> Optional[CollapsibleStepCard]:
        for card in self._progress_cards:
            if contains_text in getattr(card, "title_text", ""):
                return card
        return None

    def _finish_last_progress_card(self):
        last = self._last_progress_card()
        if last is not None and getattr(last, "status", "") == "running":
            last.set_content(
                last.title_text,
                last.detail_text,
                status="done",
                action_label=last.action_btn.text() if hasattr(last, "action_btn") else "",
                action_callback=getattr(last, "_action_callback", None),
            )
            self._sync_progress_review_tab()

    def _parse_step_text(self, step: str) -> tuple[str, str]:
        clean = (step or "").strip()
        if "：" in clean:
            title, detail = clean.split("：", 1)
            return title.strip(), detail.strip()
        if ":" in clean:
            title, detail = clean.split(":", 1)
            return title.strip(), detail.strip()
        return clean, ""

    def _tool_display_name(self, tool_name: str) -> str:
        mapping = {
            "context_snapshot": "上下文快照",
            "symbol_technical_snapshot": "技术面摘要",
            "symbol_news_snapshot": "消息面摘要",
            "symbol_fundamental_snapshot": "基本面摘要",
            "symbol_analysis_packet": "深度分析资料",
            "current_kline_image": "K线截图",
            "position_snapshot": "持仓快照",
            "watchlist_snapshot": "自选快照",
            "compare_symbols": "标的对比",
        }
        return mapping.get(tool_name, tool_name)

    def _truncate_text(self, text: str, limit: int = 220) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ..."

    def _open_local_evidence_path(self, path: str):
        if not path:
            return
        normalized = os.path.abspath(path)
        if not os.path.exists(normalized):
            QMessageBox.warning(self, "提示", f"证据文件不存在：\n{normalized}")
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(normalized))
        if not opened:
            QMessageBox.warning(self, "提示", f"无法打开证据文件：\n{normalized}")

    def _attach_tool_subcards(self, prepared, parent_card: Optional[CollapsibleStepCard] = None):
        if parent_card is None:
            parent_card = self._find_progress_card("执行领域工具链")
        if parent_card is None:
            return
        parent_card.clear_children()
        for idx, item in enumerate(prepared.evidence_items, start=1):
            tool_label = self._tool_display_name(item.tool_name)
            detail_lines = [
                f"工具标识: {item.tool_name}",
                f"证据标题: {item.title}",
                f"摘要: {item.summary}",
            ]
            preview = self._truncate_text(item.content, 260)
            if preview:
                detail_lines.extend(["", "关键内容预览:", preview])
            metadata = item.metadata or {}
            file_path = str(metadata.get("image_path") or metadata.get("file_path") or "").strip()
            action_callback = None
            action_label = ""
            preview_path = ""
            if file_path:
                detail_lines.extend(["", f"原始证据路径: {file_path}"])
                action_label = "打开证据文件/图片"
                action_callback = lambda p=file_path: self._open_local_evidence_path(p)
            image_path = str(metadata.get("image_path") or "").strip()
            if image_path and os.path.exists(image_path):
                preview_path = image_path
            child_card = CollapsibleStepCard(
                title=f"子步骤 {idx}: {tool_label}",
                detail="\n".join(detail_lines),
                status="done",
                action_label=action_label,
                action_callback=action_callback,
                preview_path=preview_path,
            )
            parent_card.add_child_card(child_card)
        if prepared.evidence_items:
            parent_card.expand()
        self._sync_progress_review_tab()

    def _set_progress_steps(self, title: str, steps: List[str]):
        self.progress_hint_label.setText(title)
        if hasattr(self, "process_trace_combo"):
            idx = self.process_trace_combo.findData("__live__")
            if idx >= 0:
                self.process_trace_combo.setCurrentIndex(idx)
        self._clear_progress_cards()
        for idx, step in enumerate(steps, start=1):
            step_title, detail = self._parse_step_text(step)
            card = CollapsibleStepCard(
                title=f"步骤 {idx}: {step_title}",
                detail=detail or step,
                status="running" if idx == len(steps) else "done",
            )
            if idx == len(steps):
                card.expand()
            self._progress_cards.append(card)
            self.progress_cards_layout.insertWidget(self.progress_cards_layout.count() - 1, card)
        self.progress_scroll.verticalScrollBar().setValue(0)
        self._sync_progress_review_tab()

    def _append_progress_step(self, step: str):
        previous = self._last_progress_card()
        if previous is not None and getattr(previous, "status", "") == "running":
            previous.set_content(
                previous.title_text,
                previous.detail_text,
                status="done",
                action_label=previous.action_btn.text() if hasattr(previous, "action_btn") else "",
                action_callback=getattr(previous, "_action_callback", None),
            )
        existing_count = max(0, self.progress_cards_layout.count() - 1)
        step_title, detail = self._parse_step_text(step)
        card = CollapsibleStepCard(
            title=f"步骤 {existing_count + 1}: {step_title}",
            detail=detail or step,
            status="running",
        )
        card.expand()
        self._progress_cards.append(card)
        self.progress_cards_layout.insertWidget(self.progress_cards_layout.count() - 1, card)
        QTimer.singleShot(
            0,
            lambda: self.progress_scroll.verticalScrollBar().setValue(
                self.progress_scroll.verticalScrollBar().maximum()
            ),
        )
        self._sync_progress_review_tab()

    def _build_prepared_steps_summary(
        self,
        context: AgentRuntimeContext,
        prepared,
        *,
        model_name: str,
        scenario_label: str,
    ) -> List[str]:
        symbol_name = context.symbol.name or "-"
        symbol_code = context.symbol.code or "-"
        summary_lines = [
            f"识别任务场景：{scenario_label}，目标标的为 {symbol_name}({symbol_code})。",
            f"读取运行上下文：账户{'已连接' if context.broker.connected else '未连接'}，当前任务模式为交易决策。",
        ]
        if prepared.executed_tools:
            summary_lines.append(
                "执行领域工具链：" + " -> ".join(prepared.executed_tools)
            )
        if prepared.evidence_items:
            evidence_bits = []
            for item in prepared.evidence_items[:6]:
                evidence_bits.append(f"{item.title}（{item.summary}）")
            summary_lines.append("提取关键证据摘要：" + "；".join(evidence_bits))
        if prepared.evidence_report_path:
            summary_lines.append(f"生成证据存档：{prepared.evidence_report_path}")
        summary_lines.extend([
            "将结构化证据、输出协议和风控要求一起注入最终提示词。",
            f"调用模型 `{model_name}` 进入推理阶段，等待生成多空分析和结构化交易决策。",
        ])
        return summary_lines

    def _progress_cards_to_steps(self) -> list[EvidenceStep]:
        return [self._card_to_step(card) for card in self._progress_cards]

    def _card_to_step(self, card: CollapsibleStepCard) -> EvidenceStep:
        children: list[EvidenceStep] = []
        for idx in range(card.children_layout.count()):
            child_item = card.children_layout.itemAt(idx)
            child_widget = child_item.widget()
            if isinstance(child_widget, CollapsibleStepCard):
                children.append(self._card_to_step(child_widget))
        return EvidenceStep(
            title=str(getattr(card, "title_text", "") or ""),
            detail=str(getattr(card, "detail_text", "") or ""),
            status=str(getattr(card, "status", "done") or "done"),
            children=children,
        )

    def _prepared_tool_calls(self, prepared) -> list[ToolCallTrace]:
        tool_calls: list[ToolCallTrace] = []
        for item in list(getattr(prepared, "evidence_items", []) or []):
            metadata = dict(getattr(item, "metadata", {}) or {})
            file_path = str(metadata.get("file_path") or metadata.get("image_path") or "").strip()
            image_path = str(metadata.get("image_path") or "").strip()
            tool_calls.append(
                ToolCallTrace(
                    tool_name=str(getattr(item, "tool_name", "") or ""),
                    title=str(getattr(item, "title", "") or ""),
                    summary=str(getattr(item, "summary", "") or ""),
                    content_preview=self._truncate_text(str(getattr(item, "content", "") or ""), 1200),
                    file_path=file_path,
                    image_path=image_path,
                )
            )
        return tool_calls

    def _decision_summary_payload(self, decision: Optional[TradeDecision]) -> dict:
        if decision is None:
            return {}
        return {
            "action": str(getattr(decision, "action", "") or ""),
            "action_label": str(getattr(decision, "action_label", "") or ""),
            "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
            "current_price": float(getattr(decision, "current_price", 0.0) or 0.0),
            "target_price": float(getattr(decision, "target_price", 0.0) or 0.0),
            "stop_loss_price": float(getattr(decision, "stop_loss_price", 0.0) or 0.0),
            "reasoning": str(getattr(decision, "reasoning", "") or ""),
        }

    def _risk_summary_payload(self, risk_result) -> dict:
        if risk_result is None:
            return {}
        return {
            "passed": bool(getattr(risk_result, "passed", False)),
            "overall_risk_level": str(getattr(risk_result, "overall_risk_level", "") or ""),
            "blocked_reasons": list(getattr(risk_result, "blocked_reasons", []) or []),
            "warnings": list(getattr(risk_result, "warnings", []) or []),
        }

    def _save_evidence_trace(
        self,
        *,
        result: Dict[str, Any],
        prepared=None,
        steps: Optional[list[EvidenceStep]] = None,
        status: str = "done",
        trace_id: str = "",
        started_at: Optional[datetime] = None,
    ) -> str:
        symbol_code = str(result.get("symbol_code", "") or "").strip()
        symbol_name = str(result.get("symbol_name", "") or "").strip()
        if not symbol_code and not symbol_name:
            return ""
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = self._active_session_id or self._active_scan_run_id or trace_id or uuid4().hex
        trace = EvidenceTrace(
            trace_id=trace_id or f"{symbol_code}_{datetime.now().strftime('%H%M%S')}",
            session_id=session_id,
            decision_record_id=str(result.get("decision_record_id", "") or ""),
            symbol_code=symbol_code,
            symbol_name=symbol_name,
            mode=str(self._current_mode or ""),
            scan_scope=str(result.get("scan_scope", "") or self._active_scan_scope or ""),
            source=str(self._active_scan_source or "manual"),
            run_at=started_at.strftime("%Y-%m-%d %H:%M:%S") if isinstance(started_at, datetime) else now_text,
            completed_at=now_text,
            status=status,
            model_name=str((getattr(self, "_active_model_cfg", {}) or {}).get("model") or self.get_current_model_name()),
            steps=list(steps or self._progress_cards_to_steps()),
            tool_calls=self._prepared_tool_calls(prepared) if prepared is not None else [],
            artifacts=[
                {"type": "evidence_report", "path": str(getattr(prepared, "evidence_report_path", "") or "")}
            ] if prepared is not None and getattr(prepared, "evidence_report_path", "") else [],
            response_preview=self._truncate_text(str(result.get("response_text", "") or ""), 2000),
            decision_summary=self._decision_summary_payload(result.get("decision")),
            risk_summary=self._risk_summary_payload(result.get("risk_result")),
        )
        try:
            path = self.decision_lifecycle.save_evidence_trace(trace)
        except Exception as exc:
            logger.warning("保存 AI 证据轨迹失败: %s", exc, exc_info=True)
            return ""
        self._current_trace_path = str(path)
        self._refresh_process_trace_list(select_path=str(path))
        return str(path)

    def _build_worker_trace_steps(
        self,
        *,
        result: Dict[str, Any],
        prepared,
        started_at: Optional[datetime],
        first_chunk_at: Optional[datetime],
        elapsed: float,
    ) -> list[EvidenceStep]:
        scan_item = dict(result.get("scan_item", {}) or {})
        symbol = f"{result.get('symbol_name', '')}({result.get('symbol_code', '')})"
        first_delay = ""
        if isinstance(started_at, datetime) and isinstance(first_chunk_at, datetime):
            first_delay = f"，首包耗时 {(first_chunk_at - started_at).total_seconds():.2f}s"
        tool_children = [
            self._tool_call_to_step(item, idx)
            for idx, item in enumerate(self._prepared_tool_calls(prepared), start=1)
        ]
        decision = result.get("decision")
        risk_result = result.get("risk_result")
        if decision is not None:
            action_label = TRADE_ACTION_LABELS.get(decision.action, decision.action)
            decision_detail = (
                f"操作: {action_label}\n"
                f"置信度: {decision.confidence:.0%}\n"
                f"理由: {decision.reasoning or '-'}"
            )
        else:
            decision_detail = "未能从模型输出解析出有效结构化决策。"
        if risk_result is not None:
            decision_detail += f"\n风控: {risk_result.overall_risk_level.upper()} / {'通过' if risk_result.passed else '未通过'}"
            if risk_result.blocked_reasons:
                decision_detail += "\n阻断原因: " + "；".join(risk_result.blocked_reasons)
        return [
            EvidenceStep(
                title=f"接收巡检任务: {symbol}",
                detail=(
                    f"会话: {self._active_scan_run_id or '-'}\n"
                    f"来源: {self._active_scan_source or 'manual'}\n"
                    f"范围: {SCAN_SCOPE_LABELS.get(str(result.get('scan_scope') or ''), result.get('scan_scope') or '-')}\n"
                    f"原始持仓/候选信息: {_make_json_safe(scan_item)}"
                ),
                status="done",
            ),
            EvidenceStep(
                title="构建运行上下文",
                detail="已构建包含标的、账户、行情、指标和任务运行上下文的 AgentRuntimeContext。",
                status="done",
            ),
            EvidenceStep(
                title="执行领域工具链",
                detail=(
                    " -> ".join(list(getattr(prepared, "executed_tools", []) or []))
                    or "未记录工具链"
                ),
                status="done",
                children=tool_children,
            ),
            EvidenceStep(
                title="模型推理返回",
                detail=f"响应长度: {len(str(result.get('response_text', '') or ''))} 字；总耗时 {elapsed:.2f}s{first_delay}",
                status="done",
            ),
            EvidenceStep(
                title="结构化决策与风控",
                detail=decision_detail,
                status="done" if decision is not None else "warning",
            ),
        ]

    def _reset_current_result(self):
        self._current_decision = None
        self._current_risk_result = None
        self._current_decision_record_id = ""
        self._current_prepared_request = None

    def _install_gui_thread_kline_snapshot(self, context: AgentRuntimeContext) -> str:
        """在 GUI 线程完成 K 线截图，后台证据线程只复用截图路径。"""

        raw = getattr(context, "raw", {}) or {}
        hooks = raw.get("_agent_tool_hooks", {}) if isinstance(raw, dict) else {}
        if not isinstance(hooks, dict):
            return ""
        capture_hook = hooks.get("capture_current_kline_image")
        if not callable(capture_hook):
            return ""

        symbol = getattr(context, "symbol", None)
        code = str(getattr(symbol, "code", "") or "").strip()
        asset_type = str(getattr(symbol, "asset_type", "") or "").strip()
        image_path = ""
        try:
            image_path = str(capture_hook(code, asset_type) or "")
        except TypeError:
            try:
                image_path = str(capture_hook() or "")
            except Exception as exc:
                logger.debug("GUI 线程 K 线截图失败: %s", exc, exc_info=True)
        except Exception as exc:
            logger.debug("GUI 线程 K 线截图失败: %s", exc, exc_info=True)

        def _cached_capture(*_args, **_kwargs):
            return image_path or None

        hooks["capture_current_kline_image"] = _cached_capture
        return image_path

    def _start_prepare_request_thread(
        self,
        *,
        base_system_prompt: str,
        context: AgentRuntimeContext,
        task_mode: str,
        chat_history: List[Dict[str, Any]],
        latest_user_content: Any,
        prepared_slot,
        failed_slot,
    ) -> _AgentPrepareRequestThread:
        worker = _AgentPrepareRequestThread(
            base_system_prompt=base_system_prompt,
            context=context,
            task_mode=task_mode,
            chat_history=chat_history,
            latest_user_content=latest_user_content,
            parent=self,
        )
        worker.prepared.connect(prepared_slot)
        worker.failed.connect(failed_slot)
        worker.start()
        return worker

    @staticmethod
    def _new_shared_evidence_cache() -> Dict[str, Any]:
        return {
            "lock": threading.RLock(),
            "results": {},
        }

    def _install_shared_evidence_cache(self, context: AgentRuntimeContext) -> None:
        if not self._scan_shared_evidence_cache:
            return
        raw = getattr(context, "raw", None)
        if isinstance(raw, dict):
            raw["_agent_shared_evidence_cache"] = self._scan_shared_evidence_cache

    def _on_analyze_clicked(self):
        model_cfg = self._resolve_model_config()
        if not model_cfg:
            return

        if self._current_mode == DECISION_MODE_POSITION_SCAN:
            codes = self._collect_scan_codes_for_freshness("position")
            self._run_with_freshness_check(codes, lambda: self._start_position_scan(model_cfg))
            return
        if self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN:
            items = self._load_candidate_pool_items(refresh=True)
            if not items:
                QMessageBox.warning(self, "提示", "当前候选池为空，请检查本地行情数据或股票池配置")
                return
            codes = [item.get("code", "") for item in items if item.get("code")]
            self._run_with_freshness_check(codes, lambda items=items: self._start_candidate_pool_scan(model_cfg, items))
            return
        QMessageBox.warning(self, "提示", "当前仅支持持仓巡检和候选池巡检")

    def _collect_scan_codes_for_freshness(self, scan_type: str) -> list:
        codes: list[str] = []
        if scan_type == "position":
            account_panel = self._find_account_panel()
            if account_panel:
                try:
                    positions = account_panel.get_live_positions()
                    codes = [p.get("code", "") for p in positions if p.get("code")]
                except Exception:
                    pass
        elif scan_type == "watchlist":
            group_name = self.watchlist_group_combo.currentText()
            if group_name:
                try:
                    wm = WatchlistManager()
                    codes = wm.get_group_stocks(group_name)
                except Exception:
                    pass
        elif scan_type == "candidate_pool":
            try:
                codes = [item.get("code", "") for item in self._load_candidate_pool_items(refresh=True)]
            except Exception:
                codes = []
        return [c for c in codes if c]

    def _run_with_freshness_check(self, codes: list, proceed_callback):
        try:
            from trading_app.services.market_data.data_freshness_service import check_parquet_freshness
        except ImportError:
            from trading_app.services.market_data.data_freshness_service import check_parquet_freshness

        unique_codes = []
        seen_codes = set()
        for code in codes:
            normalized = str(code or "").strip().split(".", 1)[0]
            if not normalized or normalized in seen_codes:
                continue
            seen_codes.add(normalized)
            unique_codes.append(normalized)

        if not unique_codes:
            proceed_callback()
            return

        stale_items = []
        for code in unique_codes:
            fresh, info = check_parquet_freshness(code)
            if not fresh:
                stale_items.append((code, info))

        if not stale_items:
            proceed_callback()
            return

        stale_preview = "\n".join(f"  {c}: 最新 {d}" for c, d in stale_items[:8])
        if len(stale_items) > 8:
            stale_preview += f"\n  ... 还有 {len(stale_items) - 8} 只"

        trade_window = self._find_trade_window()
        if not trade_window or not hasattr(trade_window, "freshness_guard"):
            QMessageBox.critical(
                self,
                "本地数据未更新",
                f"检测到 {len(stale_items)} 只标的的 K 线数据不是最新的：\n\n"
                f"{stale_preview}\n\n"
                "当前环境未连接 DataFreshnessGuard，无法安全自动更新。为避免实盘策略使用过期数据，本次分析已阻断。",
            )
            return

        dlg = QMessageBox(self)
        dlg.setWindowTitle("本地数据未更新")
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setText(
            f"检测到 {len(stale_items)} 只标的的 K 线数据不是最新的：\n\n"
            f"{stale_preview}\n\n"
                "实盘策略中枢不允许使用过期 K 线继续分析，请先完成数据更新。"
        )
        btn_update = dlg.addButton("检测 miniQMT 并更新数据", QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = dlg.addButton("取消", QMessageBox.ButtonRole.DestructiveRole)
        dlg.setDefaultButton(btn_update)
        dlg.exec()

        clicked = dlg.clickedButton()
        if clicked == btn_cancel:
            return

        guard = trade_window.freshness_guard
        self.analyze_btn.setEnabled(False)
        self.progress_label.setText("正在检测 miniQMT 数据源并更新...")
        self.stack.setCurrentIndex(1)

        def _on_done(ok, msg):
            try:
                guard.update_finished.disconnect(_on_done)
            except (TypeError, RuntimeError):
                pass
            self.progress_label.setText(msg)
            if not ok:
                self.analyze_btn.setEnabled(True)
                self.stack.setCurrentIndex(0)

        guard.update_finished.connect(_on_done)
        guard.ensure_fresh_then_run(
            unique_codes,
            proceed_callback,
            include_indices=True,
            prefer_realtime=True,
            require_minute_freshness=False,
        )

    def _start_single_decision(
        self,
        context: AgentRuntimeContext,
        model_cfg: Dict[str, str],
        *,
        user_prompt: str | None = None,
        scan_item: Optional[Dict[str, Any]] = None,
    ):
        self._active_scan_scope = SCAN_SCOPE_AI_MANAGED
        self._active_scan_label = ""
        self._active_scan_allow_auto_execute = True
        self._active_scan_broker_context = None
        self._active_model_cfg = dict(model_cfg or {})
        self._active_session_id = f"single_{uuid4().hex}"
        self._upsert_decision_session(
            session_id=self._active_session_id,
            title=f"单票决策 {context.symbol.name or context.symbol.code or '-'}",
            source="single",
            mode=self._current_mode,
            scan_scope=SCAN_SCOPE_AI_MANAGED,
            task_id="",
            status="running",
            summary="单票 AI 决策生成中",
        )
        self.analyze_btn.setEnabled(False)
        self._reset_current_result()
        self.stack.setCurrentIndex(1)
        self._full_response = ""
        self._context_for_decision = context
        self._current_scan_item = scan_item
        self._stream_started = False
        scenario_label = "持仓巡检单票分析" if scan_item else "单股交易决策"
        self.progress_label.setText(
            f"正在收集 {context.symbol.name}({context.symbol.code}) 的多维度数据..."
        )
        self._set_progress_steps(
            "执行步骤概要",
            [
                f"接收请求并识别场景：{scenario_label}。",
                f"解析目标标的：{context.symbol.name or '-'}({context.symbol.code or '-'})。",
                "准备运行上下文，包括账户、图表、行情和可用持仓信息。",
                "开始执行领域工具，采集技术面、消息面、基本面和图表证据。",
            ],
        )

        system_prompt = AgentPromptBuilder.build_system_prompt(
            self._system_prompt_seed(),
            context,
            task_mode=TASK_MODE_TRADE_DECISION,
        )
        latest_user_content = user_prompt or AgentPromptBuilder.build_quick_task_prompt(
            TASK_MODE_TRADE_DECISION,
            context,
        )
        screenshot_path = self._install_gui_thread_kline_snapshot(context)
        if screenshot_path:
            self._append_progress_step("已在 GUI 线程完成 K 线截图，后台线程将复用该截图证据。")
        else:
            self._append_progress_step("K 线截图不可用或失败，后台线程将继续采集其他证据。")
        self.progress_label.setText("正在后台准备证据，请稍候...")
        self._prepare_thread = self._start_prepare_request_thread(
            base_system_prompt=system_prompt,
            context=context,
            task_mode=TASK_MODE_TRADE_DECISION,
            chat_history=[],
            latest_user_content=latest_user_content,
            prepared_slot=lambda prepared, ctx=context, cfg=dict(model_cfg), label=scenario_label: self._on_single_prepare_finished(
                prepared,
                ctx,
                cfg,
                label,
            ),
            failed_slot=self._on_single_prepare_failed,
        )

    def _on_single_prepare_finished(
        self,
        prepared,
        context: AgentRuntimeContext,
        model_cfg: Dict[str, str],
        scenario_label: str,
    ) -> None:
        if self._prepare_thread is not None:
            self._prepare_thread.deleteLater()
            self._prepare_thread = None
        self._current_prepared_request = prepared
        self._set_progress_steps(
            "执行步骤概要",
            self._build_prepared_steps_summary(
                context,
                prepared,
                model_name=model_cfg["model"],
                scenario_label=scenario_label,
            ),
        )
        self._attach_tool_subcards(prepared)
        self.progress_label.setText("数据收集完成，AI 正在分析决策...")
        self._append_progress_step("模型已开始流式生成分析结果，正在持续接收输出片段。")

        ChatThread = _get_chat_thread_class()
        self._chat_thread = ChatThread(
            model_cfg["api_key"],
            model_cfg["base_url"],
            model_cfg["model"],
            prepared.system_prompt,
            prepared.messages,
            stream=True,
            request_timeout_seconds=SCAN_SUBAGENT_REQUEST_TIMEOUT_SECONDS,
            log_context=f"single:{self._current_mode}:{context.symbol.code or 'unknown'}",
        )
        self._chat_thread.message_received.connect(self._on_stream_message)
        self._chat_thread.finished_signal.connect(self._on_analysis_finished)
        self._chat_thread.start()

    def _on_single_prepare_failed(self, message: str) -> None:
        if self._prepare_thread is not None:
            self._prepare_thread.deleteLater()
            self._prepare_thread = None
        self.analyze_btn.setEnabled(True)
        self.stack.setCurrentIndex(0)
        self.progress_label.setText("证据准备失败")
        self._append_progress_step(f"后台证据准备失败：{message}")
        self._complete_decision_session(
            self._active_session_id,
            summary=f"单票决策失败: {message}",
            status="error",
        )
        QMessageBox.warning(self, "证据准备失败", message or "后台证据准备失败")

    def _start_position_scan(
        self,
        model_cfg: Dict[str, str],
        *,
        scan_source: str = "manual",
        scheduled_task_id: str = "",
        items: Optional[List[Dict[str, Any]]] = None,
        scan_scope: str = "",
        scan_label: str = "",
        allow_auto_execute: bool = True,
        broker_context: Optional[BrokerContext] = None,
    ):
        if self._run_context_override is None:
            self._set_run_context_override(None)
        account_panel = self._find_account_panel()
        if account_panel is None:
            QMessageBox.warning(self, "提示", "未找到账户面板")
            return ""
        resolved_scope = str(
            scan_scope
            or getattr(account_panel, "position_scope", "")
            or SCAN_SCOPE_AI_MANAGED
        )
        positions = [dict(item) for item in (items or [])]
        if not positions:
            positions = account_panel.get_live_positions()
        positions = [
            {
                **dict(item),
                "scan_scope": str(item.get("scan_scope") or resolved_scope or SCAN_SCOPE_AI_MANAGED),
            }
            for item in positions
        ]
        if not positions:
            QMessageBox.warning(self, "提示", "当前无可巡检持仓，请先连接券商并确认持仓数据")
            return ""
        if broker_context is None and account_panel is not None:
            try:
                broker_context = account_panel.get_broker_context()
            except Exception:
                broker_context = None
        effective_label = scan_label or (
            self.position_scan_label if resolved_scope == SCAN_SCOPE_UNMANAGED else self.position_scan_label
        )

        scan_run_id = self._begin_scan_session(
            items=positions,
            scan_source=scan_source,
            scheduled_task_id=scheduled_task_id,
            scan_scope=resolved_scope,
            scan_label=effective_label,
            allow_auto_execute=allow_auto_execute,
            broker_context=broker_context,
        )
        if not scan_run_id:
            return ""
        self._active_model_cfg = model_cfg
        self.scan_table.setRowCount(0)
        self.analysis_display.clear()
        self._populate_decision_card(None, None)
        self.stack.setCurrentIndex(1)
        self.progress_label.setText(f"准备开始{effective_label}，共 {len(positions)} 只持仓...")
        self._set_progress_steps(
            "执行步骤概要",
            [
                f"接收{effective_label}请求，本轮共识别到 {len(positions)} 只有效持仓。",
                f"选择并行子代理模式处理，最大并发数设为 {SCAN_SUBAGENT_CONCURRENCY}。",
                "每只持仓都会单独完成：上下文构建 -> 证据采集 -> 模型推理 -> 决策提取 -> 风控评估。",
                "本轮仅生成巡检建议，不会直接发起新增买入。" if not allow_auto_execute else "当前巡检表会在每只股票完成后实时追加结果。",
            ],
        )
        self.result_tabs.setCurrentWidget(self.scan_table)
        self._launch_scan_subagents()
        return scan_run_id

    def _start_watchlist_scan(self, model_cfg: Dict[str, str]):
        group_name = self.watchlist_group_combo.currentText()
        if not group_name:
            QMessageBox.warning(self, "提示", "请选择一个自选分组")
            return
        try:
            wm = WatchlistManager()
            codes = wm.get_group_stocks(group_name)
        except Exception as exc:
            QMessageBox.warning(self, "提示", f"读取自选分组失败: {exc}")
            return
        if not codes:
            QMessageBox.warning(self, "提示", f"分组「{group_name}」中暂无股票")
            return

        items: List[Dict[str, Any]] = []
        for code in codes:
            name = self._resolve_runtime_symbol_name(code)
            items.append({"code": code, "name": name or code})

        self._scan_queue = items
        self._scan_results = []
        self._current_scan_index = 0
        self._scan_in_progress = True
        self._scan_total_count = len(items)
        self._scan_completed_count = 0
        self._scan_active_workers = {}
        self._scan_worker_states = {}
        self._active_model_cfg = model_cfg
        self.scan_table.setRowCount(0)
        self.analysis_display.clear()
        self._populate_decision_card(None, None)
        self.stack.setCurrentIndex(1)
        self.progress_label.setText(f"准备开始自选巡检「{group_name}」，共 {len(items)} 只...")
        self._set_progress_steps(
            "执行步骤概要",
            [
                f"接收自选巡检请求（分组: {group_name}），本轮共 {len(items)} 只标的。",
                f"选择并行子代理模式处理，最大并发数设为 {SCAN_SUBAGENT_CONCURRENCY}。",
                "每只标的都会单独完成：上下文构建 -> 证据采集 -> 模型推理 -> 决策提取 -> 风控评估。",
                "当前巡检表会在每只股票完成后实时追加结果。",
            ],
        )
        self.result_tabs.setCurrentWidget(self.scan_table)
        self._launch_scan_subagents()

    def _load_candidate_pool_items(self, *, refresh: bool = False) -> List[Dict[str, Any]]:
        try:
            cfg = self.stock_pool_service.get_config()
            return self.stock_pool_service.get_candidate_items(
                refresh=refresh,
                limit=int(cfg.ai_review_limit or 10),
                run_context=self._get_effective_run_context().to_dict(),
            )
        except Exception as exc:
            logger.exception("加载候选池失败")
            QMessageBox.warning(self, "提示", f"加载候选池失败: {exc}")
            return []

    def _start_candidate_pool_scan(
        self,
        model_cfg: Dict[str, str],
        items: Optional[List[Dict[str, Any]]] = None,
        *,
        scan_source: str = "manual",
        scheduled_task_id: str = "",
    ):
        if self._run_context_override is None:
            self._set_run_context_override(None)
        candidate_items = list(items or self._load_candidate_pool_items(refresh=True))
        if not candidate_items:
            QMessageBox.warning(self, "提示", "当前候选池为空，请先生成候选池")
            return ""

        scan_run_id = self._begin_scan_session(
            items=candidate_items,
            scan_source=scan_source,
            scheduled_task_id=scheduled_task_id,
        )
        if not scan_run_id:
            return ""
        self._active_model_cfg = model_cfg
        self.scan_table.setRowCount(0)
        self.analysis_display.clear()
        self._populate_decision_card(None, None)
        self.stack.setCurrentIndex(1)
        snapshot = self.stock_pool_service.get_snapshot()
        generated_at = str(snapshot.get("generated_at", "") or "")
        self.progress_label.setText(f"准备开始候选池巡检，共 {len(candidate_items)} 只...")
        self._set_progress_steps(
            "执行步骤概要",
            [
                f"候选池已生成，最新快照时间: {generated_at or '未知'}。",
                f"本轮按量化总分选取前 {len(candidate_items)} 只标的进入 AI 复核。",
                f"选择并行子代理模式处理，最大并发数设为 {SCAN_SUBAGENT_CONCURRENCY}。",
                "每只标的会单独完成：上下文构建 -> 证据采集 -> 模型推理 -> 决策提取 -> 风控评估。",
            ],
        )
        self.result_tabs.setCurrentWidget(self.scan_table)
        self._launch_scan_subagents()
        return scan_run_id

    def _begin_scan_session(
        self,
        *,
        items: List[Dict[str, Any]],
        scan_source: str,
        scheduled_task_id: str,
        scan_scope: str = SCAN_SCOPE_AI_MANAGED,
        scan_label: str = "",
        allow_auto_execute: bool = True,
        broker_context: Optional[BrokerContext] = None,
    ) -> str:
        if self._scan_in_progress:
            logger.warning(
                "忽略新的扫描启动请求: source=%s task_id=%s current_run=%s current_source=%s",
                scan_source,
                scheduled_task_id,
                self._active_scan_run_id,
                self._active_scan_source,
            )
            return ""
        self._scan_queue = list(items)
        self._scan_results = []
        self._current_scan_index = 0
        self._scan_in_progress = True
        self._scan_total_count = len(items)
        self._scan_completed_count = 0
        self._scan_active_workers = {}
        self._scan_worker_states = {}
        self._scan_shared_evidence_cache = self._new_shared_evidence_cache()
        self._active_scan_run_id = uuid4().hex
        self._active_session_id = self._active_scan_run_id
        self._active_scan_source = str(scan_source or "manual")
        self._active_scan_task_id = str(scheduled_task_id or "")
        self._active_scan_scope = str(scan_scope or SCAN_SCOPE_AI_MANAGED)
        self._active_scan_label = str(scan_label or "")
        self._active_scan_allow_auto_execute = bool(allow_auto_execute)
        self._active_scan_broker_context = broker_context
        self._upsert_decision_session(
            session_id=self._active_scan_run_id,
            title=self._active_scan_label or ("候选池巡检" if self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN else "持仓巡检"),
            source=self._active_scan_source,
            mode=self._current_mode,
            scan_scope=self._active_scan_scope,
            task_id=self._active_scan_task_id,
            status="running",
            summary=f"本轮共 {len(items)} 只标的",
        )
        return self._active_scan_run_id

    def _launch_scan_subagents(self):
        is_candidate_pool = self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN
        while self._scan_queue and len(self._scan_active_workers) < SCAN_SUBAGENT_CONCURRENCY:
            item = self._scan_queue.pop(0)
            context = self._build_runtime_context_for_symbol(item["code"], item["name"])
            if self._active_scan_broker_context is not None:
                context.broker = self._active_scan_broker_context
            self._install_shared_evidence_cache(context)
            if is_candidate_pool:
                prompt = self._build_candidate_pool_scan_prompt(context, item)
            else:
                prompt = self._build_position_scan_prompt(context, item)
            worker_id = f"{item['code']}::{self._current_scan_index}"
            self._current_scan_index += 1

            system_prompt = AgentPromptBuilder.build_system_prompt(
                self._system_prompt_seed(),
                context,
                task_mode=TASK_MODE_TRADE_DECISION,
            )
            screenshot_path = self._install_gui_thread_kline_snapshot(context)
            worker = self._start_prepare_request_thread(
                base_system_prompt=system_prompt,
                context=context,
                task_mode=TASK_MODE_TRADE_DECISION,
                chat_history=[],
                latest_user_content=prompt,
                prepared_slot=lambda prepared, wid=worker_id: self._on_scan_prepare_finished(wid, prepared),
                failed_slot=lambda message, wid=worker_id: self._on_scan_prepare_failed(wid, message),
            )
            self._scan_active_workers[worker_id] = worker
            self._scan_worker_states[worker_id] = {
                "response": "",
                "context": context,
                "scan_item": item,
                "prepared": None,
                "started_at": datetime.now(),
                "first_chunk_at": None,
            }
            if is_candidate_pool:
                scan_label = "候选池巡检"
            else:
                scan_label = "持仓决策"
            logger.info(
                "启动巡检子任务: %s(%s) stream=%s timeout=%.1fs",
                item["name"],
                item["code"],
                False,
                SCAN_SUBAGENT_REQUEST_TIMEOUT_SECONDS,
            )
            self._append_progress_step(
                f"启动子代理：{item['name']}({item['code']})，后台准备证据后生成{scan_label}。"
            )
            if screenshot_path:
                self._append_progress_step(
                    f"{item['name']}({item['code']}) 已在 GUI 线程完成 K 线截图，后台线程将复用该截图。"
                )
            else:
                self._append_progress_step(
                    f"{item['name']}({item['code']}) K 线截图不可用，继续准备其他证据。"
                )

        self._update_scan_progress_label()

    def _on_scan_prepare_finished(self, worker_id: str, prepared) -> None:
        state = self._scan_worker_states.get(worker_id)
        prepare_worker = self._scan_active_workers.get(worker_id)
        if prepare_worker is not None:
            prepare_worker.deleteLater()
        if not state:
            self._scan_active_workers.pop(worker_id, None)
            self._update_scan_progress_label()
            return

        state["prepared"] = prepared
        item = state.get("scan_item", {})
        self._attach_tool_subcards(prepared, parent_card=self._last_progress_card())
        ChatThread = _get_chat_thread_class()
        worker = ChatThread(
            self._active_model_cfg["api_key"],
            self._active_model_cfg["base_url"],
            self._active_model_cfg["model"],
            prepared.system_prompt,
            prepared.messages,
            stream=False,
            request_timeout_seconds=SCAN_SUBAGENT_REQUEST_TIMEOUT_SECONDS,
            log_context=f"scan:{item.get('code', '')}:{item.get('name', '')}",
        )
        self._scan_active_workers[worker_id] = worker
        worker.message_received.connect(
            lambda content, is_error, wid=worker_id: self._on_scan_worker_message(wid, content, is_error)
        )
        worker.finished_signal.connect(
            lambda wid=worker_id: self._on_scan_worker_finished(wid)
        )
        worker.start()
        self._update_scan_progress_label()

    def _on_scan_prepare_failed(self, worker_id: str, message: str) -> None:
        state = self._scan_worker_states.get(worker_id)
        if state is not None:
            state["response"] = f"\n\n[错误] 后台证据准备失败: {message}"
            state["prepared"] = None
            state["first_chunk_at"] = datetime.now()
        self._on_scan_worker_finished(worker_id)

    def _update_scan_progress_label(self):
        running_names = [
            state["scan_item"].get("name") or state["scan_item"].get("code")
            for state in self._scan_worker_states.values()
        ]
        running_text = "、".join(running_names[:3]) if running_names else "无"
        if self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN:
            mode_label = "候选池巡检"
        else:
            mode_label = self._active_scan_label or "持仓巡检"
        self.progress_label.setText(
            f"{mode_label}中: 已完成 {self._scan_completed_count}/{self._scan_total_count} | "
            f"运行中 {len(self._scan_active_workers)} 个子代理 | 当前: {running_text}"
        )

    def _build_position_scan_prompt(self, context: AgentRuntimeContext, position: Dict[str, Any]) -> str:
        cost = float(position.get("cost_price", 0) or 0)
        profit_rate = float(position.get("profit_rate", 0) or 0) * 100
        can_use = int(position.get("can_use_volume", 0) or 0)
        volume = int(position.get("volume", 0) or 0)
        market_value = float(position.get("market_value", 0) or 0)
        base_prompt = AgentPromptBuilder.build_quick_task_prompt(TASK_MODE_TRADE_DECISION, context)
        extra_lines = [
            "",
            "这是持仓巡检场景，请结合当前已经持有该股票的事实做判断。",
            f"- 持仓来源: {SCAN_SCOPE_LABELS.get(str(position.get('scan_scope') or self._active_scan_scope), '当前账户持仓')}",
            f"- 当前持仓数量: {volume} 股，可卖数量: {can_use} 股",
            f"- 持仓成本价: {cost:.3f}" if cost > 0 else "- 持仓成本价: 未知",
            f"- 当前持仓盈亏: {profit_rate:+.2f}%",
            f"- 当前持仓市值: ¥{market_value:,.2f}",
            "",
            "请重点判断：继续持有、加仓、减仓、卖出、还是继续观察。",
            "如果建议卖出或减仓，请明确给出触发依据；如果建议继续持有，也要说明需要继续跟踪的风险信号。",
        ]
        if str(position.get("scan_scope") or self._active_scan_scope) == SCAN_SCOPE_UNMANAGED:
            extra_lines.extend(
                [
                    "",
                    "这是未管理账户的持仓巡检，本轮只输出持仓建议，不考虑候选池，也不考虑新开仓。",
                ]
            )
        return "\n".join([base_prompt, *extra_lines]).strip()

    def _build_watchlist_scan_prompt(self, context: AgentRuntimeContext, item: Dict[str, Any]) -> str:
        base_prompt = AgentPromptBuilder.build_quick_task_prompt(TASK_MODE_TRADE_DECISION, context)
        extra_lines = [
            "",
            "这是自选股巡检场景，当前尚未持有该股票，请从买入机会角度进行评估。",
            "请重点判断：当前是否适合买入、应该继续观望、还是应该从自选中移除。",
            "如果建议买入，请给出建议的买入价位区间、止损价、仓位建议；",
            "如果建议观望，请说明需要等待什么条件或信号才值得介入。",
        ]
        return "\n".join([base_prompt, *extra_lines]).strip()

    def _build_candidate_pool_scan_prompt(self, context: AgentRuntimeContext, item: Dict[str, Any]) -> str:
        base_prompt = AgentPromptBuilder.build_quick_task_prompt(TASK_MODE_TRADE_DECISION, context)
        factors = item.get("factors", {}) or {}
        reasons = list(item.get("reasons", []) or [])
        extra_lines = [
            "",
            "这是自动候选池巡检场景，该股票由量化规则初筛后进入候选池，请在规则筛选结论基础上继续做买入机会复核。",
            f"- 候选池排名: #{int(item.get('rank', 0) or 0)} / 量化总分: {float(item.get('score', 0.0) or 0.0):.2f}",
            f"- 所属行业: {item.get('industry', '') or '未知'}",
            f"- 20日涨幅: {float(factors.get('momentum_20', 0.0) or 0.0):+.2%}",
            f"- 60日涨幅: {float(factors.get('momentum_60', 0.0) or 0.0):+.2%}",
            f"- 相对基准超额: {float(factors.get('excess_return_20', 0.0) or 0.0):+.2%}",
            f"- 量能比: {float(factors.get('volume_ratio', 0.0) or 0.0):.2f}",
            f"- 20日回撤: {float(factors.get('drawdown_20', 0.0) or 0.0):+.2%}",
            f"- 入池原因: {'；'.join(reasons) if reasons else '量化初筛通过'}",
            "",
            "请重点判断：当前是否适合买入、应继续观望、还是应从候选池剔除。",
            "本场景下若暂不买入，请优先使用 action=watch（观望）或 action=reject（剔除），不要输出 hold。",
            "如果建议买入，请给出仓位建议、止损位和触发条件；如果建议观望或剔除，请明确说明否决原因。",
        ]
        return "\n".join([base_prompt, *extra_lines]).strip()

    def _on_scan_worker_message(self, worker_id: str, content: str, is_error: bool):
        state = self._scan_worker_states.get(worker_id)
        if not state:
            return
        if state.get("first_chunk_at") is None:
            state["first_chunk_at"] = datetime.now()
            started_at = state.get("started_at")
            if isinstance(started_at, datetime):
                delay = (state["first_chunk_at"] - started_at).total_seconds()
                item = state.get("scan_item", {})
                logger.info(
                    "巡检子任务收到完整结果: %s(%s) delay=%.2fs error=%s",
                    item.get("name"),
                    item.get("code"),
                    delay,
                    is_error,
                )
        if is_error:
            state["response"] += f"\n\n[错误] {content}"
        else:
            state["response"] += content
        self._update_scan_progress_label()

    def _on_stream_message(self, content: str, is_error: bool):
        if not self._stream_started:
            self._stream_started = True
            self._append_progress_step("模型已返回首段内容，进入结果生成与结构化提取阶段。")
        if is_error:
            self._full_response += f"\n\n[错误] {content}"
        else:
            self._full_response += content
        if self._current_mode == DECISION_MODE_POSITION_SCAN and self._current_scan_item:
            self.progress_label.setText(
                f"巡检 {self._current_scan_item.get('name', '')} 中... ({len(self._full_response)} 字)"
            )
        else:
            self.progress_label.setText(f"AI 分析中... ({len(self._full_response)} 字)")

    def _save_decision_record_for_result(
        self,
        result: Dict[str, Any],
        *,
        outcome: str = DecisionOutcome.INSPECTED.value,
    ) -> str:
        existing_id = str(result.get("decision_record_id", "") or "").strip()
        if existing_id:
            return existing_id

        symbol_code = str(result.get("symbol_code", "") or "").strip()
        symbol_name = str(result.get("symbol_name", "") or "").strip()
        if not symbol_code and not symbol_name:
            return ""

        decision = result.get("decision")
        if decision is None:
            decision = TradeDecision(
                action=TradeAction.HOLD.value,
                symbol_code=symbol_code,
                symbol_name=symbol_name,
            )
            context = result.get("context")
            latest_close = float(getattr(getattr(context, "symbol", None), "latest_close", 0.0) or 0.0)
            if latest_close > 0:
                decision.current_price = latest_close

        risk_result = result.get("risk_result")
        if risk_result is None:
            risk_result = self.risk_guard.evaluate(decision, BrokerContext())
            if result.get("decision") is not None:
                result["risk_result"] = risk_result

        try:
            record = self.decision_lifecycle.save_decision_record(decision, risk_result, outcome)
        except Exception as exc:
            logger.warning("保存 AI 决策记录失败: %s", exc, exc_info=True)
            return ""

        record_id = record.record_id if record else ""
        result["decision_record_id"] = record_id
        return record_id

    def _on_scan_worker_finished(self, worker_id: str):
        worker = self._scan_active_workers.pop(worker_id, None)
        state = self._scan_worker_states.pop(worker_id, None)
        if worker is not None:
            worker.deleteLater()
        if not state:
            self._update_scan_progress_label()
            return

        started_at = state.get("started_at")
        first_chunk_at = state.get("first_chunk_at")
        elapsed = (datetime.now() - started_at).total_seconds() if isinstance(started_at, datetime) else -1.0
        first_delay = (first_chunk_at - started_at).total_seconds() if isinstance(started_at, datetime) and isinstance(first_chunk_at, datetime) else -1.0
        item = state.get("scan_item", {})
        logger.info(
            "巡检子任务完成: %s(%s) elapsed=%.2fs first_result=%.2fs response_len=%d",
            item.get("name"),
            item.get("code"),
            elapsed,
            first_delay,
            len(state.get("response", "")),
        )

        result = self._build_analysis_result(
            state.get("response", ""),
            state["context"],
            state["scan_item"],
        )
        self._append_scan_result(result)
        if self.scan_table.currentRow() < 0:
            self._display_result(result, switch_to_details=False, emit_decision=False)

        self._save_decision_record_for_result(result)
        trace_steps = self._build_worker_trace_steps(
            result=result,
            prepared=state.get("prepared"),
            started_at=started_at,
            first_chunk_at=first_chunk_at,
            elapsed=elapsed,
        )
        trace_path = self._save_evidence_trace(
            result=result,
            prepared=state.get("prepared"),
            steps=trace_steps,
            status="done" if result["decision"] is not None else "warning",
            trace_id=worker_id.replace("::", "_"),
            started_at=started_at if isinstance(started_at, datetime) else None,
        )
        if trace_path:
            result["evidence_trace_path"] = trace_path
        self._append_decision_session_item(
            session_id=self._active_scan_run_id,
            result=result,
            item_type="scan_result",
        )
        if result["decision"] is not None:
            action_label = TRADE_ACTION_LABELS.get(result["decision"].action, result["decision"].action)
            risk_level = (
                result["risk_result"].overall_risk_level.upper()
                if result["risk_result"] is not None else "-"
            )
            self._append_progress_step(
                f"子代理完成：{result['symbol_name']}({result['symbol_code']}) -> {action_label}，"
                f"置信度 {result['decision'].confidence:.0%}，风险 {risk_level}。"
            )
        else:
            self._append_progress_step(
                f"子代理完成：{result['symbol_name']}({result['symbol_code']})，但未能提取有效结构化决策。"
            )

        self._scan_completed_count += 1
        if not self._scan_queue and not self._scan_active_workers:
            self._scan_in_progress = False
            self.analyze_btn.setEnabled(True)
            self.stack.setCurrentIndex(2)
            if self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN:
                scan_label = "候选池巡检"
            else:
                scan_label = self._active_scan_label or "持仓巡检"
            self.decision_status_label.setText(f"✅ {scan_label}完成，共 {len(self._scan_results)} 只")
            self.decision_status_label.setStyleSheet("color: green; font-weight: bold;")
            self._append_progress_step(f"全部子代理已完成，本轮{scan_label}结束，结果已写入当前巡检和决策记录。")
            self._finish_last_progress_card()
            self._complete_decision_session(
                self._active_scan_run_id,
                summary=f"{scan_label}完成，共 {len(self._scan_results)} 只",
                status="done",
            )
            self._refresh_history()
            self.result_tabs.setCurrentWidget(self.scan_table)
            if self._scan_results:
                self.scan_table.selectRow(0)
            self.scan_completed.emit({
                "mode": self._current_mode,
                "results": list(self._scan_results),
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scan_run_id": self._active_scan_run_id,
                "scan_source": self._active_scan_source,
                "scheduled_task_id": self._active_scan_task_id,
                "scan_scope": self._active_scan_scope,
                "scan_label": self._active_scan_label,
                "allow_auto_execute": self._active_scan_allow_auto_execute,
                "session_id": self._active_scan_run_id,
            })
            self._clear_run_context_override()
            self._scan_shared_evidence_cache = None
            self._try_scan_notification()
            return

        self._launch_scan_subagents()

    def _on_analysis_finished(self):
        self.stack.setCurrentIndex(2)
        result = self._build_analysis_result(self._full_response, self._context_for_decision, self._current_scan_item)
        self.analyze_btn.setEnabled(True)
        self._save_decision_record_for_result(result)
        if result["decision"] is not None:
            action_label = TRADE_ACTION_LABELS.get(result["decision"].action, result["decision"].action)
            risk_level = (
                result["risk_result"].overall_risk_level.upper()
                if result["risk_result"] is not None else "-"
            )
            self._append_progress_step(
                f"模型输出已解析完成：建议 {action_label}，置信度 {result['decision'].confidence:.0%}，风险 {risk_level}。"
            )
        else:
            self._append_progress_step("模型输出已返回，但未能解析出有效的结构化交易决策。")
        self._finish_last_progress_card()
        self._display_result(result, switch_to_details=True, emit_decision=True)
        trace_path = self._save_evidence_trace(
            result=result,
            prepared=self._current_prepared_request,
            status="done" if result["decision"] is not None else "warning",
            trace_id=f"single_{result.get('symbol_code', '')}_{datetime.now().strftime('%H%M%S')}",
        )
        if trace_path:
            result["evidence_trace_path"] = trace_path
        self._append_decision_session_item(
            session_id=self._active_session_id,
            result=result,
            item_type="single_decision",
        )
        self._complete_decision_session(
            self._active_session_id,
            summary=f"单票决策完成: {result.get('symbol_name', '')}({result.get('symbol_code', '')})",
            status="done" if result["decision"] is not None else "warning",
        )
        self._refresh_history()
        self._clear_run_context_override()

    def _build_analysis_result(
        self,
        response_text: str,
        context: AgentRuntimeContext,
        scan_item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decision = TradeDecisionExtractor.extract(response_text)
        if decision is not None:
            decision = self._normalize_decision_for_mode(decision)
        broker_ctx = self._active_scan_broker_context or BrokerContext()
        if self._active_scan_broker_context is None:
            account_panel = self._find_account_panel()
            if account_panel:
                broker_ctx = account_panel.get_broker_context()

        target_symbol_code = ""
        target_symbol_name = ""
        if scan_item:
            target_symbol_code = str((scan_item or {}).get("code", "") or context.symbol.code or "").strip()
            target_symbol_name = str((scan_item or {}).get("name", "") or context.symbol.name or "").strip()
        elif context.symbol.is_available:
            target_symbol_code = str(context.symbol.code or "").strip()
            target_symbol_name = str(context.symbol.name or "").strip()

        if decision is not None:
            if not decision.symbol_code and context.symbol.is_available:
                decision.symbol_code = context.symbol.code
            if not decision.symbol_name and context.symbol.name:
                decision.symbol_name = context.symbol.name
            if scan_item and target_symbol_code:
                extracted_code = str(decision.symbol_code or "").strip()
                if self._normalize_symbol_code(extracted_code) != self._normalize_symbol_code(target_symbol_code):
                    logger.warning(
                        "巡检结果标的代码与当前任务不一致，已强制校正: target=%s(%s) extracted=%s(%s)",
                        target_symbol_name or target_symbol_code,
                        target_symbol_code,
                        decision.symbol_name or extracted_code,
                        extracted_code,
                    )
                decision.symbol_code = target_symbol_code
                if target_symbol_name:
                    decision.symbol_name = target_symbol_name
            if decision.current_price <= 0 and context.symbol.latest_close > 0:
                decision.current_price = context.symbol.latest_close
            previous_close = float((scan_item or {}).get("latest_close", 0.0) or 0.0)
            if previous_close > 0:
                setattr(decision, "previous_close", previous_close)
            risk_result = self.risk_guard.evaluate(decision, broker_ctx)
        else:
            risk_result = None

        symbol_code = (
            decision.symbol_code if decision else target_symbol_code or context.symbol.code or (scan_item or {}).get("code", "")
        )
        symbol_name = (
            decision.symbol_name if decision else target_symbol_name or context.symbol.name or (scan_item or {}).get("name", "")
        )
        return {
            "response_text": response_text,
            "context": context,
            "decision": decision,
            "risk_result": risk_result,
            "scan_item": scan_item,
            "symbol_code": symbol_code,
            "symbol_name": symbol_name,
            "scan_scope": str((scan_item or {}).get("scan_scope", "") or self._active_scan_scope),
        }

    def _normalize_decision_for_mode(self, decision: TradeDecision) -> TradeDecision:
        action = str(getattr(decision, "action", "") or "").lower().strip()
        if self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN:
            if action == TradeAction.HOLD.value:
                decision.action = TradeAction.WATCH.value
        elif action in (TradeAction.WATCH.value, TradeAction.REJECT.value):
            decision.action = TradeAction.HOLD.value
        return decision

    def _display_result(self, result: Dict[str, Any], *, switch_to_details: bool, emit_decision: bool):
        decision = result["decision"]
        self._render_response_text(result["response_text"], decision)
        risk_result = result["risk_result"]
        self._current_decision = decision
        self._current_risk_result = risk_result
        self._current_decision_record_id = str(result.get("decision_record_id", "") or "")
        self._populate_decision_card(decision, risk_result)
        self._apply_action_state(decision, risk_result)
        if emit_decision and decision is not None:
            self.decision_ready.emit({
                "decision": decision,
                "risk_result": risk_result,
                "decision_record_id": str(result.get("decision_record_id", "") or ""),
            })
        if switch_to_details:
            self.result_tabs.setCurrentWidget(self.decision_card_widget)

    _ACTION_COLORS = {
        "buy": "#4caf50", "add": "#4caf50",
        "sell": "#f44336", "reduce": "#ff9800",
        "hold": "#90caf9",
        "watch": "#90caf9",
        "reject": "#9e9e9e",
    }

    @classmethod
    def _decision_summary_html(cls, decision: Optional[TradeDecision]) -> str:
        if decision is None:
            return "<p style='color:#999;font-style:italic;'>未能提取到有效的结构化决策。</p>"

        action_color = cls._ACTION_COLORS.get(decision.action, "#90caf9")
        ret_pct = decision.expected_return_pct
        loss_pct = decision.max_loss_pct

        def _row(label: str, value: str, *, value_color: str = "") -> str:
            vc = f" style='color:{value_color};font-weight:bold;'" if value_color else ""
            return (
                f"<tr>"
                f"<td style='padding:4px 10px;color:#aaa;white-space:nowrap;'>{label}</td>"
                f"<td style='padding:4px 10px;'{vc}>{value}</td>"
                f"</tr>"
            )

        target_extra = f"<span style='color:#4caf50;font-size:0.9em;'>（预期 {ret_pct:+.2f}%）</span>" if ret_pct is not None else ""
        stop_extra = f"<span style='color:#f44336;font-size:0.9em;'>（最大亏损 {loss_pct:+.2f}%）</span>" if loss_pct is not None else ""

        rows = "".join([
            _row("操作建议", f"<span style='color:{action_color};font-size:1.1em;font-weight:bold;'>{decision.action_label}</span>"),
            _row("标的", f"{decision.symbol_name}（{decision.symbol_code}）"),
            _row("置信度", f"{decision.confidence:.0%}", value_color=action_color),
            _row("当前价", f"{decision.current_price:.2f}"),
            _row("目标价", f"{decision.target_price:.2f} {target_extra}"),
            _row("止损价", f"{decision.stop_loss_price:.2f} {stop_extra}"),
            _row("仓位建议", f"{decision.position_pct:.0%}"),
            _row("风险评分", f"{decision.risk_score:.2f}"),
            _row("时间维度", decision.horizon_label),
        ])
        if decision.reasoning:
            rows += _row("核心逻辑", decision.reasoning)

        return (
            f"<table cellspacing='0' style='border:1px solid #333;border-radius:4px;"
            f"margin:6px 0;width:100%;'>"
            f"<tbody>{rows}</tbody></table>"
        )

    _ANALYSIS_CSS = (
        "body{color:#d0d0d0;font-family:sans-serif;font-size:14px;}"
        "h1,h2,h3,h4{color:#e0e0e0;margin:12px 0 6px 0;}"
        "ul,ol{margin:4px 0 4px 18px;padding:0;}"
        "li{margin:2px 0;}"
        "table{border-collapse:collapse;}"
        "td,th{border-bottom:1px solid #2a2a2a;}"
    )

    def _render_response_text(self, response_text: str, decision: Optional[TradeDecision] = None):
        import re
        decision_html = self._decision_summary_html(decision)
        cleaned = re.sub(
            r"<trade_decision>\s*.*?\s*</trade_decision>",
            "{{DECISION_TABLE}}",
            response_text,
            flags=re.DOTALL,
        ).strip()
        try:
            import markdown as md_lib
            body_html = md_lib.markdown(cleaned, extensions=["fenced_code", "tables"])
        except Exception:
            body_html = f"<pre>{cleaned}</pre>"
        body_html = body_html.replace("{{DECISION_TABLE}}", decision_html)
        full_html = f"<html><head><style>{self._ANALYSIS_CSS}</style></head><body>{body_html}</body></html>"
        self.analysis_display.setHtml(full_html)

    def _apply_action_state(self, decision: Optional[TradeDecision], risk_result):
        if decision is None:
            self.decision_status_label.setText("⚠ 未能提取有效决策")
            self.decision_status_label.setStyleSheet("color: orange; font-weight: bold;")
            return
        if self._scan_in_progress:
            self.decision_status_label.setText(
                f"🔄 巡检进行中: {TRADE_ACTION_LABELS.get(decision.action, decision.action)}"
            )
            self.decision_status_label.setStyleSheet("color: #0078d4; font-weight: bold;")
            return
        if decision.is_actionable and risk_result and risk_result.passed:
            self.decision_status_label.setText("✅ 风控通过，可执行")
            self.decision_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif decision.is_actionable:
            self.decision_status_label.setText("⛔ 风控未通过")
            self.decision_status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.decision_status_label.setText(f"ℹ 建议: {TRADE_ACTION_LABELS.get(decision.action, decision.action)}")
            self.decision_status_label.setStyleSheet("color: #666; font-weight: bold;")

    def _populate_decision_card(self, decision: Optional[TradeDecision], risk_result):
        while self.decision_card_layout.count():
            child = self.decision_card_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if decision is None:
            lbl = QLabel("未提取到结构化决策。请查看“AI 原文”内容。")
            lbl.setStyleSheet("color: #888; padding: 20px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.decision_card_layout.addWidget(lbl)
            return

        action_label = TRADE_ACTION_LABELS.get(decision.action, decision.action)

        # Title
        title = QLabel(f"📊 {action_label}  {decision.symbol_name}({decision.symbol_code})")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px 0;")
        self.decision_card_layout.addWidget(title)

        # Details form
        details = QGroupBox("决策参数")
        form = QFormLayout(details)
        form.setSpacing(4)
        form.addRow("操作:", QLabel(action_label))
        form.addRow("当前价:", QLabel(f"{decision.current_price:.2f}" if decision.current_price > 0 else "-"))
        form.addRow("目标价:", QLabel(f"{decision.target_price:.2f}" if decision.target_price > 0 else "-"))
        form.addRow("止损价:", QLabel(f"{decision.stop_loss_price:.2f}" if decision.stop_loss_price > 0 else "-"))

        ret = decision.expected_return_pct
        if ret is not None:
            color = "green" if ret > 0 else "red"
            form.addRow("预期收益:", QLabel(f"<span style='color:{color}'>{ret:+.2f}%</span>"))
        loss = decision.max_loss_pct
        if loss is not None:
            form.addRow("最大亏损:", QLabel(f"<span style='color:red'>{loss:.2f}%</span>"))

        form.addRow("置信度:", QLabel(f"{decision.confidence:.0%}"))
        form.addRow("建议仓位:", QLabel(f"{decision.position_pct:.0%}"))
        form.addRow("风险评分:", QLabel(f"{decision.risk_score:.2f}"))
        form.addRow("持有周期:", QLabel(decision.horizon_label))

        if decision.reasoning:
            r_lbl = QLabel(decision.reasoning)
            r_lbl.setWordWrap(True)
            form.addRow("理由:", r_lbl)
        if decision.bull_case:
            b_lbl = QLabel(decision.bull_case)
            b_lbl.setWordWrap(True)
            form.addRow("看多:", b_lbl)
        if decision.bear_case:
            br_lbl = QLabel(decision.bear_case)
            br_lbl.setWordWrap(True)
            form.addRow("看空:", br_lbl)
        if decision.invalidation:
            inv_lbl = QLabel(decision.invalidation)
            inv_lbl.setWordWrap(True)
            form.addRow("失效条件:", inv_lbl)

        self.decision_card_layout.addWidget(details)

        # Risk checks
        if risk_result:
            risk_group = QGroupBox("风控审核")
            risk_layout = QVBoxLayout(risk_group)
            icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(
                risk_result.overall_risk_level, "⚪"
            )
            risk_layout.addWidget(QLabel(f"{icon} 综合风险: {risk_result.overall_risk_level.upper()}"))
            for check in risk_result.checks:
                c_icon = "✅" if check.passed else ("⛔" if check.level == "block" else "⚠️")
                risk_layout.addWidget(QLabel(f"  {c_icon} {check.name}: {check.message}"))
            if risk_result.blocked_reasons:
                blk = QLabel("⛔ " + "; ".join(risk_result.blocked_reasons))
                blk.setStyleSheet("color: red; font-weight: bold;")
                blk.setWordWrap(True)
                risk_layout.addWidget(blk)
            self.decision_card_layout.addWidget(risk_group)

        self.decision_card_layout.addStretch()

    def _append_scan_result(self, result: Dict[str, Any]):
        row = self.scan_table.rowCount()
        self.scan_table.insertRow(row)
        self._scan_results.append(result)

        self._populate_scan_result_row(self.scan_table, row, result)

    def _build_scan_result_row_values(self, result: Dict[str, Any], row: int) -> List[str]:
        decision = result["decision"]
        risk_result = result["risk_result"]
        scan_item = result["scan_item"] or {}
        cost_price = float(scan_item.get("cost_price", 0) or 0)
        action_label = "解析失败" if decision is None else TRADE_ACTION_LABELS.get(decision.action, decision.action)
        confidence_text = "-" if decision is None else f"{decision.confidence:.0%}"
        current_price_text = "-" if decision is None or decision.current_price <= 0 else f"{decision.current_price:.2f}"
        cost_text = "-" if cost_price <= 0 else f"{cost_price:.2f}"
        risk_text = "-" if risk_result is None else risk_result.overall_risk_level.upper()
        status_text = _build_scan_status_text(decision, risk_result)

        return [
            str(row + 1),
            result["symbol_code"],
            result["symbol_name"],
            action_label,
            confidence_text,
            current_price_text,
            cost_text,
            risk_text,
            status_text,
        ]

    def _populate_scan_result_row(self, table: QTableWidget, row: int, result: Dict[str, Any]) -> None:
        values = self._build_scan_result_row_values(result, row)
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(value))

    def _render_scan_result_table(self, table: QTableWidget, results: List[Dict[str, Any]]) -> None:
        table.setRowCount(0)
        for row, result in enumerate(list(results or [])):
            table.insertRow(row)
            self._populate_scan_result_row(table, row, result)

    def _on_scan_selection_changed(self):
        row = self.scan_table.currentRow()
        if row < 0 or row >= len(self._scan_results):
            return
        result = self._scan_results[row]
        self._display_result(result, switch_to_details=False, emit_decision=True)
        self.result_tabs.setCurrentWidget(self.decision_card_widget)

    def _on_scan_result_context_menu(self, table: QTableWidget, pos, results: List[Dict[str, Any]]) -> None:
        row = table.rowAt(pos.y())
        if row < 0 or row >= len(results or []):
            return
        result = dict(results[row] or {})
        code = str(result.get("symbol_code", "") or "").strip()
        name = str(result.get("symbol_name", "") or "").strip()
        if not code:
            return
        menu = QMenu(self)
        view_action = menu.addAction(f"查看K线 {name}({code})" if name else f"查看K线 {code}")
        evidence_path = str(result.get("evidence_trace_path", "") or "").strip()
        evidence_action = None
        if evidence_path:
            evidence_action = menu.addAction("查看过程证据")
        chosen = menu.exec(table.viewport().mapToGlobal(pos))
        if chosen == view_action:
            self.market_view_requested.emit(code, name)
        elif evidence_action is not None and chosen == evidence_action:
            self._refresh_process_trace_list(select_path=evidence_path)
            self.result_tabs.setCurrentWidget(self.process_review_widget)

    @staticmethod
    def _deserialize_risk_result(payload: Dict[str, Any]) -> Optional[RiskCheckResult]:
        if not isinstance(payload, dict) or not payload:
            return None
        checks = []
        for item in list(payload.get("checks", []) or []):
            if not isinstance(item, dict):
                continue
            checks.append(
                RiskCheckItem(
                    name=str(item.get("name", "") or ""),
                    passed=bool(item.get("passed", False)),
                    level=str(item.get("level", "info") or "info"),
                    message=str(item.get("message", "") or ""),
                )
            )
        return RiskCheckResult(
            passed=bool(payload.get("passed", False)),
            checks=checks,
            overall_risk_level=str(payload.get("overall_risk_level", "low") or "low"),
            warnings=[str(item) for item in list(payload.get("warnings", []) or []) if str(item)],
            blocked_reasons=[str(item) for item in list(payload.get("blocked_reasons", []) or []) if str(item)],
        )

    def _build_runtime_scan_result_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        decision_payload = dict(record.get("decision", {}) or {})
        risk_payload = dict(record.get("risk_result", {}) or {})
        decision = TradeDecision.from_dict(decision_payload) if decision_payload else None
        if decision is not None:
            if not decision.symbol_code:
                decision.symbol_code = str(record.get("symbol_code", "") or "")
            if not decision.symbol_name:
                decision.symbol_name = str(record.get("symbol_name", "") or "")
        return {
            "symbol_code": str(record.get("symbol_code", "") or (decision.symbol_code if decision else "")),
            "symbol_name": str(record.get("symbol_name", "") or (decision.symbol_name if decision else "")),
            "decision": decision,
            "risk_result": self._deserialize_risk_result(risk_payload),
            "scan_item": dict(record.get("scan_item", {}) or {}),
            "response_text": str(record.get("response_text", "") or ""),
            "decision_record_id": str(record.get("decision_record_id", "") or ""),
            "evidence_trace_path": str(record.get("evidence_trace_path", "") or ""),
        }

    def _upsert_session_from_scheduled_record(self, record: Dict[str, Any]) -> None:
        session_id = str(record.get("scan_run_id", "") or "").strip()
        if not session_id:
            return
        items: list[DecisionSessionItem] = []
        for raw in list(record.get("results", []) or []):
            if not isinstance(raw, dict):
                continue
            result = self._build_runtime_scan_result_from_record(raw)
            decision = result.get("decision")
            action = str(getattr(decision, "action", "") or "")
            items.append(
                DecisionSessionItem(
                    item_id=str(raw.get("decision_record_id") or raw.get("evidence_trace_path") or uuid4().hex),
                    item_type="scheduled_scan",
                    symbol_code=str(result.get("symbol_code", "") or ""),
                    symbol_name=str(result.get("symbol_name", "") or ""),
                    decision_record_id=str(result.get("decision_record_id", "") or ""),
                    evidence_trace_path=str(result.get("evidence_trace_path", "") or ""),
                    action=TRADE_ACTION_LABELS.get(action, action),
                    status_text=str(raw.get("status_text", "") or _build_scan_status_text(decision, result.get("risk_result"))),
                    created_at=str(record.get("completed_at", "") or ""),
                    payload=dict(raw),
                )
            )
        self._upsert_decision_session(
            session_id=session_id,
            title=str(record.get("scan_label", "") or record.get("task_name", "") or "定时巡检"),
            source=str(record.get("scan_source", "") or "scheduled"),
            mode=str(record.get("mode", "") or ""),
            scan_scope=str(record.get("scan_scope", "") or ""),
            task_id=str(record.get("task_id", "") or ""),
            status="done",
            summary=str(record.get("summary_text", "") or ""),
            started_at=str(record.get("completed_at", "") or ""),
            completed_at=str(record.get("completed_at", "") or ""),
            items=items,
        )

    def set_scheduled_scan_records(self, records: List[Dict[str, Any]], *, focus_latest: bool = False) -> None:
        self._scheduled_scan_records = list(records or [])
        self.scheduled_scan_batches_table.setRowCount(0)
        self._scheduled_scan_record_items = []
        self.scheduled_scan_detail_table.setRowCount(0)
        if not self._scheduled_scan_records:
            self.scheduled_scan_summary_label.setText("暂无定时巡检记录")
            self.scheduled_scan_detail_label.setText("选择一组定时巡检记录后，可在下方查看明细")
            return

        self.scheduled_scan_summary_label.setText(
            f"已记录 {len(self._scheduled_scan_records)} 组最近定时巡检结果，选择批次后可查看对应明细。"
        )
        for row, record in enumerate(self._scheduled_scan_records):
            self._upsert_session_from_scheduled_record(record)
            self.scheduled_scan_batches_table.insertRow(row)
            values = [
                str(record.get("completed_at", "") or "-"),
                str(record.get("task_name", "") or "-"),
                str(record.get("scan_label", "") or "-"),
                str(int(record.get("scan_total", 0) or 0)),
                str(int(record.get("actionable", 0) or 0)),
                str(int(record.get("risk_blocked", 0) or 0)),
                str(record.get("summary_text", "") or "-"),
            ]
            for col, value in enumerate(values):
                self.scheduled_scan_batches_table.setItem(row, col, QTableWidgetItem(value))
        self.scheduled_scan_batches_table.selectRow(0)
        self._refresh_session_list()
        if focus_latest:
            self.result_tabs.setCurrentWidget(self.session_widget)

    def _on_scheduled_scan_batch_selection_changed(self) -> None:
        row = self.scheduled_scan_batches_table.currentRow()
        if row < 0 or row >= len(self._scheduled_scan_records):
            self._scheduled_scan_record_items = []
            self.scheduled_scan_detail_label.setText("选择一组定时巡检记录后，可在下方查看明细")
            self.scheduled_scan_detail_table.setRowCount(0)
            return
        record = self._scheduled_scan_records[row]
        self._scheduled_scan_record_items = [
            self._build_runtime_scan_result_from_record(item)
            for item in list(record.get("results", []) or [])
            if isinstance(item, dict)
        ]
        self.scheduled_scan_detail_label.setText(
            f"{record.get('scan_label', '定时巡检')} | 共 {len(self._scheduled_scan_record_items)} 只 | "
            f"完成于 {record.get('completed_at', '-')}"
        )
        self._render_scan_result_table(self.scheduled_scan_detail_table, self._scheduled_scan_record_items)
        if self._scheduled_scan_record_items:
            self.scheduled_scan_detail_table.selectRow(0)

    def _on_scheduled_scan_detail_selection_changed(self) -> None:
        row = self.scheduled_scan_detail_table.currentRow()
        if row < 0 or row >= len(self._scheduled_scan_record_items):
            return
        result = self._scheduled_scan_record_items[row]
        self._display_result(result, switch_to_details=False, emit_decision=False)

    def _refresh_history(self):
        records = self.decision_lifecycle.query_recent_decisions(limit=50)
        self._history_records = records
        self.history_table.setRowCount(len(records))
        for row, rec in enumerate(records):
            d = rec.decision or {}
            risk = rec.risk_result or {}
            self.history_table.setItem(row, 0, QTableWidgetItem(rec.created_at))
            self.history_table.setItem(row, 1, QTableWidgetItem(f"{rec.symbol_name}({rec.symbol_code})"))
            self.history_table.setItem(row, 2, QTableWidgetItem(
                TRADE_ACTION_LABELS.get(d.get("action", ""), d.get("action", ""))
            ))
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{d.get('confidence', 0):.0%}"))
            self.history_table.setItem(row, 4, QTableWidgetItem(
                f"{rec.entry_price:.2f}" if rec.entry_price > 0 else "-"
            ))
            pnl_item = QTableWidgetItem(f"{rec.actual_pnl_pct:+.2f}%" if rec.closed_at else "-")
            if rec.closed_at:
                pnl_item.setForeground(QBrush(QColor("#4caf50") if rec.actual_pnl_pct >= 0 else QColor("#f44336")))
            self.history_table.setItem(row, 5, pnl_item)
            pnl_amt = QTableWidgetItem(f"¥{rec.actual_pnl:+,.2f}" if rec.closed_at else "-")
            if rec.closed_at:
                pnl_amt.setForeground(QBrush(QColor("#4caf50") if rec.actual_pnl >= 0 else QColor("#f44336")))
            self.history_table.setItem(row, 6, pnl_amt)
            self.history_table.setItem(row, 7, QTableWidgetItem(
                risk.get("overall_risk_level", "-").upper()
            ))
            self.history_table.setItem(row, 8, QTableWidgetItem(rec.outcome))
        self._refresh_stats_bar()

    def _refresh_stats_bar(self):
        try:
            stats = self.decision_lifecycle.get_decision_stats()
        except Exception:
            self.stats_bar.setText("统计数据加载失败")
            return
        total = stats.get("total_decisions", 0)
        executed = stats.get("executed_count", 0)
        closed = stats.get("closed_count", 0)
        win_rate = stats.get("win_rate", 0)
        avg_pnl = stats.get("avg_pnl_pct", 0)
        total_pnl = stats.get("total_pnl", 0)

        wr_color = "#4caf50" if win_rate >= 0.5 else "#f44336" if win_rate > 0 else "#888"
        pnl_color = "#4caf50" if total_pnl > 0 else "#f44336" if total_pnl < 0 else "#888"
        self.stats_bar.setText(
            f"📊 决策总数: <b>{total}</b> | "
            f"已执行: <b>{executed}</b> | "
            f"已平仓: <b>{closed}</b> | "
            f"胜率: <b style='color:{wr_color}'>{win_rate:.1%}</b> | "
            f"平均盈亏: <b>{avg_pnl:+.2f}%</b> | "
            f"累计盈亏: <b style='color:{pnl_color}'>¥{total_pnl:,.2f}</b>"
        )

    def _export_csv(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "导出决策CSV", f"decisions_{datetime.now().strftime('%Y%m%d')}.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        from pathlib import Path as _P
        count = self.decision_lifecycle.export_decisions_csv(_P(path))
        QMessageBox.information(self, "导出完成", f"已导出 {count} 条决策记录到\n{path}")

    def _export_html(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "导出复盘报告", f"decision_report_{datetime.now().strftime('%Y%m%d')}.html",
            "HTML files (*.html)",
        )
        if not path:
            return
        from pathlib import Path as _P
        count = self.decision_lifecycle.export_decisions_html_report(_P(path))
        QMessageBox.information(self, "导出完成", f"已导出 {count} 条决策复盘报告到\n{path}")
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_history_context_menu(self, pos):
        row = self.history_table.rowAt(pos.y())
        records = getattr(self, "_history_records", [])
        if row < 0 or row >= len(records):
            return
        rec = records[row]
        menu = QMenu(self)
        view_action = menu.addAction(f"查看K线 {rec.symbol_name}({rec.symbol_code})")
        close_action = None
        if rec.outcome == DecisionOutcome.EXECUTED.value and not rec.closed_at:
            action = (rec.decision or {}).get("action", "")
            if action in ("buy", "add"):
                close_action = menu.addAction("📊 手动平仓（输入卖出价）")
        chosen = menu.exec(self.history_table.viewport().mapToGlobal(pos))
        if chosen == view_action:
            self.market_view_requested.emit(str(rec.symbol_code or ""), str(rec.symbol_name or ""))
            return
        if chosen != close_action or close_action is None:
            return
        action = (rec.decision or {}).get("action", "")
        if action not in ("buy", "add"):
            return

        from PyQt6.QtWidgets import QInputDialog
        price_str, ok = QInputDialog.getText(
            self, "手动平仓",
            f"请输入 {rec.symbol_name}({rec.symbol_code}) 的卖出价:",
        )
        if not ok or not price_str.strip():
            return
        try:
            exit_price = float(price_str.strip())
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的价格数字")
            return
        if exit_price <= 0:
            return

        success = self.decision_lifecycle.close_position(rec.record_id, exit_price)
        if success:
            self._refresh_history()
            QMessageBox.information(
                self, "平仓完成",
                f"{rec.symbol_name} 已平仓，入场价 {rec.entry_price:.2f}，"
                f"出场价 {exit_price:.2f}"
            )
        else:
            QMessageBox.warning(self, "提示", "平仓失败，请检查记录")

    def _try_scan_notification(self):
        try:
            from trading_app.services.ai.ai_decision_notifier import notify_scan_complete
        except ImportError:
            try:
                from trading_app.services.ai.ai_decision_notifier import notify_scan_complete
            except ImportError:
                return
        if self._current_mode == DECISION_MODE_CANDIDATE_POOL_SCAN:
            scan_type = "candidate_pool_scan"
        else:
            scan_type = "position_scan"
        try:
            notify_scan_complete(scan_type, self._scan_results, group_name="")
        except Exception as exc:
            logger.debug("Scan notification failed: %s", exc)

    def _find_account_panel(self) -> Optional[AccountPanel]:
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "account_panel"):
                return parent.account_panel
            parent = parent.parent() if hasattr(parent, "parent") and callable(parent.parent) else None
        return None

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_history()


# ───────────────────────────────────────────────────────────────────────────
#  Scheduler Settings Dialog
# ───────────────────────────────────────────────────────────────────────────
