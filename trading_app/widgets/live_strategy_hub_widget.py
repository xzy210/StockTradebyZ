from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

from PyQt6.QtCore import QEvent, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFontMetrics, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QSystemTrayIcon,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from common.broker_connection_panel import BrokerConnectionPanel
from common.trading_runtime import get_live_trading_engine
from trading_app.services.execution.live_event_persistence import LiveEventPersistenceHandler
from trading_app.services.ai.auto_trade_config_service import get_auto_trade_config_service
from trading_app.services.ops.live_strategy_end_of_day_service import LiveStrategyEndOfDayService
from trading_app.services.live_event_qt_bridge import QtLiveEventBridge
from trading_app.services.ops.live_strategy_logging import get_live_strategy_log_path
from trading_app.services.live_strategy_center import (
    AlertEventService,
    HubStateService,
    LiveStrategyHubController,
    LiveStrategyPlugin,
    LiveStrategyPluginRegistry,
    LiveStrategyPortfolioService,
    LiveStrategyTaskSpec,
    PanelLiveStrategyAdapter,
    TaskOrchestratorService,
    get_live_strategy_center_storage,
)
from trading_app.services.live_strategy_center.builtin_portfolio_plugins import (
    create_ai_stock_portfolio_provider,
    create_etf_rotation_portfolio_provider,
)
from trading_app.services.live_strategy_center.builtin_unmanaged_plugin import (
    create_unmanaged_position_review_plugin,
)
from trading_app.services.ops.qmt_startup_orchestrator import QmtStartupOrchestrator
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service
from trading_app.widgets.live_strategy_account_settings_dialog import LiveStrategyAccountSettingsDialog
from trading_app.widgets.live_strategy_alert_center_widget import LiveStrategyAlertCenterWidget
from trading_app.widgets.live_strategy_exception_order_widget import LiveStrategyExceptionOrderWidget
from trading_app.widgets.live_log_viewer_widget import LiveLogViewerWidget
from trading_app.widgets.live_strategy_performance_widget import LiveStrategyPerformanceWidget
from trading_app.widgets.live_strategy_status_bar_widget import LiveStrategyStatusBarWidget
from trading_app.widgets.live_strategy_task_center_widget import LiveStrategyTaskCenterWidget
from trading_app.widgets.live_strategy_hub_overview import _LiveStrategyOverviewWidget
from trading_app.widgets.live_strategy_hub_workers import _EndOfDayWorker, _KlineRefreshWorker
from trading_app.widgets.ai_trade_decision_widget import AITradeDecisionPanel, UnmanagedPositionPanel
from trading_app.widgets.readonly_market_view_dialog import ReadOnlyMarketViewDialog
from live_rotation.widget import ETFRotationLiveWidget
from live_rotation.holiday_calendar import is_trading_day

class LiveStrategyHubWidget(QWidget):
    """Unified live strategy workspace with grouped navigation."""

    TAB_OVERVIEW = "overview"
    TAB_ALERTS = "alerts"
    TAB_TASKS = "tasks"
    TAB_EXCEPTIONS = "exceptions"
    TAB_PERFORMANCE = "performance"
    TAB_AI = "ai"
    TAB_UNMANAGED = "unmanaged"
    TAB_ETF = "etf"
    TAB_LOGS = "logs"
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        *,
        context_provider=None,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
        name_map: Optional[Dict[str, str]] = None,
        etf_name_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(parent)
        self.symbol_name_resolver = symbol_name_resolver
        self.name_map = dict(name_map or {})
        self.etf_name_map = dict(etf_name_map or {})
        self._market_view_dialogs: list[ReadOnlyMarketViewDialog] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.broker_panel = BrokerConnectionPanel(self)
        self.live_trading_engine = get_live_trading_engine(broker_service=self.broker_panel.broker)
        self.live_event_persistence = LiveEventPersistenceHandler(self.live_trading_engine.event_engine)
        self.live_event_persistence.start()
        self.live_event_bridge = QtLiveEventBridge(self.live_trading_engine.event_engine, self)

        eod_bar = QWidget(self)
        eod_layout = QHBoxLayout(eod_bar)
        eod_layout.setContentsMargins(0, 0, 0, 0)
        eod_layout.setSpacing(4)
        self.eod_status_label = QLabel("")
        self.eod_status_label.setFixedWidth(160)
        self.eod_status_label.setWordWrap(False)
        self.eod_status_label.setStyleSheet("color:#888;font-size:12px;")
        eod_layout.addWidget(self.eod_status_label, 1)
        self.run_eod_btn = QPushButton("执行日终")
        self.run_eod_btn.setFixedHeight(32)
        self.run_eod_btn.clicked.connect(self._run_end_of_day_cycle)
        eod_layout.addWidget(self.run_eod_btn)
        self._set_eod_text("日终: 待命")
        self.broker_panel.set_trailing_widget(eod_bar)
        # 状态栏接管展示后，原先的券商面板/日终按钮行不再上屏，但内部对象仍需保留
        # （broker 服务、eod_status_label、run_eod_btn 被其它方法引用）。
        self.broker_panel.hide()

        self.status_bar_widget = LiveStrategyStatusBarWidget(self)
        layout.addWidget(self.status_bar_widget)

        self.nav_tree = QTreeWidget(self)
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.setIndentation(10)
        self.nav_tree.setFixedWidth(118)
        self.nav_tree.setStyleSheet(
            "QTreeWidget { background:#151515; color:#d1d5db; border:1px solid #2b2b2b; border-radius:4px; }"
            "QTreeWidget::branch { background:#151515; }"
            "QTreeWidget::item { height:26px; padding:1px 4px; }"
            "QTreeWidget::item:hover { background:#242424; color:#ffffff; border-radius:3px; }"
            "QTreeWidget::item:selected { background:#0078d4; color:#ffffff; border-radius:3px; }"
        )
        self.nav_tree.currentItemChanged.connect(self._on_navigation_item_changed)
        self.market_quick_widget = self._build_market_quick_widget()
        self.page_stack = QStackedWidget(self)
        self.page_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._nav_items: dict[str, QTreeWidgetItem] = {}
        self._page_index_by_key: dict[str, int] = {}
        self._page_titles: dict[str, str] = {}
        self._suppress_nav_signal = False

        self.ai_panel = AITradeDecisionPanel(
            context_provider=context_provider,
            parent=self,
            symbol_name_resolver=symbol_name_resolver,
            name_map=name_map,
            etf_name_map=etf_name_map,
            shared_broker_panel=self.broker_panel,
            manage_startup=False,
        )
        self.etf_panel = ETFRotationLiveWidget(
            parent=self,
            broker_panel=self.broker_panel,
            manage_startup=False,
        )
        self.unmanaged_panel = UnmanagedPositionPanel(
            context_provider=context_provider,
            parent=self,
            symbol_name_resolver=symbol_name_resolver,
            name_map=name_map,
            etf_name_map=etf_name_map,
            shared_broker_panel=self.broker_panel,
            shared_ai_panel=self.ai_panel,
        )

        self.strategy_spec_service = get_strategy_spec_service()
        self.ai_strategy_spec = self.strategy_spec_service.ai_stock()
        self.etf_strategy_spec = self.strategy_spec_service.etf_rotation()
        self.unmanaged_strategy_spec = self.strategy_spec_service.unmanaged()

        self.ai_strategy_adapter = PanelLiveStrategyAdapter.from_panel(
            self.ai_panel,
            strategy_id=self.ai_strategy_spec.strategy_id,
            strategy_name=self.ai_strategy_spec.strategy_name,
            virtual_account_id=self.ai_strategy_spec.virtual_account_id,
            automation_paused_provider=lambda: bool(getattr(self.ai_panel, "_paused_scheduler_task_ids", []) or []),
        )
        self.etf_strategy_adapter = PanelLiveStrategyAdapter.from_panel(
            self.etf_panel,
            strategy_id=self.etf_strategy_spec.strategy_id,
            strategy_name=self.etf_strategy_spec.strategy_name,
            virtual_account_id=self.etf_strategy_spec.virtual_account_id,
            automation_paused_provider=lambda: getattr(self.etf_panel, "_center_auto_pause_snapshot", None) is not None,
            rotation_pool_provider=self._get_etf_rotation_pool,
        )
        self.strategy_plugin_registry = LiveStrategyPluginRegistry()
        self._register_builtin_strategy_plugins()
        self.strategy_adapters = self.strategy_plugin_registry.adapters()
        self.portfolio_service = LiveStrategyPortfolioService(
            strategy_adapters=self.strategy_adapters,
            portfolio_providers=self.strategy_plugin_registry.portfolio_providers(),
            symbol_name_resolver=symbol_name_resolver,
        )

        self.end_of_day_service = LiveStrategyEndOfDayService(
            parent=self,
            rotation_etf_pool=[],
            portfolio_service=self.portfolio_service,
        )
        for adapter in self.strategy_adapters:
            self.end_of_day_service.register_strategy(adapter.strategy_id, adapter.strategy_name, adapter.run_end_of_day)
        self.end_of_day_service.status_changed.connect(self._on_status_message, Qt.ConnectionType.QueuedConnection)
        self.end_of_day_service.cycle_finished.connect(self._on_end_of_day_finished, Qt.ConnectionType.QueuedConnection)
        self._eod_worker: Optional[_EndOfDayWorker] = None
        self._kline_worker: Optional[_KlineRefreshWorker] = None

        self.startup_orchestrator = QmtStartupOrchestrator(self.broker_panel.broker, self)
        self.startup_orchestrator.status_changed.connect(self._on_startup_status)
        self.startup_orchestrator.finished.connect(self._on_startup_finished)
        self._morning_freshness_timer = QTimer(self)
        self._morning_freshness_timer.setSingleShot(True)
        self._morning_freshness_timer.timeout.connect(self._run_morning_freshness_check)
        self._schedule_next_morning_freshness_check()

        self.center_storage = get_live_strategy_center_storage()
        self.alert_event_service = AlertEventService(self.center_storage, self)
        self.live_event_bridge.events_changed.connect(self.alert_event_service.events_changed.emit)
        self.task_orchestrator_service = TaskOrchestratorService(self.center_storage, self)
        self.hub_state_service = HubStateService(self)
        self._auto_trade_config_service = get_auto_trade_config_service()
        self.hub_controller = LiveStrategyHubController(
            task_service=self.task_orchestrator_service,
            hub_state_service=self.hub_state_service,
            eod_service=self.end_of_day_service,
            strategy_adapters=self.strategy_adapters,
            startup_orchestrator=self.startup_orchestrator,
            parent=self,
        )
        self.end_of_day_service.set_automation_controls(
            pause=self.hub_controller.pause_center_automation,
            resume=self.hub_controller.resume_center_automation,
        )
        self.hub_controller.sync_rotation_pool()

        self.alert_event_service.connect_broker_service(self.broker_panel.broker)
        self.alert_event_service.connect_qmt_startup(self.startup_orchestrator)
        self.alert_event_service.connect_end_of_day(self.end_of_day_service)
        self.alert_event_service.connect_ai_panel(self.ai_panel)
        self.alert_event_service.connect_etf_panel(self.etf_panel)

        self.alert_center_widget = LiveStrategyAlertCenterWidget(self.alert_event_service, self)
        self.task_center_widget = LiveStrategyTaskCenterWidget(self.task_orchestrator_service, self)
        self.exception_order_widget = LiveStrategyExceptionOrderWidget(
            self.broker_panel.broker,
            self,
        )
        self.performance_widget = LiveStrategyPerformanceWidget(
            self.portfolio_service,
            ai_panel=self.ai_panel,
            etf_panel=self.etf_panel,
            parent=self,
        )
        log_path = get_live_strategy_log_path()
        self.log_viewer = LiveLogViewerWidget(log_path, self)
        self.overview_widget = _LiveStrategyOverviewWidget(self)

        self._tab_widgets = {
            self.TAB_OVERVIEW: self.overview_widget,
            self.TAB_ALERTS: self.alert_center_widget,
            self.TAB_TASKS: self.task_center_widget,
            self.TAB_EXCEPTIONS: self.exception_order_widget,
            self.TAB_PERFORMANCE: self.performance_widget,
            self.TAB_LOGS: self.log_viewer,
        }
        self._page_titles = {
            self.TAB_OVERVIEW: "总览",
            self.TAB_ALERTS: "实盘事件",
            self.TAB_TASKS: "实盘任务",
            self.TAB_EXCEPTIONS: "异常订单",
            self.TAB_PERFORMANCE: "实盘收益",
            self.TAB_LOGS: "运行日志",
        }
        for tab_key, tab_title, tab_widget in self.strategy_plugin_registry.tab_specs():
            self._tab_widgets[tab_key] = tab_widget
            self._page_titles[tab_key] = tab_title
        self._setup_navigation_workspace(layout)

        self.overview_widget.navigate_requested.connect(self.switch_to_tab)
        self.overview_widget.account_settings_requested.connect(self._open_account_settings_dialog)
        self.alert_center_widget.navigate_requested.connect(self.switch_to_tab)
        self.alert_center_widget.market_view_requested.connect(self._open_readonly_market_view)
        self.exception_order_widget.navigate_requested.connect(self.switch_to_tab)
        self.exception_order_widget.market_view_requested.connect(self._open_readonly_market_view)
        self.ai_panel.market_view_requested.connect(self._open_readonly_market_view)
        self.etf_panel.market_view_requested.connect(self._open_readonly_market_view)
        self.unmanaged_panel.market_view_requested.connect(self._open_readonly_market_view)
        self.status_bar_widget.navigate_requested.connect(self.switch_to_tab)
        self.status_bar_widget.mode_change_requested.connect(self._set_auto_trade_mode)
        self.status_bar_widget.automation_toggle_requested.connect(self._toggle_center_automation)
        self.status_bar_widget.account_settings_requested.connect(self._open_account_settings_dialog)

        self.hub_controller.register_center_tasks(
            startup_action=self._start_startup_orchestration,
            morning_freshness_action=self._run_morning_freshness_check,
            end_of_day_action=self._run_end_of_day_cycle,
            startup_message_provider=lambda: self.broker_panel.client_status_label.text()
            if hasattr(self.broker_panel, "client_status_label") else "",
            strategy_task_specs=self.strategy_plugin_registry.task_specs(),
        )
        self.hub_state_service.bind(
            broker_service=self.broker_panel.broker,
            startup_orchestrator=self.startup_orchestrator,
            eod_service=self.end_of_day_service,
            alert_service=self.alert_event_service,
            task_service=self.task_orchestrator_service,
            strategy_adapters=self.strategy_adapters,
        )
        self.hub_state_service.state_changed.connect(self.status_bar_widget.refresh_view)
        self.hub_state_service.state_changed.connect(self.overview_widget.refresh_view)
        try:
            self.hub_state_service.refresh_state()
        except Exception:
            pass

        QTimer.singleShot(600, self._start_startup_orchestration)
        QTimer.singleShot(1200, self._refresh_center_public_views)

    def _open_market_view_from_input(self) -> None:
        query = self.market_symbol_input.text().strip()
        if not query:
            QMessageBox.information(self, "只读行情", "请输入要查看的标的代码或名称。")
            return
        resolved = self._resolve_market_query(query)
        if resolved is None:
            QMessageBox.information(self, "只读行情", f"未找到匹配的标的：{query}")
            return
        code, name = resolved
        self._open_readonly_market_view(code, name)

    def _open_readonly_market_view(self, code: str, name: str = "") -> None:
        normalized = str(code or "").strip().upper()
        if "." in normalized:
            normalized = normalized.split(".", 1)[0]
        if not normalized:
            QMessageBox.information(self, "只读行情", "未找到可查看的标的代码。")
            return
        dialog = ReadOnlyMarketViewDialog(self, symbol_name_resolver=self._resolve_symbol_name)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_args, d=dialog: self._forget_market_view_dialog(d))
        self._market_view_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QTimer.singleShot(0, lambda: dialog.load_symbol(normalized, name or self._resolve_symbol_name(normalized)))

    def _forget_market_view_dialog(self, dialog: ReadOnlyMarketViewDialog) -> None:
        try:
            self._market_view_dialogs.remove(dialog)
        except ValueError:
            pass

    def _resolve_symbol_name(self, code: str) -> str:
        normalized = str(code or "").strip().upper()
        if "." in normalized:
            normalized = normalized.split(".", 1)[0]
        for mapping in (self.name_map, self.etf_name_map):
            if normalized in mapping:
                return str(mapping.get(normalized) or "")
        if callable(self.symbol_name_resolver):
            try:
                return str(self.symbol_name_resolver(normalized) or "")
            except Exception:
                return ""
        return ""

    def _resolve_market_query(self, query: str) -> Optional[tuple[str, str]]:
        text = str(query or "").strip()
        if not text:
            return None
        normalized = text.upper()
        if "." in normalized:
            normalized = normalized.split(".", 1)[0]
        if normalized.isdigit():
            code = normalized.zfill(6)
            return code, self._resolve_symbol_name(code)

        exact_match: Optional[tuple[str, str]] = None
        fuzzy_match: Optional[tuple[str, str]] = None
        for mapping in (self.name_map, self.etf_name_map):
            for raw_code, raw_name in mapping.items():
                code = str(raw_code or "").strip().upper()
                if "." in code:
                    code = code.split(".", 1)[0]
                code = code.zfill(6) if code.isdigit() else code
                name = str(raw_name or "").strip()
                if not code or not name:
                    continue
                if name == text:
                    exact_match = (code, name)
                    break
                if fuzzy_match is None and text in name:
                    fuzzy_match = (code, name)
            if exact_match is not None:
                break
        return exact_match or fuzzy_match

    def _register_builtin_strategy_plugins(self) -> None:
        ai_spec = self.ai_strategy_spec
        unmanaged_spec = self.unmanaged_strategy_spec
        etf_spec = self.etf_strategy_spec
        self.strategy_plugin_registry.register(
            LiveStrategyPlugin(
                plugin_id=ai_spec.plugin_id,
                plugin_name=ai_spec.plugin_name,
                adapter=self.ai_strategy_adapter,
                widget=self.ai_panel,
                tab_key=ai_spec.plugin_tab_key or self.TAB_AI,
                tab_title=ai_spec.plugin_tab_title or "AI实盘决策",
                task_specs=(
                    LiveStrategyTaskSpec(
                        task_key="daily_ai_strategy_cycle",
                        task_type="ai",
                        title="每日 AI 实盘决策任务",
                        provider=self._task_provider_ai_scheduler,
                        strategy_id=ai_spec.strategy_id,
                        strategy_name=ai_spec.strategy_name,
                        actions={
                            "立即执行": lambda: self._run_strategy_action(
                                lambda: self.ai_panel.scheduler.run_now("daily_ai_strategy_cycle"),
                                "已触发 AI 实盘决策定时任务",
                            ),
                            "检查并执行输出": lambda: self._run_strategy_action(
                                lambda: self.hub_controller.execute_strategy_signals(ai_spec.strategy_id),
                                "已触发 AI 实盘决策输出并统一执行",
                            ),
                            "暂停调度": self.ai_strategy_adapter.pause_automation,
                            "恢复调度": self.ai_strategy_adapter.resume_automation,
                        },
                        order=10,
                    ),
                ),
                portfolio_providers=(
                    create_ai_stock_portfolio_provider(self.ai_panel, order=10),
                ),
                metadata=ai_spec.to_plugin_metadata(),
                order=10,
            )
        )
        self.strategy_plugin_registry.register(
            create_unmanaged_position_review_plugin(
                self.unmanaged_panel,
                tab_key=unmanaged_spec.plugin_tab_key or self.TAB_UNMANAGED,
                task_provider=self._task_provider_unmanaged_ai_scheduler,
                run_scan_action=lambda: self._run_strategy_action(
                    lambda: self.ai_panel.scheduler.run_now("daily_unmanaged_position_scan"),
                    "已触发未管理持仓 AI 巡检",
                ),
                order=20,
            )
        )
        self.strategy_plugin_registry.register(
            LiveStrategyPlugin(
                plugin_id=etf_spec.plugin_id,
                plugin_name=etf_spec.plugin_name,
                adapter=self.etf_strategy_adapter,
                widget=self.etf_panel,
                tab_key=etf_spec.plugin_tab_key or self.TAB_ETF,
                tab_title=etf_spec.plugin_tab_title or "ETF轮动实盘",
                task_specs=(
                    LiveStrategyTaskSpec(
                        task_key="etf_rotation_auto_check",
                        task_type="etf",
                        title="ETF 轮动实盘自动检查",
                        provider=self._task_provider_etf_rotation,
                        strategy_id=etf_spec.strategy_id,
                        strategy_name=etf_spec.strategy_name,
                        actions={
                            "仅检查信号": lambda: self._run_strategy_action(
                                lambda: self.etf_panel.engine.run_signal_check(),
                                "已触发 ETF 轮动实盘信号检查",
                            ),
                            "检查并执行": lambda: self._run_strategy_action(
                                lambda: self.hub_controller.execute_strategy_signals(etf_spec.strategy_id),
                                "已触发 ETF 轮动实盘信号检查并统一执行",
                            ),
                            "暂停调度": self.etf_strategy_adapter.pause_automation,
                            "恢复调度": self.etf_strategy_adapter.resume_automation,
                        },
                        order=10,
                    ),
                ),
                portfolio_providers=(
                    create_etf_rotation_portfolio_provider(
                        self.etf_panel,
                        self.etf_strategy_adapter,
                        order=10,
                    ),
                ),
                metadata=etf_spec.to_plugin_metadata(),
                order=30,
            )
        )

    def _setup_navigation_workspace(self, root_layout: QVBoxLayout) -> None:
        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        nav_host = QWidget(content)
        nav_layout = QVBoxLayout(nav_host)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)
        nav_layout.addWidget(self.nav_tree, 1)
        nav_layout.addWidget(self.market_quick_widget)
        content_layout.addWidget(nav_host)
        content_layout.addWidget(self.page_stack, 1)
        root_layout.addWidget(content, 1)

        self._add_pages_to_stack()
        self._build_navigation_tree()
        self._select_navigation_item(self.TAB_OVERVIEW)

    def _build_market_quick_widget(self) -> QWidget:
        widget = QWidget(self)
        widget.setFixedWidth(118)
        widget.setStyleSheet(
            "QWidget{background:#151515;border:1px solid #2b2b2b;border-radius:4px;}"
            "QLabel{color:#9ca3af;border:none;background:transparent;font-size:11px;}"
            "QLineEdit{background:#0f172a;color:#e5e7eb;border:1px solid #334155;border-radius:3px;padding:3px;}"
            "QPushButton{background:#334155;color:#ffffff;border:1px solid #475569;border-radius:3px;padding:4px;}"
            "QPushButton:hover{background:#475569;}"
        )
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        layout.addWidget(QLabel("只读行情", widget))
        self.market_symbol_input = QLineEdit(widget)
        self.market_symbol_input.setPlaceholderText("代码/名称")
        self.market_symbol_input.returnPressed.connect(self._open_market_view_from_input)
        layout.addWidget(self.market_symbol_input)
        self.open_market_view_btn = QPushButton("看K线", widget)
        self.open_market_view_btn.setToolTip("打开只读 K线 / 分时 / 盘口视图，不会触发下单")
        self.open_market_view_btn.clicked.connect(self._open_market_view_from_input)
        layout.addWidget(self.open_market_view_btn)
        return widget

    def _add_pages_to_stack(self) -> None:
        preferred_order = [
            self.TAB_OVERVIEW,
            self.TAB_AI,
            self.TAB_ETF,
            self.TAB_UNMANAGED,
            self.TAB_EXCEPTIONS,
            self.TAB_ALERTS,
            self.TAB_TASKS,
            self.TAB_PERFORMANCE,
            self.TAB_LOGS,
        ]
        for key in preferred_order:
            self._add_page_to_stack(key)
        for key in list(self._tab_widgets.keys()):
            self._add_page_to_stack(key)

    def _add_page_to_stack(self, key: str) -> None:
        normalized = str(key or "").strip()
        if not normalized or normalized in self._page_index_by_key:
            return
        widget = self._tab_widgets.get(normalized)
        if widget is None:
            return
        self._page_index_by_key[normalized] = self.page_stack.addWidget(widget)

    def _build_navigation_tree(self) -> None:
        self.nav_tree.clear()
        self._nav_items.clear()

        self._add_nav_root("总览", self.TAB_OVERVIEW)

        strategy_root = self._add_nav_root("实盘策略", self.TAB_AI)
        self._add_nav_child(strategy_root, "AI实盘决策", self.TAB_AI)
        self._add_nav_child(strategy_root, "ETF轮动实盘", self.TAB_ETF)
        self._add_nav_child(strategy_root, "未管理持仓巡检", self.TAB_UNMANAGED)
        for key in self._extra_strategy_page_keys():
            self._add_nav_child(strategy_root, self._page_titles.get(key, key), key)

        self._add_nav_root("异常订单", self.TAB_EXCEPTIONS)
        self._add_nav_root("实盘事件", self.TAB_ALERTS)
        self._add_nav_root("实盘任务", self.TAB_TASKS)
        self._add_nav_root("实盘收益", self.TAB_PERFORMANCE)
        self._add_nav_root("运行日志", self.TAB_LOGS)

        self.nav_tree.expandAll()

    def _extra_strategy_page_keys(self) -> list[str]:
        grouped = {
            self.TAB_OVERVIEW,
            self.TAB_AI,
            self.TAB_ETF,
            self.TAB_UNMANAGED,
            self.TAB_EXCEPTIONS,
            self.TAB_ALERTS,
            self.TAB_TASKS,
            self.TAB_PERFORMANCE,
            self.TAB_LOGS,
        }
        return [key for key in self._tab_widgets.keys() if key not in grouped]

    def _add_nav_root(self, title: str, key: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title])
        item.setData(0, Qt.ItemDataRole.UserRole, key)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        self.nav_tree.addTopLevelItem(item)
        self._nav_items.setdefault(key, item)
        return item

    def _add_nav_child(self, parent: QTreeWidgetItem, title: str, key: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title])
        item.setData(0, Qt.ItemDataRole.UserRole, key)
        parent.addChild(item)
        self._nav_items[key] = item
        return item



    def _on_navigation_item_changed(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if self._suppress_nav_signal or current is None:
            return
        key = str(current.data(0, Qt.ItemDataRole.UserRole) or "").strip()
        if key:
            self.switch_to_tab(key)

    def _select_navigation_item(self, key: str) -> None:
        item = self._nav_items.get(str(key or "").strip())
        if item is None or self.nav_tree.currentItem() is item:
            return
        self._suppress_nav_signal = True
        try:
            self.nav_tree.setCurrentItem(item)
        finally:
            self._suppress_nav_signal = False

    @staticmethod
    def _run_strategy_action(callback: Callable[[], None], message: str) -> str:
        callback()
        return message

    def _task_provider_ai_scheduler(self) -> dict:
        try:
            rows = self.ai_strategy_adapter.get_task_summaries()
        except Exception:
            rows = []
        return dict(rows[0] if rows else {})

    def _task_provider_unmanaged_ai_scheduler(self) -> dict:
        try:
            return dict(self.ai_strategy_adapter.get_task_summary("daily_unmanaged_position_scan") or {})
        except Exception:
            return {}

    def _task_provider_etf_rotation(self) -> dict:
        try:
            rows = self.etf_strategy_adapter.get_task_summaries()
        except Exception:
            rows = []
        return dict(rows[0] if rows else {})

    def switch_to_tab(self, tab_name: str) -> None:
        normalized = str(tab_name or "").strip().lower()
        alias_map = {
            "overview": self.TAB_OVERVIEW,
            "home": self.TAB_OVERVIEW,
            "dashboard": self.TAB_OVERVIEW,
            "alert": self.TAB_ALERTS,
            "alerts": self.TAB_ALERTS,
            "tasks": self.TAB_TASKS,
            "task": self.TAB_TASKS,
            "exceptions": self.TAB_EXCEPTIONS,
            "orders": self.TAB_EXCEPTIONS,
            "performance": self.TAB_PERFORMANCE,
            "logs": self.TAB_LOGS,
            "log": self.TAB_LOGS,
            "ai": self.TAB_AI,
            "unmanaged": self.TAB_UNMANAGED,
            "etf": self.TAB_ETF,
        }
        target_key = alias_map.get(normalized, normalized if normalized in self._tab_widgets else self.TAB_AI)
        index = self._page_index_by_key.get(target_key)
        if index is not None:
            self.page_stack.setCurrentIndex(index)
            self._select_navigation_item(target_key)
            if target_key == self.TAB_LOGS:
                self.log_viewer.refresh_log()

    def set_symbol(self, code: str, name: str = "") -> None:
        self.switch_to_tab(self.TAB_AI)
        self.ai_panel.set_symbol(code, name)

    def _refresh_center_public_views(self) -> str:
        return self.hub_controller.refresh_public_views([
            self.alert_center_widget.refresh_events,
            self.exception_order_widget.refresh_orders,
            self.performance_widget.refresh_view,
        ])

    def _set_auto_trade_mode(self, mode: str) -> str:
        target_mode = str(mode or "off").strip().lower()
        current_cfg = self._auto_trade_config_service.get_config()
        current_mode = str(current_cfg.auto_trade_mode or "off").strip().lower()
        if target_mode == current_mode:
            return f"统一执行模式保持为 {current_mode}"
        if target_mode == "live" and not self._confirm_switch_to_live_mode():
            self.hub_controller.refresh_state()
            return "已取消切换到实盘模式"
        cfg = self._auto_trade_config_service.update_config(auto_trade_mode=target_mode)
        self.hub_controller.refresh_state()
        message = f"统一执行模式已切换为 {cfg.auto_trade_mode}"
        self.status_changed.emit(message)
        return message

    def _confirm_switch_to_live_mode(self) -> bool:
        state = self.hub_state_service.get_state()
        broker_connected = bool(state.get("broker_connected", False))
        qmt_running = bool(state.get("qmt_running", False))
        manual_enabled = bool(state.get("manual_orders_enabled", True))
        require_trading_time = bool(state.get("require_trading_time", True))
        alert_counts = dict(state.get("alert_counts", {}) or {})
        open_alerts = int(alert_counts.get("open", 0) or 0)
        exception_count = int(state.get("exception_order_count", 0) or 0)
        risk_summary = dict(state.get("risk_summary", {}) or {})
        risk_tooltip = str(risk_summary.get("tooltip", "") or "")
        details = [
            f"券商连接：{'已连接' if broker_connected else '未连接'}",
            f"QMT状态：{'运行中' if qmt_running else '未就绪'}",
            f"手动委托：{'开启' if manual_enabled else '关闭'}",
            f"交易时段闸：{'开启' if require_trading_time else '关闭'}",
            f"未处理告警：{open_alerts}",
            f"异常订单：{exception_count}",
        ]
        if risk_tooltip:
            details.append("")
            details.append(risk_tooltip)
        text = "即将切换到实盘模式，系统可能向券商提交真实委托。\n\n" + "\n".join(details) + "\n\n确认继续？"
        result = QMessageBox.question(
            self,
            "确认切换到实盘模式",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _toggle_center_automation(self, resume: bool) -> str:
        message = self.hub_controller.toggle_center_automation(resume)
        if message:
            self.status_changed.emit(message)
        return message

    def _open_account_settings_dialog(self) -> None:
        """Show the shared account-level gateway settings dialog.

        打开后修改的是 ``AutoTradeConfig``，对所有策略（AI + ETF + 其他条件单）
        生效；关闭后立刻刷新 hub 状态，避免状态栏与新配置不一致。
        """
        dialog = LiveStrategyAccountSettingsDialog(self, service=self._auto_trade_config_service)
        if dialog.exec():
            self.hub_controller.refresh_state()

    def _start_startup_orchestration(self) -> None:
        if self.startup_orchestrator.is_running:
            return
        started = self.startup_orchestrator.start()
        if started:
            self.broker_panel.show_client_workflow_status("连接自检中...", success=None)
            self.task_orchestrator_service.record_runtime(
                "startup_check",
                status="running",
                message="连接自检中...",
                trigger="manual_or_auto",
                started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

    def _schedule_next_morning_freshness_check(self) -> None:
        """Schedule one trading-day-only 09:35 market-data freshness check."""
        now = datetime.now()
        target = now.replace(hour=9, minute=35, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        while not is_trading_day(target.date()):
            target += timedelta(days=1)
        delay_ms = max(int((target - now).total_seconds() * 1000), 1000)
        self._morning_freshness_timer.start(delay_ms)

    def _run_morning_freshness_check(self) -> None:
        self._schedule_next_morning_freshness_check()
        self.task_orchestrator_service.record_runtime(
            "morning_freshness",
            status="running",
            message="盘中新鲜度检查已开始",
            trigger="timer_or_manual",
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if self.startup_orchestrator.is_running:
            self.broker_panel.show_client_workflow_status("09:35 盘中数据检查跳过：当前自检进行中", success=None)
            self.task_orchestrator_service.record_runtime(
                "morning_freshness",
                status="skipped",
                message="当前自检进行中，跳过盘中新鲜度检查",
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return
        started = self.startup_orchestrator.start()
        if started:
            self.broker_panel.show_client_workflow_status("09:35 盘中数据检查中...", success=None)

    def _on_startup_status(self, message: str) -> None:
        self.broker_panel.show_client_workflow_status(message, success=None)

    def _on_startup_finished(self, success: bool, message: str) -> None:
        self.broker_panel.show_client_workflow_status(message, success=success)
        self.broker_panel.refresh_client_status()
        self.task_orchestrator_service.record_runtime(
            "startup_check",
            status="completed" if success else "failed",
            message=message,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if success:
            QTimer.singleShot(800, self._try_end_of_day_catchup)

    def _run_end_of_day_cycle(self) -> None:
        self._sync_rotation_pool()
        self.task_orchestrator_service.record_runtime(
            "end_of_day_cycle",
            status="running",
            message="开始执行统一日终流程",
            trigger="manual",
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._start_end_of_day_worker("manual")

    def _try_end_of_day_catchup(self) -> None:
        self._sync_rotation_pool()
        self.task_orchestrator_service.record_runtime(
            "end_of_day_cycle",
            status="running",
            message="检查并补跑缺失的日终流程",
            trigger="catchup",
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._start_kline_refresh_catchup()

    def _sync_rotation_pool(self) -> None:
        """Push the latest ETF rotation pool into the EOD service."""
        self.hub_controller.sync_rotation_pool()

    def _get_etf_rotation_pool(self) -> list[str]:
        engine = getattr(self.etf_panel, "engine", None)
        config = getattr(engine, "config", None)
        return list(getattr(config, "etf_pool", []) or [])

    def _start_kline_refresh_catchup(self) -> None:
        if self._kline_worker is not None and self._kline_worker.isRunning():
            return
        snapshot_date = self.end_of_day_service._today()  # noqa: SLF001
        cycle_state = self.end_of_day_service._get_cycle_state(snapshot_date)  # noqa: SLF001
        phases = dict(cycle_state.get("phases", {}) or {})
        reconcile_state = self.end_of_day_service.daily_auto_trade.get_day_state_section("reconcile", day=snapshot_date)
        reconcile_status = str(reconcile_state.get("status", "") or "")
        refresh_status = str((phases.get("phase4_kline_refresh", {}) or {}).get("status", "") or "")
        if reconcile_status != "completed":
            message = self.end_of_day_service.daily_auto_trade.should_run_reconcile_catchup()[1]
            self._on_status_message(message)
            self.task_orchestrator_service.record_runtime(
                "end_of_day_cycle",
                status="skipped",
                message=message,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return
        if refresh_status == "completed":
            message = "今日日终K线刷新已完成，无需补跑"
            self._on_status_message(message)
            self.task_orchestrator_service.record_runtime(
                "end_of_day_cycle",
                status="completed",
                message=message,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return
        missing_non_refresh_phases = [
            phase_key
            for phase_key in ("phase2_strategy_hooks", "phase3_portfolio_snapshots")
            if str((phases.get(phase_key, {}) or {}).get("status", "") or "") != "completed"
        ]
        if missing_non_refresh_phases:
            message = "日终补跑存在非K线阶段未完成，请手动执行完整日终流程"
            self._on_status_message(message)
            self.task_orchestrator_service.record_runtime(
                "end_of_day_cycle",
                status="skipped",
                message=message,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return
        self.run_eod_btn.setEnabled(False)
        self._on_status_message("Phase 4: 补跑全量K线数据刷新...")
        self.end_of_day_service._update_cycle_state(  # noqa: SLF001
            snapshot_date,
            status="running",
            started_at=cycle_state.get("started_at", "") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            completed_at="",
            last_trigger="catchup",
            workflow_version="unified_v1",
        )
        self.end_of_day_service._update_cycle_phase(  # noqa: SLF001
            phase_key="phase4_kline_refresh",
            status="running",
            snapshot_date=snapshot_date,
            trigger="catchup",
            message="开始全量K线数据刷新",
        )
        self._kline_worker = _KlineRefreshWorker(self._get_etf_rotation_pool(), self)
        self._kline_worker.status_message.connect(self._on_status_message, Qt.ConnectionType.QueuedConnection)
        self._kline_worker.finished_refresh.connect(self._on_kline_refresh_catchup_finished, Qt.ConnectionType.QueuedConnection)
        self._kline_worker.failed_refresh.connect(self._on_kline_refresh_catchup_failed, Qt.ConnectionType.QueuedConnection)
        self._kline_worker.finished.connect(self._cleanup_kline_refresh_worker)
        self._kline_worker.start()

    def _start_end_of_day_worker(self, mode: str) -> None:
        if self._eod_worker is not None and self._eod_worker.isRunning():
            return
        self.run_eod_btn.setEnabled(False)
        if mode == "catchup":
            self._on_status_message("检查并补跑缺失的日终流程...")
        else:
            self._on_status_message("开始执行统一日终流程...")
        self._eod_worker = _EndOfDayWorker(self.end_of_day_service, mode, self)
        self._eod_worker.finished_cycle.connect(self._on_end_of_day_worker_finished)
        self._eod_worker.failed_cycle.connect(self._on_end_of_day_worker_failed)
        self._eod_worker.finished.connect(self._cleanup_end_of_day_worker)
        self._eod_worker.start()

    def _on_end_of_day_worker_finished(self, mode: str, success: bool, message: str, payload: object) -> None:
        if mode == "catchup" and not success:
            self._on_status_message(message)
            self.run_eod_btn.setEnabled(True)
            cycle_state = self.end_of_day_service._get_cycle_state()  # noqa: SLF001
            overall_status = str(cycle_state.get("status", "") or "").strip().lower()
            runtime_status = "completed" if overall_status == "completed" else ("failed" if overall_status == "failed" else "skipped")
            self.task_orchestrator_service.record_runtime(
                "end_of_day_cycle",
                status=runtime_status,
                message=message,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return
        if mode == "catchup" and success:
            self._on_status_message(f"🔁 {message}")
            return
        if not success:
            self.run_eod_btn.setEnabled(True)
            self._on_status_message(f"❌ {message}")

    def _on_end_of_day_worker_failed(self, _mode: str, message: str) -> None:
        self.run_eod_btn.setEnabled(True)
        self._on_status_message(f"❌ 日终线程异常: {message}")
        self.task_orchestrator_service.record_runtime(
            "end_of_day_cycle",
            status="failed",
            message=message,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _cleanup_end_of_day_worker(self) -> None:
        worker = self._eod_worker
        if worker is None:
            return
        worker.deleteLater()
        self._eod_worker = None

    def _on_kline_refresh_catchup_finished(self, success: bool, message: str) -> None:
        snapshot_date = self.end_of_day_service._today()  # noqa: SLF001
        self.end_of_day_service._update_cycle_phase(  # noqa: SLF001
            phase_key="phase4_kline_refresh",
            status="completed" if success else "failed",
            snapshot_date=snapshot_date,
            trigger="catchup",
            message=message,
        )
        unlock_success = True
        unlock_message = "次日任务已在前序流程解锁"
        cycle_state = self.end_of_day_service._get_cycle_state(snapshot_date)  # noqa: SLF001
        phases = dict(cycle_state.get("phases", {}) or {})
        if success and str((phases.get("phase5_next_day_unlock", {}) or {}).get("status", "") or "") != "completed":
            unlock_success, unlock_message = self.end_of_day_service._run_next_day_unlock_phase(  # noqa: SLF001
                snapshot_date=snapshot_date,
                trigger="catchup",
            )
        final_state = self.end_of_day_service._refresh_cycle_overall_state(  # noqa: SLF001
            snapshot_date=snapshot_date,
            trigger="catchup",
        )
        overall_status = str(final_state.get("status", "") or "")
        final_success = success and unlock_success and overall_status == "completed"
        final_message = message if unlock_success else f"{message}；{unlock_message}"
        self.run_eod_btn.setEnabled(True)
        self._on_status_message(("🔁 " if final_success else "❌ ") + final_message)
        self.task_orchestrator_service.record_runtime(
            "end_of_day_cycle",
            status="completed" if final_success else "failed",
            message=final_message,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._refresh_center_public_views()

    def _on_kline_refresh_catchup_failed(self, message: str) -> None:
        snapshot_date = self.end_of_day_service._today()  # noqa: SLF001
        final_message = f"K线全量刷新异常: {message}"
        self.end_of_day_service._update_cycle_phase(  # noqa: SLF001
            phase_key="phase4_kline_refresh",
            status="failed",
            snapshot_date=snapshot_date,
            trigger="catchup",
            message=final_message,
        )
        self.run_eod_btn.setEnabled(True)
        self._on_status_message(f"❌ {final_message}")
        self.task_orchestrator_service.record_runtime(
            "end_of_day_cycle",
            status="failed",
            message=final_message,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._refresh_center_public_views()

    def _cleanup_kline_refresh_worker(self) -> None:
        worker = self._kline_worker
        if worker is None:
            return
        worker.deleteLater()
        self._kline_worker = None

    def _on_end_of_day_finished(self, success: bool, message: str, _payload: dict) -> None:
        self.run_eod_btn.setEnabled(True)
        self.hub_controller.refresh_strategies_after_eod()
        color = "#4caf50" if success else "#d9534f"
        self.eod_status_label.setStyleSheet(f"color:{color};font-size:12px;")
        self._set_eod_text(f"日终: {message}")
        self.status_changed.emit(message)
        self.task_orchestrator_service.record_runtime(
            "end_of_day_cycle",
            status="completed" if success else "failed",
            message=message,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._refresh_center_public_views()

    def _on_status_message(self, message: str) -> None:
        self.eod_status_label.setStyleSheet("color:#888;font-size:12px;")
        self._set_eod_text(f"日终: {message}")
        self.status_changed.emit(message)

    def _set_eod_text(self, text: str) -> None:
        self.eod_status_label.setToolTip(text)
        metrics = QFontMetrics(self.eod_status_label.font())
        self.eod_status_label.setText(
            metrics.elidedText(text, Qt.TextElideMode.ElideRight, self.eod_status_label.width())
        )

    def closeEvent(self, event) -> None:
        try:
            self.live_event_bridge.stop()
            self.live_event_persistence.stop()
        except Exception:
            pass
        try:
            self.startup_orchestrator.cancel()
        except Exception:
            pass
        try:
            self._morning_freshness_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)


class LiveStrategyHubWindow(QMainWindow):
    """Window wrapper for the unified live strategy workspace."""

    def __init__(
        self,
        parent=None,
        *,
        context_provider=None,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
        name_map: Optional[Dict[str, str]] = None,
        etf_name_map: Optional[Dict[str, str]] = None,
        initial_tab: str = LiveStrategyHubWidget.TAB_OVERVIEW,
    ):
        super().__init__(parent)
        self.setWindowTitle("实盘策略中枢")
        self.resize(1480, 900)
        self.setMinimumSize(1150, 800)
        self._allow_close = False
        self._tray_notice_shown = False
        self._tray_icon: Optional[QSystemTrayIcon] = None

        self.workspace = LiveStrategyHubWidget(
            self,
            context_provider=context_provider,
            symbol_name_resolver=symbol_name_resolver,
            name_map=name_map,
            etf_name_map=etf_name_map,
        )
        self.setCentralWidget(self.workspace)
        self.workspace.switch_to_tab(initial_tab)
        self.workspace.status_changed.connect(self.statusBar().showMessage)
        self.statusBar().showMessage("就绪")
        self._setup_window_icon()
        self._setup_system_tray()

    def switch_to_tab(self, tab_name: str) -> None:
        self.workspace.switch_to_tab(tab_name)

    def set_symbol(self, code: str, name: str = "") -> None:
        self.workspace.set_symbol(code, name)

    def _setup_window_icon(self) -> None:
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "icon.jpeg")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                self.setWindowIcon(icon)
                app = QApplication.instance()
                if app is not None:
                    app.setWindowIcon(icon)

    def _setup_system_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray_icon = QSystemTrayIcon(self.windowIcon(), self)
        tray_icon.setToolTip("实盘策略中枢")

        menu = QMenu(self)
        show_action = QAction("显示主窗口", self)
        show_action.triggered.connect(self._show_from_tray)
        menu.addAction(show_action)

        show_alert_action = QAction("打开实盘事件", self)
        show_alert_action.triggered.connect(lambda: self._show_tab_from_tray(self.workspace.TAB_ALERTS))
        menu.addAction(show_alert_action)

        show_log_action = QAction("显示运行日志", self)
        show_log_action.triggered.connect(self._show_logs_from_tray)
        menu.addAction(show_log_action)

        hide_action = QAction("隐藏到托盘", self)
        hide_action.triggered.connect(self._hide_to_tray)
        menu.addAction(hide_action)

        menu.addSeparator()

        run_eod_action = QAction("执行日终", self)
        run_eod_action.triggered.connect(self.workspace._run_end_of_day_cycle)
        menu.addAction(run_eod_action)

        menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self._exit_from_tray)
        menu.addAction(exit_action)

        tray_icon.setContextMenu(menu)
        tray_icon.activated.connect(self._on_tray_activated)
        tray_icon.show()
        self._tray_icon = tray_icon

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.DoubleClick,
            QSystemTrayIcon.ActivationReason.Trigger,
        ):
            if self.isVisible() and not self.isMinimized():
                self._hide_to_tray()
            else:
                self._show_from_tray()

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _show_logs_from_tray(self) -> None:
        self._show_tab_from_tray(self.workspace.TAB_LOGS)

    def _show_tab_from_tray(self, tab_name: str) -> None:
        self._show_from_tray()
        self.workspace.switch_to_tab(tab_name)

    def _hide_to_tray(self) -> None:
        if self._tray_icon is None:
            self.hide()
            return
        self.hide()
        if not self._tray_notice_shown:
            self._tray_icon.showMessage(
                "实盘策略中枢",
                "程序已隐藏到系统托盘，可从右下角托盘图标恢复。",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )
            self._tray_notice_shown = True

    def _exit_from_tray(self) -> None:
        self._allow_close = True
        if self._tray_icon is not None:
            self._tray_icon.hide()
        self.close()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        tray_icon = getattr(self, "_tray_icon", None)
        if tray_icon is None:
            return
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized():
            QTimer.singleShot(0, self._hide_to_tray)

    def closeEvent(self, event) -> None:
        tray_icon = getattr(self, "_tray_icon", None)
        allow_close = bool(getattr(self, "_allow_close", False))
        if allow_close or tray_icon is None:
            if tray_icon is not None:
                tray_icon.hide()
            super().closeEvent(event)
            return
        event.ignore()
        self._hide_to_tray()
