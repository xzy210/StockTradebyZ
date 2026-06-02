# main_window.py - 主窗口
"""
来财主窗口
"""
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QMenuBar, QMenu,
    QToolBar, QGroupBox, QCheckBox, QComboBox,
    QDateEdit, QPushButton, QMessageBox, QApplication,
    QInputDialog, QDialog, QDialogButtonBox, QTabWidget,
    QProgressDialog, QProgressBar
)
from PyQt6.QtCore import Qt, QDate, QSize, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut, QIcon

# 本地模块
from trading_app.controllers.realtime_controller import RealtimeController
from trading_app.controllers.sync_controller import SyncController
from trading_app.controllers.trading_bridge import TradingBridge
from common.broker_session_service import get_broker_session_service
from trading_app.widgets.kline_widget import KLineWidget, should_update_realtime_kline
from trading_app.widgets.stock_list_widget import StockListWidget
from trading_app.widgets.timeshare_widget import TimeShareWidget
from trading_app.widgets.trading_simulator_widget import TradingSimulatorWidget
from trading_app.widgets.ai_agent_widget import AIAgentWidget
from trading_app.widgets.live_strategy_hub_widget import LiveStrategyHubWindow
from trading_app.widgets.update_dialog import UpdateDialog
from trading_app.widgets.notification_dialog import NotificationDialog, MarketCloseReminderDialog
from trading_app.widgets.scheduled_task_dialog import ScheduledTaskDialog
from trading_app.widgets.watchlist_panel_widget import WatchlistPanelWidget
from trading_app.widgets.etf_list_widget import ETFListWidget
from trading_app.widgets.watchlist_widget import WatchlistWidget
from trading_app.widgets.chip_distribution_widget import ChipDistributionDialog
from trading_app.widgets.sector_window import SectorWindow
from trading_app.watchlist_manager import WatchlistManager
from common.data_portal import get_data_portal, get_stock_cache
from trading_app.indicators import attach_all_indicators
from trading_app.data_updater import DataUpdateThread, ETFUpdateThread
from trading_app.scheduler import ScheduledTaskManager
from trading_app.services.agent_evidence_service import TEMP_KLINE_PREFIX
from trading_app.services.quote_service import QuoteData
from trading_app.services.data_update_result import DataUpdateResult


class DataPreloadThread(QThread):
    """
    后台预加载股票数据的线程
    在应用启动时将所有股票K线数据加载到内存缓存
    """
    progress_updated = pyqtSignal(int, int, str)  # current, total, code
    finished_signal = pyqtSignal(bool, int, str)  # success, loaded_count, message
    
    def __init__(self, data_dir: str, stock_codes: list):
        super().__init__()
        self.data_dir = data_dir
        self.stock_codes = stock_codes
    
    def run(self):
        try:
            cache = get_stock_cache()
            
            def progress_callback(current, total, code):
                self.progress_updated.emit(current, total, code)
            
            loaded_count = cache.preload_all(
                self.data_dir,
                self.stock_codes,
                progress_callback=progress_callback,
                max_workers=8
            )
            
            self.finished_signal.emit(
                True, 
                loaded_count, 
                f"成功预加载 {loaded_count}/{len(self.stock_codes)} 只股票数据"
            )
        except Exception as e:
            self.finished_signal.emit(False, 0, f"预加载失败: {e}")


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口图标
        icon_path = Path(__file__).resolve().parent.parent / "icon.jpeg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # 数据目录配置
        self.data_dir = self.get_data_dir()
        self.stocklist_path = self.get_stocklist_path()
        
        # 数据
        self.stock_list = []
        self.name_map = {}
        self.current_code = ""
        self.current_name = ""
        
        # 自选股管理
        self.watchlist_manager = WatchlistManager()
        
        # ETF 数据
        self.etf_list = []
        self.etf_name_map = {}
        self.etf_categories = []
        self.current_etf_code = ""
        self.current_etf_name = ""
        
        # 指数数据
        self.current_index_code = ""
        self.current_index_name = ""
        
        self.current_view = "stock"  # "stock", "etf", or "index"
        
        # 设置
        self.ma_windows = [5, 10, 20]
        
        self.setupUI()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_shortcuts()

        # 热门板块/交易窗口引用
        self.sector_window: Optional[SectorWindow] = None
        self.broker_window = None
        self._live_strategy_window: Optional[LiveStrategyHubWindow] = None

        # 先初始化 controller，再加载数据；load_stock_list() 会触发实时行情装配
        self.trading_bridge = TradingBridge(self, self)
        self.realtime_controller = RealtimeController(self, self.trading_bridge, self)
        self.sync_controller = SyncController(
            self,
            realtime_controller=self.realtime_controller,
            trading_bridge=self.trading_bridge,
            parent=self,
        )
        self.quote_service = self.realtime_controller.quote_service

        self.trading_bridge.initialize()
        self.realtime_controller.initialize()

        # 初始化定时任务管理器（需在 agent context provider 注册之前，
        # 因为 build_agent_runtime_context 会读取 scheduler_manager.config）
        self.scheduler_manager = ScheduledTaskManager(self.data_dir, self.stocklist_path)
        self.scheduler_manager.task_finished.connect(self.on_scheduled_task_finished)

        self.agent_widget.set_context_provider(self.build_agent_runtime_context)
        
        # 加载数据
        self.load_stock_list()
        
        self.update_thread = None
        self.update_dialog = None
        self.preload_thread = None
        self._last_update_result: Optional[DataUpdateResult] = None
        self._updating_codes = []  # 记录正在更新的股票代码
        
        # 启动数据预加载
        self.start_data_preload()

        # 收盘提醒相关
        self._market_close_reminder_timer = QTimer(self)
        self._market_close_reminder_timer.timeout.connect(self.check_market_close_reminder)
        # 每分钟检查一次，在15:05-15:30之间
        self._market_close_reminder_timer.start(60000)
        # 启动时也检查一次
        QTimer.singleShot(5000, self.check_market_close_reminder)

    def get_data_dir(self) -> str:
        """获取数据目录路径"""
        # 支持多种路径
        possible_paths = [
            Path(__file__).parent.parent / "data",  # 相对于trading_app目录
            Path("./data"),
            Path("../data"),
        ]
        
        for p in possible_paths:
            if p.exists():
                return str(p)
        
        return str(possible_paths[0])
    
    def get_stocklist_path(self) -> str:
        """获取股票列表文件路径"""
        possible_paths = [
            Path(__file__).parent.parent / "stocklist" / "stocklist.csv",
            Path("./stocklist/stocklist.csv"),
            Path("../stocklist/stocklist.csv"),
        ]
        
        for p in possible_paths:
            if p.exists():
                return str(p)
        
        return str(possible_paths[0])
    
    def setupUI(self):
        """设置界面"""
        self.setWindowTitle("来财")
        self.setMinimumSize(1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板（股票列表 + 设置）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # 股票/ETF 切换 Tab
        self.left_tabs = QTabWidget()
        self.left_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px 16px;
                border: 1px solid #3c3c3c;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3c3c3c;
            }
        """)
        self.left_tabs.currentChanged.connect(self.on_left_tab_changed)
        
        # Tab 1: 股票列表
        self.stock_list_widget = StockListWidget()
        self.stock_list_widget.stockSelected.connect(self.on_stock_selected)
        self.stock_list_widget.refreshRequested.connect(self.on_refresh_strategy)
        self.stock_list_widget.set_watchlist_manager(self.watchlist_manager)
        # 添加右键菜单
        self.stock_list_widget.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.stock_list_widget.list_widget.customContextMenuRequested.connect(self.show_stock_list_context_menu)
        self.left_tabs.addTab(self.stock_list_widget, "📈 股票")
        
        # Tab 2: ETF列表
        self.etf_list_widget = ETFListWidget()
        self.etf_list_widget.etfSelected.connect(self.on_etf_selected)
        self.etf_list_widget.set_watchlist_manager(self.watchlist_manager)
        # 添加右键菜单
        self.etf_list_widget.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.etf_list_widget.list_widget.customContextMenuRequested.connect(self.show_etf_list_context_menu)
        self.left_tabs.addTab(self.etf_list_widget, "📊 ETF")
        
        # Tab 3: 自选列表（股票+ETF混合）
        self.watchlist_widget = WatchlistWidget()
        self.watchlist_widget.itemSelected.connect(self.on_watchlist_item_selected)
        self.watchlist_widget.set_watchlist_manager(self.watchlist_manager)
        # 添加右键菜单
        self.watchlist_widget.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.watchlist_widget.list_widget.customContextMenuRequested.connect(self.show_watchlist_context_menu)
        self.left_tabs.addTab(self.watchlist_widget, "⭐ 自选")
        
        # Tab 4: 指数列表
        from trading_app.widgets.index_list_widget import IndexListWidget
        self.index_list_widget = IndexListWidget()
        self.index_list_widget.indexSelected.connect(self.on_index_selected)
        self.left_tabs.addTab(self.index_list_widget, "📊 指数")
        
        left_layout.addWidget(self.left_tabs, stretch=1)
        
        # 指标设置
        indicator_group = QGroupBox("指标设置")
        indicator_layout = QVBoxLayout(indicator_group)
        
        self.volume_checkbox = QCheckBox("成交量")
        self.volume_checkbox.setChecked(True)
        self.volume_checkbox.stateChanged.connect(self.on_indicator_changed)
        indicator_layout.addWidget(self.volume_checkbox)
        
        self.macd_checkbox = QCheckBox("MACD")
        self.macd_checkbox.setChecked(True)
        self.macd_checkbox.stateChanged.connect(self.on_indicator_changed)
        indicator_layout.addWidget(self.macd_checkbox)
        
        self.kdj_checkbox = QCheckBox("KDJ")
        self.kdj_checkbox.setChecked(False)
        self.kdj_checkbox.stateChanged.connect(self.on_indicator_changed)
        indicator_layout.addWidget(self.kdj_checkbox)
        
        left_layout.addWidget(indicator_group)
        
        # 日期范围
        date_group = QGroupBox("日期范围")
        date_layout = QVBoxLayout(date_group)
        
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("起始:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate(2016, 1, 1))
        start_layout.addWidget(self.start_date_edit)
        date_layout.addLayout(start_layout)
        
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("结束:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        end_layout.addWidget(self.end_date_edit)
        date_layout.addLayout(end_layout)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_chart)
        date_layout.addWidget(refresh_btn)
        
        left_layout.addWidget(date_group)
        
        splitter.addWidget(left_panel)
        
        # 右侧面板（K线图 + 面板）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs)
        
        # Tab 1: K线图
        self.kline_widget = KLineWidget()
        self.right_tabs.addTab(self.kline_widget, "📈 K线图")
        
        # Tab 2: 分时图
        self.timeshare_widget = TimeShareWidget()
        self.timeshare_widget.refreshStatusChanged.connect(self.on_timeshare_refresh_status_changed)
        self.right_tabs.addTab(self.timeshare_widget, "📊 分时图")
        
        # Tab 3: 面板
        self.watchlist_panel = WatchlistPanelWidget(
            self.watchlist_manager, 
            self.name_map, 
            self.data_dir
        )
        self.watchlist_panel.stockSelected.connect(self.on_panel_stock_selected)
        self.right_tabs.addTab(self.watchlist_panel, "📋 面板")

        # 连接列表变化信号到面板
        self.stock_list_widget.displayListChanged.connect(self.on_display_list_changed)
        self.stock_list_widget.groupChanged.connect(self.on_group_changed_for_panel)
        
        # 连接自选Tab的信号到面板
        self.watchlist_widget.groupChanged.connect(self.on_watchlist_group_changed_for_panel)
        self.watchlist_widget.displayListChanged.connect(self.on_watchlist_display_list_changed)
        
        # 连接Tab切换信号
        self.right_tabs.currentChanged.connect(self.on_right_tab_changed)

        splitter.addWidget(right_panel)
        
        # 智能体面板
        self.agent_widget = AIAgentWidget()
        self.agent_widget.setVisible(False)
        splitter.addWidget(self.agent_widget)
        
        # 保存 splitter 引用
        self.splitter = splitter
        
        # 设置分割比例
        splitter.setSizes([150, 1050, 0])
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 预加载进度条 (放在状态栏右侧)
        self.preload_status_label = QLabel("")
        self.preload_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.preload_progress_bar = QProgressBar()
        self.preload_progress_bar.setMaximumHeight(12)
        self.preload_progress_bar.setMaximumWidth(150)
        self.preload_progress_bar.setTextVisible(False)
        self.preload_progress_bar.setVisible(False)
        
        self.statusBar().addPermanentWidget(self.preload_status_label)
        self.statusBar().addPermanentWidget(self.preload_progress_bar)
        
        # 模拟器窗口列表，防止被垃圾回收
        self.simulator_windows = []
        self.sector_window = None
    
    def setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        refresh_action = QAction("刷新图表(&R)", self)
        refresh_action.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh_action.triggered.connect(self.refresh_chart)
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        self.volume_action = QAction("成交量", self, checkable=True, checked=True)
        self.volume_action.triggered.connect(lambda checked: self.volume_checkbox.setChecked(checked))
        view_menu.addAction(self.volume_action)
        
        self.macd_action = QAction("MACD", self, checkable=True, checked=True)
        self.macd_action.triggered.connect(lambda checked: self.macd_checkbox.setChecked(checked))
        view_menu.addAction(self.macd_action)
        
        self.kdj_action = QAction("KDJ", self, checkable=True, checked=False)
        self.kdj_action.triggered.connect(lambda checked: self.kdj_checkbox.setChecked(checked))
        view_menu.addAction(self.kdj_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        simulator_action = QAction("模拟训练(&S)", self)
        simulator_action.triggered.connect(self.open_simulator)
        tools_menu.addAction(simulator_action)
        
        sector_action = QAction("🔥 热门板块(&B)", self)
        sector_action.setShortcut("Ctrl+B")
        sector_action.triggered.connect(self.open_sector_window)
        tools_menu.addAction(sector_action)

        tools_menu.addSeparator()

        broker_action = QAction("交易(&T)", self)
        broker_action.triggered.connect(self.open_broker_account)
        tools_menu.addAction(broker_action)

        live_strategy_action = QAction("📊 实盘策略中枢(&D)", self)
        live_strategy_action.triggered.connect(self.open_live_strategy_center)
        tools_menu.addAction(live_strategy_action)

        tools_menu.addSeparator()

        notification_action = QAction("消息推送(&N)", self)
        notification_action.triggered.connect(self.open_notification_dialog)
        tools_menu.addAction(notification_action)
        
        scheduled_task_action = QAction("定时任务(&L)", self)
        scheduled_task_action.triggered.connect(self.open_scheduled_task_dialog)
        tools_menu.addAction(scheduled_task_action)
        
        tools_menu.addSeparator()
        
        auto_stop_loss_action = QAction("自动止损设置(&A)", self)
        auto_stop_loss_action.triggered.connect(self.open_auto_stop_loss_config)
        tools_menu.addAction(auto_stop_loss_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # 均线选择
        toolbar.addWidget(QLabel(" 均线: "))
        
        self.ma_combo = QComboBox()
        self.ma_combo.addItems([
            "MA5/10/20",
            "MA5/10/20/60",
            "MA10/20/60",
            "MA5/10/30/60",
        ])
        self.ma_combo.currentIndexChanged.connect(self.on_ma_changed)
        toolbar.addWidget(self.ma_combo)
        
        toolbar.addSeparator()
        
        # 上一只/下一只按钮
        prev_btn = QPushButton("◀ 上一只")
        prev_btn.clicked.connect(self._select_previous_item)
        toolbar.addWidget(prev_btn)
        
        next_btn = QPushButton("下一只 ▶")
        next_btn.clicked.connect(self._select_next_item)
        toolbar.addWidget(next_btn)
        
        toolbar.addSeparator()
        
        # 交易按钮
        trade_btn = QPushButton("💰 交易")
        trade_btn.clicked.connect(self.open_broker_account)
        trade_btn.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; padding: 6px 12px;")
        toolbar.addWidget(trade_btn)
        
        toolbar.addSeparator()
        
        # 智能体按钮
        agent_btn = QPushButton("🤖 智能体")
        agent_btn.clicked.connect(self.open_ai_agent)
        toolbar.addWidget(agent_btn)

        # 实盘策略中枢按钮
        ai_trade_btn = QPushButton("📊 实盘中枢")
        ai_trade_btn.setToolTip("实盘策略中枢 — AI实盘决策 / ETF轮动实盘")
        ai_trade_btn.clicked.connect(self.open_live_strategy_center)
        ai_trade_btn.setStyleSheet("background-color: #107c10; color: white; font-weight: bold; padding: 6px 12px;")
        toolbar.addWidget(ai_trade_btn)

        toolbar.addSeparator()
        
        # 热门板块按钮
        sector_btn = QPushButton("🔥 板块")
        sector_btn.setToolTip("热门板块 (Ctrl+B)")
        sector_btn.clicked.connect(self.open_sector_window)
        sector_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 6px 12px;")
        toolbar.addWidget(sector_btn)

        toolbar.addSeparator()

        # 同步数据按钮
        self.sync_btn = QPushButton("🔄 同步数据")
        self.sync_btn.setToolTip("全量同步数据 (Ctrl+Shift+R)")
        self.sync_btn.clicked.connect(self.start_full_data_sync)
        self.sync_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 6px 12px;")
        toolbar.addWidget(self.sync_btn)
    
    def setup_shortcuts(self):
        """设置快捷键"""
        # 上下键切换股票/ETF/自选
        QShortcut(Qt.Key.Key_Up, self, self._select_previous_item)
        QShortcut(Qt.Key.Key_Down, self, self._select_next_item)

        # F5 刷新
        QShortcut(Qt.Key.Key_F5, self, self.refresh_chart)

        # Ctrl+Shift+R 全量同步数据
        sync_shortcut = QShortcut(QKeySequence("Ctrl+Shift+R"), self)
        sync_shortcut.activated.connect(self.start_full_data_sync)
    
    def _select_previous_item(self):
        """根据当前Tab选中上一个项目"""
        current_tab = self.left_tabs.currentIndex()
        if current_tab == 0:  # 股票Tab
            self.stock_list_widget.select_previous()
        elif current_tab == 1:  # ETF Tab
            self.etf_list_widget.select_previous()
        elif current_tab == 2:  # 自选Tab
            self.watchlist_widget.select_previous()
        elif current_tab == 3:  # 指数Tab
            self.index_list_widget.select_previous()
    
    def _select_next_item(self):
        """根据当前Tab选中下一个项目"""
        current_tab = self.left_tabs.currentIndex()
        if current_tab == 0:  # 股票Tab
            self.stock_list_widget.select_next()
        elif current_tab == 1:  # ETF Tab
            self.etf_list_widget.select_next()
        elif current_tab == 2:  # 自选Tab
            self.watchlist_widget.select_next()
        elif current_tab == 3:  # 指数Tab
            self.index_list_widget.select_next()
    
    def load_stock_list(self):
        """加载股票列表"""
        self.statusBar().showMessage("正在加载股票列表...")
        QApplication.processEvents()
        
        # 加载股票代码列表
        self.stock_list = get_data_portal().list_symbols(
            asset_type="stock",
            data_dir=self.data_dir,
        )
        
        # 更新名称映射
        self.name_map = get_data_portal().get_name_map(
            asset_type="stock",
            stocklist_path=self.stocklist_path,
        )
        
        # 更新组件
        self.stock_list_widget.set_stock_list(self.stock_list, self.name_map)
        self.watchlist_panel.update_name_map(self.name_map)
        
        # 更新自选股分组下拉框
        self.stock_list_widget.update_group_combo()
        
        self.statusBar().showMessage(f"已加载 {len(self.stock_list)} 只股票")
        
        # 加载ETF列表
        self.load_etf_list()
        
        # 默认选中第一只
        if self.stock_list:
            first_code = self.stock_list[0]
            self.stock_list_widget.select_stock(first_code)
            self.on_stock_selected(first_code, self.name_map.get(first_code, ""))
        
        # 启动实时行情服务
        self._start_quote_service()

        # 启动时预订阅自选股实时行情
        QTimer.singleShot(3000, self._subscribe_watchlist_quotes_on_startup)
    
    def load_etf_list(self):
        """加载ETF列表"""
        # 加载ETF代码列表
        self.etf_list = get_data_portal().list_symbols(
            asset_type="etf",
            data_dir=self.data_dir,
        )
        
        # 加载ETF名称映射和分类
        self.etf_name_map = get_data_portal().get_name_map(asset_type="etf")
        self.etf_categories = get_data_portal().get_categories(asset_type="etf")
        
        # 更新ETF列表组件
        self.etf_list_widget.set_etf_data(
            self.etf_list, 
            self.etf_name_map, 
            self.etf_categories
        )
        
        # 更新自选分组下拉框
        self.etf_list_widget.update_group_combo()
        
        # 更新自选列表组件的名称映射
        self.watchlist_widget.set_name_maps(self.name_map, self.etf_name_map, self.etf_list)
        self.watchlist_widget.update_group_combo()
        
        # 更新面板的ETF名称映射
        self.watchlist_panel.update_etf_name_map(self.etf_name_map, self.etf_list)
        
        if self.etf_list:
            self.statusBar().showMessage(
                f"已加载 {len(self.stock_list)} 只股票, {len(self.etf_list)} 只ETF"
            )
    
    def start_data_preload(self):
        """启动数据预加载"""
        if not self.stock_list:
            return
        
        # 设置状态栏进度条
        self.preload_progress_bar.setRange(0, len(self.stock_list))
        self.preload_progress_bar.setValue(0)
        self.preload_progress_bar.setVisible(True)
        self.preload_status_label.setText("正在预加载数据...")
        self.preload_status_label.setVisible(True)
        
        # 创建并启动预加载线程
        self.preload_thread = DataPreloadThread(self.data_dir, self.stock_list)
        self.preload_thread.progress_updated.connect(self.on_preload_progress)
        self.preload_thread.finished_signal.connect(self.on_preload_finished)
        self.preload_thread.start()
    
    def on_preload_progress(self, current: int, total: int, code: str):
        """更新预加载进度"""
        self.preload_progress_bar.setValue(current)
        self.preload_status_label.setText(f"预加载中: {code} ({current}/{total}) ")
    
    def on_preload_finished(self, success: bool, loaded_count: int, message: str):
        """预加载完成回调"""
        self.preload_progress_bar.setVisible(False)
        
        if success:
            self.preload_status_label.setText(f"✓ 数据预加载完成 ({loaded_count}只) ")
            # 5秒后隐藏完成提示
            QTimer.singleShot(5000, lambda: self.preload_status_label.setVisible(False))
        else:
            self.preload_status_label.setText(f"⚠ 预加载失败 ")
            self.statusBar().showMessage(f"⚠ {message}")
        
        # 刷新当前图表（使用缓存数据）
        if success and self.current_code:
            self.load_and_display_chart()
    
    def on_stock_selected(self, code: str, name: str):
        """处理股票选择"""
        self.current_code = code
        self.current_name = name
        self.current_view = "stock"
        
        self.load_and_display_chart()
        self.refresh_agent_context()
        
        # Update timeshare widget if it's visible
        current_tab_index = self.right_tabs.currentIndex()
        if current_tab_index == 1:  # Timeshare tab
            self.load_timeshare_data()
    
    def on_etf_selected(self, code: str, name: str):
        """处理ETF选择"""
        self.current_etf_code = code
        self.current_etf_name = name
        self.current_view = "etf"
        
        # Load K-line chart first
        self.load_and_display_etf_chart()
        self.refresh_agent_context()
        
        # If currently on timeshare tab, also load timeshare data
        if self.right_tabs.currentIndex() == 1:
            self.load_etf_timeshare_data()
    
    def on_index_selected(self, code: str, name: str):
        """处理指数选择"""
        self.current_index_code = code
        self.current_index_name = name
        self.current_view = "index"
        
        # Load K-line chart
        self.load_and_display_index_chart()
        self.refresh_agent_context()
    
    def on_left_tab_changed(self, index: int):
        """处理左侧股票/ETF/自选/指数Tab切换"""
        # 确保watchlist_panel已初始化
        if not hasattr(self, 'watchlist_panel'):
            return
            
        if index == 0:  # 股票Tab
            self.current_view = "stock"
            # 更新面板显示股票分组
            current_group = self.stock_list_widget.current_group
            is_group = bool(current_group) and not current_group.startswith("全部")
            self.watchlist_panel.set_group_mode(is_group)
            if is_group:
                stocks = self.stock_list_widget.filtered_list
                self.watchlist_panel.set_stocks(stocks)
            # 如果有选中的股票，刷新显示
            if self.current_code:
                self.load_and_display_chart()
                # If currently on timeshare tab, also load timeshare data
                if self.right_tabs.currentIndex() == 1:
                    self.load_timeshare_data()
        elif index == 1:  # ETF Tab
            self.current_view = "etf"
            # ETF Tab时关闭面板的分组模式（因为ETF分组是类别，不是自选）
            self.watchlist_panel.set_group_mode(False)
            # 如果有选中的ETF，刷新显示
            if self.current_etf_code:
                self.load_and_display_etf_chart()
                # If currently on timeshare tab, also load timeshare data
                if self.right_tabs.currentIndex() == 1:
                    self.load_etf_timeshare_data()
            elif self.etf_list:
                # 选中第一只ETF
                first_etf = self.etf_list[0]
                self.etf_list_widget.select_etf(first_etf)
                self.on_etf_selected(first_etf, self.etf_name_map.get(first_etf, ""))
        elif index == 2:  # 自选Tab
            # 自选Tab: 更新面板显示当前分组
            current_group = self.watchlist_widget.get_current_group()
            is_group = bool(current_group)
            self.watchlist_panel.set_group_mode(is_group)
            if is_group:
                stocks = self.watchlist_widget.filtered_list
                self.watchlist_panel.set_stocks(stocks)
        elif index == 3:  # 指数Tab
            self.current_view = "index"
            self.watchlist_panel.set_group_mode(False)
            # 如果有选中的指数，刷新显示
            if hasattr(self, 'current_index_code') and self.current_index_code:
                self.load_and_display_index_chart()
            elif hasattr(self, 'index_list_widget'):
                # 选中第一个指数
                indices = get_data_portal().list_assets(asset_type="index", include_status=False)
                if indices:
                    first_index = indices[0]
                    self.index_list_widget.select_index(first_index['code'])
                    self.on_index_selected(first_index['code'], first_index['name'])

        if self.right_tabs.currentIndex() == 2:
            self.watchlist_panel.set_auto_refresh_active(index == 2)
        self.refresh_agent_context()

    def on_watchlist_item_selected(self, code: str, name: str, is_etf: bool):
        """处理自选列表项目选中（支持股票和ETF混合）"""
        if is_etf:
            # ETF处理
            self.current_etf_code = code
            self.current_etf_name = name
            self.current_view = "etf"
            self.load_and_display_etf_chart()
            self.refresh_agent_context()
            # If currently on timeshare tab, also load timeshare data
            if self.right_tabs.currentIndex() == 1:
                self.load_etf_timeshare_data()
        else:
            # 股票处理
            self.current_code = code
            self.current_name = name
            self.current_view = "stock"
            self.load_and_display_chart()
            self.refresh_agent_context()
            # If currently on timeshare tab, also load timeshare data
            if self.right_tabs.currentIndex() == 1:
                self.load_timeshare_data()

    def on_sector_stock_selected(self, code: str, name: str):
        """处理热门板块中成分股被选中"""
        # 切换到股票 Tab
        self.left_tabs.setCurrentIndex(0)  # 股票 Tab
        
        # 选中股票
        self.stock_list_widget.select_stock(code)
        
        # 获取股票名称
        if not name:
            name = self.name_map.get(code, "")
        
        # 触发股票选中逻辑
        self.on_stock_selected(code, name)
        
        self.statusBar().showMessage(f"📈 已跳转至 {code} {name}")

    def on_sector_selected(self, sector_name: str):
        """处理板块被选中"""
        # 可以在这里添加板块选中后的逻辑
        # 例如：显示板块相关信息、记录等
        display_name = sector_name.replace("申万", "")
        self.statusBar().showMessage(f"🔥 已选中板块: {display_name}")

    def on_panel_stock_selected(self, code: str, name: str, is_etf: bool = False):
        """处理面板中的股票/ETF选择"""
        # 切换到 K线图 Tab
        self.right_tabs.setCurrentIndex(0)
        
        # 检查当前是否在自选Tab，如果是则保持在自选Tab
        current_left_tab = self.left_tabs.currentIndex()
        is_watchlist_tab = (current_left_tab == 2)
        
        if is_etf:
            if not is_watchlist_tab:
                # 非自选Tab: 切换到ETF Tab并选择
                self.left_tabs.setCurrentIndex(1)  # ETF Tab
                self.etf_list_widget.select_etf(code)
            self.on_etf_selected(code, name)
        else:
            if not is_watchlist_tab:
                # 非自选Tab: 切换到股票Tab并选择
                self.left_tabs.setCurrentIndex(0)  # Stock Tab
                self.stock_list_widget.select_stock(code)
            self.on_stock_selected(code, name)

    def on_right_tab_changed(self, index: int):
        """Handle right panel tab changes"""
        if index == 1:  # Timeshare tab
            # Load timeshare data based on current view (stock or ETF)
            if self.current_view == "etf" and self.current_etf_code:
                self.load_etf_timeshare_data()
            else:
                self.load_timeshare_data()
        elif index == 0:  # K-line tab
            # Stop timeshare auto-refresh when switching away
            self.timeshare_widget.stop_auto_refresh()
        elif index == 2:  # Panel tab
            is_watchlist_tab = self.left_tabs.currentIndex() == 2
            self.watchlist_panel.set_auto_refresh_active(is_watchlist_tab)
        else:
            if self.left_tabs.currentIndex() == 2:
                self.watchlist_panel.set_auto_refresh_active(False)

    def load_timeshare_data(self):
        """Load timeshare data for current stock"""
        if not self.current_code:
            return
        
        # Get today's date
        import datetime
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_date = datetime.date.today()
        
        # Get previous close from K-line data
        prev_close = None
        if self.kline_widget.data is not None and not self.kline_widget.data.empty:
            df = self.kline_widget.data
            
            # Get the last record's date
            last_date = df.iloc[-1]['date']
            if hasattr(last_date, 'date'):
                last_date = last_date.date()
            elif isinstance(last_date, str):
                last_date = datetime.datetime.strptime(last_date[:10], "%Y-%m-%d").date()
            
            # If the last K-line data is today, use second last as prev_close
            # Otherwise, use the last one as prev_close (it's yesterday or earlier)
            if last_date == today_date:
                if len(df) >= 2:
                    prev_close = df.iloc[-2]['close']
                else:
                    prev_close = df.iloc[-1]['open']
            else:
                prev_close = df.iloc[-1]['close']
        
        self.statusBar().showMessage(f"正在加载 {self.current_code} {self.current_name} 分时图...")
        
        self.timeshare_widget.load_data(
            code=self.current_code,
            date_str=today_str,
            data_dir=self.data_dir,
            prev_close=prev_close
        )
    
    def load_etf_timeshare_data(self):
        """Load timeshare data for current ETF"""
        if not self.current_etf_code:
            return
        
        # Get today's date
        import datetime
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_date = datetime.date.today()
        
        # Get previous close from K-line data
        prev_close = None
        if self.kline_widget.data is not None and not self.kline_widget.data.empty:
            df = self.kline_widget.data
            
            # Get the last record's date
            last_date = df.iloc[-1]['date']
            if hasattr(last_date, 'date'):
                last_date = last_date.date()
            elif isinstance(last_date, str):
                last_date = datetime.datetime.strptime(last_date[:10], "%Y-%m-%d").date()
            
            # If the last K-line data is today, use second last as prev_close
            # Otherwise, use the last one as prev_close
            if last_date == today_date:
                if len(df) >= 2:
                    prev_close = df.iloc[-2]['close']
                else:
                    prev_close = df.iloc[-1]['open']
            else:
                prev_close = df.iloc[-1]['close']
        
        self.statusBar().showMessage(f"正在加载 {self.current_etf_code} {self.current_etf_name} 分时图...")
        
        self.timeshare_widget.load_data(
            code=self.current_etf_code,
            date_str=today_str,
            data_dir=self.data_dir,
            prev_close=prev_close
        )

    def on_timeshare_refresh_status_changed(self, is_refreshing: bool, message: str):
        """Handle timeshare refresh status changes"""
        if is_refreshing:
            self.statusBar().showMessage(f"分时图: {message}")

    def on_group_changed_for_panel(self, group_name):
        """同步分组状态到面板（实时行情由面板 checkbox 控制）"""
        is_group = bool(group_name)  # 非空字符串表示选中了某个分组
        self.watchlist_panel.set_group_mode(is_group)
        
        # 如果是分组模式，立即更新一次数据
        if is_group:
            stocks = self.stock_list_widget.filtered_list
            self.watchlist_panel.set_stocks(stocks)
            # 注意：实时行情订阅不再自动开启，由用户点击面板上的 "📡 实时行情" checkbox 控制
        self.refresh_agent_context()

    def on_display_list_changed(self, stocks):
        """当左侧列表过滤或搜索变化时，同步到面板"""
        self.watchlist_panel.set_stocks(stocks)
        self.refresh_agent_context()
    
    def on_watchlist_group_changed_for_panel(self, group_name):
        """处理自选Tab的分组切换，同步到面板"""
        # Only respond when the watchlist tab is active
        if self.left_tabs.currentIndex() != 2:
            return
        
        is_group = bool(group_name)
        self.watchlist_panel.set_group_mode(is_group)
        
        if is_group:
            stocks = self.watchlist_widget.filtered_list
            self.watchlist_panel.set_stocks(stocks)
        self.refresh_agent_context()
    
    def on_watchlist_display_list_changed(self, stocks):
        """处理自选Tab的列表变化，同步到面板"""
        # Only respond when the watchlist tab is active
        if self.left_tabs.currentIndex() != 2:
            return
        self.watchlist_panel.set_stocks(stocks)
        self.refresh_agent_context()
    
    # ========== 实时行情服务集成 ==========
    
    def _start_quote_service(self):
        self.realtime_controller.initialize()
    
    def _update_quote_subscription(self, group_name: str, stocks: list):
        """
        更新实时行情订阅（仅针对自选分组）
        
        Args:
            group_name: 分组名称，空字符串表示非分组模式
            stocks: 要订阅的股票列表
        """
        if not hasattr(self, 'quote_service') or self.quote_service is None:
            return
        
        if not self.quote_service.is_available:
            return
        
        try:
            if group_name and stocks:
                owner_id = f"main:group:{group_name}"
                self.quote_service.replace_subscription(owner_id, stocks, start_service=True)
                self.statusBar().showMessage(f"📡 已订阅 {group_name} 分组 ({len(stocks)} 只)", 3000)
            else:
                # 只清理本 owner，不停止全局服务
                for owner_id in self.quote_service.list_owner_ids():
                    if owner_id.startswith("main:group:"):
                        self.quote_service.clear_owner_subscription(owner_id)
                self.statusBar().showMessage("📡 分组实时订阅已清理", 2000)
                    
        except Exception as e:
            pass  # 静默处理
    
    def _on_quote_updated(self, quote_data: QuoteData):
        self.realtime_controller._on_quote_updated(quote_data)
    
    def _on_quote_status_changed(self, connected: bool, message: str):
        self.realtime_controller._on_quote_status_changed(connected, message)

    def _init_conditional_order_monitor(self):
        self.trading_bridge.initialize()
    
    def _init_auto_stop_loss_service(self, conditional_service):
        self.trading_bridge.initialize()

    # ========== 全量数据同步功能 ==========

    def _prepare_runtime_for_full_sync(self):
        self.sync_controller.prepare_runtime_for_full_sync()

    def _restore_runtime_after_full_sync(self):
        self.sync_controller.restore_runtime_after_full_sync()

    def start_full_data_sync(self):
        self.sync_controller.start_full_data_sync()

    def _on_sync_progress(self, phase_name: str, current: int, total: int):
        self.sync_controller.on_sync_progress(phase_name, current, total)

    def _on_sync_log(self, message: str):
        self.sync_controller.on_sync_log(message)

    def _on_sync_finished(self, success: bool, message: str):
        self.sync_controller.on_sync_finished(success, message)

    def _refresh_all_caches(self):
        self.sync_controller.refresh_all_caches()

    # ========== 收盘提醒功能 ==========

    def _check_market_close_reminder(self):
        self.sync_controller.check_market_close_reminder()

    def _is_data_up_to_date(self) -> bool:
        return self.sync_controller.is_data_up_to_date()

    def _get_last_data_date(self) -> str:
        return self.sync_controller.get_last_data_date()

    def _show_market_close_reminder(self, last_data_date: str):
        self.sync_controller.show_market_close_reminder(last_data_date)

    def _on_reminder_sync_now(self):
        self.sync_controller.on_reminder_sync_now()

    def _on_reminder_later(self):
        self.sync_controller.on_reminder_later()

    def _on_reminder_ignore(self):
        self.sync_controller.on_reminder_ignore()

    # ========== 启动时预订阅自选股 ==========

    def _subscribe_watchlist_quotes_on_startup(self):
        self.realtime_controller.subscribe_watchlist_on_startup()

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        # 0. 停止收盘提醒定时器
        if hasattr(self, '_market_close_reminder_timer'):
            self._market_close_reminder_timer.stop()

        self.sync_controller.shutdown()

        # 1. 停止定时任务检查
        if hasattr(self, 'scheduler_manager'):
            self.scheduler_manager.stop()

        # 2. 停止分时图自动刷新
        if hasattr(self, 'timeshare_widget'):
            self.timeshare_widget.stop_auto_refresh()

        # 3. 停止K线图实时行情订阅
        if hasattr(self, 'kline_widget'):
            self.kline_widget.stop_realtime()

        self.realtime_controller.shutdown()

        # 5. 关闭独立板块窗口
        try:
            if self.sector_window:
                self.sector_window.close()
                self.sector_window = None
        except Exception:
            pass

        # 6. 停止数据更新线程
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()
            self.update_thread.wait(2000) # 最多等待2秒

        self.trading_bridge.shutdown()

        # 7. 停止数据预加载线程
        if self.preload_thread and self.preload_thread.isRunning():
            # 预加载线程通常没那么紧急，但也应该停止
            pass

        # 8. 停止所有模拟器窗口
        for window in self.simulator_windows:
            try:
                window.close()
            except:
                pass
                
        event.accept()

    def load_and_display_chart(self):
        """加载并显示K线图"""
        if not self.current_code:
            return
        
        self.statusBar().showMessage(f"正在加载 {self.current_code} {self.current_name}...")
        QApplication.processEvents()
        
        # 获取日期范围
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        
        # 加载数据
        df = get_data_portal().get_daily_bars(
            self.current_code,
            data_dir=self.data_dir,
            adjust="qfq",
            start=start_date,
            end=end_date,
            asset_type="stock",
        )
        
        if df is None or df.empty:
            self.statusBar().showMessage(f"未找到 {self.current_code} 的数据")
            QMessageBox.warning(self, "数据错误", f"未找到 {self.current_code} 的数据")
            return
        
        # 添加技术指标
        df = attach_all_indicators(
            df,
            ma_windows=self.ma_windows,
            include_macd=self.macd_checkbox.isChecked(),
            include_kdj=self.kdj_checkbox.isChecked(),
            include_bbi=False,
            vol_ma_window=5
        )
        
        # 切换股票时停止实时行情
        if self.kline_widget.is_realtime_enabled:
            self.kline_widget.stop_realtime()
        
        # 更新K线图
        self.kline_widget.set_indicators(
            show_volume=self.volume_checkbox.isChecked(),
            show_macd=self.macd_checkbox.isChecked(),
            show_kdj=self.kdj_checkbox.isChecked()
        )
        self.kline_widget.set_ma_windows(self.ma_windows)
        self.kline_widget.set_data(df, self.current_code, self.current_name)
        
        # 更新窗口标题
        self.setWindowTitle(f"来财 - {self.current_code} {self.current_name}")
        
        self.statusBar().showMessage(
            f"{self.current_code} {self.current_name} | "
            f"数据范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')} | "
            f"共 {len(df)} 根K线"
        )
        
        # 根据交易时间自动开启实时行情
        last_data_date = df['date'].max().date()
        if should_update_realtime_kline(last_data_date):
            if self.kline_widget.start_realtime():
                self.statusBar().showMessage(f"📡 已开启 {self.current_code} 实时行情", 3000)
    
    def load_and_display_etf_chart(self):
        """加载并显示ETF K线图"""
        if not self.current_etf_code:
            return
        
        self.statusBar().showMessage(f"正在加载 {self.current_etf_code} {self.current_etf_name}...")
        QApplication.processEvents()
        
        # 获取日期范围
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        
        # 加载ETF数据
        df = get_data_portal().get_daily_bars(
            self.current_etf_code,
            data_dir=self.data_dir,
            start=start_date,
            end=end_date,
            asset_type="etf",
        )
        
        if df is None or df.empty:
            self.statusBar().showMessage(f"未找到 {self.current_etf_code} 的数据")
            QMessageBox.warning(self, "数据错误", f"未找到 {self.current_etf_code} 的ETF数据，请先更新ETF数据")
            return
        
        # 添加技术指标
        df = attach_all_indicators(
            df,
            ma_windows=self.ma_windows,
            include_macd=self.macd_checkbox.isChecked(),
            include_kdj=self.kdj_checkbox.isChecked(),
            include_bbi=False,
            vol_ma_window=5
        )
        
        # 切换股票时停止实时行情
        if self.kline_widget.is_realtime_enabled:
            self.kline_widget.stop_realtime()
        
        # 更新K线图
        self.kline_widget.set_indicators(
            show_volume=self.volume_checkbox.isChecked(),
            show_macd=self.macd_checkbox.isChecked(),
            show_kdj=self.kdj_checkbox.isChecked()
        )
        self.kline_widget.set_ma_windows(self.ma_windows)
        self.kline_widget.set_data(df, self.current_etf_code, self.current_etf_name)
        
        # 更新窗口标题
        self.setWindowTitle(f"来财 - ETF {self.current_etf_code} {self.current_etf_name}")
        
        self.statusBar().showMessage(
            f"ETF {self.current_etf_code} {self.current_etf_name} | "
            f"数据范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')} | "
            f"共 {len(df)} 根K线"
        )
        
        # 根据交易时间自动开启实时行情
        last_data_date = df['date'].max().date()
        if should_update_realtime_kline(last_data_date):
            if self.kline_widget.start_realtime():
                self.statusBar().showMessage(f"📡 已开启 {self.current_etf_code} 实时行情", 3000)
    
    def load_and_display_index_chart(self):
        """加载并显示指数K线图"""
        if not hasattr(self, 'current_index_code') or not self.current_index_code:
            return
        
        self.statusBar().showMessage(f"正在加载 {self.current_index_code} {self.current_index_name}...")
        QApplication.processEvents()
        
        # 获取日期范围
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        
        # 加载指数数据
        df = get_data_portal().get_daily_bars(
            self.current_index_code,
            data_dir=self.data_dir,
            start=start_date,
            end=end_date,
            asset_type="index",
            use_cache=False,
        )
        
        if df is None or df.empty:
            self.statusBar().showMessage(f"未找到 {self.current_index_code} 的数据，请先更新指数数据")
            QMessageBox.warning(self, "数据错误", 
                f"未找到 {self.current_index_code} ({self.current_index_name}) 的数据\n"
                "请在指数列表中点击'更新'按钮下载指数数据")
            return
        
        # 添加技术指标
        df = attach_all_indicators(
            df,
            ma_windows=self.ma_windows,
            include_macd=self.macd_checkbox.isChecked(),
            include_kdj=self.kdj_checkbox.isChecked(),
            include_bbi=False,
            vol_ma_window=5
        )
        
        # 切换时停止实时行情
        if self.kline_widget.is_realtime_enabled:
            self.kline_widget.stop_realtime()
        
        # 更新K线图
        self.kline_widget.set_indicators(
            show_volume=self.volume_checkbox.isChecked(),
            show_macd=self.macd_checkbox.isChecked(),
            show_kdj=self.kdj_checkbox.isChecked()
        )
        self.kline_widget.set_ma_windows(self.ma_windows)
        self.kline_widget.set_data(df, self.current_index_code, self.current_index_name, is_index=True)
        
        # 更新窗口标题
        self.setWindowTitle(f"来财 - 指数 {self.current_index_code} {self.current_index_name}")
        
        self.statusBar().showMessage(
            f"指数 {self.current_index_code} {self.current_index_name} | "
            f"数据范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')} | "
            f"共 {len(df)} 根K线"
        )
        
        # 根据交易时间自动开启实时行情
        last_data_date = df['date'].max().date()
        if should_update_realtime_kline(last_data_date):
            if self.kline_widget.start_realtime():
                self.statusBar().showMessage(f"📡 已开启 {self.current_index_code} {self.current_index_name} 实时指数行情", 3000)
    
    def on_indicator_changed(self, state):
        """处理指标复选框变化"""
        # 同步菜单状态
        self.volume_action.setChecked(self.volume_checkbox.isChecked())
        self.macd_action.setChecked(self.macd_checkbox.isChecked())
        self.kdj_action.setChecked(self.kdj_checkbox.isChecked())
        
        # 重新加载图表（根据当前视图）
        if self.current_view == "etf":
            self.load_and_display_etf_chart()
        elif self.current_view == "index":
            self.load_and_display_index_chart()
        else:
            self.load_and_display_chart()
    
    def on_ma_changed(self, index):
        """处理均线选择变化"""
        ma_options = [
            [5, 10, 20],
            [5, 10, 20, 60],
            [10, 20, 60],
            [5, 10, 30, 60],
        ]
        
        if 0 <= index < len(ma_options):
            self.ma_windows = ma_options[index]
            # 根据当前视图刷新
            if self.current_view == "etf":
                self.load_and_display_etf_chart()
            else:
                self.load_and_display_chart()
    
    def refresh_chart(self):
        """刷新图表"""
        if self.current_view == "etf":
            self.load_and_display_etf_chart()
        elif self.current_view == "index":
            self.load_and_display_index_chart()
        else:
            self.load_and_display_chart()
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "来财\n\n"
            "基于 PyQt6 和 pyqtgraph 开发\n\n"
            "功能特性:\n"
            "• K线图显示\n"
            "• 均线指标 (MA5/MA10/MA20等)\n"
            "• MACD 指标\n"
            "• KDJ 指标\n"
            "• 成交量显示\n"
            "• 十字光标\n"
            "• 股票搜索\n"
        )

    def refresh_all_today_kline(self):
        """一键刷新今日所有股票的实时日线"""
        # 获取配置
        config = self.scheduler_manager.config
        data_source = config.get("data_source", "xtquant")
        token = config.get("tushare_token", os.environ.get("TUSHARE_TOKEN", ""))
        
        # 显示更新对话框并自动开始
        self.update_dialog = UpdateDialog(self, token)
        self.update_dialog.setWindowTitle("同步今日实时数据")
        self.update_dialog.start_update.connect(self.start_data_update)
        self.update_dialog.stop_update.connect(self.stop_data_update)
        
        # 切换到正确的数据源
        index = self.update_dialog.source_combo.findData(data_source)
        if index >= 0:
            self.update_dialog.source_combo.setCurrentIndex(index)
            
        # 设置为增量更新
        self.update_dialog.full_update_cb.setChecked(False)
        
        # 显示对话框
        self.update_dialog.show()
        
        # 自动开始
        self.update_dialog.on_start_clicked()

    def start_data_update(self, token, full_update, exclude_boards, start_date="", 
                          data_source="tushare", period="1d"):
        """开始数据更新
        
        Args:
            token: Tushare token（仅 tushare 数据源需要）
            full_update: 是否全量更新
            exclude_boards: 要排除的板块列表
            start_date: 起始日期
            data_source: 数据源 ("tushare" 或 "xtquant")
            period: K线周期 ("1d", "1m", "5m", "15m", "30m", "60m")
        """
        if self.update_thread and self.update_thread.isRunning():
            return

        self._updating_codes = []  # 批量更新时清空，后续需要完全重新预加载
        
        self.update_thread = DataUpdateThread(
            self.data_dir,
            self.stocklist_path,
            token,
            full_update,
            exclude_boards,
            start_date=start_date if start_date else None,
            data_source=data_source,
            period=period
        )
        self.update_thread.progress_updated.connect(self.update_dialog.update_progress)
        self.update_thread.log_message.connect(self.update_dialog.append_log)
        self.update_thread.result_signal.connect(self._on_update_result)
        self.update_thread.finished_signal.connect(self.on_update_finished)
        self.update_thread.start()

    def stop_data_update(self):
        """停止数据更新"""
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()

    def _on_update_result(self, result: object):
        if isinstance(result, DataUpdateResult):
            self._last_update_result = result

    def on_update_finished(self, success, message):
        """数据更新完成"""
        result = self._last_update_result
        display_message = result.to_ui_message() if isinstance(result, DataUpdateResult) else message
        if self.update_dialog:
            self.update_dialog.on_finished(success, display_message)
        
        if success:
            # 重新加载股票列表
            self.load_stock_list()
            
            # 刷新缓存
            cache = get_stock_cache()
            if cache.is_loaded():
                if self._updating_codes:
                    # 只更新单只或少量股票，只刷新这些股票的缓存
                    for code in self._updating_codes:
                        cache.reload_stock(code, self.data_dir)
                    self.statusBar().showMessage(f"✓ 已更新 {len(self._updating_codes)} 只股票的缓存")
                    self.load_and_display_chart()
                else:
                    # 批量更新，重新预加载所有数据
                    self.start_data_preload()
            else:
                self.load_and_display_chart()
            
            self._updating_codes = []  # 清空更新记录

    def show_stock_list_context_menu(self, position):
        """显示股票列表右键菜单"""
        item = self.stock_list_widget.list_widget.itemAt(position)
        if not item:
            return
            
        # 确保选中了该项
        self.stock_list_widget.list_widget.setCurrentItem(item)
        code = item.data(Qt.ItemDataRole.UserRole)
        name = self.name_map.get(code, "")
        
        menu = QMenu(self)
        
        # 交易下单
        trade_action = menu.addAction(f"💰 去交易 {name}({code})")
        trade_action.triggered.connect(lambda checked, c=code: self.open_broker_account(c))
        ai_decision_action = menu.addAction(f"📊 AI实盘决策 {name}({code})")
        ai_decision_action.triggered.connect(lambda checked, c=code, n=name: self._open_ai_decision_for(c, n))
        
        menu.addSeparator()
        
        # 筹码分布
        chip_action = menu.addAction("📊 筹码分布")
        chip_action.triggered.connect(lambda checked, c=code, n=name: self.show_chip_distribution(c, n))

        # 模拟训练
        simulate_action = menu.addAction("模拟训练")
        simulate_action.triggered.connect(self.open_simulator)
        menu.addSeparator()
        
        current_group = self.stock_list_widget.get_current_group()
        
        if current_group:
            # In a watchlist group - show remove option
            remove_action = menu.addAction("从当前分组移除")
            remove_action.triggered.connect(self.remove_current_from_group)
            menu.addSeparator()
        
        # 更新数据菜单
        update_menu = menu.addMenu("更新数据")
        
        refresh_today_action = update_menu.addAction("刷新今日当前时刻K线")
        refresh_today_action.triggered.connect(self.refresh_all_today_kline)
        
        update_menu.addSeparator()
        
        inc_update_action = update_menu.addAction("增量更新 (补齐历史数据)")
        inc_update_action.triggered.connect(self.update_current_stock_incremental)
        
        full_update_action = update_menu.addAction("重新拉取 (指定日期)...")
        full_update_action.triggered.connect(self.update_current_stock_full)
        
        menu.addSeparator()

        # Add to watchlist submenu
        add_to_fav_menu = menu.addMenu("添加到自选股")
        
        groups = self.watchlist_manager.get_all_groups()
        if not groups:
            add_to_fav_menu.addAction("无分组").setEnabled(False)
        else:
            for group in groups:
                action = add_to_fav_menu.addAction(group)
                action.triggered.connect(lambda checked, g=group: self.add_current_to_watchlist(g))
                
        new_group_action = add_to_fav_menu.addAction("新建分组...")
        new_group_action.triggered.connect(self.create_group_and_add)
        
        menu.exec(self.stock_list_widget.list_widget.mapToGlobal(position))

    def show_etf_list_context_menu(self, position):
        """显示ETF列表右键菜单"""
        item = self.etf_list_widget.list_widget.itemAt(position)
        if not item:
            return
            
        # 确保选中了该项
        self.etf_list_widget.list_widget.setCurrentItem(item)
        code = item.data(Qt.ItemDataRole.UserRole)
        name = self.etf_name_map.get(code, "")
        
        menu = QMenu(self)
        
        # 交易下单
        trade_action = menu.addAction(f"💰 去交易 {name}({code})")
        trade_action.triggered.connect(lambda checked, c=code: self.open_broker_account(c))
        ai_decision_action = menu.addAction(f"📊 AI实盘决策 {name}({code})")
        ai_decision_action.triggered.connect(lambda checked, c=code, n=name: self._open_ai_decision_for(c, n))
        
        menu.addSeparator()
        
        # 筹码分布
        chip_action = menu.addAction("📊 筹码分布")
        chip_action.triggered.connect(lambda checked, c=code, n=name: self.show_chip_distribution(c, n))
        
        menu.addSeparator()
        
        # 如果当前在自选分组模式，显示移除选项
        current_group = self.etf_list_widget.get_current_group()
        if current_group:
            remove_action = menu.addAction(f"从 '{current_group}' 移除")
            remove_action.triggered.connect(self.remove_current_etf_from_group)
            menu.addSeparator()
        
        # 添加到自选分组子菜单
        add_to_fav_menu = menu.addMenu("添加到自选分组")
        
        groups = self.watchlist_manager.get_all_groups()
        if not groups:
            no_group_action = add_to_fav_menu.addAction("无分组")
            no_group_action.setEnabled(False)
        else:
            for group in groups:
                action = add_to_fav_menu.addAction(f"⭐ {group}")
                action.triggered.connect(lambda checked, g=group: self.add_current_etf_to_watchlist(g))
                
        add_to_fav_menu.addSeparator()
        new_group_action = add_to_fav_menu.addAction("✚ 新建分组...")
        new_group_action.triggered.connect(self.create_group_and_add_etf)
        
        menu.exec(self.etf_list_widget.list_widget.mapToGlobal(position))

    def show_watchlist_context_menu(self, position):
        """显示自选列表右键菜单"""
        item = self.watchlist_widget.list_widget.itemAt(position)
        if not item:
            return
            
        # 确保选中了该项
        self.watchlist_widget.list_widget.setCurrentItem(item)
        code = item.data(Qt.ItemDataRole.UserRole)
        is_etf = item.data(Qt.ItemDataRole.UserRole + 1)
        name = self.watchlist_widget._get_name(code)
        
        menu = QMenu(self)
        
        # 交易下单
        trade_action = menu.addAction(f"💰 去交易 {name}({code})")
        trade_action.triggered.connect(lambda checked, c=code: self.open_broker_account(c))
        ai_decision_action = menu.addAction(f"📊 AI实盘决策 {name}({code})")
        ai_decision_action.triggered.connect(lambda checked, c=code, n=name: self._open_ai_decision_for(c, n))
        
        menu.addSeparator()
        
        # 筹码分布
        chip_action = menu.addAction("📊 筹码分布")
        chip_action.triggered.connect(lambda checked, c=code, n=name: self.show_chip_distribution(c, n))
        
        menu.addSeparator()
        
        # 如果当前选中了分组，显示移除选项
        current_group = self.watchlist_widget.get_current_group()
        if current_group:
            # Check if protected
            if not self.watchlist_manager.is_protected_group(current_group):
                remove_action = menu.addAction(f"从 '{current_group}' 移除")
                remove_action.triggered.connect(lambda checked, c=code: self.remove_from_watchlist_group(c))
                menu.addSeparator()
        
        # 添加到自选分组子菜单
        add_to_fav_menu = menu.addMenu("添加到其他分组")
        
        groups = self.watchlist_manager.get_all_groups()
        if not groups:
            no_group_action = add_to_fav_menu.addAction("无分组")
            no_group_action.setEnabled(False)
        else:
            for group in groups:
                if group != current_group:  # 排除当前分组
                    action = add_to_fav_menu.addAction(f"⭐ {group}")
                    action.triggered.connect(lambda checked, g=group, c=code: self.add_to_watchlist_group(c, g))
                    
        add_to_fav_menu.addSeparator()
        new_group_action = add_to_fav_menu.addAction("✚ 新建分组...")
        new_group_action.triggered.connect(lambda checked, c=code: self.create_group_and_add_from_watchlist(c))
        
        menu.exec(self.watchlist_widget.list_widget.mapToGlobal(position))

    def add_to_watchlist_group(self, code: str, group_name: str):
        """添加代码到指定自选分组"""
        success, msg = self.watchlist_manager.add_to_group(group_name, code)
        if success:
            self.statusBar().showMessage(msg)
            # Refresh watchlist widget
            self.watchlist_widget.update_group_combo()
        else:
            QMessageBox.warning(self, "提示", msg)

    def remove_from_watchlist_group(self, code: str):
        """从当前自选分组移除"""
        current_group = self.watchlist_widget.get_current_group()
        if not current_group:
            return
            
        reply = QMessageBox.question(
            self, "确认移除", 
            f"确定要从 '{current_group}' 移除 {code} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, msg = self.watchlist_widget.remove_from_current_group(code)
            if success:
                self.statusBar().showMessage(msg)
                # Also update stock_list_widget and etf_list_widget combos
                self.stock_list_widget.update_group_combo()
                self.etf_list_widget.update_group_combo()
            else:
                QMessageBox.warning(self, "错误", msg)

    def create_group_and_add_from_watchlist(self, code: str):
        """创建新分组并添加代码"""
        name, ok = QInputDialog.getText(self, "新建分组", "请输入分组名称:")
        if ok and name:
            success, msg = self.watchlist_manager.create_group(name)
            if success:
                # Add code to new group
                self.watchlist_manager.add_to_group(name, code)
                self.statusBar().showMessage(f"已创建分组 '{name}' 并添加 {code}")
                # Update all group combos
                self.stock_list_widget.update_group_combo()
                self.etf_list_widget.update_group_combo()
                self.watchlist_widget.update_group_combo()
            else:
                QMessageBox.warning(self, "错误", msg)

    def add_current_to_watchlist(self, group_name):
        """添加当前选中的股票到自选股"""
        code = self.stock_list_widget.get_selected_stock()
        if not code:
            return
            
        success, msg = self.watchlist_manager.add_to_group(group_name, code)
        if success:
            self.statusBar().showMessage(msg)
            # If currently showing this group, refresh it
            if self.stock_list_widget.get_current_group() == group_name:
                self.stock_list_widget.on_group_combo_changed(
                    self.stock_list_widget.group_combo.currentIndex()
                )
            # 同步更新自选列表组件
            if self.watchlist_widget.get_current_group() == group_name:
                self.watchlist_widget.on_group_combo_changed(
                    self.watchlist_widget.group_combo.currentIndex()
                )
        else:
            QMessageBox.warning(self, "提示", msg)
    def remove_current_from_group(self):
        """从当前分组移除选中的股票"""
        code = self.stock_list_widget.get_selected_stock()
        if not code:
            return
            
        group_name = self.stock_list_widget.get_current_group()
        if not group_name:
            return
            
        reply = QMessageBox.question(
            self, "确认移除", 
            f"确定要从 '{group_name}' 移除 {code} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, msg = self.stock_list_widget.remove_stock_from_current_group(code)
            if success:
                self.statusBar().showMessage(msg)
                # 同步更新自选列表组件
                if self.watchlist_widget.get_current_group() == group_name:
                    self.watchlist_widget.on_group_combo_changed(
                        self.watchlist_widget.group_combo.currentIndex()
                    )
            else:
                QMessageBox.warning(self, "错误", msg)
    def create_group_and_add(self):
        """新建分组并添加当前股票"""
        name, ok = QInputDialog.getText(self, "新建分组", "请输入分组名称:")
        if ok and name:
            success, msg = self.watchlist_manager.create_group(name)
            if success:
                self.stock_list_widget.update_group_combo()
                self.etf_list_widget.update_group_combo()  # 同步更新ETF分组列表
                self.watchlist_widget.update_group_combo()  # 同步更新自选列表
                self.add_current_to_watchlist(name)
            else:
                QMessageBox.warning(self, "错误", msg)
    # ==================== ETF 自选分组相关方法 ====================
    
    def add_current_etf_to_watchlist(self, group_name: str):
        """添加当前选中的ETF到自选分组"""
        code = self.etf_list_widget.get_selected_etf()
        if not code:
            return
            
        success, msg = self.watchlist_manager.add_to_group(group_name, code)
        if success:
            self.statusBar().showMessage(msg)
            # 如果当前正在显示这个分组，刷新显示
            if self.etf_list_widget.get_current_group() == group_name:
                self.etf_list_widget.on_group_combo_changed(
                    self.etf_list_widget.group_combo.currentIndex()
                )
            # 同时更新ETF列表的分组下拉框
            self.etf_list_widget.update_group_combo()
            # 同步更新自选列表组件
            if self.watchlist_widget.get_current_group() == group_name:
                self.watchlist_widget.on_group_combo_changed(
                    self.watchlist_widget.group_combo.currentIndex()
                )
        else:
            QMessageBox.warning(self, "提示", msg)
    
    def remove_current_etf_from_group(self):
        """从当前自选分组移除选中的ETF"""
        code = self.etf_list_widget.get_selected_etf()
        if not code:
            return
            
        group_name = self.etf_list_widget.get_current_group()
        if not group_name:
            return
        
        name = self.etf_name_map.get(code, code)
        reply = QMessageBox.question(
            self, "确认移除", 
            f"确定要从 '{group_name}' 移除 {name}({code}) 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, msg = self.etf_list_widget.remove_etf_from_current_group(code)
            if success:
                self.statusBar().showMessage(msg)
                # 同步更新自选列表组件
                if self.watchlist_widget.get_current_group() == group_name:
                    self.watchlist_widget.on_group_combo_changed(
                        self.watchlist_widget.group_combo.currentIndex()
                    )
            else:
                QMessageBox.warning(self, "错误", msg)
    
    def create_group_and_add_etf(self):
        """新建分组并添加当前ETF"""
        name, ok = QInputDialog.getText(self, "新建自选分组", "请输入分组名称:")
        if ok and name:
            success, msg = self.watchlist_manager.create_group(name)
            if success:
                self.stock_list_widget.update_group_combo()  # 同步更新股票列表的分组
                self.etf_list_widget.update_group_combo()
                self.watchlist_widget.update_group_combo()  # 同步更新自选列表
                self.add_current_etf_to_watchlist(name)
            else:
                QMessageBox.warning(self, "错误", msg)
    def update_current_stock_incremental(self):
        """增量更新当前股票"""
        code = self.stock_list_widget.get_selected_stock()
        if not code:
            return
        self.start_single_stock_update(code, full_update=False)

    def update_current_stock_full(self):
        """全量更新当前股票（指定日期）"""
        code = self.stock_list_widget.get_selected_stock()
        if not code:
            return
            
        # Ask for start date
        dialog = QDialog(self)
        dialog.setWindowTitle("选择起始日期")
        layout = QVBoxLayout(dialog)
        
        date_edit = QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDate(QDate.currentDate().addYears(-1)) # Default 1 year ago
        date_edit.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(QLabel("起始日期:"))
        layout.addWidget(date_edit)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            start_date = date_edit.date().toString("yyyyMMdd")
            self.start_single_stock_update(code, full_update=True, start_date=start_date)

    def open_simulator(self):
        """打开模拟训练窗口"""
        code = self.stock_list_widget.get_selected_stock()
        if not code:
            return
            
        # 创建独立窗口
        simulator_window = QMainWindow(self)
        simulator_window.setWindowTitle(f"模拟交易 - {code}")
        simulator_window.resize(1200, 800)
        simulator_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # 实例化模拟器组件
        simulator_widget = TradingSimulatorWidget(self.data_dir)
        simulator_window.setCentralWidget(simulator_widget)
        
        # 预选股票
        index = simulator_widget.stock_combo.findData(code)
        if index >= 0:
            simulator_widget.stock_combo.setCurrentIndex(index)
            
        simulator_window.show()
        
        # 保存引用并处理关闭事件
        self.simulator_windows.append(simulator_window)
        # 当窗口关闭时从列表中移除引用
        simulator_window.destroyed.connect(lambda: self.simulator_windows.remove(simulator_window) if simulator_window in self.simulator_windows else None)

    def _show_strategy_app_hint(self, feature_name: str):
        """提示用户到策略应用使用相关功能"""
        QMessageBox.information(
            self,
            f"{feature_name} 已迁移",
            f"{feature_name} 功能已迁移到「策略研究」应用。\n\n"
            f"请运行 python run_app.py 启动策略研究台。\n\n"
            f"此应用专注于 K线行情查看和实盘交易。"
        )

    def open_screener(self):
        """打开智能选股窗口（已迁移）"""
        self._show_strategy_app_hint("智能选股")
    
    def open_technical_screener(self):
        """打开技术指标选股窗口（已迁移）"""
        self._show_strategy_app_hint("技术指标选股")
    
    def open_ai_tool(self):
        """打开 AI 智能交易工具（已迁移）"""
        self._show_strategy_app_hint("AI 智能交易训练")

    def open_etf_grid_strategy(self):
        """打开 ETF 网格回测窗口（已迁移）"""
        self._show_strategy_app_hint("ETF网格回测")

    def open_backtest_window(self):
        """打开策略回测窗口（已迁移）"""
        self._show_strategy_app_hint("策略回测")

    def open_cross_sectional_backtest_window(self):
        """打开截面策略回测窗口（已迁移）"""
        self._show_strategy_app_hint("截面策略回测")

    def open_factor_library_window(self):
        """打开因子库窗口（已迁移）"""
        self._show_strategy_app_hint("因子库")

    def open_sector_window(self):
        """打开热门板块独立窗口"""
        # 检查是否已经打开了板块窗口
        if self.sector_window and self.sector_window.isVisible():
            self.sector_window.activateWindow()
            self.sector_window.raise_()
            return
        
        # 创建新窗口
        self.sector_window = SectorWindow(self)
        self.sector_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        # 连接股票选择信号
        self.sector_window.stockSelected.connect(self._on_sector_stock_selected)
        
        # 窗口关闭时清理引用
        self.sector_window.destroyed.connect(lambda: setattr(self, 'sector_window', None))
        
        self.sector_window.show()
        
        # 自动启动服务
        self.sector_window.start_service()
    
    def _on_sector_stock_selected(self, code: str, name: str):
        """板块窗口中选择股票时的处理"""
        # 切换到股票视图
        self.left_tabs.setCurrentIndex(0)  # 切换到股票列表Tab
        
        # 查找并选中该股票
        stock_code_with_suffix = None
        for stock in self.stock_list:
            if stock.startswith(code):
                stock_code_with_suffix = stock
                break
        
        if stock_code_with_suffix:
            self.on_stock_selected(stock_code_with_suffix)
            # 在股票列表中定位
            self.stock_list_widget.select_stock(stock_code_with_suffix)
        else:
            # 股票不在列表中，直接尝试加载
            self.current_code = code
            self.current_name = name or code
            self.current_view = "stock"
            self.load_and_display_chart()

    def open_broker_account(self, stock_code: str = None):
        self.trading_bridge.open_broker_account(stock_code)
        self.broker_window = self.trading_bridge.broker_window

    def _on_broker_window_destroyed(self):
        self.trading_bridge._on_broker_window_destroyed()
        self.broker_window = self.trading_bridge.broker_window
    
    def on_broker_positions_updated(self, position_codes: list):
        self.trading_bridge.sync_broker_positions(position_codes)
        self.refresh_agent_context()

    def _build_market_close_reminder_dialog(self, last_data_date: str):
        return MarketCloseReminderDialog(self, last_data_date)

    def check_market_close_reminder(self):
        self.sync_controller.check_market_close_reminder()

    def refresh_agent_context(self):
        if hasattr(self, "agent_widget") and self.agent_widget:
            self.agent_widget.refresh_context()

    def build_agent_runtime_context(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "symbol": self._build_agent_symbol_context(),
            "watchlist": self._build_agent_watchlist_context(),
            "broker": self._build_agent_broker_context(),
            "market_data": self._build_agent_market_data_context(),
            "_agent_tool_hooks": self._build_agent_tool_hooks(),
        }

    def _build_agent_tool_hooks(self) -> dict:
        return {
            "get_current_symbol_df": self._get_agent_symbol_dataframe,
            "capture_current_kline_image": self._capture_agent_kline_image,
        }

    @staticmethod
    def _normalize_agent_hook_symbol_code(code: str) -> str:
        return str(code or "").split(".")[0].upper()

    def _build_agent_market_data_context(self) -> dict:
        scheduler_manager = getattr(self, "scheduler_manager", None)
        config = getattr(scheduler_manager, "config", {}) or {}
        token = config.get("tushare_token", os.environ.get("TUSHARE_TOKEN", ""))
        return {
            "tushare_token": token,
            "data_source": config.get("data_source", "xtquant"),
        }

    def _get_agent_symbol_dataframe(self, code: str = "", asset_type: str = ""):
        requested_code = self._normalize_agent_hook_symbol_code(code)
        current_code = ""
        if asset_type == "ETF":
            current_code = self.current_etf_code
        elif asset_type == "指数":
            current_code = self.current_index_code
        else:
            current_code = self.current_code

        if requested_code and self._normalize_agent_hook_symbol_code(current_code) != requested_code:
            return None
        df = getattr(self.kline_widget, "data", None)
        if df is None or df.empty:
            return None
        return df.copy()

    def _capture_agent_kline_image(self, code: str = "", asset_type: str = "") -> str | None:
        requested_code = self._normalize_agent_hook_symbol_code(code)
        current_code = ""
        if asset_type == "ETF":
            current_code = self.current_etf_code
        elif asset_type == "指数":
            current_code = self.current_index_code
        else:
            current_code = self.current_code

        if not current_code:
            return None
        if requested_code and self._normalize_agent_hook_symbol_code(current_code) != requested_code:
            return None

        QApplication.processEvents()
        pixmap = self.kline_widget.grab()
        if pixmap.isNull():
            return None

        import tempfile
        import time

        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / f"{TEMP_KLINE_PREFIX}{current_code}_{int(time.time())}.png"
        if not pixmap.save(str(file_path), "PNG"):
            return None
        return str(file_path)

    def _build_agent_symbol_context(self) -> dict:
        code = ""
        name = ""
        asset_type = self.current_view
        if self.current_view == "stock":
            code = self.current_code
            name = self.current_name
            asset_type = "股票"
        elif self.current_view == "etf":
            code = self.current_etf_code
            name = self.current_etf_name
            asset_type = "ETF"
        elif self.current_view == "index":
            code = self.current_index_code
            name = self.current_index_name
            asset_type = "指数"

        latest_close = 0.0
        latest_change_pct = 0.0
        latest_volume = 0.0
        data_points = 0
        date_start = ""
        date_end = ""
        if getattr(self.kline_widget, "data", None) is not None and not self.kline_widget.data.empty:
            df = self.kline_widget.data
            data_points = len(df)
            latest = df.iloc[-1]
            latest_close = float(latest.get("close", 0.0) or 0.0)
            latest_volume = float(latest.get("volume", 0.0) or 0.0)
            if data_points >= 2:
                prev_close = float(df.iloc[-2].get("close", 0.0) or 0.0)
                if prev_close > 0:
                    latest_change_pct = (latest_close - prev_close) / prev_close * 100
            first_date = df.iloc[0].get("date")
            last_date = latest.get("date")
            date_start = first_date.strftime("%Y-%m-%d") if hasattr(first_date, "strftime") else str(first_date)[:10]
            date_end = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)[:10]

        indicators = []
        if self.volume_checkbox.isChecked():
            indicators.append("成交量")
        if self.macd_checkbox.isChecked():
            indicators.append("MACD")
        if self.kdj_checkbox.isChecked():
            indicators.append("KDJ")

        return {
            "code": code,
            "name": name,
            "asset_type": asset_type,
            "current_view": self.current_view,
            "latest_close": latest_close,
            "latest_change_pct": latest_change_pct,
            "latest_volume": latest_volume,
            "data_points": data_points,
            "date_start": date_start,
            "date_end": date_end,
            "indicators": indicators,
        }

    def _build_agent_watchlist_context(self) -> dict:
        source_tab_map = {
            0: "股票列表",
            1: "ETF列表",
            2: "自选列表",
            3: "指数列表",
        }
        source_tab = source_tab_map.get(self.left_tabs.currentIndex(), "")
        group_name = ""
        visible_codes = []
        visible_items = []
        if self.left_tabs.currentIndex() == 0:
            group_name = self.stock_list_widget.get_current_group()
            full_list = list(self.stock_list_widget.filtered_list)
            visible_codes = full_list[:20]
            visible_items = [
                {"code": code, "name": self.name_map.get(code, code), "asset_type": "股票"}
                for code in full_list[:20]
            ]
        elif self.left_tabs.currentIndex() == 1:
            group_name = self.etf_list_widget.get_current_group()
            full_list = list(self.etf_list_widget.filtered_list)
            visible_codes = full_list[:20]
            visible_items = [
                {"code": code, "name": self.etf_name_map.get(code, code), "asset_type": "ETF"}
                for code in full_list[:20]
            ]
        elif self.left_tabs.currentIndex() == 2:
            group_name = self.watchlist_widget.get_current_group()
            full_list = list(self.watchlist_widget.filtered_list)
            visible_codes = full_list[:20]
            visible_items = []
            for code in full_list[:20]:
                is_etf = code in self.etf_name_map or code.startswith(("51", "52", "56", "58", "15", "16", "18"))
                visible_items.append({
                    "code": code,
                    "name": self.etf_name_map.get(code, code) if is_etf else self.name_map.get(code, code),
                    "asset_type": "ETF" if is_etf else "股票",
                })

        return {
            "source_tab": source_tab,
            "group_name": group_name,
            "visible_count": len(full_list) if 'full_list' in locals() else len(visible_codes),
            "visible_codes": visible_codes,
            "visible_items": visible_items,
        }

    def _build_agent_broker_context(self) -> dict:
        broker_service = get_broker_session_service()
        if not broker_service.is_connected:
            return {"connected": False}

        try:
            asset = broker_service.query_stock_asset()
            positions = broker_service.query_stock_positions() or []
            top_positions = []
            for pos in positions[:5]:
                code = str(getattr(pos, "stock_code", ""))
                simple_code = code.split(".")[0] if "." in code else code
                top_positions.append({
                    "code": simple_code,
                    "volume": int(getattr(pos, "volume", 0) or 0),
                    "cost_price": 0.0,
                    "market_value": float(getattr(pos, "market_value", 0.0) or 0.0),
                })
            return {
                "connected": True,
                "account_id": str(getattr(asset, "account_id", "")),
                "total_asset": float(getattr(asset, "total_asset", 0.0) or 0.0),
                "available_cash": float(getattr(asset, "cash", 0.0) or 0.0),
                "position_count": len(positions),
                "top_positions": top_positions,
            }
        except Exception:
            return {"connected": False}

    def open_ai_agent(self):
        """打开/关闭嵌入式智能体面板"""
        if self.agent_widget.isVisible():
            self.agent_widget.setVisible(False)
            # 恢复之前的比例，或者设为默认
            self.splitter.setSizes([150, 1050, 0])
            self.statusBar().showMessage("智能体面板已隐藏")
        else:
            self.agent_widget.setVisible(True)
            self.refresh_agent_context()
            # 设置显示比例，左:中:右 = 150 : 700 : 350
            self.splitter.setSizes([150, 700, 350])
            self.statusBar().showMessage("智能体面板已显示")
            # 自动滚动到底部
            if hasattr(self.agent_widget, 'scroll_area'):
                self.agent_widget.scroll_area.verticalScrollBar().setValue(
                    self.agent_widget.scroll_area.verticalScrollBar().maximum()
                )
            # 聚焦输入框
            if hasattr(self.agent_widget, 'message_input'):
                self.agent_widget.message_input.setFocus()

    def open_live_strategy_center(self, initial_tab: str = "ai"):
        """打开统一实盘策略中枢窗口"""
        if self._live_strategy_window and self._live_strategy_window.isVisible():
            self._live_strategy_window.switch_to_tab(initial_tab)
            self._live_strategy_window.activateWindow()
            self._live_strategy_window.raise_()
            if initial_tab == "ai" and self.current_code:
                self._live_strategy_window.set_symbol(self.current_code, self.current_name)
            return

        self._live_strategy_window = LiveStrategyHubWindow(
            parent=self,
            context_provider=self.build_agent_runtime_context,
            symbol_name_resolver=lambda code: self.name_map.get(code, "") or self.name_map.get(code.split(".")[0], "")
            or self.etf_name_map.get(code, "") or self.etf_name_map.get(code.split(".")[0], ""),
            name_map=self.name_map,
            etf_name_map=self.etf_name_map,
            initial_tab=initial_tab,
        )
        self._live_strategy_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._live_strategy_window.destroyed.connect(self._on_live_strategy_window_destroyed)
        if initial_tab == "ai" and self.current_code:
            self._live_strategy_window.set_symbol(self.current_code, self.current_name)
        self._live_strategy_window.show()

    def open_ai_trade_decision_center(self):
        """兼容旧入口：打开统一中心并切到 AI 页"""
        self.open_live_strategy_center(initial_tab="ai")

    def _on_live_strategy_window_destroyed(self):
        self._live_strategy_window = None

    def _open_ai_decision_for(self, code: str, name: str = ""):
        self.open_ai_trade_decision_center()
        if self._live_strategy_window:
            self._live_strategy_window.set_symbol(code, name)

    def open_notification_dialog(self, stocks_data=None):
        """打开消息推送对话框
        
        Args:
            stocks_data: 可选的选股数据列表
        """
        dialog = NotificationDialog(self, stocks_data=stocks_data)
        dialog.exec()

    def open_scheduled_task_dialog(self):
        """打开定时任务配置对话框"""
        dialog = ScheduledTaskDialog(self.scheduler_manager, self)
        dialog.set_dark_style()
        dialog.exec()
    
    def open_auto_stop_loss_config(self):
        """打开自动止损配置对话框"""
        from trading_app.widgets.auto_stop_loss_config_dialog import AutoStopLossConfigDialog
        
        auto_stop_loss_service = get_auto_stop_loss_service()
        dialog = AutoStopLossConfigDialog(auto_stop_loss_service, self)
        dialog.exec()

    def on_scheduled_task_finished(self, success, message):
        """定时任务完成回调"""
        if success:
            self.statusBar().showMessage(f"✓ 定时任务执行成功: {message}", 5000)
            # 任务执行完可能更新了数据，如果是手动执行的可以考虑刷新，
            # 但定时任务通常在后台，这里只提示一下
            # self.load_stock_list() 
        else:
            self.statusBar().showMessage(f"⚠ 定时任务执行失败: {message}", 10000)

    def on_screener_stock_selected(self, code):
        """处理选股结果点击"""
        # 在主窗口选中该股票
        self.stock_list_widget.select_stock(code)
        
        # 获取股票名称
        name = self.name_map.get(code, "")
        
        # 立即触发选中逻辑，更新 K 线图
        self.on_stock_selected(code, name)
        
        # 激活主窗口
        self.activateWindow()
        self.raise_()

    def on_strategy_finished(self, strategy_name, codes):
        """处理选股策略完成，同步到自选股分组"""
        group_name = f"策略: {strategy_name}"
        
        # 更新或创建分组
        self.watchlist_manager.update_group_stocks(group_name, codes)
        
        # 更新 UI
        self.stock_list_widget.update_group_combo()
        
        # 如果当前正在显示这个分组，触发刷新显示
        if self.stock_list_widget.get_current_group() == group_name:
            self.stock_list_widget.on_group_combo_changed(
                self.stock_list_widget.group_combo.findData(group_name)
            )
            
        self.statusBar().showMessage(f"已同步 {len(codes)} 只股票到分组 '{group_name}'")

    def show_chip_distribution(self, code: str, name: str = ""):
        """显示筹码分布图弹窗"""
        # 获取当前价格
        current_price = 0.0
        if self.kline_widget.data is not None and not self.kline_widget.data.empty:
            current_price = self.kline_widget.data.iloc[-1]['close']
        
        dialog = ChipDistributionDialog(code, name, current_price, self)
        dialog.show()
    
    def on_refresh_strategy(self, strategy_name):
        """处理从股票列表触发的策略刷新（已迁移）"""
        self._show_strategy_app_hint(f"选股策略 [{strategy_name}]")

    def start_single_stock_update(self, code, full_update=False, start_date=None):
        """启动单只股票更新"""
        default_token = os.environ.get("TUSHARE_TOKEN", "")
        self.update_dialog = UpdateDialog(self, default_token)
        self.update_dialog.setWindowTitle(f"更新股票 {code}")
        
        # Hide options that don't apply for single stock update
        self.update_dialog.full_update_cb.setVisible(False)
        self.update_dialog.exclude_gem_cb.setVisible(False)
        self.update_dialog.exclude_star_cb.setVisible(False)
        self.update_dialog.exclude_bj_cb.setVisible(False)
        
        # Connect dialog signals to a custom handler
        try:
            self.update_dialog.start_update.disconnect()
        except:
            pass
            
        # 新的信号有 6 个参数: token, full_update, exclude_boards, start_date, data_source, period
        self.update_dialog.start_update.connect(
            lambda t, f, e, sd, ds, p: self.run_single_stock_thread(
                code, t, full_update, start_date, ds, p
            )
        )
        
        self.update_dialog.exec()

    def run_single_stock_thread(self, code, token, full_update, start_date, 
                                 data_source="tushare", period="1d"):
        """运行单只股票更新线程
        
        Args:
            code: 股票代码
            token: Tushare token
            full_update: 是否全量更新
            start_date: 起始日期
            data_source: 数据源
            period: K线周期
        """
        if self.update_thread and self.update_thread.isRunning():
            return

        self._updating_codes = [code]  # 记录正在更新的股票代码
        
        self.update_thread = DataUpdateThread(
            self.data_dir,
            self.stocklist_path,
            token,
            full_update=full_update,
            codes=[code],
            start_date=start_date,
            data_source=data_source,
            period=period
        )
        self.update_thread.progress_updated.connect(self.update_dialog.update_progress)
        self.update_thread.log_message.connect(self.update_dialog.append_log)
        self.update_thread.result_signal.connect(self._on_update_result)
        self.update_thread.finished_signal.connect(self.on_update_finished)
        self.update_thread.start()
