"""
Broker Account Widget - Query broker account and trading records via xtquant
"""
import os
import json
import random
import logging
import traceback
from pathlib import Path
from datetime import datetime

# 检查 xtquant 是否可用
try:
    import xtquant
    from xtquant import xtdata
    HAS_XTQUANT = True
except ImportError:
    HAS_XTQUANT = False

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QGroupBox, QFormLayout, QLineEdit, QFrame,
    QSplitter, QFileDialog, QSizePolicy, QTextEdit, QScrollArea,
    QSpinBox, QDoubleSpinBox, QComboBox, QDialog, QDialogButtonBox,
    QMenu, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QColor, QBrush, QFont

from common.broker_session_service import get_broker_session_service
from trading_app.widgets.order_book_widget import OrderBookWidget
from trading_app.widgets.conditional_order_dialog import ConditionalOrderWidget, AddConditionalOrderDialog
from trading_app.widgets.trade_history_widget import TradeHistoryWidget
from trading_app.widgets.daily_pnl_widget import DailyPnlWidget
from trading_app.services.execution.conditional_order_service import get_conditional_order_service, OrderConditionType
from trading_app.services.execution.trade_execution_service import ExecutionRequest, get_trade_execution_service
from trading_app.services.execution.trade_record_service import TradeSource, get_trade_record_service

# Setup logging directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'broker_connection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BrokerConfigDialog(QDialog):
    """券商连接配置对话框"""
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = dict(config or {})
        self.setWindowTitle("券商连接配置")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # QMT data path
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self._config.get("qmt_path", ""))
        self.path_edit.setPlaceholderText("D:\\中金财富QMT个人版交易端\\userdata_mini")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        form_layout.addRow("QMT数据路径:", path_layout)

        # Account
        self.account_edit = QLineEdit(self._config.get("account", ""))
        self.account_edit.setPlaceholderText("输入资金账号")
        form_layout.addRow("资金账号:", self.account_edit)

        self.process_name_edit = QLineEdit(self._config.get("process_name", ""))
        self.process_name_edit.setPlaceholderText("可选，例如 miniqmt.exe")
        form_layout.addRow("进程名提示:", self.process_name_edit)
        
        layout.addLayout(form_layout)
        
        # Spacer
        layout.addSpacing(10)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择QMT数据目录")
        if path:
            self.path_edit.setText(path)

    def get_config(self):
        config = dict(self._config)
        config.update({
            "qmt_path": self.path_edit.text().strip(),
            "account": self.account_edit.text().strip(),
            "process_name": self.process_name_edit.text().strip(),
        })
        return config


class BrokerConnectThread(QThread):
    """Thread for connecting to broker"""
    connected = pyqtSignal(bool, str)  # success, message
    log_message = pyqtSignal(str)  # progress log message
    
    def __init__(self, qmt_path: str, account: str):
        super().__init__()
        self.qmt_path = qmt_path
        self.account = account
        self.xt_trader = None
        self.acc = None
    
    def _log(self, message: str):
        """Helper method to send log message"""
        logger.info(message)
        self.log_message.emit(message)
    
    def run(self):
        try:
            self._log("正在启动券商连接服务...")
            from xtquant import xttrader
            from xtquant.xttype import StockAccount
            
            # 生成随机会话ID
            session_id = int(random.randint(100000, 999999))
            
            # 创建实例
            self.xt_trader = xttrader.XtQuantTrader(self.qmt_path, session_id)
            self.xt_trader.start()
            
            # 连接QMT
            connect_result = self.xt_trader.connect()
            if connect_result != 0:
                error_msg = "连接QMT交易端失败，请确认QMT已启动并登录"
                self._log(f"✗ {error_msg}")
                self.connected.emit(False, error_msg)
                return
            
            # 创建并订阅账户
            self.acc = StockAccount(self.account)
            res = self.xt_trader.subscribe(self.acc)
            
            if res != 0:
                error_msg = f"账户订阅失败 (代码: {res})，请检查资金账号"
                self._log(f"✗ {error_msg}")
                self.connected.emit(False, error_msg)
                return
            
            self._log("✓ 券商账户连接成功")
            self.connected.emit(True, "连接成功")
            
        except ImportError:
            error_msg = "未安装 xtquant 库，请先安装"
            self._log(f"✗ {error_msg}")
            self.connected.emit(False, error_msg)
        except Exception as e:
            error_msg = f"连接异常: {str(e)}"
            logger.error(f"Exception: {e}")
            self._log(f"✗ {error_msg}")
            self.connected.emit(False, error_msg)

class QueryThread(QThread):
    """Thread for querying broker data"""
    finished = pyqtSignal(str, object, str)  # query_type, data, error_message
    log_message = pyqtSignal(str)
    
    def __init__(self, query_type: str, xt_trader, acc):
        super().__init__()
        self.query_type = query_type
        self.xt_trader = xt_trader
        self.acc = acc
    
    def _log(self, message: str):
        """Helper method to send log message"""
        logger.info(message)
        self.log_message.emit(message)
    
    def run(self):
        try:
            if self.query_type == "positions":
                data = self.xt_trader.query_stock_positions(self.acc)
            elif self.query_type == "orders":
                data = self.xt_trader.query_stock_orders(self.acc)
            elif self.query_type == "trades":
                data = self.xt_trader.query_stock_trades(self.acc)
            else:
                self.finished.emit(self.query_type, [], "未知的查询类型")
                return
            
            self.finished.emit(self.query_type, data, "")
            
        except Exception as e:
            error_msg = f"{str(e)}"
            logger.error(f"[{self.query_type}] 查询失败: {e}")
            self.finished.emit(self.query_type, [], error_msg)


class TradeThread(QThread):
    """Thread for placing orders through the unified execution gateway."""
    finished = pyqtSignal(bool, str, int)  # success, message, order_id
    log_message = pyqtSignal(str)

    def __init__(self, stock_code: str, stock_name: str, order_type: int,
                 order_volume: int, price_type: int, price: float):
        super().__init__()
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.order_type = order_type  # 23=买入, 24=卖出
        self.order_volume = order_volume
        self.price_type = price_type
        self.price = price

    def _log(self, message: str):
        """Helper method to send log message"""
        logger.info(message)
        self.log_message.emit(message)

    @staticmethod
    def _normalize_stock_code(stock_code: str) -> str:
        code = str(stock_code or "").strip().upper()
        if "." not in code and len(code) == 6:
            if code.startswith(("5", "6", "9")):
                return f"{code}.SH"
            if code.startswith(("0", "1", "2", "3")):
                return f"{code}.SZ"
        return code

    @staticmethod
    def _resolve_xt_price_type(ui_price_type: int) -> int:
        try:
            from xtquant import xtconstant
            if int(ui_price_type) == 0:
                return int(xtconstant.FIX_PRICE)
            return int(getattr(xtconstant, "LATEST_PRICE", getattr(xtconstant, "MARKET_PRICE", 1)))
        except Exception:
            return 0 if int(ui_price_type) == 0 else 1

    def run(self):
        try:
            stock_code = self._normalize_stock_code(self.stock_code)
            if not stock_code:
                error_msg = "股票代码不能为空"
                self._log(f"✗ {error_msg}")
                self.finished.emit(False, error_msg, -1)
                return

            direction = "买入" if int(self.order_type) == 23 else "卖出"
            actual_price_type = self._resolve_xt_price_type(self.price_type)
            price_desc = f"限价 {self.price:.3f}" if int(self.price_type) == 0 else f"市价(校验价 {self.price:.3f})"
            self._log(f"准备{direction} {stock_code} {self.order_volume}股 ({price_desc})，提交统一执行网关...")

            result = get_trade_execution_service().execute(
                ExecutionRequest(
                    stock_code=stock_code,
                    stock_name=self.stock_name or stock_code,
                    order_type=int(self.order_type),
                    order_volume=int(self.order_volume),
                    price_type=actual_price_type,
                    price=float(self.price or 0.0),
                    source=TradeSource.MANUAL.value,
                    trigger="manual",
                    strategy_name="ManualTrade",
                    remark="券商账户页手动委托",
                    require_approval=False,
                    approved=False,
                    metadata={"owner_type": "manual"},
                )
            )
            order_id = int(result.broker_order_id or -1)
            if result.success:
                success_msg = result.message or f"{direction}委托成功"
                self._log(f"✓ {success_msg}")
                self.finished.emit(True, success_msg, order_id)
                return

            error_msg = result.message or f"{direction}委托失败"
            self._log(f"✗ {error_msg}")
            self.finished.emit(False, error_msg, order_id)
        except Exception as e:
            error_msg = f"下单失败: {str(e)}"
            logger.error(f"下单失败: {e}")
            logger.error(traceback.format_exc())
            self._log(f"✗ {error_msg}")
            self.finished.emit(False, error_msg, -1)


class CancelOrderThread(QThread):
    """Thread for canceling orders"""
    finished = pyqtSignal(bool, str)  # success, message
    log_message = pyqtSignal(str)
    
    def __init__(self, xt_trader, acc, order_id: int):
        super().__init__()
        self.xt_trader = xt_trader
        self.acc = acc
        self.order_id = order_id
    
    def _log(self, message: str):
        """Helper method to send log message"""
        logger.info(message)
        self.log_message.emit(message)
    
    def run(self):
        try:
            self._log(f"正在撤单...")
            result = self.xt_trader.cancel_order_stock(self.acc, self.order_id)
            
            if result != 0:
                error_msg = f"撤单失败"
                self._log(f"✗ {error_msg}")
                self.finished.emit(False, error_msg)
            else:
                success_msg = f"撤单成功"
                self._log(f"✓ {success_msg}")
                self.finished.emit(True, success_msg)
                
        except Exception as e:
            error_msg = f"撤单异常: {str(e)}"
            logger.error(f"撤单失败: {e}")
            self._log(f"✗ {error_msg}")
            self.finished.emit(False, error_msg)


class BrokerAccountWidget(QWidget):
    """交易窗口 - 券商账户查询和交易"""
    
    # 信号：持仓数据更新，发送持仓股票代码列表
    positionsUpdated = pyqtSignal(list)  # List[str] 股票代码列表
    
    def __init__(self, parent=None, name_map=None, broker_session_service=None):
        super().__init__(parent)
        
        self.config_path = Path(__file__).parent.parent / "config" / "broker_config.json"
        self.xt_trader = None
        self.acc = None
        self.is_connected = False
        self.name_map = name_map or {}
        self.broker_session_service = broker_session_service or get_broker_session_service()
        self._connection_in_progress = False
        
        # Query threads for different data types
        self.positions_query_thread = None
        self.orders_query_thread = None
        self.trades_query_thread = None
        
        # Trade threads
        self.trade_thread = None
        self.cancel_order_thread = None
        
        # Order book refresh timer
        self.order_book_timer = QTimer(self)
        self.order_book_timer.timeout.connect(self.refresh_trade_order_book)
        self.order_book_timer.setInterval(5000) # 5 seconds
        
        # 条件单服务
        self.conditional_order_service = get_conditional_order_service()
        self.trade_execution_service = get_trade_execution_service()
        self.conditional_order_service.log_message.connect(self.append_log)
        self.conditional_order_service.order_triggered.connect(self.on_conditional_order_triggered)
        self.conditional_order_service.order_executed.connect(self.on_conditional_order_executed)
        
        # 交易记录服务
        self.trade_record_service = get_trade_record_service()
        self.trade_record_service.log_message.connect(self.append_log)
        
        logger.info("="*60)
        logger.info("初始化交易窗口")
        logger.info("="*60)
        
        self.load_config()
        self.setup_ui()
        self.broker_session_service.connection_changed.connect(self._on_broker_session_changed)
        self.broker_session_service.log_message.connect(self.append_log)
        self.broker_session_service.config_changed.connect(self._on_broker_config_changed)
        self.broker_session_service.trade_occurred.connect(self._on_broker_trade_callback)
        self.broker_session_service.order_changed.connect(self._on_broker_order_callback)
        self._sync_from_session()
        
        # 延迟自动连接
        if self.qmt_path and self.account and not self.is_connected:
            QTimer.singleShot(500, self.connect_broker)
    
    def load_config(self):
        """Load broker configuration"""
        config = self.broker_session_service.get_config()
        self.qmt_path = config.get("qmt_path", "")
        self.account = config.get("account", "")
        self.process_name = config.get("process_name", "")
        logger.info(f"✓ 配置加载成功: qmt_path={self.qmt_path}, account={self.account}")
    
    def save_config(self):
        """Save broker configuration"""
        try:
            self.broker_session_service.save_config({
                "qmt_path": self.qmt_path,
                "account": self.account,
                "process_name": getattr(self, "process_name", ""),
            })
            logger.info(f"✓ 配置已保存: {self.config_path}")
        except Exception as e:
            logger.error(f"✗ 保存配置失败: {e}")
            print(f"Failed to save broker config: {e}")

    def _on_broker_config_changed(self, config: dict):
        self.qmt_path = config.get("qmt_path", self.qmt_path)
        self.account = config.get("account", self.account)
        self.process_name = config.get("process_name", getattr(self, "process_name", ""))

    def _sync_from_session(self):
        self.xt_trader = self.broker_session_service.xt_trader
        self.acc = self.broker_session_service.account_obj
        self.is_connected = self.broker_session_service.is_connected
        if self.is_connected:
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("color: #5cb85c; font-weight: bold;")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.buy_btn.setEnabled(True)
            self.sell_btn.setEnabled(True)
            self.setup_conditional_order_executor()
        else:
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("color: #888;")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.buy_btn.setEnabled(False)
            self.sell_btn.setEnabled(False)

    def _on_broker_session_changed(self, connected: bool, message: str):
        self._sync_from_session()
        if connected:
            self._connection_in_progress = False
            self.on_connected(True, message)
            return

        if message in ("正在连接券商...", "券商连接正在进行中"):
            self.status_label.setText("连接中...")
            self.status_label.setStyleSheet("color: #f0ad4e;")
            self.connect_btn.setEnabled(False)
            return

        if self._connection_in_progress:
            self._connection_in_progress = False
            self.on_connected(False, message)
            return

        if message == "券商已断开":
            self.clear_all_tables()
            self.append_log("已断开连接")
    
    def setup_ui(self):
        """Setup UI"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Main horizontal splitter for left and right columns
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # ========== LEFT COLUMN ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Connection settings group
        settings_group = QGroupBox("券商连接")
        settings_layout = QHBoxLayout(settings_group)
        
        # Connection status and buttons in one row
        status_container = QHBoxLayout()
        status_label_prefix = QLabel("状态:")
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: #888; font-weight: bold;")
        status_container.addWidget(status_label_prefix)
        status_container.addWidget(self.status_label)
        status_container.addSpacing(15)
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.connect_broker)
        self.connect_btn.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; padding: 5px 12px;")
        status_container.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.clicked.connect(self.disconnect_broker)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("padding: 5px 12px;")
        status_container.addWidget(self.disconnect_btn)
        
        self.config_btn = QPushButton("⚙ 配置")
        self.config_btn.clicked.connect(self.open_config_dialog)
        self.config_btn.setStyleSheet("padding: 5px 12px;")
        status_container.addWidget(self.config_btn)
        
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setStyleSheet("padding: 5px 12px;")
        status_container.addWidget(self.clear_log_btn)
        
        status_container.addStretch()
        settings_layout.addLayout(status_container)
        
        left_layout.addWidget(settings_group)
        
        # Log display area
        log_group = QGroupBox("连接日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setMaximumHeight(300)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #555;
                border-radius: 3px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        left_layout.addWidget(log_group)
        left_layout.addStretch()
        
        main_splitter.addWidget(left_widget)
        
        # ========== RIGHT COLUMN ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # Trading panel
        trading_group = QGroupBox("交易下单")
        trading_main_layout = QHBoxLayout(trading_group)
        
        # Left side: Order form
        trading_form_widget = QWidget()
        trading_layout = QFormLayout(trading_form_widget)
        trading_layout.setContentsMargins(0, 0, 0, 0)
        
        # Stock code with name display
        stock_code_layout = QHBoxLayout()
        self.trade_stock_code_edit = QLineEdit()
        self.trade_stock_code_edit.setPlaceholderText("输入股票代码，如：000001")
        self.trade_stock_code_edit.setMaximumWidth(100)
        self.trade_stock_code_edit.textChanged.connect(self.on_trade_stock_changed)
        stock_code_layout.addWidget(self.trade_stock_code_edit)
        
        # Stock name label
        self.trade_stock_name_label = QLabel("")
        self.trade_stock_name_label.setStyleSheet("color: #f0ad4e; font-weight: bold; font-size: 13px; padding-left: 8px;")
        stock_code_layout.addWidget(self.trade_stock_name_label)
        stock_code_layout.addStretch()
        trading_layout.addRow("股票代码:", stock_code_layout)
        
        # Price type
        price_type_layout = QHBoxLayout()
        self.price_type_combo = QComboBox()
        self.price_type_combo.addItems(["限价", "市价"])
        self.price_type_combo.currentIndexChanged.connect(self.on_price_type_changed)
        price_type_layout.addWidget(self.price_type_combo)
        price_type_layout.addStretch()
        trading_layout.addRow("价格类型:", price_type_layout)
        
        # Price (for limit order)
        price_layout = QHBoxLayout()
        self.trade_price_spin = QDoubleSpinBox()
        self.trade_price_spin.setDecimals(3)
        self.trade_price_spin.setMinimum(0.001)
        self.trade_price_spin.setMaximum(9999.999)
        self.trade_price_spin.setSingleStep(0.01)
        self.trade_price_spin.setValue(0.0)
        price_layout.addWidget(self.trade_price_spin)
        price_layout.addStretch()
        trading_layout.addRow("委托价格:", price_layout)
        
        # Volume
        volume_layout = QHBoxLayout()
        self.trade_volume_spin = QSpinBox()
        self.trade_volume_spin.setMinimum(100)
        self.trade_volume_spin.setMaximum(1000000)
        self.trade_volume_spin.setSingleStep(100)
        self.trade_volume_spin.setValue(100)
        self.trade_volume_spin.valueChanged.connect(self._update_total_amount)
        volume_layout.addWidget(self.trade_volume_spin)
        volume_layout.addStretch()
        trading_layout.addRow("委托数量(股):", volume_layout)
        
        # Total amount (委托价格 × 委托数量)
        self.total_amount_label = QLabel("¥ 0.00")
        self.total_amount_label.setStyleSheet("color: #f0ad4e; font-weight: bold; font-size: 14px;")
        self.trade_price_spin.valueChanged.connect(self._update_total_amount)
        trading_layout.addRow("委托金额:", self.total_amount_label)
        
        # Trade buttons
        trade_btn_layout = QHBoxLayout()
        self.buy_btn = QPushButton("买入")
        self.buy_btn.clicked.connect(self.on_buy_order)
        self.buy_btn.setStyleSheet("background-color: #ec0000; color: white; font-weight: bold; padding: 8px 16px;")
        self.buy_btn.setEnabled(False)
        trade_btn_layout.addWidget(self.buy_btn)
        
        self.sell_btn = QPushButton("卖出")
        self.sell_btn.clicked.connect(self.on_sell_order)
        self.sell_btn.setStyleSheet("background-color: #00da3c; color: white; font-weight: bold; padding: 8px 16px;")
        self.sell_btn.setEnabled(False)
        trade_btn_layout.addWidget(self.sell_btn)
        
        # 条件单按钮
        self.conditional_order_btn = QPushButton("⏰ 条件单")
        self.conditional_order_btn.clicked.connect(self.on_add_conditional_order)
        self.conditional_order_btn.setStyleSheet("background-color: #f0ad4e; color: white; font-weight: bold; padding: 8px 16px;")
        self.conditional_order_btn.setToolTip("设置止盈止损条件单，自动监控触发")
        trade_btn_layout.addWidget(self.conditional_order_btn)
        
        trade_btn_layout.addStretch()
        trading_layout.addRow("", trade_btn_layout)
        
        trading_main_layout.addWidget(trading_form_widget, stretch=3)
        
        # Vertical divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #333;")
        trading_main_layout.addWidget(line)
        
        # Right side: Order book
        self.trade_order_book = OrderBookWidget()
        self.trade_order_book.setFixedWidth(180)
        trading_main_layout.addWidget(self.trade_order_book, stretch=1)
        
        right_layout.addWidget(trading_group)
        
        # Tabs for positions and orders
        self.data_tabs = QTabWidget()
        
        # Common table style
        table_style = """
            QTableWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                gridline-color: #333;
                border: none;
                selection-background-color: #264f78;
                selection-color: #ffffff;
                alternate-background-color: #252526;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #d4d4d4;
                padding: 4px;
                border: 1px solid #333;
            }
            QTableCornerButton::section {
                background-color: #2d2d2d;
                border: 1px solid #333;
            }
        """
        
        # Positions tab
        self.positions_table = QTableWidget()
        self.positions_table.setStyleSheet(table_style)
        self.positions_table.setColumnCount(12)
        self.positions_table.setHorizontalHeaderLabels([
            "证券代码", "证券名称", "持仓数量", "可用数量", "成本价", 
            "最新价", "市值", "盈亏", "盈亏比例", "昨日持仓", 
            "冻结数量", "在途股份"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.positions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.positions_table.itemDoubleClicked.connect(self.on_position_double_clicked)
        self.positions_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.positions_table.customContextMenuRequested.connect(self.show_positions_context_menu)
        self.data_tabs.addTab(self.positions_table, "📊 持仓信息")
        
        # Orders tab
        self.orders_table = QTableWidget()
        self.orders_table.setStyleSheet(table_style)
        self.orders_table.setColumnCount(11)
        self.orders_table.setHorizontalHeaderLabels([
            "委托编号", "证券代码", "证券名称", "委托方向", "委托价格",
            "委托数量", "成交数量", "委托状态", "委托时间", "备注", "操作"
        ])
        self.orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # 设置操作列的最小宽度，确保按钮文字完整显示
        self.orders_table.horizontalHeader().setMinimumSectionSize(70)
        # 操作列使用固定宽度
        self.orders_table.setColumnWidth(10, 70)
        self.orders_table.setAlternatingRowColors(True)
        self.orders_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.orders_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.data_tabs.addTab(self.orders_table, "📝 当日委托")
        
        # Trades tab
        self.trades_table = QTableWidget()
        self.trades_table.setStyleSheet(table_style)
        self.trades_table.setColumnCount(9)
        self.trades_table.setHorizontalHeaderLabels([
            "成交编号", "委托编号", "证券代码", "证券名称", "成交方向",
            "成交价格", "成交数量", "成交金额", "成交时间"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.trades_table.setAlternatingRowColors(True)
        self.data_tabs.addTab(self.trades_table, "💰 当日成交")
        
        # Account info tab
        self.account_table = QTableWidget()
        self.account_table.setStyleSheet(table_style)
        self.account_table.setColumnCount(2)
        self.account_table.setHorizontalHeaderLabels(["项目", "数值"])
        self.account_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.account_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.account_table.setAlternatingRowColors(True)
        self.account_table.setRowCount(5)
        self.account_table.verticalHeader().setVisible(False)
        
        # Initialize table items
        self.account_table.setItem(0, 0, QTableWidgetItem("资金账号"))
        self.account_table.setItem(1, 0, QTableWidgetItem("总资产"))
        self.account_table.setItem(2, 0, QTableWidgetItem("可用资金"))
        self.account_table.setItem(3, 0, QTableWidgetItem("持仓市值"))
        self.account_table.setItem(4, 0, QTableWidgetItem("浮动盈亏"))
        
        # Create label items for values (we'll update these)
        self.lbl_account_id_item = QTableWidgetItem("-")
        self.lbl_total_assets_item = QTableWidgetItem("-")
        self.lbl_available_cash_item = QTableWidgetItem("-")
        self.lbl_market_value_item = QTableWidgetItem("-")
        self.lbl_profit_loss_item = QTableWidgetItem("-")
        
        self.account_table.setItem(0, 1, self.lbl_account_id_item)
        self.account_table.setItem(1, 1, self.lbl_total_assets_item)
        self.account_table.setItem(2, 1, self.lbl_available_cash_item)
        self.account_table.setItem(3, 1, self.lbl_market_value_item)
        self.account_table.setItem(4, 1, self.lbl_profit_loss_item)
        
        # Store references for easy access
        self.lbl_account_id = self.lbl_account_id_item
        self.lbl_total_assets = self.lbl_total_assets_item
        self.lbl_available_cash = self.lbl_available_cash_item
        self.lbl_market_value = self.lbl_market_value_item
        self.lbl_profit_loss = self.lbl_profit_loss_item
        
        self.data_tabs.addTab(self.account_table, "💰 账户资产")
        
        # 条件单Tab页
        self.conditional_order_widget = ConditionalOrderWidget()
        self.conditional_order_widget.set_service(self.conditional_order_service)
        self.data_tabs.addTab(self.conditional_order_widget, "⏰ 条件单")
        
        # 交易历史Tab页
        self.trade_history_widget = TradeHistoryWidget()
        self.trade_history_widget.stock_selected.connect(self.on_trade_history_stock_selected)
        self.data_tabs.addTab(self.trade_history_widget, "📜 交易历史")
        
        # 每日盈亏Tab页
        self.daily_pnl_widget = DailyPnlWidget()
        self.daily_pnl_widget.snapshot_requested.connect(self.on_auto_save_pnl_snapshot)
        self.data_tabs.addTab(self.daily_pnl_widget, "📈 每日盈亏")
        
        # Refresh buttons
        refresh_widget = QWidget()
        refresh_layout = QHBoxLayout(refresh_widget)
        refresh_layout.setContentsMargins(5, 5, 5, 5)
        
        refresh_positions_btn = QPushButton("刷新持仓")
        refresh_positions_btn.clicked.connect(self.refresh_positions)
        refresh_layout.addWidget(refresh_positions_btn)
        
        refresh_orders_btn = QPushButton("刷新委托")
        refresh_orders_btn.clicked.connect(self.refresh_orders)
        refresh_layout.addWidget(refresh_orders_btn)
        
        refresh_trades_btn = QPushButton("刷新成交")
        refresh_trades_btn.clicked.connect(self.refresh_trades)
        refresh_layout.addWidget(refresh_trades_btn)
        
        refresh_all_btn = QPushButton("刷新全部")
        refresh_all_btn.clicked.connect(self.refresh_all)
        refresh_all_btn.setStyleSheet("background-color: #0078d4; color: white;")
        refresh_layout.addWidget(refresh_all_btn)
        
        export_btn = QPushButton("导出数据")
        export_btn.clicked.connect(self.export_data)
        refresh_layout.addWidget(export_btn)
        
        refresh_layout.addStretch()
        
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(self.data_tabs)
        data_layout.addWidget(refresh_widget)
        
        right_layout.addWidget(data_widget)
        
        main_splitter.addWidget(right_widget)
        
        # Set splitter sizes (left: 400, right: 600)
        main_splitter.setSizes([400, 600])
        
        # Initial log message
        self.append_log("交易界面已就绪")
        if not HAS_XTQUANT:
            self.append_log("⚠ 警告: xtquant 库未安装")
        
        if self.qmt_path and self.account:
            self.append_log(f"已加载配置 (账号: {self.account[:3]}****{self.account[-3:] if len(self.account)>6 else ''})")
        else:
            self.append_log("请先点击 ⚙ 配置券商连接信息")
    
    def set_stock_code(self, stock_code: str):
        """Set stock code in the trading panel"""
        if '.' in stock_code:
            stock_code = stock_code.split('.')[0]
        self.trade_stock_code_edit.setText(stock_code)
        self.append_log(f"已自动填充股票代码: {stock_code}")
        # Focus on volume input for convenience
        self.trade_volume_spin.setFocus()
        self.trade_volume_spin.selectAll()
        # 触发盘口刷新
        self.refresh_trade_order_book()

    def on_position_double_clicked(self, item):
        """Handle double click on position row"""
        row = item.row()
        stock_code_item = self.positions_table.item(row, 0)
        if stock_code_item:
            self.set_stock_code(stock_code_item.text())

    def show_positions_context_menu(self, pos):
        """Show context menu for positions table"""
        item = self.positions_table.itemAt(pos)
        if not item:
            return
            
        row = item.row()
        stock_code_item = self.positions_table.item(row, 0)
        stock_name_item = self.positions_table.item(row, 1)
        total_volume_item = self.positions_table.item(row, 2)
        can_use_volume_item = self.positions_table.item(row, 3)
        
        if not stock_code_item:
            return
            
        stock_code = stock_code_item.text()
        # 去掉市场后缀
        if '.' in stock_code:
            stock_code = stock_code.split('.')[0]
        stock_name = stock_name_item.text() if stock_name_item else ""
        can_use = can_use_volume_item.text() if can_use_volume_item else "0"
        total_vol_text = total_volume_item.text() if total_volume_item else "0"
        
        try:
            can_use_volume = int(float(can_use.replace(',', '')))
        except:
            can_use_volume = 0
        try:
            total_volume = int(float(total_vol_text.replace(',', '')))
        except:
            total_volume = 0
        
        menu = QMenu(self)
        
        buy_action = menu.addAction(f"买入 {stock_name}({stock_code})")
        sell_action = menu.addAction(f"卖出 {stock_name}({stock_code})")
        menu.addSeparator()
        sell_all_action = menu.addAction(f"全仓卖出 ({can_use}股)")
        
        # 条件单菜单
        menu.addSeparator()
        conditional_menu = menu.addMenu("⏰ 设置条件单")
        take_profit_action = conditional_menu.addAction("📈 设置止盈")
        stop_loss_action = conditional_menu.addAction("📉 设置止损")
        conditional_action = conditional_menu.addAction("⚙ 自定义条件单...")
        conditional_menu.addSeparator()
        batch_stop_loss_action = conditional_menu.addAction("🛡️ 为全部持仓创建止损单")
        
        action = menu.exec(self.positions_table.viewport().mapToGlobal(pos))
        
        if action == buy_action:
            self.set_stock_code(stock_code)
        elif action == sell_action:
            self.set_stock_code(stock_code)
        elif action == sell_all_action:
            self.set_stock_code(stock_code)
            try:
                self.trade_volume_spin.setValue(can_use_volume)
            except:
                pass
        elif action == take_profit_action:
            self.add_conditional_order_from_position(
                stock_code, stock_name, max(can_use_volume, total_volume), "take_profit"
            )
        elif action == stop_loss_action:
            self.add_conditional_order_from_position(
                stock_code, stock_name, max(can_use_volume, total_volume), "stop_loss"
            )
        elif action == conditional_action:
            self.set_stock_code(stock_code)
            self.on_add_conditional_order()
        elif action == batch_stop_loss_action:
            self.batch_create_stop_loss_for_positions()

    def is_trading_time(self) -> bool:
        """检查当前是否在交易时间段"""
        now = datetime.now()
        # 检查周六日
        if now.weekday() >= 5:
            return False
        
        current_time = now.time()
        # 9:30-11:30 或 13:00-15:00
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()
        
        is_morning = morning_start <= current_time <= morning_end
        is_afternoon = afternoon_start <= current_time <= afternoon_end
        
        return is_morning or is_afternoon

    def on_trade_stock_changed(self, text: str):
        """Handle stock code change in trading panel"""
        code = text.strip()
        if len(code) >= 6:
            # Update stock name display
            self._update_trade_stock_name(code)
            # 立即刷新一次，并自动填入卖1价格
            self.refresh_trade_order_book(auto_fill_price=True)
            # 仅在交易时间内开启定时刷新
            if self.is_trading_time():
                if not self.order_book_timer.isActive():
                    self.order_book_timer.start()
            else:
                self.order_book_timer.stop()
        else:
            self.order_book_timer.stop()
            self.trade_order_book.clear_data()
            # Clear stock name when code is incomplete
            self.trade_stock_name_label.setText("")
    
    def _update_trade_stock_name(self, code: str):
        """Update stock name display based on stock code"""
        stock_name = ""
        
        # Try to get name from name_map first
        stock_name = self.name_map.get(code, "")
        
        # If not found, try with market suffix
        if not stock_name and HAS_XTQUANT:
            try:
                xt_code = code
                if '.' not in xt_code:
                    if xt_code.startswith(('5', '6', '9')):
                        xt_code = f"{xt_code}.SH"
                    elif xt_code.startswith(('0', '1', '2', '3')):
                        xt_code = f"{xt_code}.SZ"
                stock_name = xtdata.get_stock_name(xt_code)
            except Exception as e:
                logger.debug(f"Failed to get stock name for {code}: {e}")
        
        self.trade_stock_name_label.setText(stock_name if stock_name else "")

    def refresh_trade_order_book(self, auto_fill_price: bool = False):
        """Refresh order book in trading panel
        
        Args:
            auto_fill_price: 是否自动填入卖1价格到委托价格（仅首次输入股票代码时为True）
        """
        code = self.trade_stock_code_edit.text().strip()
        if not code or len(code) < 6:
            return
            
        if not HAS_XTQUANT:
            return
            
        # 如果不在交易时间，且定时器正在运行，则停止定时器
        if not self.is_trading_time() and self.order_book_timer.isActive():
            self.order_book_timer.stop()
            
        try:
            # 格式化代码
            xt_code = code
            if '.' not in xt_code:
                if xt_code.startswith(('5', '6', '9')):
                    xt_code = f"{xt_code}.SH"
                elif xt_code.startswith(('0', '1', '2', '3')):
                    xt_code = f"{xt_code}.SZ"
            
            full_tick = xtdata.get_full_tick([xt_code])
            if xt_code in full_tick:
                tick = full_tick[xt_code]
                # 获取昨收价用于颜色显示
                prev_close = tick.get('lastClose', 0)
                self.trade_order_book.update_data(tick, prev_close)
                
                # 自动填入卖1价格到委托价格
                if auto_fill_price:
                    ask_prices = tick.get('askPrice', [])
                    if ask_prices and len(ask_prices) > 0:
                        ask1_price = ask_prices[0]
                        if ask1_price > 0:
                            self.trade_price_spin.setValue(ask1_price)
        except Exception as e:
            logger.error(f"刷新交易面板盘口失败: {e}")

    def _format_xt_code(self, code: str) -> str:
        xt_code = str(code or "").strip().upper()
        if '.' not in xt_code and len(xt_code) == 6:
            if xt_code.startswith(('5', '6', '9')):
                xt_code = f"{xt_code}.SH"
            elif xt_code.startswith(('0', '1', '2', '3')):
                xt_code = f"{xt_code}.SZ"
        return xt_code

    def _resolve_trade_stock_name(self, code: str) -> str:
        raw_code = str(code or "").strip()
        xt_code = self._format_xt_code(raw_code)
        stock_name = self.name_map.get(raw_code, "") or self.name_map.get(xt_code, "") or self.name_map.get(xt_code.split('.')[0], "")
        if not stock_name and HAS_XTQUANT:
            try:
                stock_name = xtdata.get_stock_name(xt_code)
            except Exception as exc:
                logger.debug(f"Failed to get stock name for {raw_code}: {exc}")
        return stock_name or raw_code or xt_code

    def _resolve_trade_price(self, code: str, ui_price_type: int, input_price: float) -> float:
        if int(ui_price_type) == 0:
            return float(input_price or 0.0)
        xt_code = self._format_xt_code(code)
        if HAS_XTQUANT and xt_code:
            try:
                tick = xtdata.get_full_tick([xt_code])
                if xt_code in tick:
                    current_price = float(tick[xt_code].get('lastPrice', 0) or 0)
                    if current_price > 0:
                        return current_price
            except Exception as exc:
                logger.error(f"获取 {xt_code} 当前价格失败: {exc}")
        return float(input_price or 0.0)

    def append_log(self, message: str, color: str = None):
        """Append message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.log_text.append(formatted_msg)
        
        # Auto scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        logger.info(message)
    
    def clear_log(self):
        """Clear log display"""
        self.log_text.clear()
        self.append_log("日志已清空")
    
    def open_config_dialog(self):
        """Open configuration dialog"""
        dialog = BrokerConfigDialog(self.broker_session_service.get_config(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self.qmt_path = config.get("qmt_path", "")
            self.account = config.get("account", "")
            self.process_name = config.get("process_name", "")
            self.save_config()
            self.append_log("配置已更新")
    
    def connect_broker(self):
        """Connect to broker"""
        if not self.qmt_path or not self.account:
            self.append_log("✗ 错误: 请先点击 ⚙ 配置 QMT路径和资金账号")
            return
        
        # Check if QMT path exists
        if not os.path.exists(self.qmt_path):
            self.append_log(f"✗ 警告: QMT路径不存在 - {self.qmt_path}")
            return
        
        self.connect_btn.setEnabled(False)
        self.status_label.setText("连接中...")
        self.status_label.setStyleSheet("color: #f0ad4e;")
        self._connection_in_progress = True
        if not self.broker_session_service.connect_async(self.qmt_path, self.account):
            self._connection_in_progress = False
            self.connect_btn.setEnabled(True)
    
    def on_connected(self, success: bool, message: str):
        """Handle connection result"""
        if success:
            self.xt_trader = self.broker_session_service.xt_trader
            self.acc = self.broker_session_service.account_obj
            self.is_connected = True
            
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("color: #5cb85c; font-weight: bold;")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            
            # Enable trading buttons
            self.buy_btn.setEnabled(True)
            self.sell_btn.setEnabled(True)
            
            # 设置并启动条件单监控
            self.setup_conditional_order_executor()
            
            # Auto refresh
            self.refresh_all()
        else:
            self.status_label.setText(f"连接失败")
            self.status_label.setStyleSheet("color: #d9534f; font-weight: bold;")
            self.connect_btn.setEnabled(True)
            QMessageBox.warning(self, "连接失败", message)
    
    def disconnect_broker(self):
        """Disconnect from broker"""
        # Stop query threads
        for thread_name, thread in [
            ("查询线程", self.positions_query_thread),
            ("查询线程", self.orders_query_thread),
            ("查询线程", self.trades_query_thread),
            ("交易线程", self.trade_thread),
            ("撤单线程", self.cancel_order_thread)
        ]:
            if thread and thread.isRunning():
                thread.quit()
                thread.wait()
        
        self.xt_trader = None
        self.acc = None
        self.is_connected = False
        
        self.positions_query_thread = None
        self.orders_query_thread = None
        self.trades_query_thread = None
        self.trade_thread = None
        self.cancel_order_thread = None
        
        # Disable trading buttons
        self.buy_btn.setEnabled(False)
        self.sell_btn.setEnabled(False)
        
        self.status_label.setText("未连接")
        self.status_label.setStyleSheet("color: #888;")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        
        self.broker_session_service.disconnect()
        
        self.append_log("已断开连接")
        
        # Clear tables
        self.clear_all_tables()
    
    def clear_all_tables(self):
        """Clear all data tables"""
        self.positions_table.setRowCount(0)
        self.orders_table.setRowCount(0)
        self.trades_table.setRowCount(0)
        
        self.lbl_account_id.setText("-")
        self.lbl_total_assets.setText("-")
        self.lbl_available_cash.setText("-")
        self.lbl_market_value.setText("-")
        self.lbl_profit_loss.setText("-")
        self.lbl_profit_loss.setForeground(QBrush(QColor("#d4d4d4")))
    
    def refresh_account_info(self):
        """Refresh account info"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            return
        
        try:
            # Query account assets
            assets = self.xt_trader.query_stock_asset(self.acc)
            
            if assets:
                self.lbl_account_id.setText(str(assets.account_id))
                self.lbl_total_assets.setText(f"¥{assets.total_asset:,.2f}")
                self.lbl_available_cash.setText(f"¥{assets.cash:,.2f}")
                self.lbl_market_value.setText(f"¥{assets.market_value:,.2f}")
                
                # Calculate profit/loss if available
                if hasattr(assets, 'frozen_cash'):
                    profit = assets.total_asset - assets.cash - assets.market_value
                    color = QColor("#ec0000") if profit >= 0 else QColor("#00da3c")
                    self.lbl_profit_loss.setText(f"¥{profit:,.2f}")
                    self.lbl_profit_loss.setForeground(QBrush(color))
        except Exception as e:
            logger.error(f"查询账户信息失败: {e}")
    
    def refresh_positions(self):
        """Refresh positions"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            return
        
        self.positions_table.setEnabled(False)
        self.positions_query_thread = QueryThread("positions", self.xt_trader, self.acc)
        self.positions_query_thread.finished.connect(self.on_positions_query_finished)
        self.positions_query_thread.start()
    
    def on_positions_query_finished(self, query_type: str, data, error_msg: str):
        """Handle positions query result"""
        if error_msg:
            self.append_log(f"✗ 查询持仓失败: {error_msg}")
            self.positions_table.setEnabled(True)
            return
        
        try:
            self.positions_table.setRowCount(0)
            
            for i, pos in enumerate(data, 1):
                row = self.positions_table.rowCount()
                self.positions_table.insertRow(row)
                
                # 0: 证券代码
                stock_code = str(pos.stock_code)
                self.positions_table.setItem(row, 0, QTableWidgetItem(stock_code))
                
                # 1: 证券名称
                stock_name = self.name_map.get(stock_code)
                if not stock_name and "." in stock_code:
                    base_code = stock_code.split(".")[0]
                    stock_name = self.name_map.get(base_code)
                
                if not stock_name and HAS_XTQUANT:
                    try:
                        stock_name = xtdata.get_stock_name(stock_code)
                    except:
                        pass
                
                if not stock_name:
                    stock_name = getattr(pos, "stock_name", "-")
                
                self.positions_table.setItem(row, 1, QTableWidgetItem(str(stock_name)))
                
                # 2: 持仓数量
                self.positions_table.setItem(row, 2, QTableWidgetItem(str(pos.volume)))
                
                # 3: 可用数量
                self.positions_table.setItem(row, 3, QTableWidgetItem(str(pos.can_use_volume)))
                
                # 4: 成本价
                open_price = pos.open_price
                self.positions_table.setItem(row, 4, QTableWidgetItem(f"{open_price:.3f}"))
                
                # 5: 最新价
                volume = pos.volume
                market_value = pos.market_value
                last_price = market_value / volume if volume > 0 else 0
                self.positions_table.setItem(row, 5, QTableWidgetItem(f"{last_price:.3f}"))
                
                # 6: 市值
                self.positions_table.setItem(row, 6, QTableWidgetItem(f"{market_value:,.2f}"))
                
                # 7: 盈亏 & 8: 盈亏比例
                profit = market_value - (open_price * volume) if volume > 0 else 0
                profit_ratio = (profit / (open_price * volume) * 100) if (volume > 0 and open_price > 0) else 0
                
                profit_item = QTableWidgetItem(f"{profit:,.2f}")
                ratio_item = QTableWidgetItem(f"{profit_ratio:.2f}%")
                
                # 设置红绿颜色
                color = QColor("#ec0000") if profit >= 0 else QColor("#00da3c")
                profit_item.setForeground(QBrush(color))
                ratio_item.setForeground(QBrush(color))
                
                self.positions_table.setItem(row, 7, profit_item)
                self.positions_table.setItem(row, 8, ratio_item)
                
                # 9: 昨日持仓
                self.positions_table.setItem(row, 9, QTableWidgetItem(str(pos.yesterday_volume)))
                
                # 10: 冻结数量
                self.positions_table.setItem(row, 10, QTableWidgetItem(str(pos.frozen_volume)))
                
                # 11: 在途股份
                self.positions_table.setItem(row, 11, QTableWidgetItem(str(pos.on_road_volume)))
            
            self.positions_table.setEnabled(True)
            
            # 提取持仓股票代码列表并发出信号
            # 去掉 .SH/.SZ 等后缀，以便与系统其他部分保持一致
            position_codes = []
            for pos in data:
                stock_code = str(pos.stock_code)
                # 去掉后缀（如 .SH, .SZ, .BJ）
                if '.' in stock_code:
                    stock_code = stock_code.split('.')[0]
                position_codes.append(stock_code)
            
            # 发出持仓更新信号
            self.positionsUpdated.emit(position_codes)
            
        except Exception as e:
            logger.error(f"处理持仓数据失败: {e}")
            logger.error(traceback.format_exc())
            self.positions_table.setEnabled(True)
    
    def refresh_orders(self):
        """Refresh orders"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            return
        
        self.orders_table.setEnabled(False)
        self.orders_query_thread = QueryThread("orders", self.xt_trader, self.acc)
        self.orders_query_thread.finished.connect(self.on_orders_query_finished)
        self.orders_query_thread.start()
    
    def on_orders_query_finished(self, query_type: str, data, error_msg: str):
        """Handle orders query result"""
        if error_msg:
            self.append_log(f"✗ 查询委托失败: {error_msg}")
            self.orders_table.setEnabled(True)
            return
        
        try:
            self.orders_table.setRowCount(0)
            
            for i, order in enumerate(data, 1):
                row = self.orders_table.rowCount()
                self.orders_table.insertRow(row)
                
                self.orders_table.setItem(row, 0, QTableWidgetItem(str(order.order_id)))
                self.orders_table.setItem(row, 1, QTableWidgetItem(str(order.stock_code)))
                
                # Get stock name - try multiple sources
                stock_name = getattr(order, 'stock_name', '')
                if not stock_name or stock_name == '-':
                    # Try from name_map using stock code (without suffix)
                    full_code = str(order.stock_code)
                    base_code = full_code.split('.')[0] if '.' in full_code else full_code
                    stock_name = self.name_map.get(base_code, '')
                    
                    # Try with full code
                    if not stock_name:
                        stock_name = self.name_map.get(full_code, '')
                    
                    # Try using xtdata.get_stock_name as last resort
                    if not stock_name and HAS_XTQUANT:
                        try:
                            xt_code = full_code
                            if '.' not in xt_code:
                                if xt_code.startswith(('5', '6', '9')):
                                    xt_code = f"{xt_code}.SH"
                                else:
                                    xt_code = f"{xt_code}.SZ"
                            stock_name = xtdata.get_stock_name(xt_code)
                        except:
                            pass
                    
                    if not stock_name:
                        stock_name = '-'
                
                self.orders_table.setItem(row, 2, QTableWidgetItem(str(stock_name)))
                
                # Order direction
                direction = "买入" if order.order_type == 23 else "卖出" if order.order_type == 24 else str(order.order_type)
                direction_item = QTableWidgetItem(direction)
                if "买" in direction:
                    direction_item.setForeground(QBrush(QColor("#ec0000")))
                else:
                    direction_item.setForeground(QBrush(QColor("#00da3c")))
                self.orders_table.setItem(row, 3, direction_item)
                
                self.orders_table.setItem(row, 4, QTableWidgetItem(f"{order.price:.3f}"))
                self.orders_table.setItem(row, 5, QTableWidgetItem(str(order.order_volume)))
                self.orders_table.setItem(row, 6, QTableWidgetItem(str(order.traded_volume)))
                
                # Order status
                status_map = {
                    48: "未报",
                    49: "待报",
                    50: "已报",
                    51: "已报待撤",
                    52: "部成待撤",
                    53: "部撤",
                    54: "已撤",
                    55: "部成",
                    56: "已成",
                    57: "废单"
                }
                status = status_map.get(order.order_status, str(order.order_status))
                self.orders_table.setItem(row, 7, QTableWidgetItem(status))
                
                # Order time
                order_time = getattr(order, 'order_time', '-')
                self.orders_table.setItem(row, 8, QTableWidgetItem(str(order_time)))
                
                # Remarks
                remarks = getattr(order, 'status_msg', '-')
                self.orders_table.setItem(row, 9, QTableWidgetItem(str(remarks)))
                
                # Cancel button
                cancellable_statuses = [50, 51, 52, 55]
                cancel_btn = QPushButton("撤销")
                cancel_btn.setMinimumWidth(60)
                cancel_btn.setMinimumHeight(28)
                cancel_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f0ad4e;
                        color: white;
                        border: none;
                        padding: 4px 12px;
                        border-radius: 3px;
                        font-size: 10pt;
                    }
                    QPushButton:hover { background-color: #ec971f; }
                    QPushButton:pressed { background-color: #d58512; }
                    QPushButton:disabled { background-color: #555; color: #888; }
                """)
                
                if order.order_status in cancellable_statuses:
                    cancel_btn.clicked.connect(lambda checked, order_id=order.order_id: self.cancel_order_by_id(order_id))
                    cancel_btn.setEnabled(True)
                else:
                    cancel_btn.setEnabled(False)
                
                self.orders_table.setCellWidget(row, 10, cancel_btn)
            
            self.orders_table.setEnabled(True)
            
            # 自动从已成交的委托同步交易记录
            if data:
                try:
                    added_count = self.trade_record_service.sync_from_orders(
                        data, source="broker_sync", name_map=self.name_map
                    )
                    if added_count > 0:
                        # 通知交易历史界面刷新
                        if hasattr(self, 'trade_history_widget'):
                            self.trade_history_widget.refresh_data()
                except Exception as e:
                    logger.error(f"同步委托成交记录失败: {e}")
                    
        except Exception as e:
            logger.error(f"处理委托数据失败: {e}")
            self.orders_table.setEnabled(True)
    
    def refresh_trades(self):
        """Refresh trades"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            return
        
        self.trades_table.setEnabled(False)
        self.trades_query_thread = QueryThread("trades", self.xt_trader, self.acc)
        self.trades_query_thread.finished.connect(self.on_trades_query_finished)
        self.trades_query_thread.start()
    
    def on_trades_query_finished(self, query_type: str, data, error_msg: str):
        """Handle trades query result"""
        if error_msg:
            self.append_log(f"✗ 查询成交失败: {error_msg}")
            self.trades_table.setEnabled(True)
            return
        
        try:
            
            self.trades_table.setRowCount(0)
            
            for i, trade in enumerate(data, 1):
                row = self.trades_table.rowCount()
                self.trades_table.insertRow(row)
                
                self.trades_table.setItem(row, 0, QTableWidgetItem(str(trade.traded_id)))
                self.trades_table.setItem(row, 1, QTableWidgetItem(str(trade.order_id)))
                self.trades_table.setItem(row, 2, QTableWidgetItem(str(trade.stock_code)))
                
                # Get stock name if available
                stock_name = getattr(trade, 'stock_name', '-')
                self.trades_table.setItem(row, 3, QTableWidgetItem(str(stock_name)))
                
                # Trade direction
                direction = "买入" if trade.order_type == 23 else "卖出" if trade.order_type == 24 else str(trade.order_type)
                direction_item = QTableWidgetItem(direction)
                if "买" in direction:
                    direction_item.setForeground(QBrush(QColor("#ec0000")))
                else:
                    direction_item.setForeground(QBrush(QColor("#00da3c")))
                self.trades_table.setItem(row, 4, direction_item)
                
                self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade.traded_price:.3f}"))
                self.trades_table.setItem(row, 6, QTableWidgetItem(str(trade.traded_volume)))
                self.trades_table.setItem(row, 7, QTableWidgetItem(f"{trade.traded_amount:,.2f}"))
                
                # Trade time
                trade_time = getattr(trade, 'traded_time', '-')
                self.trades_table.setItem(row, 8, QTableWidgetItem(str(trade_time)))
            
            self.trades_table.setEnabled(True)
            
            # 自动同步成交记录到本地数据库（基于 traded_id 去重）
            if data:
                try:
                    added_count = self.trade_record_service.sync_broker_trades(data, source="broker_sync")
                    if added_count > 0:
                        # 通知交易历史界面刷新
                        if hasattr(self, 'trade_history_widget'):
                            self.trade_history_widget.refresh_data()
                except Exception as e:
                    logger.error(f"同步成交记录失败: {e}")
            
        except Exception as e:
            logger.error(f"处理成交数据失败: {e}")
            self.trades_table.setEnabled(True)
    
    def refresh_all(self):
        """Refresh all data"""
        self.refresh_account_info()
        self.refresh_positions()
        self.refresh_orders()
        self.refresh_trades()
        self.append_log("数据已刷新")
    
    def statusBar_message(self, msg: str):
        """Show status bar message"""
        main_window = self.window()
        if hasattr(main_window, 'statusBar'):
            main_window.statusBar().showMessage(msg, 5000)
    
    def export_data(self):
        """Export data to CSV"""
        if not self.is_connected:
            QMessageBox.warning(self, "错误", "请先连接券商")
            return
        
        try:
            import pandas as pd
            
            folder = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if not folder:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export positions
            if self.positions_table.rowCount() > 0:
                positions_data = self._table_to_list(self.positions_table)
                df = pd.DataFrame(positions_data)
                df.to_csv(os.path.join(folder, f"positions_{timestamp}.csv"), index=False, encoding='utf-8-sig')
            
            # Export orders
            if self.orders_table.rowCount() > 0:
                orders_data = self._table_to_list(self.orders_table)
                df = pd.DataFrame(orders_data)
                df.to_csv(os.path.join(folder, f"orders_{timestamp}.csv"), index=False, encoding='utf-8-sig')
            
            # Export trades
            if self.trades_table.rowCount() > 0:
                trades_data = self._table_to_list(self.trades_table)
                df = pd.DataFrame(trades_data)
                df.to_csv(os.path.join(folder, f"trades_{timestamp}.csv"), index=False, encoding='utf-8-sig')
            
            QMessageBox.information(self, "成功", f"数据已导出到 {folder}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败: {e}")
    
    def _table_to_list(self, table: QTableWidget) -> list:
        """Convert table to list of dicts"""
        headers = []
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"Column{col}")
        
        data = []
        for row in range(table.rowCount()):
            row_data = {}
            for col in range(table.columnCount()):
                # Skip widget columns (like buttons)
                widget = table.cellWidget(row, col)
                if widget is not None:
                    continue
                item = table.item(row, col)
                row_data[headers[col]] = item.text() if item else ""
            data.append(row_data)
        
        return data
    
    def on_price_type_changed(self, index: int):
        """Handle price type change"""
        # 0=限价, 1=市价
        if index == 1:  # 市价
            self.trade_price_spin.setEnabled(False)
            self.trade_price_spin.setValue(0.0)
        else:  # 限价
            self.trade_price_spin.setEnabled(True)
        self._update_total_amount()
    
    def _update_total_amount(self):
        """更新委托金额显示（委托价格 × 委托数量）"""
        price = self.trade_price_spin.value()
        volume = self.trade_volume_spin.value()
        total = price * volume
        
        if total >= 10000:
            # 超过1万显示"万"单位
            self.total_amount_label.setText(f"¥ {total/10000:.2f} 万")
        else:
            self.total_amount_label.setText(f"¥ {total:,.2f}")
        
        # 根据金额大小调整颜色
        if total >= 100000:  # 10万以上红色警示
            self.total_amount_label.setStyleSheet("color: #ff4d4d; font-weight: bold; font-size: 14px;")
        elif total >= 10000:  # 1万以上橙色
            self.total_amount_label.setStyleSheet("color: #f0ad4e; font-weight: bold; font-size: 14px;")
        else:
            self.total_amount_label.setStyleSheet("color: #00b894; font-weight: bold; font-size: 14px;")
    
    def on_buy_order(self):
        """Place buy order"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            QMessageBox.warning(self, "错误", "请先连接券商")
            return
        
        stock_code = self.trade_stock_code_edit.text().strip()
        if not stock_code:
            QMessageBox.warning(self, "错误", "请输入股票代码")
            return
        
        order_volume = self.trade_volume_spin.value()
        if order_volume <= 0:
            QMessageBox.warning(self, "错误", "委托数量必须大于0")
            return
        
        price_type = self.price_type_combo.currentIndex()  # 0=限价, 1=市价
        price = self._resolve_trade_price(stock_code, price_type, self.trade_price_spin.value())
        
        if price <= 0:
            QMessageBox.warning(self, "错误", "无法获取有效委托价格")
            return
        
        # 禁用交易按钮
        self.buy_btn.setEnabled(False)
        self.sell_btn.setEnabled(False)
        
        # 创建交易线程
        self.trade_thread = TradeThread(
            stock_code,
            self._resolve_trade_stock_name(stock_code),
            23,  # 23=买入
            order_volume,
            price_type,
            price
        )
        self.trade_thread.finished.connect(self.on_trade_finished)
        self.trade_thread.log_message.connect(self.append_log)
        self.trade_thread.start()
    
    def on_sell_order(self):
        """Place sell order"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            QMessageBox.warning(self, "错误", "请先连接券商")
            return
        
        stock_code = self.trade_stock_code_edit.text().strip()
        if not stock_code:
            QMessageBox.warning(self, "错误", "请输入股票代码")
            return
        
        order_volume = self.trade_volume_spin.value()
        if order_volume <= 0:
            QMessageBox.warning(self, "错误", "委托数量必须大于0")
            return
        
        price_type = self.price_type_combo.currentIndex()  # 0=限价, 1=市价
        price = self._resolve_trade_price(stock_code, price_type, self.trade_price_spin.value())
        
        if price <= 0:
            QMessageBox.warning(self, "错误", "无法获取有效委托价格")
            return
        
        # 禁用交易按钮
        self.buy_btn.setEnabled(False)
        self.sell_btn.setEnabled(False)
        
        # 创建交易线程
        self.trade_thread = TradeThread(
            stock_code,
            self._resolve_trade_stock_name(stock_code),
            24,  # 24=卖出
            order_volume,
            price_type,
            price
        )
        self.trade_thread.finished.connect(self.on_trade_finished)
        self.trade_thread.log_message.connect(self.append_log)
        self.trade_thread.start()
    
    def on_trade_finished(self, success: bool, message: str, order_id: int):
        """Handle trade finished"""
        # 重新启用交易按钮
        if self.is_connected:
            self.buy_btn.setEnabled(True)
            self.sell_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "成功", message)
            # 刷新委托列表和成交列表
            self.refresh_orders()
            # 延迟刷新成交（等待券商系统处理）
            QTimer.singleShot(2000, self.refresh_trades)
        else:
            QMessageBox.warning(self, "失败", message)
    
    def cancel_order_by_id(self, order_id: int):
        """Cancel order by order ID (called from table button)"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            QMessageBox.warning(self, "错误", "请先连接券商")
            return
        
        # 确认撤单
        reply = QMessageBox.question(
            self,
            "确认撤单",
            f"确定要撤销订单 {order_id} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 创建撤单线程
        self.cancel_order_thread = CancelOrderThread(
            self.xt_trader,
            self.acc,
            order_id
        )
        self.cancel_order_thread.finished.connect(self.on_cancel_order_finished)
        self.cancel_order_thread.log_message.connect(self.append_log)
        self.cancel_order_thread.start()
    
    def on_cancel_order_finished(self, success: bool, message: str):
        """Handle cancel order finished"""
        if success:
            # 立即刷新委托列表
            self.append_log("正在刷新委托列表...")
            self.refresh_orders()
            # 延迟显示消息框，确保刷新操作已启动
            QTimer.singleShot(300, lambda: QMessageBox.information(self, "成功", message))
        else:
            QMessageBox.warning(self, "失败", message)
    
    def hideEvent(self, event):
        """Handle hide event"""
        self.order_book_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        """Handle close event"""
        logger.info("关闭交易窗口")
        self.append_log("窗口关闭")
        self.order_book_timer.stop()
        # 条件单监控改为后台持续运行，关闭窗口时不停止
        # self.conditional_order_service.stop_monitoring()
        for signal, slot in (
            (self.conditional_order_service.log_message, self.append_log),
            (self.trade_record_service.log_message, self.append_log),
            (self.broker_session_service.connection_changed, self._on_broker_session_changed),
            (self.broker_session_service.log_message, self.append_log),
            (self.broker_session_service.config_changed, self._on_broker_config_changed),
        ):
            try:
                signal.disconnect(slot)
            except Exception:
                pass
        super().closeEvent(event)
    
    # ==================== 条件单功能 ====================
    
    def on_add_conditional_order(self):
        """添加条件单"""
        stock_code = self.trade_stock_code_edit.text().strip()
        if not stock_code:
            QMessageBox.warning(self, "提示", "请先输入股票代码")
            return
        
        # 获取股票名称
        stock_name = self.name_map.get(stock_code, "")
        if not stock_name and HAS_XTQUANT:
            try:
                xt_code = stock_code
                if '.' not in xt_code:
                    if xt_code.startswith(('5', '6', '9')):
                        xt_code = f"{xt_code}.SH"
                    else:
                        xt_code = f"{xt_code}.SZ"
                stock_name = xtdata.get_stock_name(xt_code)
            except:
                pass
        if not stock_name:
            stock_name = stock_code
        
        # 获取当前价格
        current_price = 0.0
        if HAS_XTQUANT:
            try:
                xt_code = stock_code
                if '.' not in xt_code:
                    if xt_code.startswith(('5', '6', '9')):
                        xt_code = f"{xt_code}.SH"
                    else:
                        xt_code = f"{xt_code}.SZ"
                tick = xtdata.get_full_tick([xt_code])
                if xt_code in tick:
                    current_price = float(tick[xt_code].get('lastPrice', 0))
            except Exception as e:
                logger.error(f"获取当前价格失败: {e}")
        
        # 获取持仓数量、可用数量和成本价（从持仓中查找）
        available_volume = 0
        cost_price = 0.0
        for row in range(self.positions_table.rowCount()):
            pos_code_item = self.positions_table.item(row, 0)
            if pos_code_item:
                pos_code = pos_code_item.text().split('.')[0]
                if pos_code == stock_code:
                    # 持仓数量 (column 2)
                    total_vol_item = self.positions_table.item(row, 2)
                    if total_vol_item:
                        try:
                            total_vol = int(float(total_vol_item.text().replace(',', '')))
                        except:
                            total_vol = 0
                    else:
                        total_vol = 0
                    # 可用数量 (column 3)
                    can_use_item = self.positions_table.item(row, 3)
                    if can_use_item:
                        try:
                            available_volume = int(float(can_use_item.text().replace(',', '')))
                        except:
                            pass
                    # 条件单是未来触发，用持仓数量兜底
                    available_volume = max(available_volume, total_vol)
                    # 成本价 (column 4)
                    cost_item = self.positions_table.item(row, 4)
                    if cost_item:
                        try:
                            cost_price = float(cost_item.text().replace(',', ''))
                        except:
                            pass
                    break
        
        # 获取可用资金（用于买入条件单）
        total_cash = 0.0
        try:
            cash_text = self.lbl_available_cash_item.text().replace(',', '').replace('¥', '')
            if cash_text and cash_text != '-':
                total_cash = float(cash_text)
        except:
            pass
        
        # 打开添加条件单对话框
        dialog = AddConditionalOrderDialog(
            self,
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            available_volume=available_volume,
            cost_price=cost_price,
            total_cash=total_cash
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            order_data = dialog.get_order_data()
            order = self.conditional_order_service.add_order(**order_data)
            self.append_log(f"✓ 已添加条件单: {order.stock_name} {order.condition_display} "
                          f"触发价{order.trigger_price:.3f}")
            # 切换到条件单Tab
            for i in range(self.data_tabs.count()):
                if "条件单" in self.data_tabs.tabText(i):
                    self.data_tabs.setCurrentIndex(i)
                    break
    
    def add_conditional_order_from_position(self, stock_code: str, stock_name: str, 
                                            volume: int, condition_type: str):
        """从持仓快速添加条件单"""
        # 获取当前价格
        current_price = 0.0
        if HAS_XTQUANT:
            try:
                xt_code = stock_code
                if '.' not in xt_code:
                    if xt_code.startswith(('5', '6', '9')):
                        xt_code = f"{xt_code}.SH"
                    else:
                        xt_code = f"{xt_code}.SZ"
                tick = xtdata.get_full_tick([xt_code])
                if xt_code in tick:
                    current_price = float(tick[xt_code].get('lastPrice', 0))
            except:
                pass
        
        # 获取成本价（从持仓表格中查找）
        cost_price = 0.0
        for row in range(self.positions_table.rowCount()):
            pos_code_item = self.positions_table.item(row, 0)
            if pos_code_item:
                pos_code = pos_code_item.text().split('.')[0]
                if pos_code == stock_code:
                    cost_item = self.positions_table.item(row, 4)
                    if cost_item:
                        try:
                            cost_price = float(cost_item.text().replace(',', ''))
                        except:
                            pass
                    break
        
        # 获取可用资金
        total_cash = 0.0
        try:
            cash_text = self.lbl_available_cash_item.text().replace(',', '').replace('¥', '')
            if cash_text and cash_text != '-':
                total_cash = float(cash_text)
        except:
            pass
        
        dialog = AddConditionalOrderDialog(
            self,
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            available_volume=volume,
            cost_price=cost_price,
            total_cash=total_cash
        )
        
        # 预设条件类型
        if condition_type == "take_profit":
            dialog.condition_type_combo.setCurrentIndex(0)
        elif condition_type == "stop_loss":
            dialog.condition_type_combo.setCurrentIndex(1)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            order_data = dialog.get_order_data()
            order = self.conditional_order_service.add_order(**order_data)
            self.append_log(f"✓ 已添加条件单: {order.stock_name} {order.condition_display}")
    
    def batch_create_stop_loss_for_positions(self):
        """为全部持仓批量创建止损单"""
        from trading_app.services.execution.auto_stop_loss_service import get_auto_stop_loss_service
        
        auto_stop_loss_service = get_auto_stop_loss_service()
        
        # 收集所有持仓信息
        positions = []
        for row in range(self.positions_table.rowCount()):
            stock_code_item = self.positions_table.item(row, 0)
            stock_name_item = self.positions_table.item(row, 1)
            can_use_volume_item = self.positions_table.item(row, 3)
            cost_price_item = self.positions_table.item(row, 4)
            
            if not stock_code_item or not cost_price_item:
                continue
            
            stock_code = stock_code_item.text().split('.')[0]
            stock_name = stock_name_item.text() if stock_name_item else stock_code
            
            try:
                can_use_volume = int(float(can_use_volume_item.text().replace(',', ''))) if can_use_volume_item else 0
                cost_price = float(cost_price_item.text().replace(',', ''))
            except:
                continue
            
            if can_use_volume > 0 and cost_price > 0:
                positions.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'volume': can_use_volume,
                    'cost_price': cost_price
                })
        
        if not positions:
            QMessageBox.information(self, "提示", "没有可用持仓")
            return
        
        # 获取当前止损比例配置
        config = auto_stop_loss_service.config
        stop_pct = config.stop_loss_pct
        
        # 确认对话框
        msg = f"将为以下 {len(positions)} 只股票创建止损单：\n\n"
        for pos in positions[:5]:  # 最多显示5个
            stop_price = round(pos['cost_price'] * (1 - stop_pct / 100), 3)
            msg += f"• {pos['stock_name']}({pos['stock_code']}): 成本{pos['cost_price']:.3f} → 止损{stop_price:.3f}\n"
        if len(positions) > 5:
            msg += f"... 共 {len(positions)} 只\n"
        msg += f"\n止损比例: -{stop_pct}%"
        
        reply = QMessageBox.question(
            self, "批量创建止损单", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 批量创建止损单
        created_count = auto_stop_loss_service.batch_create_stop_loss(positions)
        
        if created_count > 0:
            QMessageBox.information(self, "成功", f"已创建 {created_count} 个止损单")
            self.append_log(f"✓ 批量创建止损单: {created_count}个")
            # 切换到条件单标签页
            for i in range(self.data_tabs.count()):
                if "条件单" in self.data_tabs.tabText(i):
                    self.data_tabs.setCurrentIndex(i)
                    break
        else:
            QMessageBox.information(self, "提示", "没有创建新的止损单\n（可能已存在或被豁免）")
    
    def setup_conditional_order_executor(self):
        """设置条件单交易执行器"""
        self.conditional_order_service.set_trade_executor(
            lambda stock_code, order_type, volume, price_type, price: self.trade_execution_service.execute_conditional_order(
                stock_code=stock_code,
                stock_name=stock_code.split('.')[0] if stock_code else "",
                order_type=order_type,
                order_volume=volume,
                price_type=price_type,
                price=price,
                strategy_name="ConditionalOrder",
                remark="条件单自动执行",
            )
        )
        # 监控已在程序启动时由main_window启动，这里不再重复启动
        # self.conditional_order_service.start_monitoring()
        self.append_log("✓ 条件单交易执行器已连接")
    
    def on_conditional_order_triggered(self, order):
        """条件单触发回调"""
        # 刷新委托列表
        QTimer.singleShot(1000, self.refresh_orders)
    
    def on_conditional_order_executed(self, order, success: bool, message: str):
        """条件单执行完成回调"""
        # 不再在这里记录交易，而是通过成交回报同步
        # 刷新委托和持仓（成交记录会在 refresh_trades 中自动同步）
        QTimer.singleShot(500, self.refresh_all)

    def _on_broker_trade_callback(self, trade_data: dict) -> None:
        code = trade_data.get("stock_code", "")
        price = trade_data.get("traded_price", "")
        vol = trade_data.get("traded_volume", "")
        self.append_log(f"📬 成交回报: {code} 价格={price} 数量={vol}")
        QTimer.singleShot(1000, self.refresh_positions)
        QTimer.singleShot(1500, self.refresh_trades)

    def _on_broker_order_callback(self, order_data: dict) -> None:
        code = order_data.get("stock_code", "")
        status = order_data.get("order_status")
        _STATUS_LABELS = {48: "未报", 50: "已报", 51: "部撤", 52: "已报待撤", 54: "已撤", 55: "部成", 56: "已成", 57: "废单"}
        label = _STATUS_LABELS.get(status, str(status))
        self.append_log(f"📋 委托回报: {code} 状态={label}")
        QTimer.singleShot(800, self.refresh_orders)
    
    def update_conditional_order_quotes(self, quotes: dict):
        """更新条件单行情监控"""
        if self.conditional_order_service:
            self.conditional_order_service.update_quotes(quotes)
    
    def on_trade_history_stock_selected(self, stock_code: str):
        """处理交易历史中选择股票的事件"""
        # 自动填充到交易面板
        self.set_stock_code(stock_code)
        # 切换到第一个Tab（交易下单区域可见）
        self.data_tabs.setCurrentIndex(0)
    
    def on_auto_save_pnl_snapshot(self):
        """从券商账户自动获取数据并保存每日盈亏快照"""
        if not self.is_connected or not self.xt_trader or not self.acc:
            QMessageBox.warning(self, "提示", "请先连接券商账户")
            return
        
        try:
            assets = self.xt_trader.query_stock_asset(self.acc)
            if not assets:
                QMessageBox.warning(self, "错误", "获取账户资产信息失败")
                return
            
            total_asset = assets.total_asset
            cash = assets.cash
            market_value = assets.market_value
            
            # Count positions
            position_count = 0
            try:
                positions = self.xt_trader.query_stock_positions(self.acc)
                if positions:
                    position_count = sum(1 for p in positions if p.volume > 0)
            except Exception:
                pass
            
            self.daily_pnl_widget.auto_save_snapshot(
                total_asset=total_asset,
                cash=cash,
                market_value=market_value,
                position_count=position_count
            )
            
        except Exception as e:
            logger.error(f"自动保存每日快照失败: {e}")
            QMessageBox.warning(self, "错误", f"获取账户数据失败: {e}")
