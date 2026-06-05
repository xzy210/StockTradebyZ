# sector_window.py - 热门板块独立窗口
"""
热门板块独立窗口

提供更大的显示空间，支持：
- 板块涨跌幅排行
- 板块成分股详情
- 多种板块类型切换
- 实时数据刷新
"""
from typing import Dict, List, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame, QSplitter,
    QListWidget, QListWidgetItem, QGroupBox, QStatusBar,
    QToolBar, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QBrush, QFont, QIcon

import logging
logger = logging.getLogger(__name__)


def _make_bold_font() -> QFont:
    """创建粗体字体（避免使用空字符串导致Windows字体错误）"""
    font = QFont()
    font.setBold(True)
    return font


# 导入板块服务
try:
    from trading_app.services.market_data.sector_service import (
        get_sector_service, SectorService, SectorData, 
        SECTOR_TYPES, HAS_XTQUANT
    )
except ImportError:
    HAS_XTQUANT = False
    SectorService = None
    SectorData = None


class SectorWindow(QMainWindow):
    """
    热门板块独立窗口
    
    信号：
        stockSelected: 成分股被选中，参数为(股票代码, 股票名称)
    """
    
    stockSelected = pyqtSignal(str, str)  # 股票代码, 股票名称
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._sector_service: Optional[SectorService] = None
        self._is_running = False
        
        self.setWindowTitle("🔥 热门板块")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        
        self._setup_ui()
        self._setup_toolbar()
        self._init_service()
    
    def _setup_ui(self):
        """设置界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建主分割器（左右布局）
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ========== 左侧：板块列表 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # 板块列表标题
        left_header = QHBoxLayout()
        left_title = QLabel("📊 板块排行")
        left_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        left_header.addWidget(left_title)
        left_header.addStretch()
        
        # 板块数量
        self.sector_count_label = QLabel("0 个板块")
        self.sector_count_label.setStyleSheet("color: #888; font-size: 12px;")
        left_header.addWidget(self.sector_count_label)
        
        left_layout.addLayout(left_header)
        
        # 板块表格
        self.sector_table = QTableWidget()
        self.sector_table.setColumnCount(9)
        self.sector_table.setHorizontalHeaderLabels([
            "板块", "热度", "涨跌幅", "换手率", "涨停", "涨", "跌", "成交占比", "领涨股"
        ])
        
        # 设置列宽
        header = self.sector_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 9):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
        self.sector_table.setColumnWidth(1, 55)   # 热度
        self.sector_table.setColumnWidth(2, 70)   # 涨跌幅
        self.sector_table.setColumnWidth(3, 60)   # 换手率
        self.sector_table.setColumnWidth(4, 45)   # 涨停
        self.sector_table.setColumnWidth(5, 40)   # 涨
        self.sector_table.setColumnWidth(6, 40)   # 跌
        self.sector_table.setColumnWidth(7, 65)   # 成交占比
        self.sector_table.setColumnWidth(8, 130)  # 领涨股
        
        self.sector_table.verticalHeader().setVisible(False)
        self.sector_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sector_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sector_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sector_table.setAlternatingRowColors(True)
        self.sector_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #252525;
                gridline-color: #3c3c3c;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #0078d4;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.sector_table.cellClicked.connect(self._on_sector_clicked)
        left_layout.addWidget(self.sector_table)
        
        main_splitter.addWidget(left_widget)
        
        # ========== 右侧：成分股详情 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # 成分股标题
        right_header = QHBoxLayout()
        self.stock_title_label = QLabel("📋 成分股列表")
        self.stock_title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        right_header.addWidget(self.stock_title_label)
        right_header.addStretch()
        
        self.stock_count_label = QLabel("")
        self.stock_count_label.setStyleSheet("color: #888; font-size: 12px;")
        right_header.addWidget(self.stock_count_label)
        
        right_layout.addLayout(right_header)
        
        # 选中的板块信息
        self.sector_info_frame = QFrame()
        self.sector_info_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        info_layout = QHBoxLayout(self.sector_info_frame)
        info_layout.setContentsMargins(10, 8, 10, 8)
        
        self.selected_sector_name = QLabel("请选择板块")
        self.selected_sector_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        info_layout.addWidget(self.selected_sector_name)
        
        info_layout.addStretch()
        
        self.selected_sector_change = QLabel("")
        self.selected_sector_change.setStyleSheet("font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.selected_sector_change)
        
        self.selected_sector_stats = QLabel("")
        self.selected_sector_stats.setStyleSheet("color: #888; font-size: 12px; margin-left: 15px;")
        info_layout.addWidget(self.selected_sector_stats)
        
        right_layout.addWidget(self.sector_info_frame)
        
        # 成分股表格（增强版：含龙头评分）
        self.stock_table = QTableWidget()
        self.stock_table.setColumnCount(7)
        self.stock_table.setHorizontalHeaderLabels([
            "代码", "名称", "龙头分", "涨跌幅", "换手率", "成交额", "标记"
        ])
        
        stock_header = self.stock_table.horizontalHeader()
        stock_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        stock_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        stock_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        stock_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        stock_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        stock_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        stock_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.stock_table.setColumnWidth(0, 65)   # 代码
        self.stock_table.setColumnWidth(2, 60)   # 龙头分
        self.stock_table.setColumnWidth(3, 70)   # 涨跌幅
        self.stock_table.setColumnWidth(4, 60)   # 换手率
        self.stock_table.setColumnWidth(5, 70)   # 成交额
        self.stock_table.setColumnWidth(6, 45)   # 标记
        
        self.stock_table.verticalHeader().setVisible(False)
        self.stock_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.stock_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.stock_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stock_table.setAlternatingRowColors(True)
        self.stock_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #252525;
                gridline-color: #3c3c3c;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #0078d4;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.stock_table.cellDoubleClicked.connect(self._on_stock_double_clicked)
        right_layout.addWidget(self.stock_table)
        
        main_splitter.addWidget(right_widget)
        
        # 设置分割比例
        main_splitter.setSizes([500, 600])
        
        main_layout.addWidget(main_splitter)
        
        # 状态栏
        self.statusBar().showMessage("点击工具栏「启动」按钮开始获取板块数据")
        self.statusBar().setStyleSheet("background-color: #007acc; color: #ffffff;")
        
        # 如果 xtquant 不可用，显示提示
        if not HAS_XTQUANT:
            self.statusBar().showMessage("⚠️ xtquant 未安装，热门板块功能不可用")
            self.statusBar().setStyleSheet("background-color: #e74c3c; color: #ffffff;")
    
    def _setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 8px;
                padding: 8px;
            }
            QToolBar QLabel {
                color: #ffffff;
            }
        """)
        self.addToolBar(toolbar)
        
        # 板块类型选择
        toolbar.addWidget(QLabel("板块类型: "))
        self.type_combo = QComboBox()
        self.type_combo.addItem("🏭 申万一级行业", "sw_l1")
        self.type_combo.addItem("🏢 申万二级行业", "sw_l2")
        self.type_combo.addItem("💡 概念板块", "concept")
        self.type_combo.addItem("🔥 题材板块", "thematic")
        self.type_combo.addItem("🌍 全部板块", "all")
        self.type_combo.setMinimumWidth(160)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        toolbar.addWidget(self.type_combo)
        
        toolbar.addSeparator()
        
        # 刷新间隔
        toolbar.addWidget(QLabel("刷新间隔: "))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(3, 60)
        self.interval_spin.setValue(60)
        self.interval_spin.setSuffix(" 秒")
        self.interval_spin.setFixedWidth(80)
        self.interval_spin.valueChanged.connect(self._on_interval_changed)
        toolbar.addWidget(self.interval_spin)
        
        toolbar.addSeparator()
        
        # 刷新时间
        self.time_label = QLabel("--:--:--")
        self.time_label.setStyleSheet("color: #888; font-size: 12px;")
        toolbar.addWidget(self.time_label)
        
        # 弹性空间
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        spacer.setMinimumWidth(20)
        toolbar.addWidget(spacer)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("🔄 刷新")
        self.refresh_btn.setFixedWidth(80)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        toolbar.addWidget(self.refresh_btn)
        
        # 启动/停止按钮
        self.toggle_btn = QPushButton("▶ 启动")
        self.toggle_btn.setFixedWidth(90)
        self.toggle_btn.setStyleSheet("background-color: #27ae60; font-weight: bold;")
        self.toggle_btn.clicked.connect(self._toggle_service)
        toolbar.addWidget(self.toggle_btn)
        
        if not HAS_XTQUANT:
            self.toggle_btn.setEnabled(False)
            self.refresh_btn.setEnabled(False)
    
    def _init_service(self):
        """初始化板块服务"""
        if not HAS_XTQUANT:
            return
        
        try:
            self._sector_service = get_sector_service()
            self._sector_service.sector_data_updated.connect(self._on_sector_data_updated)
            self._sector_service.connection_status_changed.connect(self._on_status_changed)
        except Exception as e:
            logger.error(f"初始化板块服务失败: {e}")
            self.statusBar().showMessage(f"⚠️ 初始化失败: {e}")
    
    def _toggle_service(self):
        """切换服务状态"""
        if not self._sector_service:
            return
        
        if self._is_running:
            self._sector_service.stop()
            self._is_running = False
            self.toggle_btn.setText("▶ 启动")
            self.toggle_btn.setStyleSheet("background-color: #27ae60; font-weight: bold;")
            self.statusBar().showMessage("已停止")
        else:
            sector_type = self.type_combo.currentData()
            if self._sector_service.start(sector_type):
                self._is_running = True
                self.toggle_btn.setText("⏹ 停止")
                self.toggle_btn.setStyleSheet("background-color: #e74c3c; font-weight: bold;")
                self.statusBar().showMessage("正在获取板块数据...")
            else:
                self.statusBar().showMessage("⚠️ 启动失败，请检查 miniQMT 是否运行")
    
    def _on_type_changed(self, index: int):
        """板块类型切换"""
        sector_type = self.type_combo.currentData()
        
        if self._sector_service and self._is_running:
            self._sector_service.set_sector_type(sector_type)
            self.statusBar().showMessage(f"正在切换到 {self.type_combo.currentText()}...")
    
    def _on_interval_changed(self, value: int):
        """刷新间隔变化"""
        if self._sector_service:
            self._sector_service.set_poll_interval(value * 1000)
    
    def _on_refresh_clicked(self):
        """刷新按钮点击"""
        if self._sector_service and self._is_running:
            self._sector_service.refresh()
            self.time_label.setText("刷新中...")
    
    def _on_sector_data_updated(self, sector_list: List['SectorData']):
        """板块数据更新"""
        self._update_sector_table(sector_list)
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))
        self.sector_count_label.setText(f"{len(sector_list)} 个板块")
        self.statusBar().showMessage(f"已更新 {len(sector_list)} 个板块数据")
    
    def _update_sector_table(self, sector_list: List['SectorData']):
        """更新板块表格"""
        self.sector_table.setRowCount(len(sector_list))
        
        for row, data in enumerate(sector_list):
            col = 0
            
            # 0. 板块名称
            name_item = QTableWidgetItem(data.name)
            name_item.setData(Qt.ItemDataRole.UserRole, data.code)
            name_item.setData(Qt.ItemDataRole.UserRole + 1, data.name)
            name_item.setData(Qt.ItemDataRole.UserRole + 2, data)  # 存储完整数据
            name_item.setFont(_make_bold_font())
            self.sector_table.setItem(row, col, name_item)
            col += 1
            
            # 1. 热度指数（新增）
            hotness_item = QTableWidgetItem(f"{data.hotness_score:.0f}")
            hotness_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # 热度颜色：高热度显示橙色/红色
            if data.hotness_score >= 80:
                hotness_item.setForeground(QBrush(QColor("#ff4d4d")))  # 红色
                hotness_item.setFont(_make_bold_font())
            elif data.hotness_score >= 60:
                hotness_item.setForeground(QBrush(QColor("#ff9500")))  # 橙色
            else:
                hotness_item.setForeground(QBrush(QColor("#888888")))  # 灰色
            self.sector_table.setItem(row, col, hotness_item)
            col += 1
            
            # 2. 涨跌幅
            change_item = QTableWidgetItem(f"{data.change_pct:+.2f}%")
            if data.change_pct > 0:
                change_item.setForeground(QBrush(QColor("#ff4d4d")))
            elif data.change_pct < 0:
                change_item.setForeground(QBrush(QColor("#00b894")))
            else:
                change_item.setForeground(QBrush(QColor("#888888")))
            change_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            change_item.setFont(_make_bold_font())
            self.sector_table.setItem(row, col, change_item)
            col += 1
            
            # 3. 换手率（新增）
            turnover_item = QTableWidgetItem(f"{data.turnover_rate:.1f}%")
            turnover_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if data.turnover_rate >= 5:
                turnover_item.setForeground(QBrush(QColor("#ff9500")))  # 高换手橙色
            else:
                turnover_item.setForeground(QBrush(QColor("#888888")))
            self.sector_table.setItem(row, col, turnover_item)
            col += 1
            
            # 4. 涨停数（新增）
            limit_item = QTableWidgetItem(str(data.limit_up_count))
            limit_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if data.limit_up_count > 0:
                limit_item.setForeground(QBrush(QColor("#ff4d4d")))
                if data.limit_up_count >= 3:
                    limit_item.setFont(_make_bold_font())
            else:
                limit_item.setForeground(QBrush(QColor("#888888")))
            self.sector_table.setItem(row, col, limit_item)
            col += 1
            
            # 5. 涨家数
            rise_item = QTableWidgetItem(str(data.rise_count))
            rise_item.setForeground(QBrush(QColor("#ff4d4d")))
            rise_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sector_table.setItem(row, col, rise_item)
            col += 1
            
            # 6. 跌家数
            fall_item = QTableWidgetItem(str(data.fall_count))
            fall_item.setForeground(QBrush(QColor("#00b894")))
            fall_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sector_table.setItem(row, col, fall_item)
            col += 1
            
            # 7. 成交占比（新增）
            amount_ratio_item = QTableWidgetItem(f"{data.amount_ratio:.1f}%")
            amount_ratio_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if data.amount_ratio >= 5:
                amount_ratio_item.setForeground(QBrush(QColor("#ff9500")))  # 高占比橙色
            else:
                amount_ratio_item.setForeground(QBrush(QColor("#888888")))
            self.sector_table.setItem(row, col, amount_ratio_item)
            col += 1
            
            # 8. 领涨股
            if data.leading_stock_name:
                leading_text = f"{data.leading_stock_name} +{data.leading_change:.1f}%"
            else:
                leading_text = f"{data.leading_stock} +{data.leading_change:.1f}%"
            leading_item = QTableWidgetItem(leading_text)
            if data.leading_change > 0:
                leading_item.setForeground(QBrush(QColor("#ff4d4d")))
            self.sector_table.setItem(row, col, leading_item)
    
    def _on_sector_clicked(self, row: int, column: int):
        """板块被点击"""
        name_item = self.sector_table.item(row, 0)
        if name_item:
            sector_code = name_item.data(Qt.ItemDataRole.UserRole)
            sector_name = name_item.data(Qt.ItemDataRole.UserRole + 1)
            sector_data = name_item.data(Qt.ItemDataRole.UserRole + 2)
            
            # 更新选中板块信息
            self._update_sector_info(sector_data)
            
            # 加载成分股
            self._load_sector_stocks(sector_code, sector_name)
    
    def _update_sector_info(self, data: 'SectorData'):
        """更新选中板块信息"""
        self.selected_sector_name.setText(f"📊 {data.name}")
        
        # 涨跌幅
        if data.change_pct > 0:
            color = "#ff4d4d"
            prefix = "+"
        elif data.change_pct < 0:
            color = "#00b894"
            prefix = ""
        else:
            color = "#888888"
            prefix = ""
        
        self.selected_sector_change.setText(f"{prefix}{data.change_pct:.2f}%")
        self.selected_sector_change.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        
        # 统计信息（增强版）
        limit_info = f"涨停:{data.limit_up_count}" if data.limit_up_count > 0 else ""
        self.selected_sector_stats.setText(
            f"热度:{data.hotness_score:.0f}  涨:{data.rise_count} 跌:{data.fall_count}  "
            f"换手:{data.turnover_rate:.1f}%  占比:{data.amount_ratio:.1f}%  {limit_info}"
        )
    
    def _load_sector_stocks(self, sector_code: str, sector_name: str):
        """
        加载板块成分股（增强版：含龙头评分）
        
        龙头评分公式:
        龙头分 = 涨幅排名分×25% + 成交额排名分×25% + 换手率排名分×20%
               + 涨停加分×15% + 市值适中度×15%
        """
        if not self._sector_service:
            return
        
        self.stock_title_label.setText(f"📋 {sector_name} 成分股")
        self.stock_table.setRowCount(0)
        self.stock_count_label.setText("加载中...")
        
        try:
            stocks = self._sector_service.get_sector_stocks(sector_code)
            
            if not stocks:
                self.stock_count_label.setText("暂无成分股")
                return
            
            # 获取成分股行情
            from xtquant import xtdata
            
            stocks_to_fetch = stocks[:100]  # 限制数量
            tick_data = xtdata.get_full_tick(stocks_to_fetch)
            
            # ========== 收集数据（增强版）==========
            stock_list = []
            for stock_code in stocks_to_fetch:
                if stock_code in tick_data and tick_data[stock_code]:
                    tick = tick_data[stock_code]
                    last_price = float(tick.get('lastPrice') or 0)
                    prev_close = float(tick.get('lastClose') or 0)
                    amount = float(tick.get('amount') or 0)
                    volume = float(tick.get('volume') or 0)
                    
                    change_pct = 0
                    if prev_close > 0 and last_price > 0:
                        change_pct = (last_price - prev_close) / prev_close * 100
                    
                    # 获取股票详情（含流通股本）
                    stock_name = ""
                    float_shares = 0
                    float_value = 0
                    turnover_rate = 0
                    
                    try:
                        detail = xtdata.get_instrument_detail(stock_code)
                        if detail:
                            stock_name = detail.get('InstrumentName', '')
                            float_shares = float(detail.get('FloatVolume') or 0)
                            if float_shares > 0:
                                turnover_rate = volume / float_shares * 100
                                float_value = float_shares * last_price
                    except:
                        pass
                    
                    # 判断涨停（主板10%，科创板/创业板20%）
                    is_kcb_cyb = stock_code.startswith(('688', '300', '301'))
                    limit_pct = 20.0 if is_kcb_cyb else 10.0
                    is_limit_up = change_pct >= limit_pct - 0.5
                    
                    simple_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
                    stock_list.append({
                        'code': simple_code,
                        'full_code': stock_code,
                        'name': stock_name,
                        'price': last_price,
                        'change_pct': change_pct,
                        'amount': amount,
                        'turnover_rate': turnover_rate,
                        'float_value': float_value,
                        'is_limit_up': is_limit_up,
                    })
            
            # ========== 计算排名得分 ==========
            n = len(stock_list)
            if n > 0:
                # 涨幅排名（越高越好）
                sorted_by_change = sorted(stock_list, key=lambda x: x['change_pct'], reverse=True)
                for i, s in enumerate(sorted_by_change):
                    s['change_rank_score'] = (n - i) / n * 100
                
                # 成交额排名（越高越好）
                sorted_by_amount = sorted(stock_list, key=lambda x: x['amount'], reverse=True)
                for i, s in enumerate(sorted_by_amount):
                    s['amount_rank_score'] = (n - i) / n * 100
                
                # 换手率排名（越高越好）
                sorted_by_turnover = sorted(stock_list, key=lambda x: x['turnover_rate'], reverse=True)
                for i, s in enumerate(sorted_by_turnover):
                    s['turnover_rank_score'] = (n - i) / n * 100
            
            # ========== 计算龙头评分 ==========
            for stock in stock_list:
                score = 0.0
                
                # 1. 涨幅得分 × 25%
                score += stock.get('change_rank_score', 0) * 0.25
                
                # 2. 成交额得分 × 25%
                score += stock.get('amount_rank_score', 0) * 0.25
                
                # 3. 换手率得分 × 20%
                score += stock.get('turnover_rank_score', 0) * 0.20
                
                # 4. 涨停加分 × 15%（涨停=100分，非涨停=0分）
                score += (100 if stock.get('is_limit_up') else 0) * 0.15
                
                # 5. 市值适中度 × 15%（30-300亿最优）
                fv = stock.get('float_value', 0)
                if 30e8 <= fv <= 300e8:
                    market_score = 100  # 最优区间
                elif 10e8 <= fv < 30e8 or 300e8 < fv <= 500e8:
                    market_score = 70   # 次优区间
                elif fv > 0:
                    market_score = 40   # 有数据但不在优选区间
                else:
                    market_score = 50   # 无数据，给中等分
                score += market_score * 0.15
                
                stock['leader_score'] = round(score, 1)
            
            # 按龙头评分排序（而非单纯涨跌幅）
            stock_list.sort(key=lambda x: x['leader_score'], reverse=True)
            
            # ========== 更新表格 ==========
            self.stock_table.setRowCount(len(stock_list))
            
            for row, stock in enumerate(stock_list):
                col = 0
                
                # 0. 代码
                code_item = QTableWidgetItem(stock['code'])
                code_item.setData(Qt.ItemDataRole.UserRole, stock['code'])
                code_item.setData(Qt.ItemDataRole.UserRole + 1, stock['name'])
                self.stock_table.setItem(row, col, code_item)
                col += 1
                
                # 1. 名称
                name_item = QTableWidgetItem(stock['name'])
                # 涨停股名称加粗
                if stock.get('is_limit_up'):
                    name_item.setFont(_make_bold_font())
                    name_item.setForeground(QBrush(QColor("#ff4d4d")))
                self.stock_table.setItem(row, col, name_item)
                col += 1
                
                # 2. 龙头分（新增）
                leader_score = stock.get('leader_score', 0)
                leader_item = QTableWidgetItem(f"{leader_score:.0f}")
                leader_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # 高分高亮
                if leader_score >= 80:
                    leader_item.setForeground(QBrush(QColor("#ff4d4d")))  # 红色
                    leader_item.setFont(_make_bold_font())
                elif leader_score >= 60:
                    leader_item.setForeground(QBrush(QColor("#ff9500")))  # 橙色
                else:
                    leader_item.setForeground(QBrush(QColor("#888888")))
                self.stock_table.setItem(row, col, leader_item)
                col += 1
                
                # 3. 涨跌幅
                change_text = f"{stock['change_pct']:+.2f}%"
                change_item = QTableWidgetItem(change_text)
                change_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if stock['change_pct'] > 0:
                    change_item.setForeground(QBrush(QColor("#ff4d4d")))
                elif stock['change_pct'] < 0:
                    change_item.setForeground(QBrush(QColor("#00b894")))
                else:
                    change_item.setForeground(QBrush(QColor("#888888")))
                if stock.get('is_limit_up'):
                    change_item.setFont(_make_bold_font())
                self.stock_table.setItem(row, col, change_item)
                col += 1
                
                # 4. 换手率（新增）
                turnover_text = f"{stock['turnover_rate']:.1f}%" if stock['turnover_rate'] > 0 else "--"
                turnover_item = QTableWidgetItem(turnover_text)
                turnover_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if stock['turnover_rate'] >= 10:
                    turnover_item.setForeground(QBrush(QColor("#ff9500")))  # 高换手橙色
                elif stock['turnover_rate'] >= 5:
                    turnover_item.setForeground(QBrush(QColor("#f1c40f")))  # 中等黄色
                else:
                    turnover_item.setForeground(QBrush(QColor("#888888")))
                self.stock_table.setItem(row, col, turnover_item)
                col += 1
                
                # 5. 成交额
                if stock['amount'] >= 100000000:
                    amount_text = f"{stock['amount'] / 100000000:.1f}亿"
                elif stock['amount'] >= 10000:
                    amount_text = f"{stock['amount'] / 10000:.0f}万"
                else:
                    amount_text = f"{stock['amount']:.0f}"
                amount_item = QTableWidgetItem(amount_text)
                amount_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                amount_item.setForeground(QBrush(QColor("#888888")))
                self.stock_table.setItem(row, col, amount_item)
                col += 1
                
                # 6. 标记（涨停/龙头）
                marks = []
                if stock.get('is_limit_up'):
                    marks.append("🔴")  # 涨停
                if row == 0:
                    marks.append("👑")  # 第一名，龙头
                elif row <= 2:
                    marks.append("⭐")  # 前三名
                mark_text = "".join(marks) if marks else ""
                mark_item = QTableWidgetItem(mark_text)
                mark_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stock_table.setItem(row, col, mark_item)
            
            # 统计龙头信息
            limit_up_count = sum(1 for s in stock_list if s.get('is_limit_up'))
            top_stock = stock_list[0] if stock_list else None
            if top_stock:
                self.stock_count_label.setText(
                    f"{len(stock_list)}只 | 涨停{limit_up_count} | 龙头: {top_stock['name']}({top_stock['leader_score']:.0f}分)"
                )
            else:
                self.stock_count_label.setText(f"{len(stock_list)} 只股票")
            
        except Exception as e:
            logger.error(f"加载成分股失败: {e}")
            self.stock_count_label.setText(f"加载失败: {e}")
    
    def _on_stock_double_clicked(self, row: int, column: int):
        """成分股双击"""
        code_item = self.stock_table.item(row, 0)
        if code_item:
            code = code_item.data(Qt.ItemDataRole.UserRole)
            name = code_item.data(Qt.ItemDataRole.UserRole + 1) or ""
            self.stockSelected.emit(code, name)
    
    def _on_status_changed(self, connected: bool, message: str):
        """服务状态变化"""
        if connected:
            self.statusBar().showMessage(message)
            self.statusBar().setStyleSheet("background-color: #007acc; color: #ffffff;")
        else:
            self.statusBar().showMessage(message)
            if "失败" in message or "错误" in message:
                self.statusBar().setStyleSheet("background-color: #e74c3c; color: #ffffff;")
    
    def start_service(self):
        """启动服务（外部调用）"""
        if not self._is_running and self._sector_service:
            self._toggle_service()
    
    def stop_service(self):
        """停止服务"""
        if self._is_running and self._sector_service:
            self._toggle_service()
    
    def closeEvent(self, event):
        """关闭事件"""
        if self._sector_service and self._is_running:
            self._sector_service.stop()
            self._is_running = False
        super().closeEvent(event)

