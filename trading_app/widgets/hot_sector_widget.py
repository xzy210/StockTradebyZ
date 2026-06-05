# hot_sector_widget.py - 热门板块组件
"""
热门板块显示组件

功能：
- 显示板块涨跌幅排行
- 支持申万行业、概念板块等多种类型
- 点击板块查看成分股
- 实时刷新板块数据
"""
from typing import Dict, List, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame, QProgressBar,
    QMenu, QSplitter, QListWidget, QListWidgetItem, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QBrush, QFont, QAction

import logging
logger = logging.getLogger(__name__)

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


class SectorCard(QFrame):
    """板块卡片组件"""
    
    clicked = pyqtSignal(str)  # 板块名称
    
    def __init__(self, sector_data: 'SectorData' = None, parent=None):
        super().__init__(parent)
        self.sector_data = sector_data
        self._setup_ui()
        if sector_data:
            self.update_data(sector_data)
    
    def _setup_ui(self):
        """设置UI"""
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            SectorCard {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
            }
            SectorCard:hover {
                border-color: #0078d4;
                background-color: #353535;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(70)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        
        # 第一行：板块名称 + 涨跌幅
        top_layout = QHBoxLayout()
        
        self.name_label = QLabel("--")
        self.name_label.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold;")
        top_layout.addWidget(self.name_label)
        
        top_layout.addStretch()
        
        self.change_label = QLabel("0.00%")
        self.change_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_layout.addWidget(self.change_label)
        
        layout.addLayout(top_layout)
        
        # 第二行：涨跌家数 + 领涨股
        bottom_layout = QHBoxLayout()
        
        self.count_label = QLabel("↑0 ↓0")
        self.count_label.setStyleSheet("color: #888; font-size: 11px;")
        bottom_layout.addWidget(self.count_label)
        
        bottom_layout.addStretch()
        
        self.leading_label = QLabel("")
        self.leading_label.setStyleSheet("color: #888; font-size: 11px;")
        bottom_layout.addWidget(self.leading_label)
        
        layout.addLayout(bottom_layout)
    
    def update_data(self, data: 'SectorData'):
        """更新数据"""
        self.sector_data = data
        
        # 板块名称（已经是友好的显示名称）
        self.name_label.setText(data.name)
        
        # 涨跌幅
        change_pct = data.change_pct
        if change_pct > 0:
            color = "#ff4d4d"  # 红色
            prefix = "+"
        elif change_pct < 0:
            color = "#00b894"  # 绿色
            prefix = ""
        else:
            color = "#888888"
            prefix = ""
        
        self.change_label.setText(f"{prefix}{change_pct:.2f}%")
        self.change_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        
        # 涨跌家数
        rise_color = "#ff4d4d"
        fall_color = "#00b894"
        self.count_label.setText(
            f"<span style='color:{rise_color}'>↑{data.rise_count}</span> "
            f"<span style='color:{fall_color}'>↓{data.fall_count}</span>"
        )
        
        # 领涨股
        if data.leading_stock:
            leading_text = f"{data.leading_stock}"
            if data.leading_change > 0:
                leading_text += f" +{data.leading_change:.1f}%"
            self.leading_label.setText(leading_text)
        else:
            self.leading_label.setText("")
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.sector_data:
            self.clicked.emit(self.sector_data.name)
        super().mousePressEvent(event)


class HotSectorWidget(QWidget):
    """
    热门板块组件
    
    显示板块涨跌幅排行，支持多种板块类型切换。
    
    信号：
        sectorSelected: 板块被选中，参数为板块名称
        stockSelected: 成分股被选中，参数为(股票代码, 股票名称)
    """
    
    sectorSelected = pyqtSignal(str)  # 板块名称
    stockSelected = pyqtSignal(str, str)  # 股票代码, 股票名称
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._sector_service: Optional[SectorService] = None
        self._sector_cards: Dict[str, SectorCard] = {}
        self._current_sector_type = "sw_l1"
        self._is_running = False
        
        self._setup_ui()
        self._init_service()
    
    def _setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 板块类型选择
        self.type_combo = QComboBox()
        self.type_combo.addItem("🏭 申万一级行业", "sw_l1")
        self.type_combo.addItem("🏢 申万二级行业", "sw_l2")
        self.type_combo.addItem("💡 概念板块", "concept")
        self.type_combo.addItem("🔥 题材板块", "thematic")
        self.type_combo.addItem("🌍 全部板块", "all")
        self.type_combo.setMinimumWidth(140)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        control_layout.addWidget(self.type_combo)
        
        control_layout.addStretch()
        
        # 刷新时间
        self.time_label = QLabel("--:--:--")
        self.time_label.setStyleSheet("color: #888; font-size: 11px;")
        control_layout.addWidget(self.time_label)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedSize(28, 28)
        self.refresh_btn.setToolTip("刷新数据")
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        control_layout.addWidget(self.refresh_btn)
        
        # 启动/停止按钮
        self.toggle_btn = QPushButton("▶ 启动")
        self.toggle_btn.setFixedWidth(70)
        self.toggle_btn.clicked.connect(self._toggle_service)
        control_layout.addWidget(self.toggle_btn)
        
        layout.addLayout(control_layout)
        
        # 状态提示
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 上半部分：板块列表（表格形式）
        self.sector_table = QTableWidget()
        self.sector_table.setColumnCount(5)
        self.sector_table.setHorizontalHeaderLabels(["板块", "涨跌幅", "涨↑", "跌↓", "领涨股"])
        self.sector_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sector_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.sector_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.sector_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.sector_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.sector_table.setColumnWidth(1, 70)
        self.sector_table.setColumnWidth(2, 45)
        self.sector_table.setColumnWidth(3, 45)
        self.sector_table.setColumnWidth(4, 100)
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
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #3c3c3c;
                font-weight: bold;
            }
        """)
        self.sector_table.cellClicked.connect(self._on_sector_clicked)
        splitter.addWidget(self.sector_table)
        
        # 下半部分：成分股列表
        stock_group = QGroupBox("成分股")
        stock_layout = QVBoxLayout(stock_group)
        stock_layout.setContentsMargins(5, 10, 5, 5)
        
        self.selected_sector_label = QLabel("请选择板块")
        self.selected_sector_label.setStyleSheet("color: #888; font-size: 11px;")
        stock_layout.addWidget(self.selected_sector_label)
        
        self.stock_list = QListWidget()
        self.stock_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #2d2d2d;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #353535;
            }
        """)
        self.stock_list.itemDoubleClicked.connect(self._on_stock_double_clicked)
        stock_layout.addWidget(self.stock_list)
        
        splitter.addWidget(stock_group)
        
        # 设置分割比例
        splitter.setSizes([300, 200])
        
        layout.addWidget(splitter)
        
        # 检查 xtquant 是否可用
        if not HAS_XTQUANT:
            self._show_status("⚠️ xtquant 未安装，热门板块功能不可用", error=True)
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
            self._show_status(f"⚠️ 初始化失败: {e}", error=True)
    
    def _show_status(self, message: str, error: bool = False):
        """显示状态信息"""
        self.status_label.setText(message)
        if error:
            self.status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        else:
            self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setVisible(True)
    
    def _toggle_service(self):
        """切换服务状态"""
        if not self._sector_service:
            return
        
        if self._is_running:
            self._sector_service.stop()
            self._is_running = False
            self.toggle_btn.setText("▶ 启动")
            self.toggle_btn.setStyleSheet("")
            self._show_status("已停止")
        else:
            sector_type = self.type_combo.currentData()
            if self._sector_service.start(sector_type):
                self._is_running = True
                self.toggle_btn.setText("⏹ 停止")
                self.toggle_btn.setStyleSheet("background-color: #e74c3c;")
                self.status_label.setVisible(False)
            else:
                self._show_status("⚠️ 启动失败，请检查 miniQMT 是否运行", error=True)
    
    def _on_type_changed(self, index: int):
        """板块类型切换"""
        sector_type = self.type_combo.currentData()
        self._current_sector_type = sector_type
        
        if self._sector_service and self._is_running:
            self._sector_service.set_sector_type(sector_type)
    
    def _on_refresh_clicked(self):
        """刷新按钮点击"""
        if self._sector_service and self._is_running:
            self._sector_service.refresh()
            self.time_label.setText("刷新中...")
    
    def _on_sector_data_updated(self, sector_list: List['SectorData']):
        """板块数据更新"""
        self._update_table(sector_list)
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))
    
    def _update_table(self, sector_list: List['SectorData']):
        """更新表格数据"""
        self.sector_table.setRowCount(len(sector_list))
        
        for row, data in enumerate(sector_list):
            # 板块名称（已经是友好的显示名称）
            name_item = QTableWidgetItem(data.name)
            name_item.setData(Qt.ItemDataRole.UserRole, data.code)  # 存储原始代码用于查询成分股
            name_item.setData(Qt.ItemDataRole.UserRole + 1, data.name)  # 存储显示名称
            self.sector_table.setItem(row, 0, name_item)
            
            # 涨跌幅
            change_item = QTableWidgetItem(f"{data.change_pct:+.2f}%")
            if data.change_pct > 0:
                change_item.setForeground(QBrush(QColor("#ff4d4d")))
            elif data.change_pct < 0:
                change_item.setForeground(QBrush(QColor("#00b894")))
            else:
                change_item.setForeground(QBrush(QColor("#888888")))
            change_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.sector_table.setItem(row, 1, change_item)
            
            # 涨家数
            rise_item = QTableWidgetItem(str(data.rise_count))
            rise_item.setForeground(QBrush(QColor("#ff4d4d")))
            rise_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sector_table.setItem(row, 2, rise_item)
            
            # 跌家数
            fall_item = QTableWidgetItem(str(data.fall_count))
            fall_item.setForeground(QBrush(QColor("#00b894")))
            fall_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sector_table.setItem(row, 3, fall_item)
            
            # 领涨股（显示股票名称或代码）
            if data.leading_stock_name:
                leading_text = f"{data.leading_stock_name}"
            else:
                leading_text = data.leading_stock
            if data.leading_change > 0:
                leading_text += f" +{data.leading_change:.1f}%"
            leading_item = QTableWidgetItem(leading_text)
            if data.leading_change > 0:
                leading_item.setForeground(QBrush(QColor("#ff4d4d")))
            elif data.leading_change < 0:
                leading_item.setForeground(QBrush(QColor("#00b894")))
            self.sector_table.setItem(row, 4, leading_item)
    
    def _on_sector_clicked(self, row: int, column: int):
        """板块被点击"""
        name_item = self.sector_table.item(row, 0)
        if name_item:
            sector_code = name_item.data(Qt.ItemDataRole.UserRole)  # 原始代码
            display_name = name_item.data(Qt.ItemDataRole.UserRole + 1)  # 显示名称
            self._load_sector_stocks(sector_code, display_name)
            self.sectorSelected.emit(display_name)
    
    def _load_sector_stocks(self, sector_code: str, display_name: str = None):
        """加载板块成分股"""
        if not self._sector_service:
            return
        
        if display_name is None:
            display_name = self._sector_service.get_sector_display_name(sector_code)
        
        self.selected_sector_label.setText(f"📊 {display_name} 成分股")
        self.stock_list.clear()
        
        try:
            stocks = self._sector_service.get_sector_stocks(sector_code)
            
            if not stocks:
                self.stock_list.addItem("暂无成分股数据")
                return
            
            # 获取成分股行情
            from xtquant import xtdata
            
            # 限制数量
            stocks_to_fetch = stocks[:50]
            tick_data = xtdata.get_full_tick(stocks_to_fetch)
            
            # 按涨跌幅排序
            stock_changes = []
            for stock_code in stocks_to_fetch:
                if stock_code in tick_data and tick_data[stock_code]:
                    tick = tick_data[stock_code]
                    last_price = float(tick.get('lastPrice') or 0)
                    prev_close = float(tick.get('lastClose') or 0)
                    change_pct = 0
                    if prev_close > 0 and last_price > 0:
                        change_pct = (last_price - prev_close) / prev_close * 100
                    
                    # 获取股票名称
                    stock_name = ""
                    try:
                        detail = xtdata.get_instrument_detail(stock_code)
                        if detail:
                            stock_name = detail.get('InstrumentName', '')
                    except:
                        pass
                    
                    stock_changes.append((stock_code, stock_name, change_pct, last_price))
                else:
                    stock_changes.append((stock_code, "", 0, 0))
            
            # 按涨跌幅排序
            stock_changes.sort(key=lambda x: x[2], reverse=True)
            
            # 添加到列表
            for stock_code, stock_name, change_pct, last_price in stock_changes:
                simple_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
                
                # 格式化显示：代码 名称 价格 涨跌幅
                if stock_name:
                    display_text = f"{simple_code} {stock_name}"
                else:
                    display_text = simple_code
                
                if last_price > 0:
                    display_text += f"  {last_price:.2f}"
                
                if change_pct > 0:
                    display_text += f"  +{change_pct:.2f}%"
                    color = "#ff4d4d"
                elif change_pct < 0:
                    display_text += f"  {change_pct:.2f}%"
                    color = "#00b894"
                else:
                    display_text += f"  {change_pct:.2f}%"
                    color = "#888888"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, simple_code)
                item.setData(Qt.ItemDataRole.UserRole + 1, stock_name)  # 存储名称
                item.setForeground(QBrush(QColor(color)))
                self.stock_list.addItem(item)
                
        except Exception as e:
            logger.error(f"加载成分股失败: {e}")
            self.stock_list.addItem(f"加载失败: {e}")
    
    def _on_stock_double_clicked(self, item: QListWidgetItem):
        """成分股双击"""
        code = item.data(Qt.ItemDataRole.UserRole)
        name = item.data(Qt.ItemDataRole.UserRole + 1) or ""
        if code:
            self.stockSelected.emit(code, name)
    
    def _on_status_changed(self, connected: bool, message: str):
        """服务状态变化"""
        if connected:
            self.status_label.setVisible(False)
        else:
            self._show_status(message, error=True)
    
    def start_service(self):
        """启动服务（外部调用）"""
        if not self._is_running and self._sector_service:
            self._toggle_service()
    
    def stop_service(self):
        """停止服务（外部调用）"""
        if self._is_running and self._sector_service:
            self._toggle_service()
    
    def closeEvent(self, event):
        """关闭事件"""
        if self._sector_service and self._is_running:
            self._sector_service.stop()
        super().closeEvent(event)

