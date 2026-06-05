# kline_widget.py - K线图组件
"""
基于 pyqtgraph 的高性能K线图组件

支持实时行情动态更新当日K线
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime, date, time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QCheckBox, QComboBox, QGroupBox, QMenu, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPen, QBrush, QAction, QFont

# 导入分时图窗口
from .timeshare_widget import TimeShareWindow
# 导入绘图工具
from .drawing_tools import DrawingManager

import pyqtgraph as pg

# 导入实时行情服务
try:
    from ..services.quote_service import get_quote_service, QuoteData, to_xt_code
except ImportError:
    from trading_app.services.market_data.quote_service import get_quote_service, QuoteData, to_xt_code

try:
    from ..services.trade_record_service import get_trade_record_service
except ImportError:
    from trading_app.services.execution.trade_record_service import get_trade_record_service

# 配置 pyqtgraph
pg.setConfigOptions(antialias=True)


def is_trading_time() -> bool:
    """
    判断当前是否在A股交易时间内
    
    交易时间：周一至周五 9:15 - 15:00（含集合竞价）
    
    Returns:
        是否在交易时间内
    """
    now = datetime.now()
    
    # 周末不是交易日
    if now.weekday() >= 5:  # 5=周六, 6=周日
        return False
    
    current_time = now.time()
    
    # 交易时间：9:15 - 11:30, 13:00 - 15:00
    morning_start = time(9, 15)
    morning_end = time(11, 30)
    afternoon_start = time(13, 0)
    afternoon_end = time(15, 0)
    
    if morning_start <= current_time <= morning_end:
        return True
    if afternoon_start <= current_time <= afternoon_end:
        return True
    
    return False


def should_update_realtime_kline(last_data_date: date) -> bool:
    """
    判断是否应该更新实时K线
    
    逻辑：
    1. 周末不更新
    2. 如果历史数据已经是今天的，且已收盘（15:00后），不需要实时更新
    3. 工作日的交易时间内（含午休时间），允许更新当天K线
    
    Args:
        last_data_date: 历史数据的最后一天日期
    
    Returns:
        是否应该更新实时K线
    """
    today = date.today()
    now = datetime.now()
    current_time = now.time()
    
    # 周末不更新
    if now.weekday() >= 5:
        return False
    
    # 定义时间边界
    market_open = time(9, 15)   # 集合竞价开始
    market_close = time(15, 0)  # 收盘时间
    
    # 如果历史数据已经是今天的，且已收盘（15:00后），不需要实时更新
    if last_data_date == today and current_time > market_close:
        return False
    
    # 工作日的开盘时间范围内（9:15-15:00，含午休），允许更新
    # 这样午休时间也能显示当天数据
    if market_open <= current_time <= market_close:
        return True
    
    # 盘前盘后不更新（9:15前或15:00后）
    return False


class CandlestickItem(pg.GraphicsObject):
    """蜡烛图图形项"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        up_color: str = "#ec0000",
        down_color: str = "#00da3c"
    ):
        """
        Args:
            data: 包含 open/high/low/close 的DataFrame（数值索引）
            up_color: 上涨颜色
            down_color: 下跌颜色
        """
        super().__init__()
        self.data = data
        self.up_color = QColor(up_color)
        self.down_color = QColor(down_color)
        self.picture = None
        self.generatePicture()
    
    def generatePicture(self):
        """预渲染K线图形"""
        from PyQt6.QtGui import QPicture, QPainter
        
        self.picture = QPicture()
        painter = QPainter(self.picture)
        
        w = 0.3  # K线宽度的一半（减小宽度避免重叠）
        
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            
            if c >= o:
                color = self.up_color
            else:
                color = self.down_color
            
            pen = QPen(color)
            pen.setCosmetic(True)  # 使用装饰笔，宽度为像素单位
            pen.setWidthF(1.0)     # 1 像素宽
            pen.setCapStyle(Qt.PenCapStyle.FlatCap) # 使用平头端点，避免线条比矩形宽
            painter.setPen(pen)
            painter.setBrush(QBrush(color))
            
            # 绘制上下影线
            # 使用 int 坐标，避免浮点数坐标在 cosmetic 模式下的潜在问题
            # 注意：pg.Point 只是 QPointF 的别名，这里我们显式使用 QPointF
            painter.drawLine(
                pg.QtCore.QPointF(float(i), float(l)),
                pg.QtCore.QPointF(float(i), float(h))
            )
            
            # 绘制实体
            body_height = abs(c - o)
            
            if body_height < 0.001:
                # 一字板：画线
                # FlatCap 确保线条长度与矩形宽度一致
                painter.drawLine(
                    pg.QtCore.QPointF(float(i - w), float(c)),
                    pg.QtCore.QPointF(float(i + w), float(c))
                )
            else:
                # 有实体：画矩形（无边框）
                body_top = min(o, c)
                rect = pg.QtCore.QRectF(
                    float(i - w),
                    float(body_top),
                    float(w * 2),
                    float(body_height)
                )
                painter.setPen(Qt.PenStyle.NoPen)  # 禁用边框
                painter.fillRect(rect, QBrush(color))
                painter.setPen(pen)  # 恢复画笔
        
        painter.end()
    
    def paint(self, painter, *args):
        if self.picture:
            self.picture.play(painter)
    
    def boundingRect(self):
        if self.picture is None:
            return pg.QtCore.QRectF()
        return pg.QtCore.QRectF(self.picture.boundingRect())


class VolumeBarItem(pg.BarGraphItem):
    """成交量柱状图项"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        up_color: str = "#ec0000",
        down_color: str = "#00da3c"
    ):
        """
        Args:
            data: 包含 open/close/volume 的DataFrame
            up_color: 上涨颜色
            down_color: 下跌颜色
        """
        x = np.arange(len(data))
        heights = data["volume"].values
        
        # 根据涨跌设置颜色
        brushes = []
        for i in range(len(data)):
            if data.iloc[i]["close"] >= data.iloc[i]["open"]:
                brushes.append(pg.mkBrush(up_color))
            else:
                brushes.append(pg.mkBrush(down_color))
        
        super().__init__(x=x, height=heights, width=0.6, brushes=brushes)


class KLineWidget(QWidget):
    """K线图组件 - 支持实时行情动态更新当日K线"""
    
    # 信号：十字光标位置变化
    crosshairMoved = pyqtSignal(int, dict)  # 索引, 数据字典
    # 信号：实时行情状态变化
    realtimeStatusChanged = pyqtSignal(bool, str)  # enabled, message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._quote_owner_id = f"kline:{id(self)}"
        
        # 设置焦点策略，以便接收键盘事件
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.data: Optional[pd.DataFrame] = None
        self.stock_code = ""
        self.stock_name = ""
        
        # 颜色配置
        self.up_color = "#ec0000"
        self.down_color = "#00da3c"
        self.ma_colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
        
        # 设置
        self.ma_windows = [5, 10, 20]
        self.show_volume = True
        self.show_macd = True
        self.show_kdj = False
        
        self.last_click_idx = -1
        
        # ========== 实时行情相关 ==========
        self._realtime_enabled = False
        self._quote_service = None
        self._today_date = date.today()
        self._today_bar_item = None  # 当日K线图形项（用于单独更新）
        self._today_vol_item = None  # 当日成交量柱状图
        self._is_index = False  # 是否为指数
        
        self.setupUI()
        
        # 初始化绘图管理器
        self.drawing_manager = DrawingManager(self)
        # 将绘图工具栏插入到布局顶部
        self.layout().insertWidget(0, self.drawing_manager.toolbar)
        
        # 浮动价格标签
        self.price_label = QLabel(self)
        self.price_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                color: #ffffff;
                padding: 2px;
                font-family: Arial;
                font-size: 10px;
                border: 1px solid #555555;
            }
        """)
        self.price_label.hide()

    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key.Key_Delete:
            if hasattr(self, 'drawing_manager'):
                self.drawing_manager.delete_selection()
        else:
            super().keyPressEvent(event)

    def setupUI(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # 信息栏
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                color: #ffffff;
                padding: 5px 10px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        # 设置大小策略，防止文本变化导致布局抖动
        self.info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.info_label.setMinimumWidth(1)
        
        layout.addWidget(self.info_label)
        
        # 创建图形布局
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#1e1e1e')
        layout.addWidget(self.graphics_layout, stretch=1)
        
        # 创建各个子图
        self.setup_plots()
    
    def setup_plots(self):
        """设置图表区域"""
        # 清空现有图表
        self.graphics_layout.clear()
        
        # 固定Y轴宽度，确保所有子图右侧对齐
        Y_AXIS_WIDTH = 60
        
        # 主图（K线 + 均线）
        self.price_plot = self.graphics_layout.addPlot(row=0, col=0)
        self.price_plot.showAxis('right')
        self.price_plot.hideAxis('left')
        self.price_plot.setLabel('right', '价格')
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setMinimumHeight(300)
        self.price_plot.getAxis('right').setWidth(Y_AXIS_WIDTH)
        
        # 成交量图
        if self.show_volume:
            self.volume_plot = self.graphics_layout.addPlot(row=1, col=0)
            self.volume_plot.showAxis('right')
            self.volume_plot.hideAxis('left')
            self.volume_plot.setLabel('right', '成交量')
            self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
            self.volume_plot.setMaximumHeight(100)
            self.volume_plot.setXLink(self.price_plot)
            self.volume_plot.getAxis('right').setWidth(Y_AXIS_WIDTH)
        else:
            self.volume_plot = None
        
        # MACD图
        if self.show_macd:
            row_idx = 2 if self.show_volume else 1
            self.macd_plot = self.graphics_layout.addPlot(row=row_idx, col=0)
            self.macd_plot.showAxis('right')
            self.macd_plot.hideAxis('left')
            self.macd_plot.setLabel('right', 'MACD')
            self.macd_plot.showGrid(x=True, y=True, alpha=0.3)
            self.macd_plot.setMaximumHeight(120)
            self.macd_plot.setXLink(self.price_plot)
            self.macd_plot.getAxis('right').setWidth(Y_AXIS_WIDTH)
        else:
            self.macd_plot = None
        
        # KDJ图
        if self.show_kdj:
            row_idx = 1
            if self.show_volume:
                row_idx += 1
            if self.show_macd:
                row_idx += 1
            self.kdj_plot = self.graphics_layout.addPlot(row=row_idx, col=0)
            self.kdj_plot.showAxis('right')
            self.kdj_plot.hideAxis('left')
            self.kdj_plot.setLabel('right', 'KDJ')
            self.kdj_plot.showGrid(x=True, y=True, alpha=0.3)
            self.kdj_plot.setMaximumHeight(100)
            self.kdj_plot.setXLink(self.price_plot)
            self.kdj_plot.getAxis('right').setWidth(Y_AXIS_WIDTH)
        else:
            self.kdj_plot = None
        
        # 设置十字光标
        self.setup_crosshair()
        
        # 禁用 pyqtgraph 默认的右键菜单
        self.price_plot.vb.setMenuEnabled(False)
        if self.volume_plot:
            self.volume_plot.vb.setMenuEnabled(False)
        if self.macd_plot:
            self.macd_plot.vb.setMenuEnabled(False)
        if self.kdj_plot:
            self.kdj_plot.vb.setMenuEnabled(False)
        
        # 使用 pyqtgraph 的场景鼠标点击事件来实现右键菜单
        self.price_plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)

    def on_mouse_clicked(self, evt):
        """处理鼠标点击事件"""
        # 如果正在绘图，不处理右键菜单
        if hasattr(self, 'drawing_manager') and self.drawing_manager.is_drawing_active:
            return

        # 只处理右键点击
        if evt.button() != Qt.MouseButton.RightButton:
            return
            
        if self.data is None:
            return
        
        # 获取点击位置
        pos = evt.scenePos()
        
        # 检查是否在主图区域内
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            x = int(round(mouse_point.x()))
            
            if 0 <= x < len(self.data):
                self.last_click_idx = x
                self.show_context_menu(evt.screenPos())

    def show_context_menu(self, screen_pos):
        """显示右键菜单"""
        if self.data is None:
            return
            
        if self.last_click_idx < 0 or self.last_click_idx >= len(self.data):
            return
        
        # 保存屏幕坐标，供延迟调用使用
        self._pending_menu_pos = (int(screen_pos.x()), int(screen_pos.y()))
        
        # 使用 QTimer.singleShot 延迟显示菜单，让 pyqtgraph 事件处理完成
        # 这样可以确保菜单在点击选项后正确关闭
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._show_context_menu_delayed)
    
    def _show_context_menu_delayed(self):
        """延迟显示右键菜单的实际实现"""
        if not hasattr(self, '_pending_menu_pos') or self.data is None:
            return
            
        if self.last_click_idx < 0 or self.last_click_idx >= len(self.data):
            return
        
        # 获取当前K线的日期信息
        row = self.data.iloc[self.last_click_idx]
        date_str = row["date"].strftime("%Y-%m-%d")
            
        menu = QMenu(self)
        action_minute = QAction(f"查看 {date_str} 分时图", self)
        action_minute.triggered.connect(self.show_minute_chart)
        menu.addAction(action_minute)
        
        # 使用屏幕坐标显示菜单
        from PyQt6.QtCore import QPoint
        menu.exec(QPoint(self._pending_menu_pos[0], self._pending_menu_pos[1]))

    def show_minute_chart(self):
        """显示分时图"""
        if self.last_click_idx < 0 or self.last_click_idx >= len(self.data):
            return
            
        row = self.data.iloc[self.last_click_idx]
        date_str = row["date"].strftime("%Y-%m-%d")
        
        # 获取前一天的收盘价（用于计算涨跌幅）
        prev_close = None
        if self.last_click_idx > 0:
            prev_row = self.data.iloc[self.last_click_idx - 1]
            prev_close = prev_row["close"]
        else:
            # 如果是第一天，使用当天开盘价
            prev_close = row["open"]
        
        # 获取数据目录 - 使用绝对路径
        from pathlib import Path
        
        # 基于当前文件位置计算项目根目录
        current_file = Path(__file__).resolve()
        trading_app_dir = current_file.parent.parent  # trading_app 目录
        project_root = trading_app_dir.parent  # 项目根目录
        data_dir = project_root / "data"
        
        if not data_dir.exists():
            # 尝试其他可能的路径
            data_dir = Path.cwd() / "data"
            
        # 创建并显示窗口
        # 注意：我们需要保持窗口引用，否则会被垃圾回收
        self.minute_window = TimeShareWindow(
            self.stock_code, date_str, str(data_dir), self, prev_close=prev_close
        )
        self.minute_window.show()
    
    def setup_crosshair(self):
        """设置十字光标"""
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888888', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#888888', width=1))
        
        self.price_plot.addItem(self.vLine, ignoreBounds=True)
        self.price_plot.addItem(self.hLine, ignoreBounds=True)
        
        # 连接鼠标移动事件
        self.price_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
    
    def on_mouse_moved(self, pos):
        """处理鼠标移动"""
        if self.data is None:
            self.price_label.hide()
            return
        
        if self.price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.price_plot.vb.mapSceneToView(pos)
            x = int(round(mouse_point.x()))
            y_value = mouse_point.y()
            
            if 0 <= x < len(self.data):
                self.vLine.setPos(x)
                self.hLine.setPos(y_value)
                
                # 更新信息栏
                row = self.data.iloc[x]
                self.update_info_label(x, row)
                
                # 更新浮动价格标签
                self.update_floating_label(y_value, pos)
                
                # 记录当前索引，供右键菜单使用
                self.last_click_idx = x
            else:
                self.price_label.hide()
        else:
            self.price_label.hide()

    def update_floating_label(self, price, scene_pos):
        """更新浮动价格标签位置和内容"""
        # 设置文本
        self.price_label.setText(f"{price:.2f}")
        self.price_label.adjustSize()
        
        # 计算位置
        # 将 Scene 坐标转换为 View 坐标 (QPoint)
        view_pos = self.graphics_layout.mapFromScene(scene_pos)
        
        # 转换为全局坐标再转回本地坐标，确保位置准确
        global_pos = self.graphics_layout.mapToGlobal(view_pos)
        local_pos = self.mapFromGlobal(global_pos)
        
        # X 轴位置：靠右对齐，留一点边距
        # 确保标签显示在右侧坐标轴区域
        x = self.width() - self.price_label.width()
        
        # Y 轴位置：中心对齐鼠标位置
        y = local_pos.y() - self.price_label.height() / 2
        
        self.price_label.move(int(x), int(y))
        self.price_label.show()
        self.price_label.raise_()
    
    def update_info_label(self, idx: int, row: pd.Series):
        """更新信息栏"""
        date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else ""
        
        # 计算涨跌幅
        change_pct = 0
        if idx > 0:
            prev_close = self.data.iloc[idx - 1]["close"]
            if prev_close > 0:
                change_pct = (row["close"] - prev_close) / prev_close * 100
        
        color = self.up_color if row["close"] >= row["open"] else self.down_color
        
        info_text = (
            f"<span style='color: #ffffff;'>{self.stock_code} {self.stock_name}</span> | "
            f"<span style='color: #aaaaaa;'>{date_str}</span> | "
            f"开: <span style='color: {color};'>{row['open']:.2f}</span> | "
            f"高: <span style='color: {color};'>{row['high']:.2f}</span> | "
            f"低: <span style='color: {color};'>{row['low']:.2f}</span> | "
            f"收: <span style='color: {color};'>{row['close']:.2f}</span> | "
            f"涨跌: <span style='color: {color};'>{change_pct:+.2f}%</span> | "
            f"量: <span style='color: #aaaaaa;'>{row['volume']/10000:.0f}万</span>"
        )
        
        # 添加均线信息
        for i, w in enumerate(self.ma_windows):
            ma_col = f"MA{w}"
            if ma_col in row and pd.notna(row[ma_col]):
                ma_color = self.ma_colors[i % len(self.ma_colors)]
                info_text += f" | <span style='color: {ma_color};'>MA{w}: {row[ma_col]:.2f}</span>"
        
        self.info_label.setText(info_text)
    
    def set_data(
        self,
        data: pd.DataFrame,
        code: str = "",
        name: str = "",
        is_index: bool = False
    ):
        """
        设置K线数据
        
        Args:
            data: 包含 date/open/high/low/close/volume 及指标的DataFrame
            code: 股票/指数代码
            name: 股票/指数名称
            is_index: 是否为指数
        """
        self.data = data.copy()
        self.stock_code = code
        self.stock_name = name
        self._is_index = is_index
        
        self.update_chart()
        
        # 默认显示最后一天的数据
        if not self.data.empty:
            last_idx = len(self.data) - 1
            self.update_info_label(last_idx, self.data.iloc[last_idx])
        
        # 恢复绘图
        if hasattr(self, 'drawing_manager'):
            self.drawing_manager.restore_drawings()
    
    def update_chart(self):
        """更新图表"""
        if self.data is None or self.data.empty:
            return
        
        # 重新设置图表区域
        self.setup_plots()
        
        # 更新绘图管理器的 PlotItem 引用
        if hasattr(self, 'drawing_manager'):
            self.drawing_manager.update_plot_item(self.price_plot)
        
        # 绘制K线
        self.draw_candlesticks()
        
        # 绘制交易标记
        self.draw_trade_marks()
        
        # 绘制均线
        self.draw_ma_lines()
        
        # 绘制成交量
        if self.show_volume and self.volume_plot:
            self.draw_volume()
        
        # 绘制MACD
        if self.show_macd and self.macd_plot:
            self.draw_macd()
        
        # 绘制KDJ
        if self.show_kdj and self.kdj_plot:
            self.draw_kdj()
        
        # 设置X轴范围（默认显示最近60天）
        data_len = len(self.data)
        visible_count = min(60, data_len)
        
        # 连接 X 轴变化信号，实现自动 Y 轴缩放
        # 断开旧的连接以防止重复连接
        try:
            self.price_plot.vb.sigXRangeChanged.disconnect(self.update_y_range)
        except Exception:
            pass
        self.price_plot.vb.sigXRangeChanged.connect(self.update_y_range)

        self.price_plot.setXRange(data_len - visible_count, data_len, padding=0.02)
        
        # 设置X轴日期标签
        self.setup_x_axis()
        
        # 初始自动缩放
        self.update_y_range()

    def update_y_range(self):
        """根据当前可见 X 轴范围，自动调整 Y 轴范围"""
        if self.data is None or self.data.empty:
            return

        # 获取可见 X 轴范围
        view_range = self.price_plot.viewRange()
        min_x, max_x = view_range[0]
        
        # 转换为整数索引
        start_idx = max(0, int(min_x))
        end_idx = min(len(self.data), int(max_x) + 1)
        
        if start_idx >= end_idx:
            return

        # 获取该范围内的数据
        subset = self.data.iloc[start_idx:end_idx]
        if subset.empty:
            return

        # 1. 更新主图 Y 轴 (K线 + 均线)
        # 基础价格
        highs = subset["high"]
        lows = subset["low"]
        
        min_y = lows.min()
        max_y = highs.max()
        
        # 考虑均线
        for w in self.ma_windows:
            ma_col = f"MA{w}"
            if ma_col in subset.columns:
                valid_ma = subset[ma_col].dropna()
                if not valid_ma.empty:
                    min_y = min(min_y, valid_ma.min())
                    max_y = max(max_y, valid_ma.max())
        
        if not pd.isna(min_y) and not pd.isna(max_y):
            # 添加一点 padding
            padding = (max_y - min_y) * 0.05 if max_y != min_y else max_y * 0.01
            self.price_plot.setYRange(min_y - padding, max_y + padding, padding=0)

        # 2. 更新成交量图 Y 轴
        if self.show_volume and self.volume_plot and "volume" in subset.columns:
            vol_max = subset["volume"].max()
            if "VOL_MA5" in subset.columns:
                valid_vma = subset["VOL_MA5"].dropna()
                if not valid_vma.empty:
                    vol_max = max(vol_max, valid_vma.max())
            
            if not pd.isna(vol_max):
                self.volume_plot.setYRange(0, vol_max * 1.05, padding=0)

        # 3. 更新 MACD 图 Y 轴
        if self.show_macd and self.macd_plot and "MACD" in subset.columns:
            vals = []
            if "MACD" in subset.columns: vals.append(subset["MACD"])
            if "DIF" in subset.columns: vals.append(subset["DIF"])
            if "DEA" in subset.columns: vals.append(subset["DEA"])
            
            current_min, current_max = float('inf'), float('-inf')
            has_data = False
            for s in vals:
                valid_s = s.dropna()
                if not valid_s.empty:
                    current_min = min(current_min, valid_s.min())
                    current_max = max(current_max, valid_s.max())
                    has_data = True
            
            if has_data:
                # 对称显示或自动范围
                # 这里使用自动范围
                padding = (current_max - current_min) * 0.05 if current_max != current_min else 0.1
                self.macd_plot.setYRange(current_min - padding, current_max + padding, padding=0)

        # 4. 更新 KDJ 图 Y 轴
        if self.show_kdj and self.kdj_plot:
            vals = []
            if "K" in subset.columns: vals.append(subset["K"])
            if "D" in subset.columns: vals.append(subset["D"])
            if "J" in subset.columns: vals.append(subset["J"])
            
            current_min, current_max = float('inf'), float('-inf')
            has_data = False
            for s in vals:
                valid_s = s.dropna()
                if not valid_s.empty:
                    current_min = min(current_min, valid_s.min())
                    current_max = max(current_max, valid_s.max())
                    has_data = True
            
            if has_data:
                padding = (current_max - current_min) * 0.05 if current_max != current_min else 5
                self.kdj_plot.setYRange(current_min - padding, current_max + padding, padding=0)

    
    def draw_candlesticks(self):
        """绘制K线"""
        candle_item = CandlestickItem(
            self.data,
            up_color=self.up_color,
            down_color=self.down_color
        )
        self.price_plot.addItem(candle_item)
    
    def draw_trade_marks(self):
        """绘制交易标记"""
        if not self.stock_code or self.data is None or self.data.empty:
            return
            
        try:
            service = get_trade_record_service()
            records = service.get_stock_records(self.stock_code)
            
            if not records:
                return
                
            # 创建日期索引映射
            date_to_idx = {}
            for i, d in enumerate(self.data["date"]):
                date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
                date_to_idx[date_str] = i
            
            # 记录每天的买卖情况
            daily_trades = {}
            
            for record in records:
                if record.trade_date in date_to_idx:
                    idx = date_to_idx[record.trade_date]
                    if idx not in daily_trades:
                        daily_trades[idx] = {'buy': False, 'sell': False}
                    
                    if record.direction == 'buy':
                        daily_trades[idx]['buy'] = True
                    else:
                        daily_trades[idx]['sell'] = True
            
            # 绘制标记
            for idx, trades in daily_trades.items():
                row = self.data.iloc[idx]
                
                # 计算偏移量 (使用收盘价的 1% 作为基础偏移)
                # 这样可以保证标记不会紧贴K线，且随股价高低自动调整间距
                offset = row['close'] * 0.01
                
                if trades['buy']:
                    # 买入标记：B，位于最低价下方
                    text = pg.TextItem(
                        text='B',
                        color=self.up_color,
                        anchor=(0.5, 0)  # 锚点在文本上方中间，即文本显示在坐标点下方
                    )
                    # 设置字体
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(9)
                    text.setFont(font)
                    
                    # 位置：最低价再往下偏移
                    text.setPos(idx, row['low'] - offset)
                    self.price_plot.addItem(text)
                
                if trades['sell']:
                    # 卖出标记：S，位于最高价上方
                    text = pg.TextItem(
                        text='S',
                        color=self.down_color,
                        anchor=(0.5, 1)  # 锚点在文本下方中间，即文本显示在坐标点上方
                    )
                    # 设置字体
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(9)
                    text.setFont(font)
                    
                    # 位置：最高价再往上偏移
                    text.setPos(idx, row['high'] + offset)
                    self.price_plot.addItem(text)

        except Exception as e:
            print(f"绘制交易标记出错: {e}")

    def draw_ma_lines(self):
        """绘制均线"""
        x = np.arange(len(self.data))
        
        for i, w in enumerate(self.ma_windows):
            ma_col = f"MA{w}"
            if ma_col in self.data.columns:
                color = self.ma_colors[i % len(self.ma_colors)]
                y = self.data[ma_col].values
                
                # 过滤掉NaN值
                valid_mask = ~np.isnan(y)
                if valid_mask.any():
                    pen = pg.mkPen(color, width=1.5)
                    self.price_plot.plot(
                        x[valid_mask], y[valid_mask],
                        pen=pen, name=f"MA{w}"
                    )
    
    def draw_volume(self):
        """绘制成交量"""
        if self.volume_plot is None:
            return
        
        vol_item = VolumeBarItem(
            self.data,
            up_color=self.up_color,
            down_color=self.down_color
        )
        self.volume_plot.addItem(vol_item)
        
        # 成交量均线
        if "VOL_MA5" in self.data.columns:
            x = np.arange(len(self.data))
            y = self.data["VOL_MA5"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                pen = pg.mkPen('#555555', width=1)
                self.volume_plot.plot(x[valid_mask], y[valid_mask], pen=pen)
    
    def draw_macd(self):
        """绘制MACD"""
        if self.macd_plot is None:
            return
        
        if "MACD" not in self.data.columns:
            return
        
        x = np.arange(len(self.data))
        macd_vals = self.data["MACD"].values
        
        # MACD柱
        colors = [
            self.up_color if v >= 0 else self.down_color
            for v in macd_vals
        ]
        brushes = [pg.mkBrush(c) for c in colors]
        
        bar_item = pg.BarGraphItem(
            x=x, height=macd_vals, width=0.6, brushes=brushes
        )
        self.macd_plot.addItem(bar_item)
        
        # DIF线
        if "DIF" in self.data.columns:
            y = self.data["DIF"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                pen = pg.mkPen('#e377c2', width=1.2)
                self.macd_plot.plot(x[valid_mask], y[valid_mask], pen=pen, name="DIF")
        
        # DEA线
        if "DEA" in self.data.columns:
            y = self.data["DEA"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                pen = pg.mkPen('#1f77b4', width=1.2)
                self.macd_plot.plot(x[valid_mask], y[valid_mask], pen=pen, name="DEA")
        
        # 添加零线
        self.macd_plot.addItem(
            pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#555555', width=1, style=Qt.PenStyle.DashLine))
        )
    
    def draw_kdj(self):
        """绘制KDJ"""
        if self.kdj_plot is None:
            return
        
        x = np.arange(len(self.data))
        
        # K线
        if "K" in self.data.columns:
            y = self.data["K"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                pen = pg.mkPen('#f1c40f', width=1.2)
                self.kdj_plot.plot(x[valid_mask], y[valid_mask], pen=pen, name="K")
        
        # D线
        if "D" in self.data.columns:
            y = self.data["D"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                pen = pg.mkPen('#3498db', width=1.2)
                self.kdj_plot.plot(x[valid_mask], y[valid_mask], pen=pen, name="D")
        
        # J线
        if "J" in self.data.columns:
            y = self.data["J"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                pen = pg.mkPen('#8e44ad', width=1.2)
                self.kdj_plot.plot(x[valid_mask], y[valid_mask], pen=pen, name="J")
    
    def setup_x_axis(self):
        """设置X轴日期标签"""
        if self.data is None:
            return
        
        # 创建日期-索引映射
        dates = self.data["date"].values
        
        # 每隔一定数量显示一个标签
        step = max(1, len(dates) // 10)
        
        ticks = []
        for i in range(0, len(dates), step):
            date_str = pd.Timestamp(dates[i]).strftime("%m-%d")
            ticks.append((i, date_str))
        
        # 设置X轴
        x_axis = self.price_plot.getAxis('bottom')
        x_axis.setTicks([ticks])
    
    def set_indicators(
        self,
        show_volume: bool = True,
        show_macd: bool = True,
        show_kdj: bool = False,
        update: bool = False
    ):
        """
        设置显示的指标
        
        Args:
            show_volume: 是否显示成交量
            show_macd: 是否显示MACD
            show_kdj: 是否显示KDJ
            update: 是否立即更新图表（默认False，由set_data统一更新）
        """
        self.show_volume = show_volume
        self.show_macd = show_macd
        self.show_kdj = show_kdj
        
        if update and self.data is not None:
            self.update_chart()
    
    def set_ma_windows(self, windows: List[int], update: bool = False):
        """
        设置均线周期
        
        Args:
            windows: 均线周期列表
            update: 是否立即更新图表（默认False，由set_data统一更新）
        """
        self.ma_windows = windows
        
        if update and self.data is not None:
            self.update_chart()
    
    # ==================== 实时行情相关方法 ====================
    
    def start_realtime(self) -> bool:
        """
        启动实时行情订阅
        
        Returns:
            是否成功启动
        """
        if self._realtime_enabled:
            return True
        
        if not self.stock_code:
            return False
        
        try:
            self._quote_service = get_quote_service()
            
            if not self._quote_service.is_available:
                self.realtimeStatusChanged.emit(False, "xtquant 未安装")
                return False
            
            # 连接信号
            self._quote_service.quote_updated.connect(self._on_quote_updated)
            self._quote_service.connection_status_changed.connect(self._on_connection_status)
            
            # 订阅当前股票/指数
            if self._quote_service.replace_subscription(
                self._quote_owner_id,
                [self.stock_code],
                start_service=True,
                is_index=self._is_index,
            ):
                self._realtime_enabled = True
                self._today_date = date.today()
                type_name = "指数" if self._is_index else "股票"
                self.realtimeStatusChanged.emit(True, f"已订阅 {self.stock_code} 实时{type_name}行情")
                return True
            else:
                self.realtimeStatusChanged.emit(False, "订阅失败")
                return False
                
        except Exception as e:
            self.realtimeStatusChanged.emit(False, f"启动失败: {e}")
            return False
    
    def stop_realtime(self):
        """停止实时行情订阅"""
        if not self._realtime_enabled:
            return
        
        try:
            if self._quote_service and self.stock_code:
                self._quote_service.clear_owner_subscription(self._quote_owner_id)
                
                # 断开信号
                try:
                    self._quote_service.quote_updated.disconnect(self._on_quote_updated)
                    self._quote_service.connection_status_changed.disconnect(self._on_connection_status)
                except:
                    pass
            
            self._realtime_enabled = False
            self.realtimeStatusChanged.emit(False, "实时行情已停止")
            
        except Exception as e:
            pass
    
    def switch_stock_realtime(self, new_code: str):
        """
        切换股票时更新实时行情订阅
        
        Args:
            new_code: 新的股票代码
        """
        if not self._realtime_enabled or not self._quote_service:
            return
        
        old_code = self.stock_code
        
        try:
            # 取消旧订阅
            if old_code:
                self._quote_service.clear_owner_subscription(self._quote_owner_id)
            
            # 订阅新股票
            if new_code:
                self._quote_service.replace_subscription(
                    self._quote_owner_id,
                    [new_code],
                    start_service=True,
                    is_index=self._is_index,
                )
                self._today_date = date.today()
                self.realtimeStatusChanged.emit(True, f"已切换订阅 {new_code}")
                
        except Exception as e:
            pass
    
    def _on_quote_updated(self, quote: QuoteData):
        """
        处理实时行情更新
        
        Args:
            quote: 实时行情数据
        """
        if not self._realtime_enabled:
            return
        
        # 检查是否是当前股票
        if quote.simple_code != self.stock_code:
            return
        
        # 更新当日K线
        self._update_today_bar(quote)
    
    def _on_connection_status(self, connected: bool, message: str):
        """处理连接状态变化"""
        self.realtimeStatusChanged.emit(connected, message)
    
    def _update_today_bar(self, quote: QuoteData):
        """
        根据实时行情更新当日K线
        
        只在交易时间内更新，非交易时间（周末、节假日、收盘后）不更新
        
        Args:
            quote: 实时行情数据
        """
        if self.data is None or self.data.empty:
            return
        
        if quote.last_price <= 0 or not bool(getattr(quote, "is_fresh", False)):
            return
        
        # 检查最后一条数据是否是今天的
        last_idx = len(self.data) - 1
        last_row = self.data.iloc[last_idx]
        last_date = pd.Timestamp(last_row['date']).date()
        
        # 判断是否应该更新实时K线
        if not should_update_realtime_kline(last_date):
            # 非交易时间或已有完整数据，不更新
            return
        
        today = date.today()
        
        # 更新数据
        if last_date == today:
            # 更新今日K线数据
            self.data.loc[self.data.index[last_idx], 'close'] = quote.last_price
            self.data.loc[self.data.index[last_idx], 'high'] = max(
                self.data.iloc[last_idx]['high'], quote.high_price if quote.high_price > 0 else quote.last_price
            )
            self.data.loc[self.data.index[last_idx], 'low'] = min(
                self.data.iloc[last_idx]['low'], quote.low_price if quote.low_price > 0 else quote.last_price
            )
            if quote.volume > 0:
                self.data.loc[self.data.index[last_idx], 'volume'] = quote.volume
        else:
            # 今日数据不存在，需要创建新的一行（只在交易时间内）
            new_row = {
                'date': pd.Timestamp(today),
                'open': quote.open_price if quote.open_price > 0 else quote.last_price,
                'high': quote.high_price if quote.high_price > 0 else quote.last_price,
                'low': quote.low_price if quote.low_price > 0 else quote.last_price,
                'close': quote.last_price,
                'volume': quote.volume if quote.volume > 0 else 0,
            }
            # 添加新行
            new_df = pd.DataFrame([new_row])
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            last_idx = len(self.data) - 1
        
        # 重绘当日K线（局部更新）
        self._redraw_today_candle(last_idx)
        
        # 更新信息栏
        self.update_info_label(last_idx, self.data.iloc[last_idx])
    
    def _redraw_today_candle(self, idx: int):
        """
        重绘当日K线（局部更新，避免重绘整个图表）
        
        Args:
            idx: 当日K线的索引
        """
        if self.data is None or idx < 0 or idx >= len(self.data):
            return
        
        row = self.data.iloc[idx]
        
        # 移除旧的当日K线图形项
        if self._today_bar_item is not None:
            self.price_plot.removeItem(self._today_bar_item)
            self._today_bar_item = None
        
        # 创建只包含当日数据的 DataFrame
        today_df = self.data.iloc[[idx]].copy()
        today_df.reset_index(drop=True, inplace=True)
        
        # 创建新的当日K线图形项
        self._today_bar_item = TodayCandlestickItem(
            idx, row['open'], row['high'], row['low'], row['close'],
            up_color=self.up_color,
            down_color=self.down_color
        )
        self.price_plot.addItem(self._today_bar_item)
        
        # 更新成交量
        if self.show_volume and self.volume_plot and 'volume' in row:
            if self._today_vol_item is not None:
                self.volume_plot.removeItem(self._today_vol_item)
                self._today_vol_item = None
            
            color = self.up_color if row['close'] >= row['open'] else self.down_color
            self._today_vol_item = pg.BarGraphItem(
                x=[idx], height=[row['volume']], width=0.6,
                brush=pg.mkBrush(color)
            )
            self.volume_plot.addItem(self._today_vol_item)
        
        # 更新 Y 轴范围
        self.update_y_range()
    
    @property
    def is_realtime_enabled(self) -> bool:
        """返回实时行情是否已开启"""
        return self._realtime_enabled


class TodayCandlestickItem(pg.GraphicsObject):
    """单根K线图形项（用于实时更新当日K线）"""
    
    def __init__(
        self,
        x: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        up_color: str = "#ec0000",
        down_color: str = "#00da3c"
    ):
        super().__init__()
        self.x = x
        self.o = open_price
        self.h = high_price
        self.l = low_price
        self.c = close_price
        self.up_color = QColor(up_color)
        self.down_color = QColor(down_color)
        self.picture = None
        self.generatePicture()
    
    def generatePicture(self):
        """渲染K线图形"""
        from PyQt6.QtGui import QPicture, QPainter
        
        self.picture = QPicture()
        painter = QPainter(self.picture)
        
        w = 0.3  # K线宽度的一半
        
        if self.c >= self.o:
            color = self.up_color
        else:
            color = self.down_color
        
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))
        
        # 绘制上下影线
        painter.drawLine(
            pg.QtCore.QPointF(float(self.x), float(self.l)),
            pg.QtCore.QPointF(float(self.x), float(self.h))
        )
        
        # 绘制实体
        body_height = abs(self.c - self.o)
        
        if body_height < 0.001:
            # 一字板：画线
            painter.drawLine(
                pg.QtCore.QPointF(float(self.x - w), float(self.c)),
                pg.QtCore.QPointF(float(self.x + w), float(self.c))
            )
        else:
            # 有实体：画矩形
            body_top = min(self.o, self.c)
            rect = pg.QtCore.QRectF(
                float(self.x - w),
                float(body_top),
                float(w * 2),
                float(body_height)
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.fillRect(rect, QBrush(color))
        
        painter.end()
    
    def paint(self, painter, *args):
        if self.picture:
            self.picture.play(painter)
    
    def boundingRect(self):
        if self.picture is None:
            return pg.QtCore.QRectF()
        return pg.QtCore.QRectF(self.picture.boundingRect())
