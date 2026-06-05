# daily_pnl_widget.py - Daily P&L tracking and return curve widget
"""
Daily profit/loss tracking and historical return visualization.

Features:
- Record daily account snapshots (total asset, cash, market value)
- Display daily P&L table with color-coded gains/losses
- Plot cumulative return curve with pyqtgraph
- Show key statistics: cumulative return, max drawdown, Sharpe ratio, win rate
- Manual and auto snapshot recording
- Export data to CSV
"""

import logging
from datetime import datetime, timedelta

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QDateEdit, QMessageBox, QSplitter, QFileDialog,
    QDoubleSpinBox, QSpinBox, QFormLayout, QDialog,
    QDialogButtonBox, QFrame
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QPen

from trading_app.services.execution.trade_record_service import (
    get_trade_record_service, DailyPnlSnapshot
)

logger = logging.getLogger(__name__)


class SnapshotDialog(QDialog):
    """Dialog for manually adding/editing a daily PnL snapshot"""

    def __init__(self, parent=None, snapshot_date: str = None,
                 total_asset: float = 0.0, cash: float = 0.0,
                 market_value: float = 0.0, position_count: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Record Daily Snapshot")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        # Date
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        if snapshot_date:
            parts = snapshot_date.split('-')
            self.date_edit.setDate(QDate(int(parts[0]), int(parts[1]), int(parts[2])))
        else:
            self.date_edit.setDate(QDate.currentDate())
        form.addRow("日期:", self.date_edit)

        # Total asset
        self.total_asset_spin = QDoubleSpinBox()
        self.total_asset_spin.setDecimals(2)
        self.total_asset_spin.setRange(0, 999999999.99)
        self.total_asset_spin.setSingleStep(1000)
        self.total_asset_spin.setValue(total_asset)
        self.total_asset_spin.setPrefix("¥ ")
        form.addRow("总资产:", self.total_asset_spin)

        # Cash
        self.cash_spin = QDoubleSpinBox()
        self.cash_spin.setDecimals(2)
        self.cash_spin.setRange(0, 999999999.99)
        self.cash_spin.setSingleStep(1000)
        self.cash_spin.setValue(cash)
        self.cash_spin.setPrefix("¥ ")
        form.addRow("可用资金:", self.cash_spin)

        # Market value
        self.market_value_spin = QDoubleSpinBox()
        self.market_value_spin.setDecimals(2)
        self.market_value_spin.setRange(0, 999999999.99)
        self.market_value_spin.setSingleStep(1000)
        self.market_value_spin.setValue(market_value)
        self.market_value_spin.setPrefix("¥ ")
        form.addRow("持仓市值:", self.market_value_spin)

        # Position count
        self.position_count_spin = QSpinBox()
        self.position_count_spin.setRange(0, 9999)
        self.position_count_spin.setValue(position_count)
        form.addRow("持仓数量:", self.position_count_spin)

        layout.addLayout(form)
        layout.addSpacing(10)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> dict:
        return {
            'snapshot_date': self.date_edit.date().toString("yyyy-MM-dd"),
            'total_asset': self.total_asset_spin.value(),
            'cash': self.cash_spin.value(),
            'market_value': self.market_value_spin.value(),
            'position_count': self.position_count_spin.value(),
        }


class DailyPnlWidget(QWidget):
    """
    Daily P&L tracking widget with return curve chart.

    Integrates into BrokerAccountWidget as a tab page.
    """

    snapshot_requested = pyqtSignal()  # Request broker to provide current account data

    def __init__(self, parent=None):
        super().__init__(parent)

        self.trade_service = get_trade_record_service()
        self.trade_service.pnl_snapshot_saved.connect(self.refresh_data)

        self.setup_ui()
        self.refresh_data()

    def setup_ui(self):
        """Build the UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # ========== Top: Summary Statistics ==========
        stats_group = QGroupBox("收益统计概览")
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setSpacing(20)

        # Helper to create stat labels
        def make_stat(label_text: str) -> tuple:
            container = QVBoxLayout()
            title = QLabel(label_text)
            title.setStyleSheet("color: #888; font-size: 11px;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value = QLabel("-")
            value.setStyleSheet("color: #fff; font-size: 16px; font-weight: bold;")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container.addWidget(title)
            container.addWidget(value)
            return container, value

        stat1, self.lbl_total_asset = make_stat("总资产")
        stat2, self.lbl_cumulative_return = make_stat("累计收益率")
        stat3, self.lbl_cumulative_profit = make_stat("累计盈亏")
        stat4, self.lbl_max_drawdown = make_stat("最大回撤")
        stat5, self.lbl_sharpe = make_stat("夏普比率")
        stat6, self.lbl_win_days = make_stat("盈利天数/总天数")
        stat7, self.lbl_today_pnl = make_stat("今日盈亏")

        for layout_item in [stat1, stat2, stat3, stat4, stat5, stat6, stat7]:
            stats_layout.addLayout(layout_item)

        main_layout.addWidget(stats_group)

        # ========== Middle: Splitter (Chart + Table) ==========
        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Return Curve Chart ---
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        # Chart title bar with buttons
        chart_header = QHBoxLayout()
        chart_title = QLabel("📈 收益率曲线")
        chart_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        chart_header.addWidget(chart_title)
        chart_header.addStretch()

        self.record_btn = QPushButton("📝 记录今日快照")
        self.record_btn.setStyleSheet(
            "background-color: #0078d4; color: white; font-weight: bold; "
            "padding: 5px 12px; border-radius: 3px;"
        )
        self.record_btn.clicked.connect(self.on_record_snapshot)
        chart_header.addWidget(self.record_btn)

        self.auto_record_btn = QPushButton("🔄 从账户同步")
        self.auto_record_btn.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; "
            "padding: 5px 12px; border-radius: 3px;"
        )
        self.auto_record_btn.setToolTip("自动从券商账户获取当前资产数据并记录快照")
        self.auto_record_btn.clicked.connect(self.on_auto_record)
        chart_header.addWidget(self.auto_record_btn)

        export_btn = QPushButton("📤 导出CSV")
        export_btn.setStyleSheet("padding: 5px 12px;")
        export_btn.clicked.connect(self.on_export_csv)
        chart_header.addWidget(export_btn)

        chart_layout.addLayout(chart_header)

        # pyqtgraph plot
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.setMinimumHeight(200)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('left', '收益率 (%)')
        self.plot_widget.setLabel('bottom', '日期')

        # Style axis
        for axis_name in ['left', 'bottom']:
            ax = self.plot_widget.getAxis(axis_name)
            ax.setPen(pg.mkPen('#888'))
            ax.setTextPen(pg.mkPen('#ccc'))

        chart_layout.addWidget(self.plot_widget)
        splitter.addWidget(chart_widget)

        # --- Daily PnL Table ---
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_header = QHBoxLayout()
        table_title = QLabel("📊 每日盈亏记录")
        table_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        table_header.addWidget(table_title)

        table_header.addStretch()

        # Date range filter
        table_header.addWidget(QLabel("日期:"))
        self.filter_start = QDateEdit()
        self.filter_start.setCalendarPopup(True)
        self.filter_start.setDate(QDate.currentDate().addMonths(-3))
        self.filter_start.setDisplayFormat("yyyy-MM-dd")
        table_header.addWidget(self.filter_start)

        table_header.addWidget(QLabel("至"))
        self.filter_end = QDateEdit()
        self.filter_end.setCalendarPopup(True)
        self.filter_end.setDate(QDate.currentDate())
        self.filter_end.setDisplayFormat("yyyy-MM-dd")
        table_header.addWidget(self.filter_end)

        filter_btn = QPushButton("🔍 查询")
        filter_btn.setStyleSheet(
            "background-color: #0078d4; color: white; font-weight: bold; padding: 4px 10px;"
        )
        filter_btn.clicked.connect(self.refresh_data)
        table_header.addWidget(filter_btn)

        delete_btn = QPushButton("🗑 删除选中")
        delete_btn.setStyleSheet("padding: 4px 10px;")
        delete_btn.clicked.connect(self.on_delete_selected)
        table_header.addWidget(delete_btn)

        table_layout.addLayout(table_header)

        # Table
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
                padding: 5px;
                border: 1px solid #333;
                font-weight: bold;
            }
            QTableCornerButton::section {
                background-color: #2d2d2d;
                border: 1px solid #333;
            }
        """

        self.pnl_table = QTableWidget()
        self.pnl_table.setStyleSheet(table_style)
        self.pnl_table.setColumnCount(9)
        self.pnl_table.setHorizontalHeaderLabels([
            "日期", "总资产", "可用资金", "持仓市值",
            "当日盈亏", "当日收益率", "累计收益率", "持仓数", "备注"
        ])
        self.pnl_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.pnl_table.horizontalHeader().setStretchLastSection(True)
        self.pnl_table.setAlternatingRowColors(True)
        self.pnl_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pnl_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pnl_table.itemDoubleClicked.connect(self.on_row_double_clicked)

        table_layout.addWidget(self.pnl_table)
        splitter.addWidget(table_widget)

        # Set splitter proportions (chart bigger)
        splitter.setSizes([350, 300])

        main_layout.addWidget(splitter)

    def refresh_data(self):
        """Refresh table and chart with current filter settings"""
        start_date = self.filter_start.date().toString("yyyy-MM-dd")
        end_date = self.filter_end.date().toString("yyyy-MM-dd")

        snapshots = self.trade_service.get_pnl_snapshots(start_date, end_date, limit=9999)

        self._update_table(snapshots)
        self._update_chart(snapshots)
        self._update_statistics(snapshots)

    def _update_table(self, snapshots: list):
        """Populate the PnL table"""
        self.pnl_table.setRowCount(0)

        # Show most recent first
        for snapshot in reversed(snapshots):
            row = self.pnl_table.rowCount()
            self.pnl_table.insertRow(row)

            # Date
            self.pnl_table.setItem(row, 0, QTableWidgetItem(snapshot.snapshot_date))

            # Total asset
            asset_item = QTableWidgetItem(f"¥{snapshot.total_asset:,.2f}")
            asset_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.pnl_table.setItem(row, 1, asset_item)

            # Cash
            cash_item = QTableWidgetItem(f"¥{snapshot.cash:,.2f}")
            cash_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.pnl_table.setItem(row, 2, cash_item)

            # Market value
            mv_item = QTableWidgetItem(f"¥{snapshot.market_value:,.2f}")
            mv_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.pnl_table.setItem(row, 3, mv_item)

            # Daily PnL
            pnl_item = QTableWidgetItem(f"{snapshot.total_profit:+,.2f}")
            pnl_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            pnl_color = QColor("#ec0000") if snapshot.total_profit >= 0 else QColor("#00da3c")
            pnl_item.setForeground(QBrush(pnl_color))
            self.pnl_table.setItem(row, 4, pnl_item)

            # Daily return %
            ret_item = QTableWidgetItem(f"{snapshot.total_profit_pct:+.2f}%")
            ret_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            ret_color = QColor("#ec0000") if snapshot.total_profit_pct >= 0 else QColor("#00da3c")
            ret_item.setForeground(QBrush(ret_color))
            self.pnl_table.setItem(row, 5, ret_item)

            # Cumulative return %
            cum_item = QTableWidgetItem(f"{snapshot.cumulative_return:+.2f}%")
            cum_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            cum_color = QColor("#ec0000") if snapshot.cumulative_return >= 0 else QColor("#00da3c")
            cum_item.setForeground(QBrush(cum_color))
            self.pnl_table.setItem(row, 6, cum_item)

            # Position count
            self.pnl_table.setItem(row, 7, QTableWidgetItem(str(snapshot.position_count)))

            # Remark
            self.pnl_table.setItem(row, 8, QTableWidgetItem(snapshot.remark))

    def _update_chart(self, snapshots: list):
        """Update the return curve chart"""
        self.plot_widget.clear()

        if len(snapshots) < 1:
            return

        # Prepare data
        dates = []
        cumulative_returns = []
        daily_returns = []

        for s in snapshots:
            dates.append(s.snapshot_date)
            cumulative_returns.append(s.cumulative_return)
            daily_returns.append(s.total_profit_pct)

        x = list(range(len(dates)))

        # Plot cumulative return curve
        pen_cum = pg.mkPen(color='#4fc3f7', width=2)
        self.plot_widget.plot(
            x, cumulative_returns,
            pen=pen_cum,
            name='累计收益率'
        )

        # Fill area under curve
        if len(x) > 1:
            fill_brush_pos = pg.mkBrush(color=(79, 195, 247, 40))
            fill_brush_neg = pg.mkBrush(color=(255, 82, 82, 40))

            # Create fill between curve and zero line
            zero_line = [0.0] * len(x)
            fill_pos = pg.FillBetweenItem(
                pg.PlotCurveItem(x, cumulative_returns, pen=pg.mkPen(None)),
                pg.PlotCurveItem(x, zero_line, pen=pg.mkPen(None)),
                brush=fill_brush_pos if cumulative_returns[-1] >= 0 else fill_brush_neg
            )
            self.plot_widget.addItem(fill_pos)

        # Re-plot on top of fill
        self.plot_widget.plot(x, cumulative_returns, pen=pen_cum)

        # Plot daily return as bar chart
        bar_colors = []
        for r in daily_returns:
            if r >= 0:
                bar_colors.append(pg.mkBrush(color=(236, 0, 0, 120)))
            else:
                bar_colors.append(pg.mkBrush(color=(0, 218, 60, 120)))

        bar_item = pg.BarGraphItem(
            x=x, height=daily_returns, width=0.5, brushes=bar_colors
        )
        self.plot_widget.addItem(bar_item)

        # Zero line
        zero_pen = pg.mkPen(color='#555', style=Qt.PenStyle.DashLine, width=1)
        self.plot_widget.addLine(y=0, pen=zero_pen)

        # Custom X axis labels
        if dates:
            # Show labels at reasonable intervals
            if len(dates) <= 30:
                step = 1
            elif len(dates) <= 90:
                step = 5
            elif len(dates) <= 365:
                step = 15
            else:
                step = 30

            ticks = []
            for i in range(0, len(dates), step):
                # Show MM-DD format
                d = dates[i]
                short_date = d[5:]  # MM-DD
                ticks.append((i, short_date))
            # Always show last date
            if ticks and ticks[-1][0] != len(dates) - 1:
                ticks.append((len(dates) - 1, dates[-1][5:]))

            ax = self.plot_widget.getAxis('bottom')
            ax.setTicks([ticks])

        # Add crosshair
        self._setup_crosshair(dates, cumulative_returns, daily_returns)

    def _setup_crosshair(self, dates: list, cum_returns: list, daily_returns: list):
        """Setup crosshair hover tooltip"""
        vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('#666', width=1))
        hline = pg.InfiniteLine(angle=0, pen=pg.mkPen('#666', width=1))
        self.plot_widget.addItem(vline, ignoreBounds=True)
        self.plot_widget.addItem(hline, ignoreBounds=True)

        label = pg.TextItem(color='#fff', anchor=(0, 1))
        label.setFont(QFont("Consolas", 9))
        self.plot_widget.addItem(label, ignoreBounds=True)
        label.hide()

        def on_mouse_moved(pos):
            if not dates:
                return
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            idx = int(round(mouse_point.x()))
            if 0 <= idx < len(dates):
                vline.setPos(idx)
                hline.setPos(cum_returns[idx])
                label.setText(
                    f"日期: {dates[idx]}\n"
                    f"累计收益: {cum_returns[idx]:+.2f}%\n"
                    f"当日收益: {daily_returns[idx]:+.2f}%"
                )
                label.setPos(idx, cum_returns[idx])
                label.show()
            else:
                label.hide()

        self.plot_widget.scene().sigMouseMoved.connect(on_mouse_moved)

    def _update_statistics(self, snapshots: list):
        """Update the summary statistics labels"""
        if not snapshots:
            self.lbl_total_asset.setText("-")
            self.lbl_cumulative_return.setText("-")
            self.lbl_cumulative_profit.setText("-")
            self.lbl_max_drawdown.setText("-")
            self.lbl_sharpe.setText("-")
            self.lbl_win_days.setText("-")
            self.lbl_today_pnl.setText("-")
            return

        latest = snapshots[-1]

        # Total asset
        if latest.total_asset >= 10000:
            self.lbl_total_asset.setText(f"¥{latest.total_asset / 10000:.2f}万")
        else:
            self.lbl_total_asset.setText(f"¥{latest.total_asset:,.2f}")

        # Cumulative return
        cum_ret = latest.cumulative_return
        cum_color = "#ec0000" if cum_ret >= 0 else "#00da3c"
        self.lbl_cumulative_return.setText(f"{cum_ret:+.2f}%")
        self.lbl_cumulative_return.setStyleSheet(
            f"color: {cum_color}; font-size: 16px; font-weight: bold;"
        )

        # Cumulative profit (total asset diff from first to last)
        first = snapshots[0]
        total_profit = latest.total_asset - first.total_asset
        profit_color = "#ec0000" if total_profit >= 0 else "#00da3c"
        if abs(total_profit) >= 10000:
            self.lbl_cumulative_profit.setText(f"{total_profit / 10000:+.2f}万")
        else:
            self.lbl_cumulative_profit.setText(f"¥{total_profit:+,.2f}")
        self.lbl_cumulative_profit.setStyleSheet(
            f"color: {profit_color}; font-size: 16px; font-weight: bold;"
        )

        # Max drawdown
        max_dd, dd_peak, dd_trough = self.trade_service.calculate_max_drawdown(snapshots)
        self.lbl_max_drawdown.setText(f"-{max_dd:.2f}%")
        self.lbl_max_drawdown.setStyleSheet(
            "color: #00da3c; font-size: 16px; font-weight: bold;"
        )
        if dd_peak and dd_trough:
            self.lbl_max_drawdown.setToolTip(f"峰值: {dd_peak}\n谷值: {dd_trough}")

        # Sharpe ratio
        sharpe = self.trade_service.calculate_sharpe_ratio(snapshots)
        sharpe_color = "#4fc3f7" if sharpe >= 1.0 else ("#fff" if sharpe >= 0 else "#f0ad4e")
        self.lbl_sharpe.setText(f"{sharpe:.2f}")
        self.lbl_sharpe.setStyleSheet(
            f"color: {sharpe_color}; font-size: 16px; font-weight: bold;"
        )

        # Win days / total days
        win_days = sum(1 for s in snapshots if s.total_profit > 0)
        loss_days = sum(1 for s in snapshots if s.total_profit < 0)
        total_days = len(snapshots)
        # Exclude first day (no PnL reference)
        trading_days = max(total_days - 1, 0)
        win_rate = (win_days / trading_days * 100) if trading_days > 0 else 0
        self.lbl_win_days.setText(f"{win_days}/{trading_days} ({win_rate:.0f}%)")
        self.lbl_win_days.setStyleSheet(
            "color: #fff; font-size: 16px; font-weight: bold;"
        )

        # Today PnL
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_pnl = 0.0
        today_pct = 0.0
        if latest.snapshot_date == today_str:
            today_pnl = latest.total_profit
            today_pct = latest.total_profit_pct

        today_color = "#ec0000" if today_pnl >= 0 else "#00da3c"
        if abs(today_pnl) >= 10000:
            self.lbl_today_pnl.setText(f"{today_pnl / 10000:+.2f}万 ({today_pct:+.2f}%)")
        else:
            self.lbl_today_pnl.setText(f"¥{today_pnl:+,.2f} ({today_pct:+.2f}%)")
        self.lbl_today_pnl.setStyleSheet(
            f"color: {today_color}; font-size: 16px; font-weight: bold;"
        )

    def on_record_snapshot(self):
        """Open manual snapshot dialog"""
        # Try to get latest snapshot data as defaults
        today = datetime.now().strftime("%Y-%m-%d")
        existing = self.trade_service.get_pnl_snapshot(today)

        if existing:
            dialog = SnapshotDialog(
                self,
                snapshot_date=existing.snapshot_date,
                total_asset=existing.total_asset,
                cash=existing.cash,
                market_value=existing.market_value,
                position_count=existing.position_count,
            )
        else:
            dialog = SnapshotDialog(self, snapshot_date=today)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            result = self.trade_service.save_daily_pnl(**data)
            if result:
                QMessageBox.information(self, "成功", f"已记录 {data['snapshot_date']} 的每日快照")
                self.refresh_data()
            else:
                QMessageBox.warning(self, "失败", "保存快照失败，请查看日志")

    def on_auto_record(self):
        """Request auto record from broker account"""
        self.snapshot_requested.emit()

    def auto_save_snapshot(self, total_asset: float, cash: float,
                           market_value: float, position_count: int):
        """
        Called by BrokerAccountWidget to auto-save daily snapshot.

        Args:
            total_asset: Total account asset
            cash: Available cash
            market_value: Market value of positions
            position_count: Number of positions
        """
        today = datetime.now().strftime("%Y-%m-%d")
        result = self.trade_service.save_daily_pnl(
            snapshot_date=today,
            total_asset=total_asset,
            cash=cash,
            market_value=market_value,
            position_count=position_count,
            remark="Auto synced from broker"
        )
        if result:
            QMessageBox.information(
                self, "同步成功",
                f"已记录今日 ({today}) 账户快照\n"
                f"总资产: ¥{total_asset:,.2f}\n"
                f"当日盈亏: ¥{result.total_profit:+,.2f} ({result.total_profit_pct:+.2f}%)"
            )
            self.refresh_data()
        else:
            QMessageBox.warning(self, "同步失败", "保存快照失败")

    def on_export_csv(self):
        """Export PnL data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"daily_pnl_{timestamp}.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出每日盈亏数据", default_name, "CSV文件 (*.csv)"
        )
        if not file_path:
            return

        start_date = self.filter_start.date().toString("yyyy-MM-dd")
        end_date = self.filter_end.date().toString("yyyy-MM-dd")

        success = self.trade_service.export_pnl_to_csv(file_path, start_date, end_date)
        if success:
            QMessageBox.information(self, "导出成功", f"数据已导出到:\n{file_path}")
        else:
            QMessageBox.warning(self, "导出失败", "导出失败，请查看日志")

    def on_delete_selected(self):
        """Delete selected snapshot row"""
        selected_rows = self.pnl_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择要删除的行")
            return

        dates_to_delete = []
        for index in selected_rows:
            date_item = self.pnl_table.item(index.row(), 0)
            if date_item:
                dates_to_delete.append(date_item.text())

        if not dates_to_delete:
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除以下日期的快照吗？\n{', '.join(dates_to_delete)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted = 0
        for d in dates_to_delete:
            if self.trade_service.delete_pnl_snapshot(d):
                deleted += 1

        if deleted > 0:
            QMessageBox.information(self, "成功", f"已删除 {deleted} 条记录")
            self.refresh_data()

    def on_row_double_clicked(self, item):
        """Double click row to edit snapshot"""
        row = item.row()
        date_item = self.pnl_table.item(row, 0)
        if not date_item:
            return

        snapshot_date = date_item.text()
        snapshot = self.trade_service.get_pnl_snapshot(snapshot_date)
        if not snapshot:
            return

        dialog = SnapshotDialog(
            self,
            snapshot_date=snapshot.snapshot_date,
            total_asset=snapshot.total_asset,
            cash=snapshot.cash,
            market_value=snapshot.market_value,
            position_count=snapshot.position_count,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            result = self.trade_service.save_daily_pnl(**data)
            if result:
                self.refresh_data()
