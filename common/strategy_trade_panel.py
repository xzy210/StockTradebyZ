from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, List, Optional

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QSizePolicy,
)

from trading_app.services.strategy_trade_view_service import get_strategy_trade_view_service
from trading_app.services.trade_record_service import get_trade_record_service

from common.broker_session_service import get_broker_session_service


class StrategyTradePanel(QWidget):
    order_requested = pyqtSignal(str, str, float)  # code, direction, price
    market_view_requested = pyqtSignal(str, str)  # code, name
    broker_sync_finished = pyqtSignal()

    def __init__(
        self,
        strategy_id: str,
        strategy_name: str,
        virtual_account_id: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.strategy_id = str(strategy_id or "").strip()
        self.strategy_name = str(strategy_name or "").strip()
        self.virtual_account_id = str(virtual_account_id or "").strip()
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.view_service = get_strategy_trade_view_service()
        self.trade_service = get_trade_record_service()
        self.trade_service.records_changed.connect(self._refresh_local_view)
        self.trade_service.pnl_snapshot_saved.connect(self._refresh_local_view)

        self._broker = get_broker_session_service()
        self._broker.trade_occurred.connect(self._on_broker_trade)
        self._broker.order_changed.connect(self._on_broker_order)
        self.broker_sync_finished.connect(self._on_broker_sync_finished)
        self._broker_sync_lock = threading.Lock()
        self._broker_sync_inflight = False
        self._broker_sync_pending = False
        self._broker_sync_force_pending = False
        self._last_broker_sync_started_at = 0.0
        self._broker_sync_interval_seconds = 30.0
        self._forced_broker_sync_interval_seconds = 3.0

        self._setup_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_all)
        self._refresh_timer.start(30_000)
        self.refresh_all()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel(f"策略交易面板 - {self.strategy_name or self.strategy_id}")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_all)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        self.positions_table = self._create_table(
            ["代码", "名称", "持仓", "可用", "成本", "现价", "市值", "浮盈亏", "盈亏%"],
            stretch_last=False,
        )
        self.positions_table.doubleClicked.connect(self._on_position_double_clicked)
        self.positions_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.positions_table.customContextMenuRequested.connect(self._on_position_context_menu)
        self.today_orders_table = self._create_table(
            ["时间", "代码", "名称", "方向", "状态", "委托", "成交", "模式", "备注"],
        )
        self.today_trades_table = self._create_table(
            ["时间", "代码", "名称", "方向", "价格", "数量", "金额", "来源", "备注"],
        )
        self.history_table = self._create_table(
            ["日期", "代码", "名称", "方向", "价格", "数量", "金额", "来源", "备注"],
        )
        self.ledger_table = self._create_table(
            ["日期", "时间", "操作", "代码", "名称", "变动金额", "佣金", "账本余额"],
        )

        self.tabs.addTab(self.positions_table, "当前持仓")
        self.tabs.addTab(self.today_orders_table, "当日委托")
        self.tabs.addTab(self.today_trades_table, "当日成交")
        self.tabs.addTab(self.history_table, "历史交易")
        self.tabs.addTab(self.ledger_table, "资金流水")
        self.tabs.addTab(self._build_equity_tab(), "收益曲线")

    def _create_table(self, headers: List[str], *, stretch_last: bool = True) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)
        table.setStyleSheet(
            """
            QTableWidget {
                background-color: #111111;
                alternate-background-color: #111111;
                color: #E5E7EB;
                gridline-color: #2A2A2A;
                border: 1px solid #2A2A2A;
                selection-background-color: #264F78;
                selection-color: #FFFFFF;
            }
            QTableWidget::item {
                background-color: #111111;
                color: #E5E7EB;
                padding: 4px 6px;
                border-bottom: 1px solid #1F2937;
            }
            QHeaderView::section {
                background-color: #1A1A1A;
                color: #D1D5DB;
                padding: 4px 6px;
                border: 1px solid #2A2A2A;
            }
            """
        )
        header = table.horizontalHeader()
        for idx in range(len(headers)):
            mode = QHeaderView.ResizeMode.Stretch if (stretch_last and idx >= len(headers) - 2) else QHeaderView.ResizeMode.ResizeToContents
            header.setSectionResizeMode(idx, mode)
        return table

    def _build_equity_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        stats_group = QGroupBox("收益概览")
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setSpacing(18)

        def _make_stat(name: str) -> QLabel:
            wrapper = QVBoxLayout()
            title = QLabel(name)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("color:#888; font-size:11px;")
            value = QLabel("-")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setStyleSheet("font-size:16px; font-weight:bold;")
            wrapper.addWidget(title)
            wrapper.addWidget(value)
            stats_layout.addLayout(wrapper)
            return value

        self.lbl_curve_latest = _make_stat("最新净值")
        self.lbl_curve_return = _make_stat("累计收益率")
        self.lbl_curve_drawdown = _make_stat("最大回撤")
        self.lbl_curve_days = _make_stat("样本天数")
        layout.addWidget(stats_group)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(12)
        splitter.setChildrenCollapsible(False)
        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setMinimumHeight(180)
        self.equity_plot.setMenuEnabled(False)
        self.equity_plot.setAntialiasing(True)
        self.equity_plot.setLabel("left", "累计收益率 (%)")
        self.equity_plot.setLabel("bottom", "样本序号")
        plot_layout.addWidget(self.equity_plot)
        splitter.addWidget(plot_host)

        self.equity_table = self._create_table(
            ["日期", "总资产", "现金", "持仓市值", "当日收益", "累计收益"],
        )
        splitter.addWidget(self.equity_table)
        splitter.setSizes([240, 200])
        splitter.setStyleSheet(
            "QSplitter::handle:vertical{background:#CBD5E1;border-radius:4px;margin:2px 0;}"
            "QSplitter::handle:vertical:hover{background:#94A3B8;}"
        )
        layout.addWidget(splitter, stretch=1)
        return page

    def refresh_all(self) -> None:
        self._schedule_broker_sync(force=False)
        self._refresh_local_view()

    def _refresh_local_view(self) -> None:
        self._apply_visual_style()
        self._refresh_positions()
        self._refresh_today_orders()
        self._refresh_today_trades()
        self._refresh_history()
        self._refresh_capital_ledger()
        self._refresh_equity_curve()

    def _on_broker_trade(self, trade_data: dict) -> None:
        try:
            added = self.trade_service.sync_broker_trades(
                [trade_data],
                strategy_id=self.strategy_id,
                virtual_account_id=self.virtual_account_id,
            )
            if added > 0:
                self.view_service.strategy_budget.rebuild_strategy_state_from_trade_records(
                    self.strategy_id,
                    strategy_name=self.strategy_name,
                    virtual_account_id=self.virtual_account_id,
                    real_total_asset=0.0,
                )
        except Exception:
            pass
        QTimer.singleShot(1500, lambda: self._refresh_after_broker_event(force_sync=True))

    def _on_broker_order(self, _order_data: dict) -> None:
        QTimer.singleShot(1000, lambda: self._refresh_after_broker_event(force_sync=True))

    def _refresh_after_broker_event(self, *, force_sync: bool) -> None:
        self._schedule_broker_sync(force=force_sync)
        self._refresh_local_view()

    def _schedule_broker_sync(self, *, force: bool) -> None:
        if not self._broker.is_connected:
            return
        now = time.monotonic()
        min_interval = (
            self._forced_broker_sync_interval_seconds
            if force else
            self._broker_sync_interval_seconds
        )
        with self._broker_sync_lock:
            if self._last_broker_sync_started_at > 0 and (now - self._last_broker_sync_started_at) < min_interval:
                return
            if self._broker_sync_inflight:
                self._broker_sync_pending = True
                self._broker_sync_force_pending = self._broker_sync_force_pending or force
                return
            self._broker_sync_inflight = True
            self._broker_sync_pending = False
            self._broker_sync_force_pending = False
            self._last_broker_sync_started_at = now
        threading.Thread(target=self._run_broker_sync, daemon=True).start()

    def _run_broker_sync(self) -> None:
        try:
            self.view_service.sync_strategy_broker_records(
                self.strategy_id,
                strategy_name=self.strategy_name,
                virtual_account_id=self.virtual_account_id,
            )
        except Exception:
            pass
        self.broker_sync_finished.emit()

    def _on_broker_sync_finished(self) -> None:
        should_reschedule = False
        force = False
        with self._broker_sync_lock:
            self._broker_sync_inflight = False
            if self._broker_sync_pending:
                should_reschedule = True
                force = self._broker_sync_force_pending
                self._broker_sync_pending = False
                self._broker_sync_force_pending = False
        self._refresh_local_view()
        if should_reschedule:
            self._schedule_broker_sync(force=force)

    def _refresh_positions(self) -> None:
        rows = self.view_service.get_strategy_positions(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
        )
        self.positions_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                item.get("stock_code", ""),
                item.get("stock_name", ""),
                str(int(item.get("volume", 0) or 0)),
                str(int(item.get("can_use_volume", 0) or 0)),
                f"{float(item.get('avg_cost', 0.0) or 0.0):.3f}" if item.get("avg_cost") else "-",
                f"{float(item.get('current_price', 0.0) or 0.0):.3f}" if item.get("current_price") else "-",
                f"{float(item.get('market_value', 0.0) or 0.0):,.2f}",
                f"{float(item.get('pnl', 0.0) or 0.0):+,.2f}",
                f"{float(item.get('pnl_pct', 0.0) or 0.0):+,.2f}%",
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if col in (2, 3, 4, 5, 6, 7, 8):
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col in (7, 8):
                    pnl = float(item.get("pnl", 0.0) or 0.0) if col == 7 else float(item.get("pnl_pct", 0.0) or 0.0)
                    cell.setForeground(QColor("#DC2626" if pnl >= 0 else "#16A34A"))
                self.positions_table.setItem(row, col, cell)
            code_item = self.positions_table.item(row, 0)
            if code_item is not None:
                code_item.setData(Qt.ItemDataRole.UserRole, str(item.get("stock_code", "") or ""))

    def _refresh_today_orders(self) -> None:
        rows = self.view_service.get_today_orders(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
        )
        self.today_orders_table.setRowCount(len(rows))
        for row, rec in enumerate(rows):
            ordered = f"{int(getattr(rec, 'order_volume', 0) or 0)}股 @ {float(getattr(rec, 'price', 0.0) or 0.0):.3f}"
            executed_volume = int(getattr(rec, "executed_volume", 0) or 0)
            executed = (
                f"{executed_volume}股 @ {float(getattr(rec, 'executed_price', 0.0) or 0.0):.3f}"
                if executed_volume > 0 else "-"
            )
            values = [
                str(getattr(rec, "created_at", "") or "")[-8:],
                getattr(rec, "stock_code", "") or "",
                getattr(rec, "stock_name", "") or "",
                "买入" if getattr(rec, "direction", "") == "buy" else "卖出",
                getattr(rec, "order_status_text", "") or getattr(rec, "status", "") or "",
                ordered,
                executed,
                getattr(rec, "execution_mode", "") or "",
                getattr(rec, "remark", "") or "",
            ]
            for col, value in enumerate(values):
                self.today_orders_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _refresh_today_trades(self) -> None:
        rows = self.view_service.get_today_trades(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
        )
        self.today_trades_table.setRowCount(len(rows))
        for row, rec in enumerate(rows):
            values = [
                self._format_trade_time(rec),
                self._get_trade_code(rec),
                self._get_trade_name(rec),
                self._get_trade_direction_label(rec),
                f"{self._get_trade_price(rec):.3f}",
                str(self._get_trade_volume(rec)),
                f"{self._get_trade_amount(rec):,.2f}",
                self._get_trade_source_label(rec),
                self._get_trade_remark(rec),
            ]
            for col, value in enumerate(values):
                self.today_trades_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _refresh_history(self) -> None:
        rows = self.view_service.get_trade_history(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
        )
        self.history_table.setRowCount(len(rows))
        for row, rec in enumerate(rows):
            values = [
                self._format_trade_date(rec),
                self._get_trade_code(rec),
                self._get_trade_name(rec),
                self._get_trade_direction_label(rec),
                f"{self._get_trade_price(rec):.3f}",
                str(self._get_trade_volume(rec)),
                f"{self._get_trade_amount(rec):,.2f}",
                self._get_trade_source_label(rec),
                self._get_trade_remark(rec),
            ]
            for col, value in enumerate(values):
                self.history_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _refresh_capital_ledger(self) -> None:
        rows = self.view_service.get_capital_ledger(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
        )
        self.ledger_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            amount = float(item.get("amount", 0.0) or 0.0)
            commission = float(item.get("commission", 0.0) or 0.0)
            fee_source = str(item.get("fee_source", "") or "").strip()
            balance = float(item.get("balance", 0.0) or 0.0)
            values = [
                str(item.get("date", "") or ""),
                str(item.get("time", "") or ""),
                str(item.get("action", "") or ""),
                str(item.get("code", "") or ""),
                str(item.get("name", "") or ""),
                f"{amount:+,.2f}",
                f"{commission:.2f} {fee_source}".strip() if commission else "-",
                f"{balance:,.2f}" if balance else "-",
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if col >= 5:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 5:
                    cell.setForeground(QColor("#DC2626" if amount >= 0 else "#16A34A"))
                self.ledger_table.setItem(row, col, cell)

    def _refresh_equity_curve(self) -> None:
        rows = self.view_service.get_equity_curve(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
        )
        self.equity_table.setRowCount(len(rows))
        self.equity_plot.clear()
        if not rows:
            self.lbl_curve_latest.setText("-")
            self.lbl_curve_return.setText("-")
            self.lbl_curve_drawdown.setText("-")
            self.lbl_curve_days.setText("0")
            return

        curve = [float(item.get("cumulative_return_pct", 0.0) or 0.0) for item in rows]
        x_values = list(range(len(curve)))
        bg, text, grid, accent = self._plot_colors()
        self.equity_plot.setBackground(bg)
        self.equity_plot.showGrid(x=True, y=True, alpha=0.18)
        plot_item = self.equity_plot.getPlotItem()
        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis.setTextPen(pg.mkPen(text))
            axis.setPen(pg.mkPen(grid))
        self.equity_plot.plot(
            x_values,
            curve,
            pen=pg.mkPen(accent, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(accent),
            symbolPen=pg.mkPen(accent),
        )
        self.equity_plot.addLine(y=0, pen=pg.mkPen(grid, style=Qt.PenStyle.DashLine))

        peak_return = curve[0] if curve else 0.0
        max_drawdown = 0.0
        for return_pct in curve:
            peak_return = max(peak_return, return_pct)
            drawdown = return_pct - peak_return
            max_drawdown = min(max_drawdown, drawdown)

        self.lbl_curve_latest.setText(f"{float(rows[-1].get('total_asset', 0.0) or 0.0):,.2f}")
        latest_return = float(rows[-1].get("cumulative_return_pct", 0.0) or 0.0)
        self.lbl_curve_return.setText(f"{latest_return:+.2f}%")
        self.lbl_curve_return.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{'#DC2626' if latest_return >= 0 else '#16A34A'};"
        )
        self.lbl_curve_drawdown.setText(f"{max_drawdown:.2f}%")
        self.lbl_curve_days.setText(str(len(rows)))

        for row, item in enumerate(reversed(rows)):
            display = [
                item.get("date", ""),
                f"{float(item.get('total_asset', 0.0) or 0.0):,.2f}",
                f"{float(item.get('cash', 0.0) or 0.0):,.2f}" if float(item.get("cash", 0.0) or 0.0) > 0 else "-",
                f"{float(item.get('market_value', 0.0) or 0.0):,.2f}" if float(item.get("market_value", 0.0) or 0.0) > 0 else "-",
                f"{float(item.get('daily_return_pct', 0.0) or 0.0):+,.2f}%",
                f"{float(item.get('cumulative_return_pct', 0.0) or 0.0):+,.2f}%",
            ]
            for col, value in enumerate(display):
                cell = QTableWidgetItem(str(value))
                if col >= 1:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col in (4, 5):
                    pct = float(item.get("daily_return_pct", 0.0) or 0.0) if col == 4 else float(item.get("cumulative_return_pct", 0.0) or 0.0)
                    cell.setForeground(QColor("#DC2626" if pct >= 0 else "#16A34A"))
                self.equity_table.setItem(row, col, cell)

    def _apply_visual_style(self) -> None:
        bg, text, grid, _accent = self._plot_colors()
        self.equity_plot.setBackground(bg)
        self.equity_plot.getPlotItem().getViewBox().setBorder(pg.mkPen(grid))
        self.equity_plot.getPlotItem().getAxis("left").setLabel(text="累计收益率 (%)", color=text.name())
        self.equity_plot.getPlotItem().getAxis("bottom").setLabel(text="样本序号", color=text.name())

    def _plot_colors(self) -> tuple[QColor, QColor, QColor, QColor]:
        palette = self.palette()
        bg = palette.color(QPalette.ColorRole.Base)
        if not bg.isValid():
            bg = palette.color(QPalette.ColorRole.Window)
        text = palette.color(QPalette.ColorRole.Text)
        if not text.isValid():
            text = QColor("#111827")
        grid = palette.color(QPalette.ColorRole.Mid)
        if not grid.isValid():
            grid = QColor("#94A3B8")
        luminance = (bg.red() * 299 + bg.green() * 587 + bg.blue() * 114) / 1000
        accent = QColor("#2563EB" if luminance > 128 else "#60A5FA")
        return bg, text, grid, accent

    def _on_position_double_clicked(self, index) -> None:
        row = index.row()
        if row < 0:
            return
        code_item = self.positions_table.item(row, 0)
        if code_item is None:
            return
        code = str(code_item.data(Qt.ItemDataRole.UserRole) or code_item.text() or "").strip()
        if not code:
            return
        self.order_requested.emit(code, "sell", 0.0)

    def _on_position_context_menu(self, pos) -> None:
        row = self.positions_table.rowAt(pos.y())
        if row < 0:
            return
        code_item = self.positions_table.item(row, 0)
        name_item = self.positions_table.item(row, 1)
        if code_item is None:
            return
        code = str(code_item.data(Qt.ItemDataRole.UserRole) or code_item.text() or "").strip()
        name = str(name_item.text() if name_item else "").strip()
        if not code:
            return
        menu = QMenu(self)
        view_action = menu.addAction(f"查看K线 {name}({code})" if name else f"查看K线 {code}")
        chosen = menu.exec(self.positions_table.viewport().mapToGlobal(pos))
        if chosen == view_action:
            self.market_view_requested.emit(code, name)

    @staticmethod
    def _record_value(rec: Any, *keys: str, default: Any = "") -> Any:
        for key in keys:
            if isinstance(rec, dict) and key in rec:
                value = rec.get(key)
            else:
                value = getattr(rec, key, None)
            if value not in (None, ""):
                return value
        return default

    def _format_trade_time(self, rec: Any) -> str:
        explicit_time = str(self._record_value(rec, "time", default="") or "").strip()
        if explicit_time:
            return explicit_time[-8:]
        created_at = str(self._record_value(rec, "created_at", default="") or "").strip()
        if created_at:
            return created_at[-8:]
        return "-"

    def _format_trade_date(self, rec: Any) -> str:
        raw_date = str(self._record_value(rec, "trade_date", "date", default="") or "").strip()
        for candidate in (
            raw_date,
            str(self._record_value(rec, "created_at", default="") or "").strip().split(" ")[0],
        ):
            if not candidate:
                continue
            try:
                parsed = datetime.strptime(candidate[:10], "%Y-%m-%d")
            except Exception:
                continue
            if parsed.year >= 2000:
                return parsed.strftime("%Y-%m-%d")
        return "-"

    def _get_trade_code(self, rec: Any) -> str:
        return str(self._record_value(rec, "stock_code", "code", default="") or "")

    def _get_trade_name(self, rec: Any) -> str:
        return str(self._record_value(rec, "stock_name", "name", default="") or "")

    def _get_trade_direction_label(self, rec: Any) -> str:
        direction = str(self._record_value(rec, "direction", "action", default="") or "").strip().lower()
        if direction in {"buy", "b", "买入"}:
            return "买入"
        if direction in {"sell", "s", "卖出"}:
            return "卖出"
        return "-"

    def _get_trade_price(self, rec: Any) -> float:
        return float(self._record_value(rec, "price", "filled_price", "ordered_price", default=0.0) or 0.0)

    def _get_trade_volume(self, rec: Any) -> int:
        return int(self._record_value(rec, "volume", "quantity", "filled_qty", "ordered_qty", default=0) or 0)

    def _get_trade_amount(self, rec: Any) -> float:
        amount = float(self._record_value(rec, "amount", default=0.0) or 0.0)
        if amount > 0:
            return amount
        return round(self._get_trade_price(rec) * self._get_trade_volume(rec), 2)

    def _get_trade_source_label(self, rec: Any) -> str:
        label = str(self._record_value(rec, "source_display", "source", default="") or "").strip()
        if label:
            return label
        if self.strategy_id == "ai_trade_decision_center":
            return "AI智能"
        return self.strategy_name or "-"

    def _get_trade_remark(self, rec: Any) -> str:
        return str(self._record_value(rec, "remark", "reason", default="") or "")
