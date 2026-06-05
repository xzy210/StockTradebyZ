from __future__ import annotations

from datetime import datetime, timedelta

try:
    from trading_app.services.execution.trade_record_service import get_trade_record_service
except ImportError:
    from trading_app.services.execution.trade_record_service import get_trade_record_service

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_EXCEPTION_STATUSES = ("blocked", "failed", "cancelled", "rejected")

_STATUS_LABELS = {
    "blocked": "阻断",
    "failed": "失败",
    "cancelled": "已撤销",
    "rejected": "已拒绝",
}

_STATUS_FG_COLORS = {
    "rejected": QColor("#ff6b6b"),
    "failed": QColor("#ff6b6b"),
    "blocked": QColor("#ffb454"),
    "cancelled": QColor("#9e9e9e"),
}

# 这些状态表明订单从未真正挂到券商（或已终结），清理时可以放心归档。
_NEVER_LIVE_STATUSES = {"blocked", "cancelled", "rejected"}


class LiveStrategyExceptionOrderWidget(QWidget):
    navigate_requested = pyqtSignal(str)
    market_view_requested = pyqtSignal(str, str)

    def __init__(self, broker_service, parent=None) -> None:
        super().__init__(parent)
        self.broker_service = broker_service
        self.trade_service = get_trade_record_service()
        self._rows: list[object] = []
        self._setup_ui()
        self.time_combo.setCurrentIndex(0)
        self.trade_service.order_record_added.connect(lambda *_: self.refresh_orders())
        self.trade_service.order_record_updated.connect(lambda *_: self.refresh_orders())
        self.refresh_orders()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("时间"))
        self.time_combo = QComboBox()
        self.time_combo.addItem("今日", "today")
        self.time_combo.addItem("最近3天", "3d")
        self.time_combo.addItem("最近7天", "7d")
        self.time_combo.addItem("全部历史", "all")
        self.time_combo.currentIndexChanged.connect(self.refresh_orders)
        action_row.addWidget(self.time_combo)

        action_row.addWidget(QLabel("状态"))
        self.status_combo = QComboBox()
        self.status_combo.addItem("全部异常", "")
        for code in _EXCEPTION_STATUSES:
            self.status_combo.addItem(_STATUS_LABELS[code], code)
        self.status_combo.currentIndexChanged.connect(self.refresh_orders)
        action_row.addWidget(self.status_combo)

        self.show_archived_chk = QCheckBox("显示已忽略")
        self.show_archived_chk.setChecked(False)
        self.show_archived_chk.stateChanged.connect(self.refresh_orders)
        action_row.addWidget(self.show_archived_chk)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_orders)
        action_row.addWidget(refresh_btn)

        cancel_btn = QPushButton("撤单")
        cancel_btn.clicked.connect(self._cancel_selected)
        action_row.addWidget(cancel_btn)

        market_btn = QPushButton("看K线")
        market_btn.clicked.connect(self._open_selected_market_view)
        action_row.addWidget(market_btn)

        ignore_btn = QPushButton("忽略选中")
        ignore_btn.clicked.connect(self._ignore_selected)
        action_row.addWidget(ignore_btn)

        cleanup_btn = QPushButton("一键清理未成交")
        cleanup_btn.setToolTip("批量归档当前列表中的 阻断/已撤销/已拒绝 记录（这类订单未真正在券商挂单）")
        cleanup_btn.clicked.connect(self._cleanup_never_live)
        action_row.addWidget(cleanup_btn)

        action_row.addStretch()

        self.lbl_count = QLabel("共 0 条")
        action_row.addWidget(self.lbl_count)
        layout.addLayout(action_row)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["时间", "请求ID", "标的", "状态", "模式", "来源", "券商委托号", "消息"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.table, 1)

    def refresh_orders(self) -> None:
        start_time, end_time = self._resolve_time_range()
        include_archived = self.show_archived_chk.isChecked() if hasattr(self, "show_archived_chk") else False
        try:
            records = self.trade_service.get_order_records(
                start_time=start_time,
                end_time=end_time,
                include_archived=include_archived,
                limit=500,
            )
        except Exception:
            records = []
        status_filter = str(self.status_combo.currentData() or "") if hasattr(self, "status_combo") else ""
        self._rows = [
            item for item in records
            if str(getattr(item, "status", "") or "").strip().lower() in _EXCEPTION_STATUSES
            and (not status_filter or str(getattr(item, "status", "") or "").strip().lower() == status_filter)
        ]
        self.table.setRowCount(len(self._rows))
        for row, item in enumerate(self._rows):
            status_raw = str(getattr(item, "status", "") or "").strip().lower()
            archived = int(getattr(item, "archived", 0) or 0)
            values = [
                str(getattr(item, "created_at", "") or ""),
                str(getattr(item, "request_id", "") or ""),
                f"{getattr(item, 'stock_name', '')}({getattr(item, 'stock_code', '')})",
                _STATUS_LABELS.get(status_raw, status_raw or "-") + (" (已忽略)" if archived else ""),
                str(getattr(item, "execution_mode", "") or ""),
                str(getattr(item, "source", "") or ""),
                str(int(getattr(item, "broker_order_id", 0) or 0)),
                str(getattr(item, "validation_message", "") or ""),
            ]
            fg = _STATUS_FG_COLORS.get(status_raw) if not archived else QColor("#9e9e9e")
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if fg is not None and col == 3:
                    cell.setForeground(QBrush(fg))
                self.table.setItem(row, col, cell)
        active_count = sum(1 for item in self._rows if not int(getattr(item, "archived", 0) or 0))
        self.lbl_count.setText(
            f"共 {len(self._rows)} 条（活跃 {active_count}）" if include_archived else f"共 {len(self._rows)} 条"
        )

    def _resolve_time_range(self) -> tuple[str, str]:
        mode = str(self.time_combo.currentData() or "today")
        now = datetime.now()
        if mode == "all":
            return "", ""
        if mode == "3d":
            start = now - timedelta(days=2)
        elif mode == "7d":
            start = now - timedelta(days=6)
        else:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")

    def _selected_records(self) -> list:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        return [self._rows[row] for row in rows if 0 <= row < len(self._rows)]

    def _selected_record(self):
        records = self._selected_records()
        return records[0] if records else None

    def _cancel_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            QMessageBox.information(self, "异常订单", "请先选择一条委托记录。")
            return
        order_id = int(getattr(record, "broker_order_id", 0) or 0)
        if order_id <= 0:
            QMessageBox.information(self, "异常订单", "这条记录没有可撤销的券商委托号。")
            return
        try:
            self.broker_service.cancel_order_stock(order_id)
        except Exception as exc:
            QMessageBox.warning(self, "异常订单", f"撤单失败: {exc}")
            return
        QMessageBox.information(self, "异常订单", f"已发送撤单请求: {order_id}")

    def _open_selected_market_view(self) -> None:
        record = self._selected_record()
        if record is None:
            QMessageBox.information(self, "异常订单", "请先选择一条委托记录。")
            return
        code = str(getattr(record, "stock_code", "") or "").strip()
        name = str(getattr(record, "stock_name", "") or "").strip()
        if not code:
            QMessageBox.information(self, "异常订单", "该记录没有可查看的标的代码。")
            return
        self.market_view_requested.emit(code, name)

    def _ignore_selected(self) -> None:
        records = self._selected_records()
        if not records:
            QMessageBox.information(self, "异常订单", "请先选择要忽略的记录（支持多选）。")
            return
        ids = [str(getattr(r, "request_id", "") or "") for r in records]
        ids = [rid for rid in ids if rid]
        if not ids:
            return
        confirm = QMessageBox.question(
            self,
            "异常订单",
            f"将标记 {len(ids)} 条记录为已忽略，不再出现在异常订单列表。是否继续？",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        count = self.trade_service.archive_order_records(ids)
        QMessageBox.information(self, "异常订单", f"已忽略 {count} 条记录。")
        self.refresh_orders()

    def _cleanup_never_live(self) -> None:
        candidates = [
            r for r in self._rows
            if str(getattr(r, "status", "") or "").strip().lower() in _NEVER_LIVE_STATUSES
            and not int(getattr(r, "archived", 0) or 0)
        ]
        if not candidates:
            QMessageBox.information(self, "异常订单", "当前列表中没有可清理的记录。")
            return
        confirm = QMessageBox.question(
            self,
            "异常订单",
            f"将批量归档 {len(candidates)} 条 阻断/已撤销/已拒绝 记录（这类订单未真正在券商挂单）。是否继续？",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        ids = [str(getattr(r, "request_id", "") or "") for r in candidates]
        ids = [rid for rid in ids if rid]
        count = self.trade_service.archive_order_records(ids)
        QMessageBox.information(self, "异常订单", f"已清理 {count} 条记录。")
        self.refresh_orders()
