from __future__ import annotations

import re
from datetime import datetime, timedelta

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
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

_LEVEL_FG_COLORS = {
    "danger": QColor("#ff6b6b"),
    "warning": QColor("#ffb454"),
    "info": QColor("#6ec1e4"),
    "success": QColor("#7ed957"),
}
_LEVEL_LABELS = {
    "danger": "严重",
    "warning": "警告",
    "info": "信息",
    "success": "成功",
}
_CATEGORY_LABELS = {
    "startup": "连接自检",
    "broker_error": "券商异常",
    "broker_disconnected": "券商断开",
    "order_exception": "异常委托",
    "order_execution": "订单执行",
    "decision_alert": "价格预警",
    "ai_task": "AI 任务",
    "end_of_day": "日终",
    "etf_trade": "ETF 交易",
}
_STATUS_LABELS = {
    "open": "未处理",
    "read": "已读",
    "ignored": "已忽略",
    "resolved": "已解决",
}


class LiveStrategyAlertCenterWidget(QWidget):
    navigate_requested = pyqtSignal(str)
    market_view_requested = pyqtSignal(str, str)

    def __init__(self, alert_service, parent=None) -> None:
        super().__init__(parent)
        self.alert_service = alert_service
        self._rows: list[dict] = []
        self._session_started_at = datetime.now()
        self._setup_ui()
        # 设置默认值时暂时屏蔽 currentIndexChanged，避免构造阶段触发多次冗余刷新。
        for combo, default_index in (
            (self.level_combo, 0),  # 默认"仅告警"
            (self.status_combo, 1),  # 默认"未处理"
            (self.category_combo, 0),
            (self.time_combo, 1),  # 默认"今日"
        ):
            combo.blockSignals(True)
            combo.setCurrentIndex(default_index)
            combo.blockSignals(False)
        self.alert_service.events_changed.connect(self.refresh_events)
        self.refresh_events()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("状态"))
        self.status_combo = QComboBox()
        self.status_combo.addItem("全部", "")
        self.status_combo.addItem("未处理", "open")
        self.status_combo.addItem("已读", "read")
        self.status_combo.addItem("已解决", "resolved")
        self.status_combo.addItem("已忽略", "ignored")
        self.status_combo.currentIndexChanged.connect(self.refresh_events)
        filter_row.addWidget(self.status_combo)

        filter_row.addWidget(QLabel("级别"))
        self.level_combo = QComboBox()
        self.level_combo.addItem("仅告警", "alert_only")
        self.level_combo.addItem("警告", "warning")
        self.level_combo.addItem("严重", "danger")
        self.level_combo.addItem("全部", "")
        self.level_combo.currentIndexChanged.connect(self.refresh_events)
        filter_row.addWidget(self.level_combo)

        filter_row.addWidget(QLabel("分类"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("全部", "")
        for item in [
            ("连接自检", "startup"),
            ("券商异常", "broker_error"),
            ("券商断开", "broker_disconnected"),
            ("异常委托", "order_exception"),
            ("订单执行", "order_execution"),
            ("价格预警", "decision_alert"),
            ("AI 任务", "ai_task"),
            ("日终", "end_of_day"),
            ("ETF 交易", "etf_trade"),
        ]:
            self.category_combo.addItem(item[0], item[1])
        self.category_combo.currentIndexChanged.connect(self.refresh_events)
        filter_row.addWidget(self.category_combo)

        filter_row.addWidget(QLabel("时间"))
        self.time_combo = QComboBox()
        self.time_combo.addItem("本次启动", "session")
        self.time_combo.addItem("今日", "today")
        self.time_combo.addItem("最近3天", "3d")
        self.time_combo.addItem("最近7天", "7d")
        self.time_combo.addItem("全部历史", "all")
        self.time_combo.currentIndexChanged.connect(self.refresh_events)
        filter_row.addWidget(self.time_combo)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_events)
        filter_row.addWidget(refresh_btn)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["时间", "级别", "分类", "来源", "关联订单", "标题", "状态", "消息"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.table, 1)

        action_row = QHBoxLayout()
        read_btn = QPushButton("标记已读")
        read_btn.clicked.connect(lambda: self._mark_selected("read"))
        action_row.addWidget(read_btn)

        resolve_btn = QPushButton("标记已解决")
        resolve_btn.clicked.connect(lambda: self._mark_selected("resolved"))
        action_row.addWidget(resolve_btn)

        ignore_btn = QPushButton("忽略")
        ignore_btn.clicked.connect(lambda: self._mark_selected("ignored"))
        action_row.addWidget(ignore_btn)

        jump_btn = QPushButton("跳转相关页面")
        jump_btn.clicked.connect(self._jump_to_related)
        action_row.addWidget(jump_btn)

        market_btn = QPushButton("看K线")
        market_btn.clicked.connect(self._open_related_market_view)
        action_row.addWidget(market_btn)
        action_row.addStretch()

        self.lbl_count = QLabel("共 0 条")
        action_row.addWidget(self.lbl_count)
        layout.addLayout(action_row)

    def refresh_events(self) -> None:
        start_time, end_time = self._resolve_time_range()
        rows = self.alert_service.list_events(
            status=str(self.status_combo.currentData() or ""),
            category=str(self.category_combo.currentData() or ""),
            start_time=start_time,
            end_time=end_time,
            limit=400,
            include_ignored=self.status_combo.currentData() == "ignored" or not self.status_combo.currentData(),
        )
        self._rows = [item for item in rows if self._matches_level(item)]
        self.table.setRowCount(len(self._rows))
        for row, item in enumerate(self._rows):
            level = str(item.get("level", "") or "").strip().lower()
            status = str(item.get("status", "") or "").strip().lower()
            category = str(item.get("category", "") or "").strip().lower()
            message = str(item.get("message", "") or "")
            order_detail = str(item.get("order_observable_detail", "") or "")
            if category == "order_execution" and order_detail:
                message = f"{order_detail}｜{message}" if message else order_detail
            values = [
                str(item.get("occurred_at", "") or ""),
                _LEVEL_LABELS.get(level, level),
                _CATEGORY_LABELS.get(category, category or "-"),
                str(item.get("source", "") or ""),
                str(item.get("order_reference", "") or "-"),
                str(item.get("title", "") or ""),
                _STATUS_LABELS.get(status, status or "-"),
                message,
            ]
            fg = _LEVEL_FG_COLORS.get(level)
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if fg is not None and col in (1, 5):
                    cell.setForeground(QBrush(fg))
                self.table.setItem(row, col, cell)
        self.lbl_count.setText(f"共 {len(self._rows)} 条")

    def _matches_level(self, item: dict) -> bool:
        selected = str(self.level_combo.currentData() or "").strip().lower()
        level = str(item.get("level", "") or "").strip().lower()
        selected_category = str(self.category_combo.currentData() or "").strip().lower()
        if not selected:
            return True
        if selected == "alert_only":
            if selected_category == "order_execution":
                return True
            return level in {"warning", "danger"}
        return level == selected

    def _resolve_time_range(self) -> tuple[str, str]:
        mode = str(self.time_combo.currentData() or "today")
        now = datetime.now()
        if mode == "all":
            return "", ""
        if mode == "session":
            start = self._session_started_at
        elif mode == "3d":
            start = now - timedelta(days=2)
        elif mode == "7d":
            start = now - timedelta(days=6)
        else:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")

    def _selected_event(self) -> dict:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return {}
        return dict(self._rows[row] or {})

    def _mark_selected(self, status: str) -> None:
        event = self._selected_event()
        if not event:
            QMessageBox.information(self, "提示", "请先选择一条事件。")
            return
        self.alert_service.mark_event_status(str(event.get("event_id", "") or ""), status)
        self.refresh_events()

    def _jump_to_related(self) -> None:
        event = self._selected_event()
        if not event:
            return
        category = str(event.get("category", "") or "")
        source = str(event.get("source", "") or "")
        if category in {"broker_error", "order_exception"}:
            self.navigate_requested.emit("exceptions")
            return
        if category == "order_execution":
            status = str(event.get("status", "") or "").strip().lower()
            self.navigate_requested.emit("exceptions" if status == "open" else "logs")
            return
        if source.startswith("ai") or category in {"decision_alert", "ai_task"}:
            self.navigate_requested.emit("ai")
            return
        if source.startswith("etf") or category == "etf_trade":
            self.navigate_requested.emit("etf")
            return
        if category in {"startup", "end_of_day"}:
            self.navigate_requested.emit("tasks")
            return
        self.navigate_requested.emit("logs")

    def _open_related_market_view(self) -> None:
        event = self._selected_event()
        if not event:
            QMessageBox.information(self, "提示", "请先选择一条事件。")
            return
        code = self._extract_symbol_code(event)
        if not code:
            QMessageBox.information(self, "提示", "这条事件没有可识别的标的代码。")
            return
        self.market_view_requested.emit(code, "")

    @staticmethod
    def _extract_symbol_code(event: dict) -> str:
        for key in ("stock_code", "symbol_code", "symbol", "code"):
            value = str(event.get(key, "") or "").strip()
            if value:
                return value.split(".", 1)[0]
        text = " ".join(
            str(event.get(key, "") or "")
            for key in ("title", "message", "order_reference", "order_observable_detail")
        )
        match = re.search(r"(?<!\d)(?:[0368]\d{5}|15\d{4}|16\d{4}|5[123568]\d{4})(?!\d)", text)
        return match.group(0) if match else ""
