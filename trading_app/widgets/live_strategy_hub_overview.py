# -*- coding: utf-8 -*-
"""????????????"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class _OverviewCard(QFrame):
    """Small reusable card used by the hub overview dashboard."""

    def __init__(self, title: str, action_text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setStyleSheet(
            "QFrame { background:#1f1f1f; border:1px solid #3a3a3a; border-radius:8px; }"
            "QLabel { border:none; background:transparent; }"
            "QPushButton { background:#0078d4; color:#ffffff; border:none; border-radius:4px; padding:6px 12px; }"
            "QPushButton:hover { background:#1688dd; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight:bold;font-size:14px;color:#f3f4f6;")
        layout.addWidget(self.title_label)

        self.body_label = QLabel("-")
        self.body_label.setWordWrap(True)
        self.body_label.setStyleSheet("color:#d1d5db;font-size:12px;line-height:150%;")
        layout.addWidget(self.body_label, 1)

        self.action_btn = QPushButton(action_text)
        self.action_btn.setVisible(bool(action_text))
        self.action_btn.setFixedHeight(32)
        self.action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.action_btn, 0, Qt.AlignmentFlag.AlignRight)

    def set_body(self, lines: list[str] | tuple[str, ...] | str) -> None:
        if isinstance(lines, str):
            text = lines
        else:
            text = "\n".join(str(item) for item in lines if str(item or "").strip())
        self.body_label.setText(text or "-")


class _LiveStrategyOverviewWidget(QWidget):
    """Dashboard page for the live strategy center."""

    navigate_requested = pyqtSignal(str)
    account_settings_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        header = QLabel("实盘策略中枢总览")
        header.setStyleSheet("font-size:18px;font-weight:bold;color:#f3f4f6;")
        outer.addWidget(header)

        hint = QLabel("从这里快速确认连接、实盘策略、风险、任务、收益和日终状态。")
        hint.setStyleSheet("color:#9ca3af;font-size:12px;")
        outer.addWidget(hint)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        outer.addLayout(grid, 1)

        self.connectivity_card = _OverviewCard("连接与执行", "账户设置", self)
        self.connectivity_card.action_btn.clicked.connect(self.account_settings_requested.emit)
        grid.addWidget(self.connectivity_card, 0, 0)

        self.strategy_card = _OverviewCard("实盘策略运行", "查看实盘策略", self)
        self.strategy_card.action_btn.clicked.connect(lambda: self.navigate_requested.emit("ai"))
        grid.addWidget(self.strategy_card, 0, 1)

        self.risk_card = _OverviewCard("持仓与风险", "处理风险", self)
        self.risk_card.action_btn.clicked.connect(lambda: self.navigate_requested.emit("unmanaged"))
        grid.addWidget(self.risk_card, 1, 0)

        self.task_card = _OverviewCard("任务调度", "查看任务", self)
        self.task_card.action_btn.clicked.connect(lambda: self.navigate_requested.emit("tasks"))
        grid.addWidget(self.task_card, 1, 1)

        self.performance_card = _OverviewCard("实盘收益", "查看实盘收益", self)
        self.performance_card.action_btn.clicked.connect(lambda: self.navigate_requested.emit("performance"))
        grid.addWidget(self.performance_card, 2, 0)

        self.log_card = _OverviewCard("实盘运行与日终", "查看日志", self)
        self.log_card.action_btn.clicked.connect(lambda: self.navigate_requested.emit("logs"))
        grid.addWidget(self.log_card, 2, 1)

        for column in range(2):
            grid.setColumnStretch(column, 1)
        outer.addStretch(0)

    def refresh_view(self, state: dict) -> None:
        state = dict(state or {})
        self._refresh_connectivity(state)
        self._refresh_strategies(state)
        self._refresh_risk(state)
        self._refresh_tasks(state)
        self._refresh_performance(state)
        self._refresh_runtime(state)

    def _refresh_connectivity(self, state: dict) -> None:
        broker_connected = bool(state.get("broker_connected", False))
        qmt_running = bool(state.get("qmt_running", False))
        startup_running = bool(state.get("startup_running", False))
        mode = str(state.get("auto_trade_mode", "off") or "off")
        manual_enabled = bool(state.get("manual_orders_enabled", True))
        require_trading_time = bool(state.get("require_trading_time", True))
        self.connectivity_card.set_body([
            f"券商连接：{'已连接' if broker_connected else '未连接'}",
            f"QMT状态：{'运行中' if qmt_running else '未就绪'}",
            f"连接自检：{'进行中' if startup_running else '空闲'}",
            f"统一执行模式：{mode}",
            f"手动委托：{'开启' if manual_enabled else '关闭'}",
            f"交易时段闸：{'开启' if require_trading_time else '关闭'}",
        ])

    def _refresh_strategies(self, state: dict) -> None:
        rows = list(state.get("strategy_statuses", []) or [])
        lines: list[str] = []
        for item in rows:
            row = dict(item or {})
            name = str(row.get("strategy_name") or row.get("strategy_id") or "未命名策略")
            paused = bool(row.get("automation_paused", False))
            status = str(row.get("status") or row.get("state") or "运行中")
            lines.append(f"{name}：{'已暂停' if paused else status}")
        if not lines:
            lines.append("暂无策略状态")
        self.strategy_card.set_body(lines)

    def _refresh_risk(self, state: dict) -> None:
        risk_summary = dict(state.get("risk_summary", {}) or {})
        alert_counts = dict(state.get("alert_counts", {}) or {})
        open_alerts = int(alert_counts.get("open", 0) or 0)
        exception_count = int(state.get("exception_order_count", 0) or 0)
        items = list(risk_summary.get("items", []) or [])
        lines = [
            str(risk_summary.get("label", "风控: -") or "风控: -"),
            f"未处理告警：{open_alerts}",
            f"异常订单：{exception_count}",
        ]
        lines.extend(str(item) for item in items[:4])
        self.risk_card.set_body(lines)

    def _refresh_tasks(self, state: dict) -> None:
        tasks = list(state.get("tasks", []) or [])
        counts: dict[str, int] = {}
        for item in tasks:
            row = dict(item or {})
            status = str(row.get("status", "unknown") or "unknown").strip().lower()
            counts[status] = counts.get(status, 0) + 1
        running = counts.get("running", 0)
        failed = counts.get("failed", 0)
        completed = counts.get("completed", 0) + counts.get("success", 0)
        pending = max(len(tasks) - running - failed - completed, 0)
        self.task_card.set_body([
            f"任务总数：{len(tasks)}",
            f"运行中：{running}",
            f"待处理：{pending}",
            f"已完成：{completed}",
            f"失败：{failed}",
        ])

    def _refresh_performance(self, state: dict) -> None:
        self.performance_card.set_body([
            "实盘收益已接入策略账户、持仓行和日终快照。",
            "点击查看 AI实盘决策、ETF轮动实盘与未管理账户的收益归属。",
            f"最近状态更新时间：{state.get('updated_at', '-')}",
        ])

    def _refresh_runtime(self, state: dict) -> None:
        eod_state = dict(state.get("eod_state", {}) or {})
        eod_status = str(eod_state.get("status", "idle") or "idle")
        eod_error = str(eod_state.get("last_error", "") or "")
        center_paused = bool(state.get("center_automation_paused", False))
        lines = [
            f"今日日终：{eod_status}",
            f"中心自动化：{'已暂停' if center_paused else '正常'}",
            f"最近状态更新时间：{state.get('updated_at', '-')}",
        ]
        if eod_error:
            lines.append(f"日终错误：{eod_error}")
        self.log_card.set_body(lines)

