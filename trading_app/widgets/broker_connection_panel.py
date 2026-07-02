from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.broker_session_service import get_broker_session_service

logger = logging.getLogger(__name__)


class _ClientStatusWorker(QThread):
    finished_status = pyqtSignal(dict)
    failed_status = pyqtSignal(str)

    def __init__(self, broker, parent=None):
        super().__init__(parent)
        self.broker = broker

    def run(self):
        try:
            self.finished_status.emit(self.broker.get_client_status())
        except Exception as exc:
            self.failed_status.emit(str(exc))


class BrokerConnectionPanel(QGroupBox):
    broker_connected = pyqtSignal()
    broker_disconnected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("miniQMT 连接", parent)
        self.broker = get_broker_session_service()
        self._status_worker = None
        self._was_connected = bool(self.broker.is_connected)
        self._setup_ui()
        self._load_config()
        self.broker.connection_changed.connect(self._on_connection_changed)
        self.broker.config_changed.connect(self._on_config_changed)
        self.broker.client_state_changed.connect(self._on_client_state_changed)
        self._apply_connection_ui(self.broker.is_connected, "券商会话已连接" if self.broker.is_connected else "券商已断开")
        self._refresh_client_status_safe()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("font-weight: bold;")
        top_row.addWidget(self.status_label)

        self.client_status_label = QLabel("客户端: 未检测")
        self.client_status_label.setWordWrap(True)
        self.client_status_label.setStyleSheet("color:#888;")
        top_row.addWidget(self.client_status_label, stretch=1)
        layout.addLayout(top_row)

        action_row = QHBoxLayout()
        self.connect_btn = QPushButton("连接券商")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        action_row.addWidget(self.connect_btn)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setMaximumWidth(32)
        self.settings_btn.setToolTip("展开/收起连接设置")
        self.settings_btn.clicked.connect(self._toggle_settings)
        action_row.addWidget(self.settings_btn)
        layout.addLayout(action_row)

        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(4)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("路径:"))
        self.edit_qmt_path = QLineEdit()
        self.edit_qmt_path.setPlaceholderText(r"例: D:\中金财富QMT个人版交易端\userdata_mini")
        path_row.addWidget(self.edit_qmt_path)
        browse_btn = QPushButton("…")
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._on_browse_qmt_path)
        path_row.addWidget(browse_btn)
        settings_layout.addLayout(path_row)

        account_row = QHBoxLayout()
        account_row.addWidget(QLabel("账户:"))
        self.edit_account = QLineEdit()
        self.edit_account.setPlaceholderText("资金账号")
        account_row.addWidget(self.edit_account)
        settings_layout.addLayout(account_row)

        self.settings_widget.setVisible(False)
        layout.addWidget(self.settings_widget)

    def _load_config(self):
        config = self.broker.get_config()
        self.edit_qmt_path.setText(config.get("qmt_path", ""))
        self.edit_account.setText(config.get("account", ""))

    def _on_config_changed(self, config: dict):
        self.edit_qmt_path.setText(config.get("qmt_path", ""))
        self.edit_account.setText(config.get("account", ""))

    def _toggle_settings(self):
        self.settings_widget.setVisible(not self.settings_widget.isVisible())

    def _on_browse_qmt_path(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "选择 miniQMT userdata_mini 目录",
            self.edit_qmt_path.text() or "C:\\",
        )
        if path:
            self.edit_qmt_path.setText(path)

    def _on_connect_clicked(self):
        if self.broker.is_connected:
            self.broker.disconnect()
            return
        qmt_path = self.edit_qmt_path.text().strip()
        account = self.edit_account.text().strip()
        if not qmt_path:
            self.settings_widget.setVisible(True)
            QMessageBox.warning(self, "提示", "请先填写 miniQMT 路径")
            return
        if not account:
            self.settings_widget.setVisible(True)
            QMessageBox.warning(self, "提示", "请先填写资金账号")
            return
        self.broker.save_config({"qmt_path": qmt_path, "account": account})
        started = self.broker.connect_async(qmt_path, account)
        if not started:
            self.connect_btn.setEnabled(True)

    def _on_connection_changed(self, connected: bool, message: str):
        self._apply_connection_ui(connected, message)
        if connected and not self._was_connected:
            self._was_connected = True
            self.settings_widget.setVisible(False)
            self.broker_connected.emit()
        elif not connected and self._was_connected and message not in ("正在连接券商...", "券商连接正在进行中"):
            self._was_connected = False
            self.broker_disconnected.emit()

    def _apply_connection_ui(self, connected: bool, message: str):
        if connected:
            account = self.broker.get_config().get("account", "")
            self.status_label.setText(f"已连接 {account}".strip())
            self.status_label.setStyleSheet("font-weight:bold;color:#16A34A;")
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("断开")
            return
        if message in ("正在连接券商...", "券商连接正在进行中"):
            self.status_label.setText("连接中...")
            self.status_label.setStyleSheet("font-weight:bold;color:#D97706;")
            self.connect_btn.setEnabled(False)
            self.connect_btn.setText("连接中...")
            return
        self.status_label.setText("未连接")
        self.status_label.setStyleSheet("font-weight:bold;color:#888;")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("连接券商")

    def _on_client_state_changed(self, _state: dict):
        self._refresh_client_status_safe()

    def _refresh_client_status_safe(self):
        if self._status_worker and self._status_worker.isRunning():
            return
        self._status_worker = _ClientStatusWorker(self.broker, parent=self)
        self._status_worker.finished_status.connect(self._apply_client_status)
        self._status_worker.failed_status.connect(self._apply_client_status_error)
        self._status_worker.start()

    def _apply_client_status(self, status: dict):
        message = status.get("message", "未检测")
        self.client_status_label.setText(f"客户端: {message}")
        running = bool(status.get("running"))
        if running:
            self.client_status_label.setStyleSheet("color:#16A34A;")
        else:
            self.client_status_label.setStyleSheet("color:#888;")
        self._status_worker = None

    def _apply_client_status_error(self, message: str):
        logger.warning("刷新 QMT 客户端状态失败: %s", message)
        self.client_status_label.setText("客户端: 状态检测失败")
        self.client_status_label.setStyleSheet("color:#DC2626;")
        self._status_worker = None

