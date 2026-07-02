from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QObject, pyqtSignal

from common.broker_session_service import BrokerSessionService
from trading_app.services.market_data.data_freshness_service import evaluate_xtquant_data_freshness

logger = logging.getLogger(__name__)


class QmtStartupOrchestrator(QObject):
    """Connect broker directly on startup and verify market-data access."""

    status_changed = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    _data_test_finished = pyqtSignal(bool, str)

    def __init__(self, broker_service: BrokerSessionService, parent=None):
        super().__init__(parent)
        self.broker_service = broker_service
        self._running = False
        self._cancelled = False
        self._waiting_for_connection = False
        self._data_test_finished.connect(self._on_data_test_finished)
        self.broker_service.connection_changed.connect(self._on_connection_changed)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        if self._running:
            return False

        config = self.broker_service.reload_config()
        qmt_path = str(config.get("qmt_path", "") or "").strip()
        account = str(config.get("account", "") or "").strip()
        if not qmt_path or not account:
            self.finished.emit(False, "券商配置缺失，请先填写 qmt_path 和 account")
            return False

        self._running = True
        self._cancelled = False
        if self.broker_service.is_connected:
            self._emit_status("券商已连接，开始检测数据链路")
            self._start_data_test()
            return True

        self._waiting_for_connection = True
        self._emit_status("开始连接券商")
        started = self.broker_service.connect_async(qmt_path, account)
        if not started:
            self._waiting_for_connection = False
            self._finish(False, "券商连接未能启动，请检查当前连接状态")
            return False
        return True

    def cancel(self) -> None:
        self._cancelled = True
        self._running = False
        self._waiting_for_connection = False

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        if self._cancelled or not self._running or not self._waiting_for_connection:
            return

        if message in ("正在连接券商...", "券商连接正在进行中"):
            self._emit_status(message)
            return

        self._waiting_for_connection = False
        if connected:
            self._emit_status("券商已连接，开始检测数据链路")
            self._start_data_test()
            return

        self._finish(False, message)

    def _start_data_test(self) -> None:
        self._emit_status("检测数据链路")

        def runner():
            try:
                report = evaluate_xtquant_data_freshness(require_minute_freshness=False)
                ok, message = report.ok, report.summary
            except Exception as exc:
                ok, message = False, f"数据链路检测异常: {exc}"
            if not self._cancelled:
                self._data_test_finished.emit(ok, message)

        threading.Thread(target=runner, daemon=True).start()

    def _on_data_test_finished(self, success: bool, message: str) -> None:
        if self._cancelled or not self._running:
            return
        self._emit_status(message)
        self._finish(success, message)

    def _finish(self, success: bool, message: str) -> None:
        if self._cancelled:
            return
        self._running = False
        self._waiting_for_connection = False
        if success:
            logger.info("QMT 连接自检完成: %s", message)
        else:
            logger.warning("QMT 连接自检失败: %s", message)
        self.finished.emit(success, message)

    def _emit_status(self, message: str) -> None:
        if self._cancelled:
            return
        logger.info("QMT 连接自检: %s", message)
        self.status_changed.emit(message)
