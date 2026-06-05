# -*- coding: utf-8 -*-
"""???????? worker?"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from trading_app.services.ops.live_strategy_end_of_day_service import LiveStrategyEndOfDayService


class _EndOfDayWorker(QThread):
    finished_cycle = pyqtSignal(str, bool, str, object)
    failed_cycle = pyqtSignal(str, str)

    def __init__(self, service: LiveStrategyEndOfDayService, mode: str, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.mode = mode

    def run(self) -> None:
        try:
            if self.mode == "manual":
                success, message, payload = self.service.run_manual_cycle()
            else:
                raise ValueError(f"unsupported end-of-day mode: {self.mode}")
            self.finished_cycle.emit(self.mode, success, message, payload)
        except Exception as exc:
            self.failed_cycle.emit(self.mode, str(exc))


class _KlineRefreshWorker(QThread):
    status_message = pyqtSignal(str)
    finished_refresh = pyqtSignal(bool, str)
    failed_refresh = pyqtSignal(str)

    def __init__(self, rotation_etf_pool: list[str], parent=None) -> None:
        super().__init__(parent)
        self.rotation_etf_pool = list(rotation_etf_pool or [])

    def run(self) -> None:
        try:
            from trading_app.services.market_data.kline_full_refresh_service import KlineFullRefreshService

            service = KlineFullRefreshService(rotation_etf_pool=self.rotation_etf_pool)
            success, message = service.run_full_refresh(status_cb=self.status_message.emit)
            self.finished_refresh.emit(success, message)
        except Exception as exc:
            self.failed_refresh.emit(str(exc))

