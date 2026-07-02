# -*- coding: utf-8 -*-
"""AI ???????? worker?"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


class _AccountRefreshWorker(QThread):
    refresh_ready = pyqtSignal(object, object)
    refresh_failed = pyqtSignal(str)

    def __init__(self, broker, parent=None):
        super().__init__(parent)
        self.broker = broker

    def run(self):
        try:
            asset = self.broker.query_stock_asset()
            positions = self.broker.query_stock_positions() or []
            asset_payload = {
                "total_asset": float(getattr(asset, "total_asset", 0) or 0.0),
                "cash": float(getattr(asset, "cash", 0) or getattr(asset, "available_cash", 0) or 0.0),
                "market_value": float(getattr(asset, "market_value", 0) or 0.0),
            }
            position_payloads = [
                {
                    "stock_code": str(getattr(pos, "stock_code", "") or ""),
                    "stock_name": str(getattr(pos, "stock_name", "") or ""),
                    "volume": int(getattr(pos, "volume", 0) or 0),
                    "can_use_volume": int(getattr(pos, "can_use_volume", 0) or 0),
                    "open_price": float(getattr(pos, "open_price", 0) or 0.0),
                    "market_value": float(getattr(pos, "market_value", 0) or 0.0),
                    "profit_rate": float(getattr(pos, "profit_rate", 0) or 0.0),
                }
                for pos in positions
            ]
            self.refresh_ready.emit(asset_payload, position_payloads)
        except Exception as exc:
            self.refresh_failed.emit(str(exc))


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


class _ReconcileCatchupWorker(QThread):
    finished_reconcile = pyqtSignal(bool, str)
    failed_reconcile = pyqtSignal(str)

    def __init__(self, daily_auto_trade, parent=None):
        super().__init__(parent)
        self.daily_auto_trade = daily_auto_trade

    def run(self):
        try:
            success, message = self.daily_auto_trade.run_reconcile_catchup_if_needed()
            self.finished_reconcile.emit(success, message)
        except Exception as exc:
            self.failed_reconcile.emit(str(exc))


# ───────────────────────────────────────────────────────────────────────────
#  Left panel: Account & Position overview
# ───────────────────────────────────────────────────────────────────────────

