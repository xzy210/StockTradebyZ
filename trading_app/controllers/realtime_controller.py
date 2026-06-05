from __future__ import annotations

import logging

from PyQt6.QtCore import QObject

from trading_app.services.market_data.quote_service import QuoteData, get_quote_service

logger = logging.getLogger(__name__)


class RealtimeController(QObject):
    """Coordinates quote service lifecycle and app-level quote fan-out."""

    STARTUP_OWNER = "main:startup-watchlist"

    def __init__(self, main_window, trading_bridge=None, parent=None):
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.trading_bridge = trading_bridge
        self.quote_service = get_quote_service()
        self._connected = False

    def initialize(self):
        if self._connected:
            return
        if not self.quote_service.is_available:
            return
        self.quote_service.quote_updated.connect(self._on_quote_updated)
        self.quote_service.connection_status_changed.connect(self._on_quote_status_changed)
        self._connected = True

    def shutdown(self):
        try:
            self.quote_service.clear_owner_subscription(self.STARTUP_OWNER)
            if self.quote_service.is_running:
                self.quote_service.stop()
        except Exception as exc:
            logger.debug("关闭实时行情服务失败: %s", exc)

    def subscribe_watchlist_on_startup(self):
        if not self.quote_service.is_available:
            return
        try:
            all_watchlist_codes = set()
            groups = self.main_window.watchlist_manager.get_all_groups()
            for group in groups:
                codes = self.main_window.watchlist_manager.get_stocks_in_group(group)
                all_watchlist_codes.update(codes)

            if not all_watchlist_codes:
                return

            self.quote_service.replace_subscription(
                self.STARTUP_OWNER,
                list(all_watchlist_codes),
                start_service=True,
            )
            self.main_window.statusBar().showMessage(
                f"📡 已预订阅 {len(all_watchlist_codes)} 只自选股实时行情",
                3000,
            )
        except Exception as exc:
            logger.debug("启动时预订阅自选股失败: %s", exc)

    def pause_for_sync(self):
        try:
            self.quote_service.clear_owner_subscription(self.STARTUP_OWNER)
            if self.quote_service.is_running:
                self.quote_service.stop()
        except Exception as exc:
            logger.debug("同步前暂停实时行情失败: %s", exc)

    def resume_after_sync(self, had_running: bool):
        if had_running:
            self.subscribe_watchlist_on_startup()

    def _on_quote_updated(self, quote_data: QuoteData):
        try:
            if self.main_window.right_tabs.currentIndex() == 1:
                self.main_window.timeshare_widget.update_realtime_quote(quote_data)

            if self.main_window.right_tabs.currentIndex() == 2:
                simple_code = quote_data.simple_code
                card = self.main_window.watchlist_panel.cards_map.get(simple_code)
                if card:
                    card.update_realtime(quote_data)

            if self.trading_bridge:
                self.trading_bridge.on_quote_updated(quote_data)
        except Exception:
            pass

    def _on_quote_status_changed(self, connected: bool, message: str):
        if connected:
            self.main_window.statusBar().showMessage(f"📡 {message}", 3000)
        else:
            self.main_window.statusBar().showMessage(f"📡 {message}", 5000)
