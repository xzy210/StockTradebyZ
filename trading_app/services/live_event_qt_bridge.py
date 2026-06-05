from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from common.trading_runtime import (
    EVENT_ACCOUNT,
    EVENT_CONNECTION,
    EVENT_ORDER,
    EVENT_ORDER_ERROR,
    EVENT_ORDER_EXECUTION,
    EVENT_POSITION,
    EVENT_TRADE,
    EVENT_TIMER,
    LiveEvent,
    LiveEventEngine,
)


class QtLiveEventBridge(QObject):
    """Bridge LiveEventEngine callbacks into the Qt main-thread signal model."""

    live_event_received = pyqtSignal(object)
    events_changed = pyqtSignal()
    order_changed = pyqtSignal(object)
    trade_occurred = pyqtSignal(object)
    account_changed = pyqtSignal(object)
    position_changed = pyqtSignal(object)
    connection_changed = pyqtSignal(object)
    broker_error = pyqtSignal(object)
    order_execution_event = pyqtSignal(object)

    def __init__(self, event_engine: LiveEventEngine, parent=None) -> None:
        super().__init__(parent)
        self.event_engine = event_engine
        self._registered = False
        self.start()

    def start(self) -> None:
        if self._registered:
            return
        self.event_engine.register_general(self._on_live_event)
        self._registered = True

    def stop(self) -> None:
        if not self._registered:
            return
        self.event_engine.unregister_general(self._on_live_event)
        self._registered = False

    def _on_live_event(self, event: LiveEvent) -> None:
        if event.type == EVENT_TIMER:
            return
        self.live_event_received.emit(event)
        if event.type == EVENT_ORDER:
            self.order_changed.emit(event.data)
            self.events_changed.emit()
        elif event.type == EVENT_TRADE:
            self.trade_occurred.emit(event.data)
            self.events_changed.emit()
        elif event.type == EVENT_ACCOUNT:
            self.account_changed.emit(event.data)
        elif event.type == EVENT_POSITION:
            self.position_changed.emit(event.data)
        elif event.type == EVENT_CONNECTION:
            self.connection_changed.emit(event.data)
            self.events_changed.emit()
        elif event.type == EVENT_ORDER_ERROR:
            self.broker_error.emit(event.data)
            self.events_changed.emit()
        elif event.type == EVENT_ORDER_EXECUTION:
            self.order_execution_event.emit(event.data)
            self.events_changed.emit()


__all__ = ["QtLiveEventBridge"]
