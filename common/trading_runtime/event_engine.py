from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from queue import Empty, Queue
from threading import RLock, Thread
from time import sleep
from typing import Optional

from .events import EVENT_TIMER, LiveEvent

logger = logging.getLogger(__name__)

LiveEventHandler = Callable[[LiveEvent], None]


class LiveEventEngine:
    """Queue-backed event engine for live trading runtime events."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = max(float(interval or 1.0), 0.1)
        self._queue: Queue[LiveEvent] = Queue()
        self._active = False
        self._thread: Optional[Thread] = None
        self._timer: Optional[Thread] = None
        self._handlers: dict[str, list[LiveEventHandler]] = defaultdict(list)
        self._general_handlers: list[LiveEventHandler] = []
        self._lock = RLock()
        self.dispatch_errors: list[Exception] = []

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True
            self._thread = Thread(target=self._run, name="LiveEventEngine", daemon=True)
            self._timer = Thread(target=self._run_timer, name="LiveEventTimer", daemon=True)
            self._thread.start()
            self._timer.start()

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            thread = self._thread
            timer = self._timer
            self._thread = None
            self._timer = None
        if timer and timer.is_alive():
            timer.join(timeout=self._interval + 1.0)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    def put(self, event: LiveEvent) -> None:
        if not isinstance(event, LiveEvent):
            raise TypeError("event must be LiveEvent")
        self._queue.put(event)

    def register(self, event_type: str, handler: LiveEventHandler) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        key = str(event_type or "")
        with self._lock:
            handlers = self._handlers[key]
            if handler not in handlers:
                handlers.append(handler)

    def unregister(self, event_type: str, handler: LiveEventHandler) -> None:
        key = str(event_type or "")
        with self._lock:
            handlers = self._handlers.get(key, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers and key in self._handlers:
                self._handlers.pop(key, None)

    def register_general(self, handler: LiveEventHandler) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            if handler not in self._general_handlers:
                self._general_handlers.append(handler)

    def unregister_general(self, handler: LiveEventHandler) -> None:
        with self._lock:
            if handler in self._general_handlers:
                self._general_handlers.remove(handler)

    def _run(self) -> None:
        while self._active:
            try:
                event = self._queue.get(block=True, timeout=0.5)
            except Empty:
                continue
            self._process(event)

    def _run_timer(self) -> None:
        while self._active:
            sleep(self._interval)
            if self._active:
                self.put(LiveEvent(EVENT_TIMER))

    def _process(self, event: LiveEvent) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
            general_handlers = list(self._general_handlers)
        for handler in [*handlers, *general_handlers]:
            try:
                handler(event)
            except Exception as exc:
                self.dispatch_errors.append(exc)
                logger.exception("Live event handler failed: type=%s handler=%r", event.type, handler)


__all__ = ["LiveEventEngine", "LiveEventHandler"]
