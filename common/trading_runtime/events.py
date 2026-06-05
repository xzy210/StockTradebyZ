from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


EVENT_TIMER = "eLiveTimer"
EVENT_TICK = "eLiveTick."
EVENT_ORDER = "eLiveOrder."
EVENT_TRADE = "eLiveTrade."
EVENT_POSITION = "eLivePosition."
EVENT_ACCOUNT = "eLiveAccount."
EVENT_CONTRACT = "eLiveContract."
EVENT_CONNECTION = "eLiveConnection."
EVENT_ORDER_ERROR = "eLiveOrderError."
EVENT_ORDER_EXECUTION = "eLiveOrderExecution."
EVENT_LOG = "eLiveLog"


@dataclass(frozen=True)
class LiveEvent:
    type: str
    data: Any = None
    gateway_name: str = ""
    symbol: str = ""
    key: str = ""
    message: str = ""
    level: str = "info"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_type(self, event_type: str) -> "LiveEvent":
        return LiveEvent(
            type=event_type,
            data=self.data,
            gateway_name=self.gateway_name,
            symbol=self.symbol,
            key=self.key,
            message=self.message,
            level=self.level,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
        )


__all__ = [
    "EVENT_ACCOUNT",
    "EVENT_CONNECTION",
    "EVENT_CONTRACT",
    "EVENT_LOG",
    "EVENT_ORDER",
    "EVENT_ORDER_ERROR",
    "EVENT_ORDER_EXECUTION",
    "EVENT_POSITION",
    "EVENT_TICK",
    "EVENT_TIMER",
    "EVENT_TRADE",
    "LiveEvent",
]
