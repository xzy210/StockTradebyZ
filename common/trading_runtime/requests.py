from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.broker_interface import BrokerOrderRequest
from common.execution_contract import OrderIntent

from .constants import Direction, Exchange, OrderType, direction_from_order_type, normalize_symbol, order_type_code, split_symbol


@dataclass(frozen=True)
class SubscribeRequest:
    symbol: str
    exchange: Exchange = Exchange.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code, exchange = split_symbol(self.symbol)
        object.__setattr__(self, "symbol", code)
        object.__setattr__(self, "exchange", self.exchange if self.exchange != Exchange.UNKNOWN else exchange)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    direction: Direction
    volume: int
    price: float = 0.0
    order_type: OrderType = OrderType.LIMIT
    exchange: Exchange = Exchange.UNKNOWN
    price_type: int = 5
    strategy_name: str = ""
    remark: str = ""
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code, exchange = split_symbol(self.symbol)
        object.__setattr__(self, "symbol", code)
        object.__setattr__(self, "exchange", self.exchange if self.exchange != Exchange.UNKNOWN else exchange)
        object.__setattr__(self, "direction", Direction(str(self.direction)))
        object.__setattr__(self, "volume", abs(int(self.volume or 0)))
        object.__setattr__(self, "price", float(self.price or 0.0))
        object.__setattr__(self, "price_type", int(self.price_type or 5))
        object.__setattr__(self, "strategy_name", str(self.strategy_name or "").strip())
        object.__setattr__(self, "remark", str(self.remark or "").strip())
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)

    @property
    def broker_order_type(self) -> int:
        return order_type_code(self.direction)

    def to_broker_order_request(self) -> BrokerOrderRequest:
        return BrokerOrderRequest(
            stock_code=self.vt_symbol,
            order_type=self.broker_order_type,
            order_volume=self.volume,
            price_type=self.price_type,
            price=self.price,
            strategy_name=self.strategy_name,
            remark=self.remark,
            request_id=self.request_id,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_broker_order_request(cls, request: BrokerOrderRequest) -> "OrderRequest":
        return cls(
            symbol=request.stock_code,
            direction=direction_from_order_type(request.order_type),
            volume=request.order_volume,
            price=request.price,
            price_type=request.price_type,
            strategy_name=request.strategy_name,
            remark=request.remark,
            request_id=request.request_id,
            metadata=dict(request.metadata or {}),
        )

    @classmethod
    def from_order_intent(cls, intent: OrderIntent, *, stock_name: str = "") -> "OrderRequest":
        kwargs = intent.to_execution_request_kwargs(stock_name=stock_name)
        return cls(
            symbol=str(kwargs.get("stock_code", "") or ""),
            direction=direction_from_order_type(kwargs.get("order_type")),
            volume=int(kwargs.get("order_volume", 0) or 0),
            price=float(kwargs.get("price", 0.0) or 0.0),
            price_type=int(kwargs.get("price_type", 5) or 5),
            strategy_name=str(kwargs.get("strategy_name", "") or ""),
            remark=str(kwargs.get("remark", "") or ""),
            request_id=str(kwargs.get("intent_id", "") or ""),
            metadata=dict(kwargs.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class CancelRequest:
    order_id: str
    symbol: str = ""
    exchange: Exchange = Exchange.UNKNOWN
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code, exchange = split_symbol(self.symbol)
        object.__setattr__(self, "symbol", code)
        object.__setattr__(self, "exchange", self.exchange if self.exchange != Exchange.UNKNOWN else exchange)
        object.__setattr__(self, "order_id", str(self.order_id or "").strip())
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)


__all__ = ["CancelRequest", "OrderRequest", "SubscribeRequest"]
