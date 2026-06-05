from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .constants import (
    ACTIVE_ORDER_STATUSES,
    Direction,
    Exchange,
    OrderStatus,
    OrderType,
    direction_from_order_type,
    normalize_exchange,
    normalize_symbol,
    plain_symbol,
    split_symbol,
    status_from_xt_code,
)


def _get_value(source: Any, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return default
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _to_datetime(value: Any = None) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%H:%M:%S":
                now = datetime.now()
                return now.replace(hour=parsed.hour, minute=parsed.minute, second=parsed.second, microsecond=0)
            return parsed
        except ValueError:
            continue
    return datetime.now()


@dataclass(frozen=True)
class BaseTradingData:
    gateway_name: str
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TickData(BaseTradingData):
    symbol: str = ""
    exchange: Exchange = Exchange.UNKNOWN
    datetime: datetime = field(default_factory=datetime.now)
    name: str = ""
    last_price: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0
    bid_price_1: float = 0.0
    bid_volume_1: float = 0.0
    ask_price_1: float = 0.0
    ask_volume_1: float = 0.0

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)


@dataclass(frozen=True)
class OrderData(BaseTradingData):
    symbol: str = ""
    exchange: Exchange = Exchange.UNKNOWN
    order_id: str = ""
    direction: Direction = Direction.BUY
    order_type: OrderType = OrderType.LIMIT
    price: float = 0.0
    volume: int = 0
    traded: int = 0
    status: OrderStatus = OrderStatus.UNKNOWN
    status_code: int = 0
    status_message: str = ""
    datetime: datetime = field(default_factory=datetime.now)
    reference: str = ""
    request_id: str = ""

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)

    @property
    def vt_orderid(self) -> str:
        return f"{self.gateway_name}.{self.order_id}" if self.gateway_name else str(self.order_id)

    def is_active(self) -> bool:
        return self.status in ACTIVE_ORDER_STATUSES


@dataclass(frozen=True)
class TradeData(BaseTradingData):
    symbol: str = ""
    exchange: Exchange = Exchange.UNKNOWN
    order_id: str = ""
    trade_id: str = ""
    direction: Direction = Direction.BUY
    price: float = 0.0
    volume: int = 0
    amount: float = 0.0
    datetime: datetime = field(default_factory=datetime.now)
    reference: str = ""
    request_id: str = ""

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)

    @property
    def vt_orderid(self) -> str:
        return f"{self.gateway_name}.{self.order_id}" if self.gateway_name else str(self.order_id)

    @property
    def vt_tradeid(self) -> str:
        trade_id = self.trade_id or f"{self.order_id}:{self.datetime.timestamp()}"
        return f"{self.gateway_name}.{trade_id}" if self.gateway_name else trade_id


@dataclass(frozen=True)
class PositionData(BaseTradingData):
    symbol: str = ""
    exchange: Exchange = Exchange.UNKNOWN
    name: str = ""
    volume: int = 0
    available: int = 0
    frozen: int = 0
    price: float = 0.0
    pnl: float = 0.0

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)

    @property
    def vt_positionid(self) -> str:
        return f"{self.gateway_name}.{self.vt_symbol}.net" if self.gateway_name else f"{self.vt_symbol}.net"


@dataclass(frozen=True)
class AccountData(BaseTradingData):
    account_id: str = ""
    balance: float = 0.0
    available: float = 0.0
    frozen: float = 0.0
    market_value: float = 0.0

    @property
    def vt_accountid(self) -> str:
        return f"{self.gateway_name}.{self.account_id}" if self.gateway_name else self.account_id


@dataclass(frozen=True)
class ContractData(BaseTradingData):
    symbol: str = ""
    exchange: Exchange = Exchange.UNKNOWN
    name: str = ""
    min_volume: int = 100
    price_tick: float = 0.01

    @property
    def vt_symbol(self) -> str:
        return normalize_symbol(f"{self.symbol}.{self.exchange.value}" if self.exchange.value else self.symbol)


@dataclass(frozen=True)
class LogData(BaseTradingData):
    message: str = ""
    level: str = "info"
    datetime: datetime = field(default_factory=datetime.now)


def order_from_xt(order: Any, *, gateway_name: str) -> OrderData:
    raw_symbol = str(_get_value(order, "stock_code", "symbol", default="") or "")
    code, exchange = split_symbol(raw_symbol)
    status_code = _to_int(_get_value(order, "order_status", "status_code", default=0))
    return OrderData(
        gateway_name=gateway_name,
        symbol=code,
        exchange=exchange,
        order_id=str(_get_value(order, "order_id", "orderid", default="") or ""),
        direction=direction_from_order_type(_get_value(order, "order_type", default=23)),
        order_type=OrderType.LIMIT,
        price=_to_float(_get_value(order, "price", "order_price", default=0.0)),
        volume=_to_int(_get_value(order, "order_volume", "volume", default=0)),
        traded=_to_int(_get_value(order, "traded_volume", "traded", default=0)),
        status=status_from_xt_code(status_code),
        status_code=status_code,
        status_message=str(_get_value(order, "status_msg", "status_message", default="") or ""),
        datetime=_to_datetime(_get_value(order, "order_time", "datetime", default=None)),
        reference=str(_get_value(order, "order_remark", "remark", "strategy_name", default="") or ""),
        request_id=str(_get_value(order, "request_id", default="") or ""),
        raw=order,
    )


def trade_from_xt(trade: Any, *, gateway_name: str) -> TradeData:
    raw_symbol = str(_get_value(trade, "stock_code", "symbol", default="") or "")
    code, exchange = split_symbol(raw_symbol)
    price = _to_float(_get_value(trade, "traded_price", "price", default=0.0))
    volume = _to_int(_get_value(trade, "traded_volume", "volume", default=0))
    return TradeData(
        gateway_name=gateway_name,
        symbol=code,
        exchange=exchange,
        order_id=str(_get_value(trade, "order_id", "orderid", default="") or ""),
        trade_id=str(_get_value(trade, "traded_id", "trade_id", "tradeid", default="") or ""),
        direction=direction_from_order_type(_get_value(trade, "order_type", default=23)),
        price=price,
        volume=volume,
        amount=_to_float(_get_value(trade, "traded_amount", "amount", default=price * volume)),
        datetime=_to_datetime(_get_value(trade, "traded_time", "datetime", default=None)),
        reference=str(_get_value(trade, "order_remark", "remark", "strategy_name", default="") or ""),
        request_id=str(_get_value(trade, "request_id", default="") or ""),
        raw=trade,
    )


def position_from_xt(position: Any, *, gateway_name: str) -> PositionData:
    raw_symbol = str(_get_value(position, "stock_code", "symbol", default="") or "")
    code, exchange = split_symbol(raw_symbol)
    volume = _to_int(_get_value(position, "volume", "current_amount", "position_volume", default=0))
    available = _to_int(_get_value(position, "can_use_volume", "enable_amount", default=volume))
    return PositionData(
        gateway_name=gateway_name,
        symbol=code,
        exchange=exchange,
        name=str(_get_value(position, "stock_name", "name", default="") or ""),
        volume=volume,
        available=available,
        frozen=max(volume - available, 0),
        price=_to_float(_get_value(position, "avg_price", "open_price", "cost_price", default=0.0)),
        pnl=_to_float(_get_value(position, "profit", "pnl", "float_profit", default=0.0)),
        raw=position,
    )


def account_from_xt(asset: Any, *, gateway_name: str, account_id: str = "") -> AccountData:
    balance = _to_float(_get_value(asset, "total_asset", "balance", default=0.0))
    available = _to_float(_get_value(asset, "cash", "available_cash", "available", default=0.0))
    return AccountData(
        gateway_name=gateway_name,
        account_id=account_id or str(_get_value(asset, "account_id", "accountid", default="") or ""),
        balance=balance,
        available=available,
        frozen=max(balance - available, 0.0),
        market_value=_to_float(_get_value(asset, "market_value", "stock_value", default=0.0)),
        raw=asset,
    )


__all__ = [
    "AccountData",
    "BaseTradingData",
    "ContractData",
    "LogData",
    "OrderData",
    "PositionData",
    "TickData",
    "TradeData",
    "account_from_xt",
    "order_from_xt",
    "position_from_xt",
    "trade_from_xt",
]
