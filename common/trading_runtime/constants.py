from __future__ import annotations

from enum import Enum


class Exchange(str, Enum):
    """Supported market suffixes used by the live trading runtime."""

    SH = "SH"
    SZ = "SZ"
    UNKNOWN = ""


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    SUBMITTING = "submitting"
    NOT_TRADED = "not_traded"
    PART_TRADED = "part_traded"
    ALL_TRADED = "all_traded"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


ACTIVE_ORDER_STATUSES = {
    OrderStatus.SUBMITTING,
    OrderStatus.NOT_TRADED,
    OrderStatus.PART_TRADED,
}


XT_ORDER_STATUS_MAP = {
    48: OrderStatus.SUBMITTING,
    49: OrderStatus.SUBMITTING,
    50: OrderStatus.NOT_TRADED,
    51: OrderStatus.SUBMITTING,
    52: OrderStatus.PART_TRADED,
    53: OrderStatus.CANCELLED,
    54: OrderStatus.CANCELLED,
    55: OrderStatus.PART_TRADED,
    56: OrderStatus.ALL_TRADED,
    57: OrderStatus.REJECTED,
}


def normalize_exchange(value: str | Exchange | None) -> Exchange:
    if isinstance(value, Exchange):
        return value
    text = str(value or "").strip().upper()
    if text in {"SH", "SSE", "XSHG", "SHSE"}:
        return Exchange.SH
    if text in {"SZ", "SZSE", "XSHE"}:
        return Exchange.SZ
    return Exchange.UNKNOWN


def split_symbol(symbol: str) -> tuple[str, Exchange]:
    value = str(symbol or "").strip().upper()
    if "." in value:
        code, suffix = value.split(".", 1)
        return code, normalize_exchange(suffix)
    if len(value) == 6:
        if value.startswith(("5", "6", "9")):
            return value, Exchange.SH
        return value, Exchange.SZ
    return value, Exchange.UNKNOWN


def normalize_symbol(symbol: str) -> str:
    code, exchange = split_symbol(symbol)
    if not code:
        return ""
    return f"{code}.{exchange.value}" if exchange.value else code


def plain_symbol(symbol: str) -> str:
    return split_symbol(symbol)[0]


def direction_from_order_type(order_type: int | str | None) -> Direction:
    text = str(order_type or "").strip().lower()
    if text in {"23", "buy", "b"}:
        return Direction.BUY
    return Direction.SELL


def order_type_code(direction: Direction | str) -> int:
    return 23 if Direction(str(direction)) == Direction.BUY else 24


def status_from_xt_code(value: int | str | None) -> OrderStatus:
    try:
        code = int(value or 0)
    except (TypeError, ValueError):
        return OrderStatus.UNKNOWN
    return XT_ORDER_STATUS_MAP.get(code, OrderStatus.UNKNOWN)
