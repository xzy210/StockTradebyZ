from __future__ import annotations


class OrderStatus:
    PENDING_SUBMIT = "pending_submit"
    PENDING_FILL = "pending_fill"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


def resolve_order_status(fill: dict) -> str:
    if fill.get("filled", False):
        return OrderStatus.FILLED
    if fill.get("filled_qty", 0) > 0:
        return OrderStatus.PARTIALLY_FILLED
    if fill.get("timed_out"):
        return OrderStatus.TIMEOUT
    return OrderStatus.REJECTED
