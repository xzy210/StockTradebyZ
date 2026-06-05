from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

ORDER_STATUS_LABELS = {
    48: "未报",
    49: "待报",
    50: "已报",
    51: "已报待撤",
    52: "部成待撤",
    53: "部撤",
    54: "已撤",
    55: "部成",
    56: "已成",
    57: "废单",
}
FINAL_REJECTED_STATUSES = {53, 54, 57}
FILLED_STATUSES = {55, 56}


class OrderLifecycleState(str, Enum):
    REQUESTED = "requested"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    CANCEL_PENDING = "cancel_pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMEOUT_PENDING = "timeout_pending"


TERMINAL_LIFECYCLE_STATES = {
    OrderLifecycleState.FILLED,
    OrderLifecycleState.CANCELLED,
    OrderLifecycleState.REJECTED,
}

LEGAL_TRANSITIONS: dict[OrderLifecycleState, set[OrderLifecycleState]] = {
    OrderLifecycleState.REQUESTED: {
        OrderLifecycleState.SUBMITTED,
        OrderLifecycleState.ACCEPTED,
        OrderLifecycleState.CANCEL_PENDING,
        OrderLifecycleState.PARTIALLY_FILLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.TIMEOUT_PENDING,
    },
    OrderLifecycleState.SUBMITTED: {
        OrderLifecycleState.ACCEPTED,
        OrderLifecycleState.CANCEL_PENDING,
        OrderLifecycleState.PARTIALLY_FILLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.TIMEOUT_PENDING,
    },
    OrderLifecycleState.ACCEPTED: {
        OrderLifecycleState.CANCEL_PENDING,
        OrderLifecycleState.PARTIALLY_FILLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.TIMEOUT_PENDING,
    },
    OrderLifecycleState.CANCEL_PENDING: {
        OrderLifecycleState.PARTIALLY_FILLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.TIMEOUT_PENDING,
    },
    OrderLifecycleState.PARTIALLY_FILLED: {
        OrderLifecycleState.CANCEL_PENDING,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.TIMEOUT_PENDING,
    },
    OrderLifecycleState.TIMEOUT_PENDING: {
        OrderLifecycleState.SUBMITTED,
        OrderLifecycleState.ACCEPTED,
        OrderLifecycleState.CANCEL_PENDING,
        OrderLifecycleState.PARTIALLY_FILLED,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
    },
    OrderLifecycleState.FILLED: set(),
    OrderLifecycleState.CANCELLED: set(),
    OrderLifecycleState.REJECTED: set(),
}


class OrderStateTransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrderStateSnapshot:
    status_code: int
    status_text: str
    status_message: str
    traded_volume: int
    traded_price: float

    @property
    def lifecycle_state(self) -> OrderLifecycleState:
        return lifecycle_state_from_snapshot(self)

    @property
    def has_trade(self) -> bool:
        return self.status_code in FILLED_STATUSES or self.traded_volume > 0

    @property
    def is_fully_filled(self) -> bool:
        return self.status_code == 56

    @property
    def is_partially_filled(self) -> bool:
        return self.has_trade and not self.is_fully_filled

    @property
    def is_rejected_terminal(self) -> bool:
        return self.status_code in FINAL_REJECTED_STATUSES

    @property
    def trade_record_status(self) -> str:
        if self.is_fully_filled:
            return "filled"
        if self.is_partially_filled:
            return "partial_fill"
        if self.is_rejected_terminal:
            return "rejected"
        return "submitted"

    @property
    def fill_event_type(self) -> str:
        return "OrderFilled" if self.is_fully_filled else "OrderPartiallyFilled"

    @property
    def fill_event_title(self) -> str:
        return "订单已成交" if self.is_fully_filled else "订单部分成交"


@dataclass(frozen=True)
class OrderLifecycleEvent:
    event_type: str
    snapshot: OrderStateSnapshot | None = None
    occurred_at: float | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderStateTransition:
    request_id: str
    previous_state: OrderLifecycleState
    current_state: OrderLifecycleState
    snapshot: OrderStateSnapshot | None
    event_type: str
    is_idempotent: bool = False
    occurred_at: float | None = None


class OrderLifecycle:
    def __init__(
        self,
        request_id: str,
        *,
        state: OrderLifecycleState = OrderLifecycleState.REQUESTED,
        updated_at: float | None = None,
        snapshot: OrderStateSnapshot | None = None,
    ) -> None:
        self.request_id = str(request_id)
        self.state = state
        self.updated_at = updated_at
        self.snapshot = snapshot
        self.transitions: list[OrderStateTransition] = []

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_LIFECYCLE_STATES

    def on_broker_event(self, event: Any, *, occurred_at: float | None = None) -> OrderStateTransition:
        lifecycle_event = normalize_lifecycle_event(event, occurred_at=occurred_at)
        if lifecycle_event.snapshot is None:
            raise ValueError("broker event must include an order state snapshot")
        target_state = lifecycle_state_from_snapshot(lifecycle_event.snapshot)
        return self._apply(
            target_state,
            snapshot=lifecycle_event.snapshot,
            event_type=lifecycle_event.event_type,
            occurred_at=lifecycle_event.occurred_at,
        )

    def on_timeout(self, *, occurred_at: float | None = None) -> OrderStateTransition:
        if self.is_terminal:
            return OrderStateTransition(
                request_id=self.request_id,
                previous_state=self.state,
                current_state=self.state,
                snapshot=self.snapshot,
                event_type="OrderTimeoutIgnored",
                is_idempotent=True,
                occurred_at=occurred_at,
            )
        return self._apply(
            OrderLifecycleState.TIMEOUT_PENDING,
            snapshot=self.snapshot,
            event_type="OrderTimeoutPending",
            occurred_at=occurred_at,
        )

    def mark_timeout_if_stale(self, *, now: float, timeout_seconds: float) -> OrderStateTransition | None:
        if self.is_terminal or self.updated_at is None:
            return None
        if float(now) - float(self.updated_at) < float(timeout_seconds):
            return None
        return self.on_timeout(occurred_at=now)

    def apply_event(self, event: OrderLifecycleEvent) -> OrderStateTransition:
        if event.event_type == "OrderTimeoutPending":
            return self.on_timeout(occurred_at=event.occurred_at)
        if event.snapshot is not None:
            return self.on_broker_event(event, occurred_at=event.occurred_at)
        target_state = _state_from_event_type(event.event_type)
        if target_state is None:
            raise ValueError(f"unsupported lifecycle event: {event.event_type}")
        return self._apply(
            target_state,
            snapshot=self.snapshot,
            event_type=event.event_type,
            occurred_at=event.occurred_at,
        )

    def _apply(
        self,
        target_state: OrderLifecycleState,
        *,
        snapshot: OrderStateSnapshot | None,
        event_type: str,
        occurred_at: float | None,
    ) -> OrderStateTransition:
        previous = self.state
        is_same_state = target_state == previous
        is_same_snapshot = snapshot == self.snapshot
        if not is_same_state and target_state not in LEGAL_TRANSITIONS.get(previous, set()):
            raise OrderStateTransitionError(
                f"illegal order state transition for {self.request_id}: {previous.value} -> {target_state.value}"
            )

        transition = OrderStateTransition(
            request_id=self.request_id,
            previous_state=previous,
            current_state=target_state,
            snapshot=snapshot,
            event_type=event_type,
            is_idempotent=is_same_state and is_same_snapshot,
            occurred_at=occurred_at,
        )
        self.state = target_state
        self.snapshot = snapshot or self.snapshot
        self.updated_at = occurred_at if occurred_at is not None else self.updated_at
        self.transitions.append(transition)
        return transition

    @classmethod
    def rebuild(
        cls,
        request_id: str,
        events: Iterable[Any],
        *,
        initial_state: OrderLifecycleState = OrderLifecycleState.REQUESTED,
    ) -> "OrderLifecycle":
        lifecycle = cls(request_id, state=initial_state)
        for event in events:
            lifecycle.apply_event(normalize_lifecycle_event(event))
        return lifecycle


def normalize_order_state(order: Any) -> OrderStateSnapshot:
    status_code = _to_int(getattr(order, "order_status", 0) or getattr(order, "status_code", 0))
    status_value = getattr(order, "status", "")
    status_text = ORDER_STATUS_LABELS.get(status_code, str(getattr(status_value, "value", status_value) or status_code))
    return OrderStateSnapshot(
        status_code=status_code,
        status_text=status_text,
        status_message=str(getattr(order, "status_msg", "") or getattr(order, "status_message", "") or ""),
        traded_volume=_to_int(getattr(order, "traded_volume", 0) or getattr(order, "traded", 0)),
        traded_price=_to_float(getattr(order, "traded_price", 0) or getattr(order, "price", 0)),
    )


def lifecycle_state_from_snapshot(snapshot: OrderStateSnapshot) -> OrderLifecycleState:
    if snapshot.status_code == 56:
        return OrderLifecycleState.FILLED
    if snapshot.status_code in {53, 54}:
        return OrderLifecycleState.CANCELLED
    if snapshot.status_code == 57:
        return OrderLifecycleState.REJECTED
    if snapshot.status_code in {52, 55} or snapshot.traded_volume > 0:
        return OrderLifecycleState.PARTIALLY_FILLED
    if snapshot.status_code == 51:
        return OrderLifecycleState.CANCEL_PENDING
    if snapshot.status_code == 50:
        return OrderLifecycleState.ACCEPTED
    if snapshot.status_code in {48, 49}:
        return OrderLifecycleState.SUBMITTED
    return OrderLifecycleState.SUBMITTED


def normalize_lifecycle_event(event: Any, *, occurred_at: float | None = None) -> OrderLifecycleEvent:
    if isinstance(event, OrderLifecycleEvent):
        if occurred_at is None:
            return event
        return OrderLifecycleEvent(
            event_type=event.event_type,
            snapshot=event.snapshot,
            occurred_at=occurred_at,
            message=event.message,
            payload=dict(event.payload),
        )
    if isinstance(event, OrderStateSnapshot):
        return OrderLifecycleEvent(
            event_type=_event_type_from_state(event.lifecycle_state),
            snapshot=event,
            occurred_at=occurred_at,
        )
    snapshot = normalize_order_state(event)
    return OrderLifecycleEvent(
        event_type=_event_type_from_state(snapshot.lifecycle_state),
        snapshot=snapshot,
        occurred_at=occurred_at,
        message=str(getattr(event, "status_msg", "") or ""),
    )


def rebuild_order_lifecycle(request_id: str, events: Iterable[Any]) -> OrderLifecycle:
    return OrderLifecycle.rebuild(request_id, events)


def _event_type_from_state(state: OrderLifecycleState) -> str:
    return {
        OrderLifecycleState.REQUESTED: "OrderRequested",
        OrderLifecycleState.SUBMITTED: "OrderSubmitted",
        OrderLifecycleState.ACCEPTED: "OrderAccepted",
        OrderLifecycleState.CANCEL_PENDING: "OrderCancelPending",
        OrderLifecycleState.PARTIALLY_FILLED: "OrderPartiallyFilled",
        OrderLifecycleState.FILLED: "OrderFilled",
        OrderLifecycleState.CANCELLED: "OrderCancelled",
        OrderLifecycleState.REJECTED: "OrderRejected",
        OrderLifecycleState.TIMEOUT_PENDING: "OrderTimeoutPending",
    }[state]


def _state_from_event_type(event_type: str) -> OrderLifecycleState | None:
    return {
        "OrderRequested": OrderLifecycleState.REQUESTED,
        "OrderSubmitted": OrderLifecycleState.SUBMITTED,
        "OrderAccepted": OrderLifecycleState.ACCEPTED,
        "OrderCancelPending": OrderLifecycleState.CANCEL_PENDING,
        "OrderPartiallyFilled": OrderLifecycleState.PARTIALLY_FILLED,
        "OrderFilled": OrderLifecycleState.FILLED,
        "OrderCancelled": OrderLifecycleState.CANCELLED,
        "OrderRejected": OrderLifecycleState.REJECTED,
        "OrderTimeoutPending": OrderLifecycleState.TIMEOUT_PENDING,
    }.get(str(event_type))


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
