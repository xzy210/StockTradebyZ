from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from common.broker_interface import BrokerCancelResult, BrokerOrderRequest, BrokerProtocol, BrokerSubmitResult

from .event_engine import LiveEventEngine
from .events import (
    EVENT_ACCOUNT,
    EVENT_CONNECTION,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_ORDER,
    EVENT_ORDER_ERROR,
    EVENT_POSITION,
    EVENT_TICK,
    EVENT_TRADE,
    LiveEvent,
)
from .objects import AccountData, ContractData, LogData, OrderData, PositionData, TickData, TradeData
from .requests import CancelRequest, OrderRequest, SubscribeRequest


class BaseBrokerGateway(BrokerProtocol, ABC):
    """Base gateway for live broker adapters."""

    default_name = ""

    def __init__(self, event_engine: LiveEventEngine, gateway_name: str = "") -> None:
        self.event_engine = event_engine
        self.gateway_name = gateway_name or self.default_name or type(self).__name__

    def on_event(self, event_type: str, data: Any = None, *, key: str = "", symbol: str = "", **metadata: Any) -> None:
        self.event_engine.put(
            LiveEvent(
                type=event_type,
                data=data,
                gateway_name=self.gateway_name,
                symbol=symbol,
                key=key,
                metadata=dict(metadata or {}),
            )
        )

    def on_tick(self, tick: TickData) -> None:
        self.on_event(EVENT_TICK, tick, symbol=tick.vt_symbol)
        self.on_event(EVENT_TICK + tick.vt_symbol, tick, symbol=tick.vt_symbol, key=tick.vt_symbol)

    def on_order(self, order: OrderData) -> None:
        self.on_event(EVENT_ORDER, order, symbol=order.vt_symbol, key=order.vt_orderid)
        self.on_event(EVENT_ORDER + order.vt_orderid, order, symbol=order.vt_symbol, key=order.vt_orderid)

    def on_trade(self, trade: TradeData) -> None:
        self.on_event(EVENT_TRADE, trade, symbol=trade.vt_symbol, key=trade.vt_tradeid)
        self.on_event(EVENT_TRADE + trade.vt_symbol, trade, symbol=trade.vt_symbol, key=trade.vt_tradeid)

    def on_position(self, position: PositionData) -> None:
        self.on_event(EVENT_POSITION, position, symbol=position.vt_symbol, key=position.vt_positionid)
        self.on_event(EVENT_POSITION + position.vt_symbol, position, symbol=position.vt_symbol, key=position.vt_positionid)

    def on_account(self, account: AccountData) -> None:
        self.on_event(EVENT_ACCOUNT, account, key=account.vt_accountid)
        self.on_event(EVENT_ACCOUNT + account.vt_accountid, account, key=account.vt_accountid)

    def on_contract(self, contract: ContractData) -> None:
        self.on_event(EVENT_CONTRACT, contract, symbol=contract.vt_symbol, key=contract.vt_symbol)

    def on_connection(self, connected: bool, message: str = "") -> None:
        self.on_event(
            EVENT_CONNECTION,
            {"connected": bool(connected), "message": message},
            key=self.gateway_name,
            message=message,
        )

    def on_order_error(self, payload: dict[str, Any]) -> None:
        self.on_event(EVENT_ORDER_ERROR, dict(payload or {}), key=str(payload.get("order_id", "") or ""))

    def write_log(self, message: str, level: str = "info") -> None:
        log = LogData(gateway_name=self.gateway_name, message=message, level=level)
        self.event_engine.put(
            LiveEvent(EVENT_LOG, log, gateway_name=self.gateway_name, message=message, level=level)
        )

    @abstractmethod
    def connect(self, setting: dict[str, Any] | None = None) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        ...

    @abstractmethod
    def send_order(self, req: OrderRequest) -> BrokerSubmitResult:
        ...

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> BrokerCancelResult:
        ...

    def submit(self, request: BrokerOrderRequest) -> BrokerSubmitResult:
        return self.send_order(OrderRequest.from_broker_order_request(request))

    def cancel(self, order_id: int) -> BrokerCancelResult:
        return self.cancel_order(CancelRequest(order_id=str(order_id or "")))


__all__ = ["BaseBrokerGateway"]
