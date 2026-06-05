from __future__ import annotations

from .event_engine import LiveEventEngine
from .events import (
    EVENT_ACCOUNT,
    EVENT_CONNECTION,
    EVENT_CONTRACT,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_TICK,
    EVENT_TRADE,
    LiveEvent,
)
from .objects import AccountData, ContractData, OrderData, PositionData, TickData, TradeData


class LiveOmsEngine:
    """In-memory live order management projection."""

    engine_name = "live_oms"

    def __init__(self, event_engine: LiveEventEngine) -> None:
        self.event_engine = event_engine
        self.ticks: dict[str, TickData] = {}
        self.orders: dict[str, OrderData] = {}
        self.active_orders: dict[str, OrderData] = {}
        self.trades: dict[str, TradeData] = {}
        self.positions: dict[str, PositionData] = {}
        self.accounts: dict[str, AccountData] = {}
        self.contracts: dict[str, ContractData] = {}
        self.connection_status: dict[str, dict] = {}
        self.register_event()

    def register_event(self) -> None:
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)
        self.event_engine.register(EVENT_CONNECTION, self.process_connection_event)

    def close(self) -> None:
        return

    def process_tick_event(self, event: LiveEvent) -> None:
        if isinstance(event.data, TickData):
            self.ticks[event.data.vt_symbol] = event.data

    def process_order_event(self, event: LiveEvent) -> None:
        if not isinstance(event.data, OrderData):
            return
        order = event.data
        self.orders[order.vt_orderid] = order
        if order.is_active():
            self.active_orders[order.vt_orderid] = order
        else:
            self.active_orders.pop(order.vt_orderid, None)

    def process_trade_event(self, event: LiveEvent) -> None:
        if isinstance(event.data, TradeData):
            self.trades[event.data.vt_tradeid] = event.data

    def process_position_event(self, event: LiveEvent) -> None:
        if isinstance(event.data, PositionData):
            self.positions[event.data.vt_positionid] = event.data

    def process_account_event(self, event: LiveEvent) -> None:
        if isinstance(event.data, AccountData):
            self.accounts[event.data.vt_accountid] = event.data

    def process_contract_event(self, event: LiveEvent) -> None:
        if isinstance(event.data, ContractData):
            self.contracts[event.data.vt_symbol] = event.data

    def process_connection_event(self, event: LiveEvent) -> None:
        self.connection_status[event.gateway_name or event.key or "default"] = dict(event.data or {})

    def get_tick(self, vt_symbol: str) -> TickData | None:
        return self.ticks.get(vt_symbol)

    def get_order(self, vt_orderid: str) -> OrderData | None:
        return self.orders.get(vt_orderid)

    def get_trade(self, vt_tradeid: str) -> TradeData | None:
        return self.trades.get(vt_tradeid)

    def get_position(self, vt_positionid: str) -> PositionData | None:
        return self.positions.get(vt_positionid)

    def get_account(self, vt_accountid: str) -> AccountData | None:
        return self.accounts.get(vt_accountid)

    def get_all_ticks(self) -> list[TickData]:
        return list(self.ticks.values())

    def get_all_orders(self) -> list[OrderData]:
        return list(self.orders.values())

    def get_all_active_orders(self) -> list[OrderData]:
        return list(self.active_orders.values())

    def get_all_trades(self) -> list[TradeData]:
        return list(self.trades.values())

    def get_all_positions(self) -> list[PositionData]:
        return list(self.positions.values())

    def get_all_accounts(self) -> list[AccountData]:
        return list(self.accounts.values())


__all__ = ["LiveOmsEngine"]
