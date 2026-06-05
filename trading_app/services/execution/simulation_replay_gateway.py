from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from common.broker_interface import BrokerCancelResult, BrokerOrderRequest, BrokerSubmitResult
from common.trading_runtime import (
    BaseBrokerGateway,
    Direction,
    EVENT_ORDER_EXECUTION,
    Exchange,
    LiveEvent,
    LiveEventEngine,
    OrderData,
    OrderRequest,
    OrderStatus,
    OrderType,
    TradeData,
)
from strategy_app.backtest.broker import SimulationBroker

from .order_execution_event_service import OrderExecutionEventService, get_order_execution_event_service


class SimulatedGateway(BaseBrokerGateway):
    """Broker gateway backed by the existing SimulationBroker."""

    default_name = "simulated"

    def __init__(
        self,
        event_engine: LiveEventEngine,
        gateway_name: str = "",
        broker: Optional[SimulationBroker] = None,
    ) -> None:
        super().__init__(event_engine, gateway_name or self.default_name)
        self.broker = broker or SimulationBroker()

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, setting: dict[str, Any] | None = None) -> None:
        self.on_connection(True, "模拟网关已连接")

    def close(self) -> None:
        self.on_connection(False, "模拟网关已关闭")

    def subscribe(self, req) -> None:
        return

    def send_order(self, req: OrderRequest) -> BrokerSubmitResult:
        return self.submit(req.to_broker_order_request())

    def submit(self, request: BrokerOrderRequest) -> BrokerSubmitResult:
        result = self.broker.submit(request)
        raw = result.raw
        order = OrderData(
            gateway_name=self.gateway_name,
            symbol=str(request.stock_code or "").split(".")[0],
            exchange=Exchange.SH if str(request.stock_code).startswith(("5", "6", "9")) else Exchange.SZ,
            order_id=str(result.broker_order_id),
            direction=Direction.BUY if request.order_type == 23 else Direction.SELL,
            order_type=OrderType.LIMIT,
            price=request.price,
            volume=request.order_volume,
            traded=0,
            status=OrderStatus.NOT_TRADED if result.accepted else OrderStatus.REJECTED,
            status_code=50 if result.accepted else 57,
            status_message=result.message,
            request_id=request.request_id,
            raw=raw,
            metadata=dict(request.metadata),
        )
        self.on_order(order)
        return result

    def cancel_order(self, req) -> BrokerCancelResult:
        return self.cancel(int(getattr(req, "order_id", 0) or 0))

    def cancel(self, order_id: int) -> BrokerCancelResult:
        result = self.broker.cancel(order_id)
        raw = result.raw
        if raw is not None:
            self.on_order(
                OrderData(
                    gateway_name=self.gateway_name,
                    symbol=str(getattr(raw, "stock_code", "") or "").split(".")[0],
                    exchange=Exchange.SH if str(getattr(raw, "stock_code", "")).startswith(("5", "6", "9")) else Exchange.SZ,
                    order_id=str(order_id),
                    direction=Direction.BUY if int(getattr(raw, "order_type", 23) or 23) == 23 else Direction.SELL,
                    order_type=OrderType.LIMIT,
                    price=float(getattr(raw, "price", 0.0) or 0.0),
                    volume=int(getattr(raw, "order_volume", 0) or 0),
                    traded=int(getattr(raw, "traded_volume", 0) or 0),
                    status=OrderStatus.CANCELLED,
                    status_code=54,
                    status_message=result.message,
                    raw=raw,
                )
            )
        return result

    def query_order(self, order_id: int) -> Any:
        return self.broker.query_order(order_id)

    def query_position(self, symbol: str = "") -> Any:
        return self.broker.query_position(symbol)

    def query_asset(self) -> Any:
        return self.broker.query_asset()

    def query_account(self):
        return self.query_asset()

    def query_positions(self) -> list:
        positions = self.query_position("")
        return list(positions or [])


class ReplayGateway(BaseBrokerGateway):
    """Replay persisted order execution events into a live event engine."""

    default_name = "replay"

    def __init__(
        self,
        event_engine: LiveEventEngine,
        gateway_name: str = "",
        storage: Optional[OrderExecutionEventService] = None,
    ) -> None:
        super().__init__(event_engine, gateway_name or self.default_name)
        self.storage = storage or get_order_execution_event_service()

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, setting: dict[str, Any] | None = None) -> None:
        self.on_connection(True, "回放网关已连接")

    def close(self) -> None:
        self.on_connection(False, "回放网关已关闭")

    def subscribe(self, req) -> None:
        return

    def send_order(self, req: OrderRequest) -> BrokerSubmitResult:
        return BrokerSubmitResult(False, -1, "回放网关不支持下单", status="replay")

    def cancel_order(self, req) -> BrokerCancelResult:
        return BrokerCancelResult(False, int(getattr(req, "order_id", 0) or 0), "回放网关不支持撤单")

    def query_order(self, order_id: int) -> Any:
        events = self.storage.query_by_broker_order_id(order_id, limit=1)
        if not events:
            return None
        event = events[-1]
        payload = event.payload or {}
        return SimpleNamespace(
            order_id=order_id,
            stock_code=event.symbol,
            order_status=payload.get("order_status_code", 0),
            status_msg=event.message,
            traded_volume=payload.get("executed_volume", payload.get("traded_volume", 0)),
            traded_price=payload.get("executed_price", payload.get("traded_price", 0.0)),
        )

    def query_position(self, symbol: str = "") -> Any:
        return []

    def query_asset(self) -> Any:
        return SimpleNamespace(cash=0.0, available_cash=0.0, total_asset=0.0)

    def replay_since(self, since, *, limit: int = 1000) -> int:
        count = 0
        for event in self.storage.replay_since(since, limit=limit):
            self.event_engine.put(
                LiveEvent(
                    EVENT_ORDER_EXECUTION,
                    event,
                    gateway_name=self.gateway_name,
                    symbol=event.symbol,
                    key=event.event_id,
                    message=event.message,
                    metadata={"category": event.category, "request_id": event.request_id},
                )
            )
            count += 1
        return count


__all__ = ["ReplayGateway", "SimulatedGateway"]
