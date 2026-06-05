from __future__ import annotations

import logging
from typing import Optional

from common.trading_runtime import (
    EVENT_CONNECTION,
    EVENT_ORDER,
    EVENT_ORDER_ERROR,
    EVENT_TRADE,
    LiveEvent,
    LiveEventEngine,
    OrderData,
    TradeData,
)

from .order_execution_event_service import OrderExecutionEvent, OrderExecutionEventService, get_order_execution_event_service

logger = logging.getLogger(__name__)


class LiveEventPersistenceHandler:
    """Persist selected live runtime events into the existing live center event ledger."""

    def __init__(
        self,
        event_engine: LiveEventEngine,
        storage: Optional[OrderExecutionEventService] = None,
    ) -> None:
        self.event_engine = event_engine
        self.storage = storage or get_order_execution_event_service()
        self._registered = False

    def start(self) -> None:
        if self._registered:
            return
        self.event_engine.register(EVENT_ORDER, self._on_order)
        self.event_engine.register(EVENT_TRADE, self._on_trade)
        self.event_engine.register(EVENT_ORDER_ERROR, self._on_order_error)
        self.event_engine.register(EVENT_CONNECTION, self._on_connection)
        self._registered = True

    def stop(self) -> None:
        if not self._registered:
            return
        self.event_engine.unregister(EVENT_ORDER, self._on_order)
        self.event_engine.unregister(EVENT_TRADE, self._on_trade)
        self.event_engine.unregister(EVENT_ORDER_ERROR, self._on_order_error)
        self.event_engine.unregister(EVENT_CONNECTION, self._on_connection)
        self._registered = False

    def _add_event(self, event: OrderExecutionEvent) -> None:
        try:
            self.storage.add_event(event)
        except Exception:
            logger.debug("Persist live event failed: %s", event.event_id, exc_info=True)

    def _on_order(self, event: LiveEvent) -> None:
        if not isinstance(event.data, OrderData):
            return
        order = event.data
        status_text = order.status.value
        self._add_event(
            OrderExecutionEvent(
                event_id=f"live-order:{order.vt_orderid}:{order.status_code}:{order.traded}",
                level="warning" if order.status.value in {"cancelled", "rejected"} else "info",
                category="order_execution",
                source="live_event_engine",
                strategy_id=str(order.metadata.get("strategy_id", "") or ""),
                symbol=order.symbol,
                request_id=order.request_id,
                broker_order_id=int(order.order_id or 0) if str(order.order_id or "").isdigit() else 0,
                title="券商委托状态更新",
                message=order.status_message or f"委托状态: {status_text}",
                status="open" if order.status.value in {"cancelled", "rejected"} else "resolved",
                payload={
                    "event_type": "OrderBrokerUpdate",
                    "order_status_code": order.status_code,
                    "order_status_text": status_text,
                    "status_message": order.status_message,
                    "traded_volume": order.traded,
                    "order_volume": order.volume,
                    "direction": order.direction.value,
                    "gateway_name": order.gateway_name,
                    "vt_orderid": order.vt_orderid,
                },
            )
        )

    def _on_trade(self, event: LiveEvent) -> None:
        if not isinstance(event.data, TradeData):
            return
        trade = event.data
        self._add_event(
            OrderExecutionEvent(
                event_id=f"live-trade:{trade.vt_tradeid}",
                level="info",
                category="order_execution",
                source="live_event_engine",
                strategy_id=str(trade.metadata.get("strategy_id", "") or ""),
                symbol=trade.symbol,
                request_id=trade.request_id,
                broker_order_id=int(trade.order_id or 0) if str(trade.order_id or "").isdigit() else 0,
                title="券商成交回报",
                message=f"{trade.symbol} 成交 {trade.volume} 股 @ {trade.price:.3f}",
                status="resolved",
                payload={
                    "event_type": "TradeBrokerUpdate",
                    "executed_price": trade.price,
                    "executed_volume": trade.volume,
                    "direction": trade.direction.value,
                    "gateway_name": trade.gateway_name,
                    "vt_tradeid": trade.vt_tradeid,
                },
            )
        )

    def _on_order_error(self, event: LiveEvent) -> None:
        payload = dict(event.data or {})
        order_id = int(payload.get("order_id", 0) or 0)
        self._add_event(
            OrderExecutionEvent(
                event_id=f"live-order-error:{event.gateway_name}:{order_id}:{payload.get('error_id', '')}",
                level="danger",
                category="broker_error",
                source="live_event_engine",
                broker_order_id=order_id,
                title="券商委托错误",
                message=str(payload.get("error_msg", "") or "券商返回委托错误"),
                status="open",
                payload={"event_type": "BrokerOrderError", **payload},
            )
        )

    def _on_connection(self, event: LiveEvent) -> None:
        data = dict(event.data or {})
        if data.get("connected"):
            return
        self._add_event(
            OrderExecutionEvent(
                event_id=f"live-connection:{event.gateway_name}:{event.timestamp.strftime('%Y%m%d%H%M')}",
                level="warning",
                category="broker_disconnected",
                source="live_event_engine",
                title="券商连接断开",
                message=str(data.get("message", "") or "券商连接断开"),
                status="open",
                payload={"event_type": "BrokerDisconnected", **data},
            )
        )


__all__ = ["LiveEventPersistenceHandler"]
