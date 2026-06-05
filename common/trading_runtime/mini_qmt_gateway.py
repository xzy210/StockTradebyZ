from __future__ import annotations

import logging
from typing import Any, Optional

from common.broker_interface import BrokerCancelResult, BrokerOrderRequest, BrokerSubmitResult, LiveBrokerAdapter
from common.broker_session_service import BrokerSessionService, get_broker_session_service

from .constants import Direction, Exchange, OrderStatus, OrderType, split_symbol
from .event_engine import LiveEventEngine
from .gateway import BaseBrokerGateway
from .objects import AccountData, OrderData, account_from_xt, order_from_xt, position_from_xt, trade_from_xt
from .requests import CancelRequest, OrderRequest, SubscribeRequest

logger = logging.getLogger(__name__)


class MiniQmtGateway(BaseBrokerGateway):
    """Gateway adapter over the existing BrokerSessionService."""

    default_name = "mini_qmt"

    def __init__(
        self,
        event_engine: LiveEventEngine,
        gateway_name: str = "",
        broker_service: Optional[BrokerSessionService] = None,
    ) -> None:
        super().__init__(event_engine, gateway_name or self.default_name)
        self.broker_service = broker_service or get_broker_session_service()
        self._adapter = LiveBrokerAdapter(self.broker_service)
        self._signals_connected = False
        self._connect_signals()

    @property
    def is_connected(self) -> bool:
        return bool(self.broker_service.is_connected)

    def connect(self, setting: dict[str, Any] | None = None) -> None:
        config = dict(self.broker_service.get_config())
        config.update(dict(setting or {}))
        qmt_path = str(config.get("qmt_path", "") or "").strip()
        account = str(config.get("account", "") or "").strip()
        ok = self.broker_service.connect_async(qmt_path, account)
        self.on_connection(bool(self.broker_service.is_connected), "券商连接已发起" if ok else "券商连接未发起")

    def close(self) -> None:
        self.broker_service.disconnect()

    def subscribe(self, req: SubscribeRequest) -> None:
        self.write_log(f"miniQMT 交易网关无需单独订阅交易标的: {req.vt_symbol}")

    def send_order(self, req: OrderRequest) -> BrokerSubmitResult:
        return self.submit(req.to_broker_order_request())

    def submit(self, request: BrokerOrderRequest) -> BrokerSubmitResult:
        result = self._adapter.submit(request)
        status = OrderStatus.SUBMITTING if result.accepted else OrderStatus.REJECTED
        order_id = str(result.broker_order_id if result.broker_order_id > 0 else request.request_id or "")
        code, exchange = split_symbol(request.stock_code)
        self.on_order(
            OrderData(
                gateway_name=self.gateway_name,
                symbol=code,
                exchange=exchange,
                order_id=order_id,
                direction=Direction.BUY if request.order_type == 23 else Direction.SELL,
                order_type=OrderType.LIMIT,
                price=request.price,
                volume=request.order_volume,
                traded=0,
                status=status,
                status_message=result.message,
                reference=request.remark or request.strategy_name,
                request_id=request.request_id,
                raw=result.raw,
                metadata={**dict(request.metadata), "submit_result": result.status},
            )
        )
        return result

    def cancel_order(self, req: CancelRequest) -> BrokerCancelResult:
        try:
            order_id = int(req.order_id or 0)
        except (TypeError, ValueError):
            order_id = 0
        result = self._adapter.cancel(order_id)
        if result.success:
            self.write_log(f"撤单请求已提交: {order_id}")
        else:
            self.on_order_error({"order_id": order_id, "error_msg": result.message})
        return result

    def cancel(self, order_id: int) -> BrokerCancelResult:
        return self.cancel_order(CancelRequest(order_id=str(order_id or "")))

    def query_order(self, order_id: int) -> Any:
        return self.broker_service.query_stock_order(int(order_id or 0))

    def query_position(self, symbol: str = "") -> Any:
        positions = list(self.broker_service.query_stock_positions() or [])
        plain = str(symbol or "").strip().upper().split(".")[0]
        if not plain:
            return positions
        for position in positions:
            if str(getattr(position, "stock_code", "") or "").strip().upper().split(".")[0] == plain:
                return position
        return None

    def query_asset(self) -> Any:
        return self.broker_service.query_stock_asset()

    def query_account(self) -> AccountData:
        account_id = str(self.broker_service.get_config().get("account", "") or "")
        account = account_from_xt(self.query_asset(), gateway_name=self.gateway_name, account_id=account_id)
        self.on_account(account)
        return account

    def query_positions(self) -> list:
        positions = [position_from_xt(pos, gateway_name=self.gateway_name) for pos in (self.broker_service.query_stock_positions() or [])]
        for position in positions:
            self.on_position(position)
        return positions

    def _connect_signals(self) -> None:
        if self._signals_connected:
            return
        self.broker_service.connection_changed.connect(self._on_connection_changed)
        self.broker_service.order_changed.connect(self._on_order_changed)
        self.broker_service.trade_occurred.connect(self._on_trade_occurred)
        self.broker_service.order_error.connect(self._on_order_error)
        self.broker_service.broker_disconnected.connect(self._on_broker_disconnected)
        self._signals_connected = True

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        self.on_connection(bool(connected), str(message or ""))
        if connected:
            try:
                self.query_account()
                self.query_positions()
            except Exception:
                logger.debug("miniQMT gateway initial query failed", exc_info=True)

    def _on_order_changed(self, payload: dict) -> None:
        self.on_order(order_from_xt(payload, gateway_name=self.gateway_name))

    def _on_trade_occurred(self, payload: dict) -> None:
        self.on_trade(trade_from_xt(payload, gateway_name=self.gateway_name))

    def _on_order_error(self, payload: dict) -> None:
        self.on_order_error(dict(payload or {}))

    def _on_broker_disconnected(self) -> None:
        self.on_connection(False, "券商连接断开")


__all__ = ["MiniQmtGateway"]
