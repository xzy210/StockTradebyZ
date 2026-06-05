from __future__ import annotations

import logging
from typing import Optional, TypeVar

from common.broker_session_service import BrokerSessionService, get_broker_session_service

from .event_engine import LiveEventEngine
from .gateway import BaseBrokerGateway
from .mini_qmt_gateway import MiniQmtGateway
from .objects import LogData
from .oms import LiveOmsEngine
from .requests import CancelRequest, OrderRequest, SubscribeRequest

logger = logging.getLogger(__name__)

GatewayType = TypeVar("GatewayType", bound=BaseBrokerGateway)


class LiveLogEngine:
    engine_name = "live_log"

    def __init__(self, event_engine: LiveEventEngine) -> None:
        self.event_engine = event_engine

    def close(self) -> None:
        return


class LiveTradingEngine:
    """Main coordinator for live broker gateways and runtime engines."""

    def __init__(self, event_engine: Optional[LiveEventEngine] = None) -> None:
        self.event_engine = event_engine or LiveEventEngine()
        self.event_engine.start()
        self.gateways: dict[str, BaseBrokerGateway] = {}
        self.engines: dict[str, object] = {}
        self.default_gateway_name = ""
        self.init_engines()

    def init_engines(self) -> None:
        oms = LiveOmsEngine(self.event_engine)
        self.engines[oms.engine_name] = oms
        log = LiveLogEngine(self.event_engine)
        self.engines[log.engine_name] = log
        self.oms: LiveOmsEngine = oms

    def add_gateway(self, gateway: GatewayType, *, default: bool = False) -> GatewayType:
        self.gateways[gateway.gateway_name] = gateway
        if default or not self.default_gateway_name:
            self.default_gateway_name = gateway.gateway_name
        return gateway

    def get_gateway(self, gateway_name: str = "") -> BaseBrokerGateway | None:
        name = gateway_name or self.default_gateway_name
        return self.gateways.get(name)

    def get_engine(self, engine_name: str) -> object | None:
        return self.engines.get(engine_name)

    def connect(self, setting: dict | None = None, gateway_name: str = "") -> None:
        gateway = self._require_gateway(gateway_name)
        gateway.connect(setting or {})

    def subscribe(self, req: SubscribeRequest, gateway_name: str = "") -> None:
        gateway = self._require_gateway(gateway_name)
        gateway.subscribe(req)

    def send_order(self, req: OrderRequest, gateway_name: str = ""):
        gateway = self._require_gateway(gateway_name)
        return gateway.send_order(req)

    def cancel_order(self, req: CancelRequest, gateway_name: str = ""):
        gateway = self._require_gateway(gateway_name)
        return gateway.cancel_order(req)

    def query_account(self, gateway_name: str = ""):
        gateway = self._require_gateway(gateway_name)
        return gateway.query_account()

    def query_positions(self, gateway_name: str = ""):
        gateway = self._require_gateway(gateway_name)
        return gateway.query_positions()

    def close(self) -> None:
        self.event_engine.stop()
        for engine in self.engines.values():
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        for gateway in self.gateways.values():
            try:
                gateway.close()
            except Exception:
                logger.debug("Close live gateway failed: %s", gateway.gateway_name, exc_info=True)

    def _require_gateway(self, gateway_name: str = "") -> BaseBrokerGateway:
        gateway = self.get_gateway(gateway_name)
        if gateway is None:
            raise KeyError(f"live broker gateway not found: {gateway_name or self.default_gateway_name}")
        return gateway


_live_trading_engine: LiveTradingEngine | None = None


def get_live_trading_engine(
    *,
    broker_service: BrokerSessionService | None = None,
    start_default_gateway: bool = True,
) -> LiveTradingEngine:
    global _live_trading_engine
    if _live_trading_engine is None:
        _live_trading_engine = LiveTradingEngine()
    if start_default_gateway and not _live_trading_engine.get_gateway():
        service = broker_service or get_broker_session_service()
        _live_trading_engine.add_gateway(
            MiniQmtGateway(_live_trading_engine.event_engine, broker_service=service),
            default=True,
        )
    return _live_trading_engine


def set_live_trading_engine(engine: LiveTradingEngine | None) -> None:
    global _live_trading_engine
    _live_trading_engine = engine


__all__ = ["LiveLogEngine", "LiveTradingEngine", "get_live_trading_engine", "set_live_trading_engine"]
