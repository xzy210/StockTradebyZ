from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.broker_interface import BrokerOrderRequest
from common.trading_runtime import (
    EVENT_ORDER,
    Direction,
    Exchange,
    LiveEvent,
    LiveEventEngine,
    LiveOmsEngine,
    OrderData,
    OrderStatus,
    order_from_xt,
)
from trading_app.services.execution.order_execution_event_service import OrderExecutionEvent, OrderExecutionEventService
from trading_app.services.execution.simulation_replay_gateway import ReplayGateway, SimulatedGateway


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")


def main() -> int:
    event_engine = LiveEventEngine(interval=60)
    event_engine.start()
    oms = LiveOmsEngine(event_engine)
    try:
        order = OrderData(
            gateway_name="test",
            symbol="600000",
            exchange=Exchange.SH,
            order_id="1",
            direction=Direction.BUY,
            volume=100,
            price=10.0,
            status=OrderStatus.NOT_TRADED,
            status_code=50,
        )
        event_engine.put(LiveEvent(EVENT_ORDER, order, gateway_name="test"))
        _wait_until(lambda: bool(oms.get_order("test.1")))
        assert oms.get_all_active_orders()

        filled = OrderData(
            gateway_name="test",
            symbol="600000",
            exchange=Exchange.SH,
            order_id="1",
            direction=Direction.BUY,
            volume=100,
            traded=100,
            price=10.0,
            status=OrderStatus.ALL_TRADED,
            status_code=56,
        )
        event_engine.put(LiveEvent(EVENT_ORDER, filled, gateway_name="test"))
        _wait_until(lambda: not oms.get_all_active_orders())

        converted = order_from_xt(
            {
                "stock_code": "600000.SH",
                "order_id": 2,
                "order_type": 23,
                "order_status": 50,
                "order_volume": 200,
                "traded_volume": 0,
                "price": 9.9,
            },
            gateway_name="mini_qmt",
        )
        assert converted.vt_orderid == "mini_qmt.2"
        assert converted.status == OrderStatus.NOT_TRADED

        sim = SimulatedGateway(event_engine)
        result = sim.submit(
            BrokerOrderRequest(
                stock_code="600001.SH",
                order_type=23,
                order_volume=100,
                price=8.8,
                request_id="sim-1",
            )
        )
        assert result.accepted and result.broker_order_id > 0
        _wait_until(lambda: bool(oms.get_order(f"simulated.{result.broker_order_id}")))

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = OrderExecutionEventService(Path(tmpdir) / "events.db")
            storage.add_event(
                OrderExecutionEvent(
                    event_id="replay-1",
                    category="order_execution",
                    source="test",
                    symbol="600000",
                    request_id="req-1",
                    broker_order_id=123,
                    title="回放测试",
                    message="回放事件",
                    payload={"event_type": "OrderSubmitted", "order_status_code": 50},
                    occurred_at="2026-01-01 09:30:00",
                )
            )
            replay = ReplayGateway(event_engine, storage=storage)
            assert replay.replay_since("2026-01-01 00:00:00") == 1

    finally:
        event_engine.stop()

    print("live_trading_runtime_smoketest_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
