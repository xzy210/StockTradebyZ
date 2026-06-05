from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.services.ai.auto_trade_config_service import AutoTradeConfig
from trading_app.services.execution.trade_execution_service import TradeExecutionService
from trading_app.services.execution.trade_record_service import get_trade_record_service


class FakeBrokerSessionService:
    def __init__(self):
        self.is_connected = True
        self.order_calls = []

    def query_stock_asset(self):
        return SimpleNamespace(cash=500000.0, available_cash=500000.0, total_asset=1000000.0)

    def query_stock_positions(self):
        return []

    def order_stock(self, *args, **kwargs):
        self.order_calls.append((args, kwargs))
        raise RuntimeError("shadow 模式不应触发真实下单")

    def query_stock_order(self, order_id: int):
        return None


def main():
    trade_service = get_trade_record_service()
    fake_broker = FakeBrokerSessionService()
    execution_service = TradeExecutionService(fake_broker)
    execution_service.config_service.get_config = lambda: AutoTradeConfig(
        manual_orders_enabled=True,
        auto_trade_mode="shadow",
        require_trading_time=False,
        duplicate_window_seconds=30,
        status_poll_seconds=2.0,
        status_poll_interval_seconds=0.5,
    )

    unique_price = 10.0 + ((int(time.time()) % 100) / 1000.0)
    result = execution_service.execute_conditional_order(
        stock_code="000001.SZ",
        stock_name="平安银行",
        order_type=23,
        order_volume=100,
        price_type=0,
        price=unique_price,
        strategy_name="ShadowSmokeTest",
        remark="shadow联调",
    )

    order_record = trade_service.get_order_record_by_request_id(result.request_id)

    print("TEST_PRICE=", unique_price)
    print("RESULT_SUCCESS=", result.success)
    print("RESULT_SHADOW=", result.shadow)
    print("RESULT_MODE=", result.execution_mode)
    print("RESULT_MESSAGE=", result.message)
    print("BROKER_ORDER_CALLS=", len(fake_broker.order_calls))
    print("ORDER_RECORD_STATUS=", getattr(order_record, "status", ""))
    print("ORDER_RECORD_MODE=", getattr(order_record, "execution_mode", ""))
    print("ORDER_RECORD_SOURCE=", getattr(order_record, "source", ""))


if __name__ == "__main__":
    main()
