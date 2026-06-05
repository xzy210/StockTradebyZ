"""Smoke test for the StrategyRiskPolicy gateway integration.

Scenarios exercised against a fake broker in shadow mode:
  1. empty registry  -> order passes (behaviour unchanged)
  2. registered allow-policy -> order passes
  3. registered block-policy -> order rejected with policy reason
  4. registered raising-policy -> order rejected defensively

Run::

    python scripts/strategy_risk_policy_smoketest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.services.ai.auto_trade_config_service import AutoTradeConfig
from trading_app.services.strategy.strategy_risk import (
    RiskPolicyDecision,
    get_strategy_risk_registry,
    reset_strategy_risk_registry,
)
from trading_app.services.execution.trade_execution_service import (
    ExecutionRequest,
    TradeExecutionService,
)
from trading_app.services.execution.trade_record_service import TradeSource


TEST_STRATEGY_ID = "smoketest_etf_rotation"
TEST_STRATEGY_NAME = "SmokeTestETF"
TEST_VIRTUAL_ACCOUNT = "virtual_etf_smoketest"


class FakeBroker:
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


class AllowPolicy:
    strategy_id = TEST_STRATEGY_ID

    def __init__(self):
        self.calls = 0

    def evaluate(self, request, context):
        self.calls += 1
        return RiskPolicyDecision.approve(reason="allow policy ok")


class BlockPolicy:
    strategy_id = TEST_STRATEGY_ID
    BLOCK_REASON = "ETF 单日交易已达上限"

    def __init__(self):
        self.calls = 0

    def evaluate(self, request, context):
        self.calls += 1
        return RiskPolicyDecision.block(self.BLOCK_REASON)


class RaisingPolicy:
    strategy_id = TEST_STRATEGY_ID

    def evaluate(self, request, context):
        raise RuntimeError("boom")


def _build_service() -> TradeExecutionService:
    fake_broker = FakeBroker()
    service = TradeExecutionService(fake_broker)
    service.config_service.get_config = lambda: AutoTradeConfig(
        manual_orders_enabled=True,
        auto_trade_mode="shadow",
        require_trading_time=False,
        duplicate_window_seconds=1,
        status_poll_seconds=1.0,
        status_poll_interval_seconds=0.2,
    )
    return service


def _make_request(price: float) -> ExecutionRequest:
    return ExecutionRequest(
        stock_code="510300.SH",
        stock_name="沪深300ETF",
        order_type=23,
        order_volume=100,
        price_type=0,
        price=price,
        source=TradeSource.CONDITIONAL.value,
        trigger="auto",
        strategy_name=TEST_STRATEGY_NAME,
        strategy_id=TEST_STRATEGY_ID,
        virtual_account_id=TEST_VIRTUAL_ACCOUNT,
        intent_id=f"smoketest-{int(time.time() * 1000)}",
        remark="policy smoketest",
    )


def _price_seed() -> float:
    return 1.0 + (time.time() % 1.0)


def case_empty_registry() -> None:
    reset_strategy_risk_registry()
    service = _build_service()
    result = service.execute(_make_request(_price_seed()))
    assert result.success, f"[empty] expected success, got {result.message}"
    assert result.shadow, "[empty] expected shadow execution"
    print("[empty_registry] OK ->", result.message)


def case_allow_policy() -> None:
    reset_strategy_risk_registry()
    policy = AllowPolicy()
    get_strategy_risk_registry().register(policy)
    service = _build_service()
    result = service.execute(_make_request(_price_seed()))
    assert result.success, f"[allow] expected success, got {result.message}"
    assert policy.calls == 1, f"[allow] policy should fire once, got {policy.calls}"
    print("[allow_policy] OK -> policy_calls=%d" % policy.calls)


def case_block_policy() -> None:
    reset_strategy_risk_registry()
    policy = BlockPolicy()
    get_strategy_risk_registry().register(policy)
    service = _build_service()
    result = service.execute(_make_request(_price_seed()))
    assert not result.success, "[block] expected failure"
    assert result.blocked, "[block] expected blocked flag"
    assert BlockPolicy.BLOCK_REASON in result.message, (
        f"[block] expected reason in message, got {result.message!r}"
    )
    assert policy.calls == 1, f"[block] policy should fire once, got {policy.calls}"
    print("[block_policy] OK ->", result.message)


def case_raising_policy() -> None:
    reset_strategy_risk_registry()
    get_strategy_risk_registry().register(RaisingPolicy())
    service = _build_service()
    result = service.execute(_make_request(_price_seed()))
    assert not result.success, "[raise] expected failure"
    assert result.blocked, "[raise] expected blocked flag"
    assert "策略风控执行异常" in result.message, (
        f"[raise] expected defensive block, got {result.message!r}"
    )
    print("[raising_policy] OK ->", result.message)


def main() -> None:
    case_empty_registry()
    case_allow_policy()
    case_block_policy()
    case_raising_policy()
    reset_strategy_risk_registry()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
