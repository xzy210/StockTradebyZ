"""Smoke test for the ETF rotation StrategyRiskPolicy.

Runs the policy directly against a lightweight fake `RotationConfig` /
`RotationState` pair to cover each rule without needing xtquant or PyQt:

  1. BUY inside trading window           -> APPROVE
  2. BUY outside trading window          -> BLOCK (trading_time)
  3. BUY when daily cap reached          -> BLOCK (daily_trades)
  4. SELL before min_hold_days satisfied -> BLOCK (min_hold_days)
  5. SELL with loss beyond threshold     -> WARN  (single_loss_limit)
  6. enable_risk_check=False             -> APPROVE (bypass)

We also exercise the gateway integration once via a fake broker + registered
policy to make sure ``TradeExecutionService`` routes to the ETF policy and
surfaces its block reasons.

Run::

    python scripts/etf_rotation_policy_smoketest.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from live_rotation.rotation_risk_policy import ETFRotationRiskPolicy
from trading_app.services.ai.auto_trade_config_service import AutoTradeConfig
from trading_app.services.strategy.strategy_risk import (
    RISK_LEVEL_WARN,
    StrategyRiskContext,
    get_strategy_risk_registry,
    reset_strategy_risk_registry,
)
from trading_app.services.execution.trade_execution_service import (
    ExecutionRequest,
    TradeExecutionService,
)
from trading_app.services.execution.trade_record_service import TradeSource


TEST_STRATEGY_ID = "smoketest_etf_rotation"


# ---------------------------------------------------------------------------
# 轻量伪造 config / state
# ---------------------------------------------------------------------------

@dataclass
class FakeRotationConfig:
    enable_risk_check: bool = True
    trading_start: str = "09:30"
    trading_end: str = "14:57"
    max_trades_per_day: int = 2
    min_hold_days: int = 3
    max_single_loss_pct: float = 5.0


@dataclass
class FakeRotationState:
    buy_price: float = 0.0
    buy_date: str = ""
    _trades_today: int = 0

    def get_trades_today(self) -> int:
        return self._trades_today


def _mid_trading_time() -> datetime:
    return datetime(2026, 4, 20, 10, 30)  # Monday 10:30


def _off_hours_time() -> datetime:
    return datetime(2026, 4, 20, 19, 0)


def _build_request(order_type: int, price: float = 1.0) -> ExecutionRequest:
    return ExecutionRequest(
        stock_code="510300.SH",
        stock_name="沪深300ETF",
        order_type=order_type,
        order_volume=100,
        price_type=0,
        price=price,
        source=TradeSource.CONDITIONAL.value,
        trigger="auto",
        strategy_name="ETF-Smoke",
        strategy_id=TEST_STRATEGY_ID,
        virtual_account_id=f"va_{TEST_STRATEGY_ID}",
        intent_id=f"etf-policy-smoke-{int(time.time() * 1000)}",
        remark="etf policy smoketest",
    )


def _build_policy(config: FakeRotationConfig, state: FakeRotationState) -> ETFRotationRiskPolicy:
    return ETFRotationRiskPolicy(
        strategy_id=TEST_STRATEGY_ID,
        config_provider=lambda: config,
        state_provider=lambda: state,
    )


# ---------------------------------------------------------------------------
# 规则单测（直接调用 policy.evaluate）
# ---------------------------------------------------------------------------

def case_approve_in_trading_window() -> None:
    cfg = FakeRotationConfig()
    state = FakeRotationState()
    policy = _build_policy(cfg, state)

    ctx = StrategyRiskContext(now=_mid_trading_time())
    with patch("live_rotation.rotation_risk_policy.is_trading_day", return_value=True):
        decision = policy.evaluate(_build_request(order_type=23), ctx)
    assert decision.passed, decision.reason
    print("[approve_in_trading_window] OK -> reason=", decision.reason)


def case_block_outside_trading_window() -> None:
    cfg = FakeRotationConfig()
    state = FakeRotationState()
    policy = _build_policy(cfg, state)

    ctx = StrategyRiskContext(now=_off_hours_time())
    with patch("live_rotation.rotation_risk_policy.is_trading_day", return_value=True):
        decision = policy.evaluate(_build_request(order_type=23), ctx)
    assert not decision.passed, "expected BLOCK when off-hours"
    assert "交易截止时间" in decision.reason, decision.reason
    assert decision.metadata.get("rule") == "trading_time"
    print("[block_outside_trading_window] OK -> reason=", decision.reason)


def case_block_daily_cap() -> None:
    cfg = FakeRotationConfig(max_trades_per_day=2)
    state = FakeRotationState(_trades_today=2)
    policy = _build_policy(cfg, state)

    ctx = StrategyRiskContext(now=_mid_trading_time())
    with patch("live_rotation.rotation_risk_policy.is_trading_day", return_value=True):
        decision = policy.evaluate(_build_request(order_type=23), ctx)
    assert not decision.passed
    assert "交易次数已达上限" in decision.reason
    assert decision.metadata.get("rule") == "daily_trades"
    print("[block_daily_cap] OK -> reason=", decision.reason)


def case_block_min_hold_days_on_sell() -> None:
    today = _mid_trading_time()
    cfg = FakeRotationConfig(min_hold_days=5)
    state = FakeRotationState(
        buy_price=1.0,
        buy_date=(today - timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    policy = _build_policy(cfg, state)

    ctx = StrategyRiskContext(now=today)
    with patch("live_rotation.rotation_risk_policy.is_trading_day", return_value=True):
        decision = policy.evaluate(_build_request(order_type=24, price=1.0), ctx)
    assert not decision.passed
    assert "持有天数不足" in decision.reason
    assert decision.metadata.get("rule") == "min_hold_days"
    print("[block_min_hold_days] OK -> reason=", decision.reason)


def case_warn_on_loss_over_limit() -> None:
    today = _mid_trading_time()
    cfg = FakeRotationConfig(min_hold_days=0, max_single_loss_pct=5.0)
    state = FakeRotationState(
        buy_price=1.0,
        buy_date=(today - timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    policy = _build_policy(cfg, state)

    # 当前价 0.9 -> -10% 损失，超过 -5%
    ctx = StrategyRiskContext(now=today)
    with patch("live_rotation.rotation_risk_policy.is_trading_day", return_value=True):
        decision = policy.evaluate(_build_request(order_type=24, price=0.9), ctx)
    assert decision.passed, "止损超限仍应放行"
    assert decision.level == RISK_LEVEL_WARN, f"expected warn, got {decision.level}"
    assert decision.metadata.get("rule") == "single_loss_limit"
    print("[warn_on_loss_over_limit] OK -> level=%s reason=%s" % (decision.level, decision.reason))


def case_bypass_when_risk_check_disabled() -> None:
    cfg = FakeRotationConfig(enable_risk_check=False)
    state = FakeRotationState(_trades_today=99)  # 本应触发 daily cap
    policy = _build_policy(cfg, state)

    ctx = StrategyRiskContext(now=_off_hours_time())
    decision = policy.evaluate(_build_request(order_type=23), ctx)
    assert decision.passed
    assert "已禁用" in decision.reason
    print("[bypass_when_disabled] OK -> reason=", decision.reason)


# ---------------------------------------------------------------------------
# Gateway 集成（TradeExecutionService 路由到 ETF policy）
# ---------------------------------------------------------------------------

class _FakeBroker:
    is_connected = True

    def query_stock_asset(self):
        return SimpleNamespace(cash=500000.0, available_cash=500000.0, total_asset=1000000.0)

    def query_stock_positions(self):
        return []

    def order_stock(self, *_args, **_kwargs):
        raise RuntimeError("shadow 模式不应触发真实下单")

    def query_stock_order(self, _order_id: int):
        return None


def case_gateway_routes_to_etf_policy() -> None:
    """End-to-end: 注册 policy 后，网关应把下单路由到 ETF 规则并回传拦截理由。

    把 trading_end 放宽到 23:59 绕开交易时段规则，让 daily_trades 规则先触发，
    这样测试不依赖运行时具体时间。

    另外显式放宽网关通用仓位上限，避免本机 risk_guard 持久化配置把请求先挡在
    ETF policy 之前，导致断言被环境状态污染。
    """
    reset_strategy_risk_registry()
    cfg = FakeRotationConfig(
        max_trades_per_day=1,
        trading_start="00:00",
        trading_end="23:59",
    )
    state = FakeRotationState(_trades_today=1)  # 已触顶
    policy = _build_policy(cfg, state)
    get_strategy_risk_registry().register(policy)

    service = TradeExecutionService(_FakeBroker())
    service.config_service.get_config = lambda: AutoTradeConfig(
        manual_orders_enabled=True,
        auto_trade_mode="shadow",
        require_trading_time=False,
        duplicate_window_seconds=1,
        status_poll_seconds=1.0,
        status_poll_interval_seconds=0.2,
    )
    service.risk_guard.config["max_single_position_pct"] = 1.0
    service.risk_guard.config["max_total_position_pct"] = 1.0
    service._validate_market_data_status = lambda _request: ""
    service._build_strategy_risk_context = lambda _request: StrategyRiskContext(now=_mid_trading_time())

    with patch("live_rotation.rotation_risk_policy.is_trading_day", return_value=True):
        result = service.execute(_build_request(order_type=23))

    assert not result.success, f"expected block, got {result.message}"
    assert result.blocked
    assert f"[policy:{TEST_STRATEGY_ID}:daily_trades]" in result.message, result.message
    assert "交易次数已达上限" in result.message, result.message
    print("[gateway_routes_to_etf_policy] OK -> message=", result.message)


# ---------------------------------------------------------------------------

def main() -> None:
    case_approve_in_trading_window()
    case_block_outside_trading_window()
    case_block_daily_cap()
    case_block_min_hold_days_on_sell()
    case_warn_on_loss_over_limit()
    case_bypass_when_risk_check_disabled()
    case_gateway_routes_to_etf_policy()
    reset_strategy_risk_registry()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
