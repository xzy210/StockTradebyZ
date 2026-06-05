"""Smoke test for the AI stock StrategyRiskPolicy integration.

Scenarios covered:
  1. AI order without decision                  -> APPROVE (skip)
  2. AI decision passes RiskGuard rules         -> APPROVE
  3. Low confidence AI decision                 -> BLOCK (rule=ai_stock_rules)
  4. Pre-evaluated RiskCheckResult already fails -> BLOCK (rule=pre_evaluated)
  5. Gateway auto-registers AI policy on init    -> block reason carried to ExecutionResult
  6. Observability: block message carries [policy:ai_stock:*] tag

Run::

    python scripts/ai_stock_policy_smoketest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.agent.agent_context_service import BrokerContext
from trading_app.services.ai.ai_stock_risk_policy import AIStockRiskPolicy
from trading_app.services.ai.auto_trade_config_service import AutoTradeConfig
from trading_app.services.execution.risk_guard_service import RiskGuardService
from trading_app.services.strategy.strategy_constants import (
    AI_STOCK_STRATEGY_ID,
    AI_STOCK_STRATEGY_NAME,
    AI_STOCK_VIRTUAL_ACCOUNT_ID,
)
from trading_app.services.strategy.strategy_risk import (
    StrategyRiskContext,
    get_strategy_risk_registry,
    reset_strategy_risk_registry,
)
from trading_app.services.ai.trade_decision_models import (
    RiskCheckItem,
    RiskCheckResult,
    TradeAction,
    TradeDecision,
)
from trading_app.services.execution.trade_execution_service import (
    ExecutionRequest,
    TradeExecutionService,
)
from trading_app.services.execution.trade_record_service import TradeSource


def _build_decision(confidence: float = 0.8, action: str = TradeAction.BUY.value) -> TradeDecision:
    return TradeDecision(
        action=action,
        symbol_code="000001.SZ",
        symbol_name="平安银行",
        confidence=confidence,
        target_price=12.0,
        stop_loss_price=9.8,
        current_price=10.0,
        position_pct=0.10,
        risk_score=0.30,
    )


def _build_request(
    *,
    decision=None,
    risk_result=None,
    price: float = 10.0,
    order_type: int = 23,
) -> ExecutionRequest:
    return ExecutionRequest(
        stock_code="000001.SZ",
        stock_name="平安银行",
        order_type=order_type,
        order_volume=100,
        price_type=0,
        price=price,
        source=TradeSource.AI_AGENT.value,
        trigger="manual",
        strategy_name=AI_STOCK_STRATEGY_NAME,
        strategy_id=AI_STOCK_STRATEGY_ID,
        virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
        intent_id=f"ai-policy-smoke-{int(time.time() * 1000)}",
        remark="ai policy smoketest",
        decision=decision,
        risk_result=risk_result,
        require_approval=True,
        approved=True,
    )


# ---------------------------------------------------------------------------
# 1-4: 直接 evaluate policy
# ---------------------------------------------------------------------------

def case_approve_without_decision() -> None:
    policy = AIStockRiskPolicy(risk_guard=RiskGuardService())
    req = _build_request(decision=None)
    result = policy.evaluate(req, StrategyRiskContext())
    assert result.passed and "跳过" in result.reason
    print("[approve_without_decision] OK -> reason=", result.reason)


def case_approve_good_decision() -> None:
    policy = AIStockRiskPolicy(risk_guard=RiskGuardService())
    broker = BrokerContext(connected=True, total_asset=1_000_000.0, available_cash=600_000.0)
    req = _build_request(decision=_build_decision(confidence=0.8))
    result = policy.evaluate(req, StrategyRiskContext(broker=broker))
    assert result.passed, result.reason
    print("[approve_good_decision] OK -> level=%s reason=%s" % (result.level, result.reason))


def case_block_low_confidence() -> None:
    policy = AIStockRiskPolicy(risk_guard=RiskGuardService())
    broker = BrokerContext(connected=True, total_asset=1_000_000.0, available_cash=600_000.0)
    req = _build_request(decision=_build_decision(confidence=0.3))
    result = policy.evaluate(req, StrategyRiskContext(broker=broker))
    assert not result.passed
    assert "置信度" in result.reason
    assert result.metadata.get("rule") == "ai_stock_rules"
    print("[block_low_confidence] OK -> reason=", result.reason)


def case_block_pre_evaluated_fail() -> None:
    policy = AIStockRiskPolicy(risk_guard=RiskGuardService())
    pre = RiskCheckResult(
        passed=False,
        checks=[RiskCheckItem(name="test", passed=False, level="block", message="test")],
        overall_risk_level="critical",
        warnings=[],
        blocked_reasons=["上游已拒绝"],
    )
    req = _build_request(decision=_build_decision(), risk_result=pre)
    result = policy.evaluate(req, StrategyRiskContext())
    assert not result.passed
    assert "上游已拒绝" in result.reason
    assert result.metadata.get("rule") == "pre_evaluated"
    print("[block_pre_evaluated] OK -> reason=", result.reason)


# ---------------------------------------------------------------------------
# 5-6: 网关集成 + 观测标签
# ---------------------------------------------------------------------------

class _FakeBroker:
    is_connected = True

    def query_stock_asset(self):
        return SimpleNamespace(cash=500_000.0, available_cash=500_000.0, total_asset=1_000_000.0)

    def query_stock_positions(self):
        return []

    def order_stock(self, *_args, **_kwargs):
        raise RuntimeError("shadow 模式不应触发真实下单")

    def query_stock_order(self, _order_id: int):
        return None


def _build_service() -> TradeExecutionService:
    service = TradeExecutionService(_FakeBroker())
    service.config_service.get_config = lambda: AutoTradeConfig(
        manual_orders_enabled=True,
        auto_trade_mode="shadow",
        require_trading_time=False,
        duplicate_window_seconds=1,
        status_poll_seconds=1.0,
        status_poll_interval_seconds=0.2,
    )
    service._validate_market_data_status = lambda _request: ""
    return service


def case_gateway_auto_registers_ai_policy() -> None:
    reset_strategy_risk_registry()
    service = _build_service()
    assert get_strategy_risk_registry().has(AI_STOCK_STRATEGY_ID), (
        "TradeExecutionService 初始化时应自动注册 AIStockRiskPolicy"
    )

    unique_price = 11.0 + ((int(time.time() * 1000) % 500) / 1000.0)
    req = _build_request(decision=_build_decision(confidence=0.2), price=unique_price)
    result = service.execute(req)
    assert not result.success, f"expected block, got {result.message}"
    assert result.blocked
    assert "置信度" in result.message, result.message
    print("[gateway_auto_registers_ai_policy] OK -> message=", result.message)


def case_block_message_has_observability_tag() -> None:
    reset_strategy_risk_registry()
    service = _build_service()

    unique_price = 10.0 + ((int(time.time() * 1000) % 500) / 1000.0)
    req = _build_request(decision=_build_decision(confidence=0.2), price=unique_price)
    result = service.execute(req)

    tag_prefix = f"[policy:{AI_STOCK_STRATEGY_ID}:"
    assert tag_prefix in result.message, f"expected observability tag, got {result.message!r}"
    assert "ai_stock_rules" in result.message, result.message
    print("[block_message_observability] OK -> message=", result.message)


# ---------------------------------------------------------------------------

def main() -> None:
    case_approve_without_decision()
    case_approve_good_decision()
    case_block_low_confidence()
    case_block_pre_evaluated_fail()
    case_gateway_auto_registers_ai_policy()
    case_block_message_has_observability_tag()
    reset_strategy_risk_registry()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
