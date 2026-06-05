from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from trading_app.services.ai.trade_decision_models import RiskCheckItem

RISK_LEVEL_INFO = "info"
RISK_LEVEL_WARN = "warn"
RISK_LEVEL_BLOCK = "block"

VALID_RISK_LEVELS = (RISK_LEVEL_INFO, RISK_LEVEL_WARN, RISK_LEVEL_BLOCK)


@dataclass
class RiskPolicyDecision:
    """Result returned by a single StrategyRiskPolicy.evaluate() call."""

    passed: bool = True
    level: str = RISK_LEVEL_INFO
    reason: str = ""
    checks: List[RiskCheckItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def approve(
        cls,
        reason: str = "",
        *,
        checks: Optional[List[RiskCheckItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RiskPolicyDecision":
        return cls(
            passed=True,
            level=RISK_LEVEL_INFO,
            reason=reason,
            checks=list(checks or []),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def warn(
        cls,
        reason: str,
        *,
        checks: Optional[List[RiskCheckItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RiskPolicyDecision":
        return cls(
            passed=True,
            level=RISK_LEVEL_WARN,
            reason=reason,
            checks=list(checks or []),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def block(
        cls,
        reason: str,
        *,
        checks: Optional[List[RiskCheckItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RiskPolicyDecision":
        return cls(
            passed=False,
            level=RISK_LEVEL_BLOCK,
            reason=reason,
            checks=list(checks or []),
            metadata=dict(metadata or {}),
        )


@dataclass
class StrategyRiskContext:
    """Runtime context passed to a StrategyRiskPolicy.

    Policies should treat every attribute as best-effort: fields may be empty
    when the gateway cannot resolve the underlying data (e.g. broker offline,
    budget not configured). Keep policy logic defensive.
    """

    now: datetime = field(default_factory=datetime.now)
    broker: Optional[Any] = None  # common.agent.agent_context_service.BrokerContext
    budget_snapshot: Dict[str, Any] = field(default_factory=dict)
    request_extras: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
