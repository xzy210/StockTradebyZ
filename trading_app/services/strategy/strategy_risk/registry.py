from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional

from .models import (
    RISK_LEVEL_BLOCK,
    RISK_LEVEL_INFO,
    RISK_LEVEL_WARN,
    RiskPolicyDecision,
    StrategyRiskContext,
)
from .policy import NoopStrategyRiskPolicy, StrategyRiskPolicy

if TYPE_CHECKING:  # pragma: no cover
    from trading_app.services.execution.trade_execution_service import ExecutionRequest

logger = logging.getLogger(__name__)

_NOOP_POLICY = NoopStrategyRiskPolicy()


class StrategyRiskRegistry:
    """Registry that routes ExecutionRequest to strategy-specific risk policies.

    Design goals:
    - Default to no-op: if a strategy_id has no registered policy the gateway
      behaviour is unchanged.
    - Composable: multiple policies can be chained under the same strategy_id,
      evaluated in registration order. The first blocker short-circuits.
    - Isolated: a policy raising an exception is treated as a hard block so a
      buggy strategy cannot silently let orders through.
    """

    def __init__(self) -> None:
        self._policies: Dict[str, List[StrategyRiskPolicy]] = {}
        self._lock = threading.RLock()

    def register(self, policy: StrategyRiskPolicy, *, override: bool = False) -> None:
        strategy_id = self._normalize_id(getattr(policy, "strategy_id", ""))
        if not strategy_id:
            raise ValueError("policy.strategy_id 不能为空")
        with self._lock:
            bucket = self._policies.setdefault(strategy_id, [])
            if override:
                bucket.clear()
            bucket.append(policy)
        logger.info(
            "StrategyRiskRegistry 注册 policy: strategy_id=%s class=%s override=%s bucket_size=%d",
            strategy_id,
            type(policy).__name__,
            override,
            len(bucket),
        )

    def unregister(
        self,
        strategy_id: str,
        policy: Optional[StrategyRiskPolicy] = None,
    ) -> None:
        sid = self._normalize_id(strategy_id)
        if not sid:
            return
        with self._lock:
            if policy is None:
                removed = self._policies.pop(sid, None)
                if removed is not None:
                    logger.info(
                        "StrategyRiskRegistry 移除 strategy_id=%s 共 %d 条 policy",
                        sid,
                        len(removed),
                    )
                return
            bucket = self._policies.get(sid, [])
            new_bucket = [p for p in bucket if p is not policy]
            if new_bucket:
                self._policies[sid] = new_bucket
            else:
                self._policies.pop(sid, None)

    def clear(self) -> None:
        with self._lock:
            self._policies.clear()

    def has(self, strategy_id: str) -> bool:
        sid = self._normalize_id(strategy_id)
        if not sid:
            return False
        with self._lock:
            return bool(self._policies.get(sid))

    def resolve(self, strategy_id: str) -> List[StrategyRiskPolicy]:
        sid = self._normalize_id(strategy_id)
        with self._lock:
            if sid and sid in self._policies:
                return list(self._policies[sid])
        return [_NOOP_POLICY]

    def evaluate(
        self,
        request: "ExecutionRequest",
        context: StrategyRiskContext,
    ) -> RiskPolicyDecision:
        strategy_id = self._normalize_id(getattr(request, "strategy_id", ""))
        policies = self.resolve(strategy_id)
        merged = RiskPolicyDecision(passed=True, level=RISK_LEVEL_INFO)
        reasons: List[str] = []

        for policy in policies:
            try:
                result = policy.evaluate(request, context)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception(
                    "StrategyRiskPolicy 执行异常 strategy_id=%s policy=%s",
                    strategy_id,
                    type(policy).__name__,
                )
                return RiskPolicyDecision.block(
                    reason=f"策略风控执行异常: {exc}",
                    metadata={"policy": type(policy).__name__},
                )
            if result is None:
                continue

            if result.checks:
                merged.checks.extend(result.checks)
            if result.metadata:
                merged.metadata.update(result.metadata)

            if not result.passed:
                logger.warning(
                    "StrategyRiskPolicy 拦截下单: strategy_id=%s policy=%s reason=%s",
                    strategy_id,
                    type(policy).__name__,
                    result.reason,
                )
                merged.passed = False
                merged.level = RISK_LEVEL_BLOCK
                merged.reason = result.reason or merged.reason or "策略风控未通过"
                return merged

            if result.level == RISK_LEVEL_WARN and merged.level == RISK_LEVEL_INFO:
                merged.level = RISK_LEVEL_WARN
            if result.reason:
                reasons.append(result.reason)

        if merged.passed and reasons and not merged.reason:
            merged.reason = "；".join(reasons)
        return merged

    @staticmethod
    def _normalize_id(value: Optional[str]) -> str:
        return str(value or "").strip()


_registry_singleton: Optional[StrategyRiskRegistry] = None
_registry_lock = threading.Lock()


def get_strategy_risk_registry() -> StrategyRiskRegistry:
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is None:
            _registry_singleton = StrategyRiskRegistry()
        return _registry_singleton


def reset_strategy_risk_registry() -> None:
    """Test-only: drop the singleton so the next call rebuilds a fresh registry."""
    global _registry_singleton
    with _registry_lock:
        _registry_singleton = None
