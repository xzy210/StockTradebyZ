from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

from .models import RiskPolicyDecision, StrategyRiskContext
from .schema import RiskConfigField

if TYPE_CHECKING:  # pragma: no cover
    from trading_app.services.execution.trade_execution_service import ExecutionRequest

logger = logging.getLogger(__name__)


@runtime_checkable
class StrategyRiskPolicy(Protocol):
    """Strategy-specific risk policy invoked by the unified trade gateway.

    Implementations must be thread-safe and side-effect-free: they may read
    external state (broker, budget, strategy internal state) but MUST NOT
    mutate order state or submit orders. The registry aggregates decisions
    and the gateway is responsible for applying them.
    """

    strategy_id: str

    def evaluate(
        self,
        request: "ExecutionRequest",
        context: StrategyRiskContext,
    ) -> RiskPolicyDecision: ...


@runtime_checkable
class ConfigurableStrategyRiskPolicy(Protocol):
    """Optional protocol for policies that expose declarative config schema.

    对外承诺三件事（顺序无关）：

    - :meth:`config_schema` 返回字段声明，驱动 UI 自动渲染
    - :meth:`get_config` 返回当前生效值（按 name 组成 dict）
    - :meth:`apply_config` 接收 UI 回写的新值，内部负责落库 / 通知 engine

    通用面板 :class:`StrategyRiskSettingsPanel` 通过 runtime_checkable 的
    ``isinstance`` 识别是否为 Configurable，未实现的 policy 自动跳过。
    """

    strategy_id: str

    def config_schema(self) -> List[RiskConfigField]: ...

    def get_config(self) -> Dict[str, Any]: ...

    def apply_config(self, values: Dict[str, Any]) -> None: ...


def is_configurable(policy: Any) -> bool:
    """Return True iff *policy* exposes the configurable-risk protocol.

    ``runtime_checkable`` ``isinstance`` on Protocol only validates method
    presence, not signatures. We keep the check lenient here so stubs used
    in tests don't need to satisfy the full signature shape.
    """
    return all(
        callable(getattr(policy, name, None))
        for name in ("config_schema", "get_config", "apply_config")
    )


class NoopStrategyRiskPolicy:
    """Default no-op policy used when no strategy-specific rules are registered.

    Returns an approval decision unconditionally. Useful as a placeholder and
    for tests that want to bypass strategy risk evaluation.
    """

    def __init__(self, strategy_id: str = "__noop__") -> None:
        self.strategy_id = strategy_id

    def evaluate(
        self,
        request: "ExecutionRequest",
        context: StrategyRiskContext,
    ) -> RiskPolicyDecision:
        return RiskPolicyDecision.approve(reason="无策略风控规则")
