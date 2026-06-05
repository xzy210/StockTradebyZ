"""AI stock strategy risk policy for the unified trade gateway.

This policy wraps the existing :class:`RiskGuardService` so AI decisions get
evaluated through the same :class:`StrategyRiskPolicy` mechanism as ETF
rotation. Rule implementation is left untouched inside ``RiskGuardService``;
this module only adapts its ``RiskCheckResult`` to the gateway-friendly
``RiskPolicyDecision`` shape.

The policy is auto-registered by :class:`TradeExecutionService`, so any order
with ``strategy_id == AI_STOCK_STRATEGY_ID`` flows through it without the
caller having to opt in.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from common.agent.agent_context_service import BrokerContext
from trading_app.services.execution.risk_guard_service import DEFAULT_CONFIG as _RISK_GUARD_DEFAULTS, RiskGuardService
from trading_app.services.strategy.strategy_constants import AI_STOCK_STRATEGY_ID
from trading_app.services.strategy.strategy_risk import (
    RiskConfigField,
    RiskPolicyDecision,
    StrategyRiskContext,
)

logger = logging.getLogger(__name__)


class AIStockRiskPolicy:
    """Strategy-level risk policy for AI-driven stock orders."""

    strategy_id: str = AI_STOCK_STRATEGY_ID

    #: Declarative schema driving ``StrategyRiskSettingsPanel``.
    #: 分组用短标签收拢"置信度阈值 / 仓位上限 / 特殊股票规则 / 涨跌停"四类。
    _CONFIG_SCHEMA: Tuple[RiskConfigField, ...] = (
        RiskConfigField(
            name="min_confidence",
            label="最低置信度",
            type="float",
            default=_RISK_GUARD_DEFAULTS["min_confidence"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            decimals=1,
            suffix=" %",
            display_scale=100.0,
        help="AI 实盘决策置信度低于此阈值的买/加仓订单会被拦截",
        ),
        RiskConfigField(
            name="max_risk_score_for_buy",
            label="买入风险评分上限",
            type="float",
            default=_RISK_GUARD_DEFAULTS["max_risk_score_for_buy"],
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            decimals=2,
            help="TradeDecision.risk_score 超过此值的买单会被拦截；取值 0~1",
        ),
        RiskConfigField(
            name="max_stop_loss_pct",
            label="止损幅度告警",
            type="float",
            default=_RISK_GUARD_DEFAULTS["max_stop_loss_pct"],
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            decimals=1,
            suffix=" %",
            display_scale=100.0,
            help="止损价与现价差距超过此比例会触发 warn（不拦截）",
        ),
        RiskConfigField(
            name="max_single_position_pct",
            label="单票仓位上限",
            type="float",
            default=_RISK_GUARD_DEFAULTS["max_single_position_pct"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            decimals=1,
            suffix=" %",
            display_scale=100.0,
            help="建议仓位超过此比例的买单会被拦截（需有券商账户上下文）",
        ),
        RiskConfigField(
            name="max_total_position_pct",
            label="总仓位上限",
            type="float",
            default=_RISK_GUARD_DEFAULTS["max_total_position_pct"],
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            decimals=1,
            suffix=" %",
            display_scale=100.0,
            help="账户总仓位已超过此比例时拦截新买单",
        ),
        RiskConfigField(
            name="block_st_stocks",
            label="拦截 ST 股",
            type="bool",
            default=_RISK_GUARD_DEFAULTS["block_st_stocks"],
            help="开启后所有 ST/*ST 标的订单都会被硬拦截",
        ),
        RiskConfigField(
            name="warn_st_stocks",
            label="ST 股告警",
            type="bool",
            default=_RISK_GUARD_DEFAULTS["warn_st_stocks"],
            help="不拦截但提示风险；仅在未开启上面的『拦截 ST 股』时有效",
        ),
        RiskConfigField(
            name="block_limit_up_buy",
            label="涨停禁买",
            type="bool",
            default=_RISK_GUARD_DEFAULTS["block_limit_up_buy"],
            help="启用后接近涨停的标的会禁止买入（最终以券商实时涨停校验为准）",
        ),
        RiskConfigField(
            name="block_limit_down_sell",
            label="跌停禁卖",
            type="bool",
            default=_RISK_GUARD_DEFAULTS["block_limit_down_sell"],
            help="启用后接近跌停的标的会禁止卖出",
        ),
        RiskConfigField(
            name="limit_up_pct",
            label="涨停判定阈值",
            type="float",
            default=_RISK_GUARD_DEFAULTS["limit_up_pct"],
            min_value=0.0,
            max_value=30.0,
            step=0.1,
            decimals=2,
            suffix=" %",
            display_scale=100.0,
            help="日涨幅 ≥ 此值视为接近涨停（主板 9.8%、创业板 19.8% 等）",
        ),
        RiskConfigField(
            name="limit_down_pct",
            label="跌停判定阈值",
            type="float",
            default=_RISK_GUARD_DEFAULTS["limit_down_pct"],
            min_value=-30.0,
            max_value=0.0,
            step=0.1,
            decimals=2,
            suffix=" %",
            display_scale=100.0,
            help="日跌幅 ≤ 此值视为接近跌停（负数）",
        ),
    )

    def __init__(self, risk_guard: Optional[RiskGuardService] = None) -> None:
        self.risk_guard = risk_guard or RiskGuardService()

    # ------------------------------------------------------------------
    #  Declarative config contract
    # ------------------------------------------------------------------

    def config_schema(self) -> List[RiskConfigField]:
        return list(self._CONFIG_SCHEMA)

    def get_config(self) -> Dict[str, Any]:
        cfg = self.risk_guard.config or {}
        return {
            f.name: cfg.get(f.name, f.default)
            for f in self._CONFIG_SCHEMA
        }

    def apply_config(self, values: Dict[str, Any]) -> None:
        """Pass through to :meth:`RiskGuardService.update_config` for persistence."""
        if not values:
            return
        clean: Dict[str, Any] = {}
        for f in self._CONFIG_SCHEMA:
            if f.name not in values:
                continue
            clean[f.name] = f.from_display(values[f.name])
        self.risk_guard.update_config(**clean)

    def evaluate(
        self,
        request: Any,
        context: StrategyRiskContext,
    ) -> RiskPolicyDecision:
        decision = getattr(request, "decision", None)
        if decision is None:
            return RiskPolicyDecision.approve(
                "AI 订单未附带 TradeDecision，跳过 AI 规则检查"
            )

        pre_eval = getattr(request, "risk_result", None)
        if pre_eval is not None:
            if not pre_eval.passed:
                reason = "；".join(pre_eval.blocked_reasons) or "AI 风控未通过"
                return RiskPolicyDecision.block(
                    reason,
                    metadata={"rule": "pre_evaluated"},
                )
            if pre_eval.warnings:
                return RiskPolicyDecision.warn(
                    "；".join(pre_eval.warnings),
                    metadata={"rule": "pre_evaluated_warnings"},
                )
            return RiskPolicyDecision.approve("预评估风控已通过")

        broker_ctx = context.broker if isinstance(context.broker, BrokerContext) else None
        try:
            result = self.risk_guard.evaluate(decision, broker=broker_ctx)
        except Exception as exc:
            logger.exception("AIStockRiskPolicy 调用 RiskGuardService 失败")
            return RiskPolicyDecision.block(
                f"AI 风控评估异常: {exc}",
                metadata={"rule": "ai_stock_evaluate_error"},
            )

        if not result.passed:
            reason = "；".join(result.blocked_reasons) or "AI 风控未通过"
            return RiskPolicyDecision.block(
                reason,
                metadata={
                    "rule": "ai_stock_rules",
                    "risk_level": result.overall_risk_level,
                },
            )
        if result.warnings:
            return RiskPolicyDecision.warn(
                "；".join(result.warnings),
                metadata={
                    "rule": "ai_stock_warnings",
                    "risk_level": result.overall_risk_level,
                },
            )
        return RiskPolicyDecision.approve(
            "AI 风控通过",
            metadata={"risk_level": result.overall_risk_level},
        )
