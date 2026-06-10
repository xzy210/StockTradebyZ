from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from pydantic import BaseModel, Field, validator


class ETFRotationParams(BaseModel):
    """Shared ETF rotation strategy, decision, and guard parameters."""

    etf_pool: List[str] = Field(
        default_factory=lambda: ["510880", "159949", "513100", "518880"],
        title="ETF池",
        description="参与轮动的 ETF 代码列表。",
        min_items=1,
    )
    factor_config: List[Tuple[str, float]] = Field(
        default_factory=lambda: [
            ("bias_momentum_fast", 0.3),
            ("slope_momentum_fast", 0.3),
            ("efficiency_momentum_fast", 0.4),
        ],
        title="因子配置",
        description="用于合成评分的动量因子名称和权重。",
        min_items=1,
    )
    rebalance_threshold: float = Field(
        1.5,
        ge=0.0,
        le=100.0,
        title="切换阈值",
        description="当最高分 ETF 分数超过当前持仓分数乘以该阈值时触发切换。",
    )
    momentum_window: int = Field(
        25,
        ge=2,
        le=500,
        title="动量窗口",
        description="ETF 动量因子的回看窗口。",
    )
    zscore_window: int = Field(
        60,
        ge=5,
        le=1000,
        title="Z-Score窗口",
        description="将因子值标准化为 z-score 的回看窗口。",
    )
    empty_threshold: float = Field(
        -0.5,
        ge=-100.0,
        le=100.0,
        title="空仓阈值",
        description="当所有 ETF 评分都低于该值时，策略保持空仓或清仓。",
    )
    min_buy_score: float = Field(
        0.0,
        ge=-100.0,
        le=100.0,
        title="最低买入分数",
        description="空仓或初始建仓时，最高评分低于该值则不买入。",
    )
    enable_empty_position: bool = Field(
        True,
        title="允许空仓",
        description="当所有 ETF 评分较弱时允许持有现金。",
    )
    rebalance_period: int = Field(
        1,
        ge=1,
        le=250,
        title="调仓周期",
        description="以交易日计量的最小调仓间隔。",
    )
    enable_trailing_stop: bool = Field(
        True,
        title="启用移动止盈",
        description="启用单持仓移动止盈保护。",
    )
    trailing_stop_pct: float = Field(
        0.08,
        ge=0.0,
        le=1.0,
        title="移动止盈比例",
        description="当当前持仓从观察到的最高价回撤该比例时卖出。",
    )
    enable_drawdown_protection: bool = Field(
        True,
        title="启用回撤保护",
        description="启用账户级回撤保护。",
    )
    max_drawdown_pct: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        title="最大回撤比例",
        description="账户回撤达到该比例后退出并进入冷却期。",
    )
    drawdown_cooldown_days: int = Field(
        10,
        ge=0,
        le=250,
        title="回撤冷却天数",
        description="触发账户回撤保护后的交易日冷却天数。",
    )

    _LEGACY_WEIGHT_KEYS: ClassVar[Dict[str, str]] = {
        "bias_weight": "bias_momentum_fast",
        "slope_weight": "slope_momentum_fast",
        "efficiency_weight": "efficiency_momentum_fast",
    }

    @validator("etf_pool")
    def _normalize_etf_pool(cls, value: List[str]) -> List[str]:
        normalized = [str(item).strip() for item in value or [] if str(item).strip()]
        if not normalized:
            raise ValueError("ETF池至少需要包含一个代码")
        return normalized

    @validator("factor_config", pre=True)
    def _normalize_factor_config(cls, value: Any) -> List[Tuple[str, float]]:
        items = list(value or [])
        normalized: List[Tuple[str, float]] = []
        for item in items:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("factor") or "").strip()
                weight = item.get("weight", 0.0)
            else:
                try:
                    name, weight = item
                except (TypeError, ValueError):
                    raise ValueError("因子配置项必须是 (名称, 权重) 格式") from None
                name = str(name).strip()
            if not name:
                raise ValueError("因子名称不能为空")
            normalized.append((name, float(weight)))
        if not normalized:
            raise ValueError("因子配置至少需要包含一个因子")
        return normalized

    @classmethod
    def field_names(cls) -> set[str]:
        return set(getattr(cls, "model_fields", None) or getattr(cls, "__fields__", {}))

    @classmethod
    def from_mapping(cls, values: dict | None = None, **overrides: Any) -> "ETFRotationParams":
        payload = dict(values or {})
        legacy_factors = []
        for key, factor_name in cls._LEGACY_WEIGHT_KEYS.items():
            if key in payload:
                legacy_factors.append((factor_name, payload.pop(key)))
        if legacy_factors and "factor_config" not in payload:
            payload["factor_config"] = legacy_factors
        payload.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**{key: value for key, value in payload.items() if key in cls.field_names()})

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


__all__ = ["ETFRotationParams"]
