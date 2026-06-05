"""ETF rotation live risk policy for the unified trade gateway.

这是 ETF 轮动实盘风控的**唯一事实源**（此前的 ``live_rotation.risk_manager``
已整体移除）。规则被封装为 :class:`ETFRotationRiskPolicy`，并由
:class:`RotationEngine` 在启动时注册到
:class:`trading_app.services.strategy.strategy_risk.StrategyRiskRegistry`。

两条触发路径：
  * 真实盘：订单经 :class:`TradeExecutionService` 统一网关时自动触发 policy；
  * 模拟盘：由 ``RotationEngine._preflight_strategy_risk_policy`` 在下单前
    显式调用同一个 registry，保证模拟盘也有风控兜底。

Regime:
  * trading time  -> BLOCK (non trading day / off hours / lunch break)
  * daily cap     -> BLOCK (>= max_trades_per_day already today)
  * min hold days -> BLOCK (sell before min holding satisfied)
  * single loss   -> WARN  (matches legacy behaviour: allow stop-loss exit)
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import Any, Callable, Dict, List, Optional, Tuple

from trading_app.services.strategy.strategy_risk import (
    RiskConfigField,
    RiskPolicyDecision,
    StrategyRiskContext,
)

from .holiday_calendar import get_non_trading_reason, is_trading_day

logger = logging.getLogger(__name__)

LUNCH_START = dtime(11, 30)
LUNCH_END = dtime(13, 0)

SELL_ORDER_TYPE = 24


def check_trading_time(config: Any, now: datetime) -> Tuple[bool, str]:
    """Return ``(ok, message)`` for the ETF rotation trading window.

    Uses the ETF-specific ``trading_start`` / ``trading_end`` from the
    :class:`RotationConfig` instead of the global gateway window so strategies
    that want a tighter cutoff (e.g. 14:57) are honoured.
    """
    if not is_trading_day(now.date()):
        return False, get_non_trading_reason(now.date()) or "非交易日"

    try:
        sh, sm = map(int, str(config.trading_start).split(":"))
        eh, em = map(int, str(config.trading_end).split(":"))
        start = dtime(sh, sm)
        end = dtime(eh, em)
    except (AttributeError, ValueError):
        return True, "交易时间配置异常，跳过检查"

    current = now.time()
    if current < start:
        return False, f"未到开盘时间（{config.trading_start}）"
    if current > end:
        return False, f"已过交易截止时间（{config.trading_end}）"
    if LUNCH_START <= current <= LUNCH_END:
        return False, "午休时段（11:30-13:00）"
    return True, "在交易时段内"


def check_daily_trades(config: Any, state: Any) -> Tuple[bool, str]:
    today_count = int(getattr(state, "get_trades_today", lambda: 0)() or 0)
    limit = int(getattr(config, "max_trades_per_day", 0) or 0)
    if limit > 0 and today_count >= limit:
        return False, f"今日交易次数已达上限（{today_count}/{limit}）"
    return True, "交易次数正常"


def check_hold_days(config: Any, state: Any, *, now: datetime) -> Tuple[bool, str]:
    min_hold = int(getattr(config, "min_hold_days", 0) or 0)
    if min_hold <= 0:
        return True, "无持有限制"
    buy_date = str(getattr(state, "buy_date", "") or "")
    if not buy_date:
        return True, "无持仓记录"
    try:
        buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
    except ValueError:
        return True, "持仓日期格式异常，跳过检查"
    hold_days = (now - buy_dt).days
    if hold_days < min_hold:
        return False, f"持有天数不足（{hold_days}/{min_hold}天）"
    return True, "持有天数满足"


def check_loss_limit(config: Any, state: Any, current_price: float) -> Tuple[bool, str]:
    """Return ``(within_limit, message)``.

    ``within_limit`` is ``False`` when the loss exceeds the configured
    threshold. The ETF policy maps a False outcome to a ``warn`` decision so
    stop-loss exits remain possible, mirroring the legacy behaviour.
    """
    buy_price = float(getattr(state, "buy_price", 0.0) or 0.0)
    if buy_price <= 0:
        return True, "无买入价记录"
    if current_price <= 0:
        return True, "无当前价记录"
    loss_pct = (current_price - buy_price) / buy_price * 100
    max_loss_pct = float(getattr(config, "max_single_loss_pct", 0.0) or 0.0)
    if max_loss_pct > 0 and loss_pct < -max_loss_pct:
        return False, f"亏损 {loss_pct:.2f}% 超过 -{max_loss_pct}%（允许止损卖出）"
    return True, "亏损在限制范围内"


class ETFRotationRiskPolicy:
    """Strategy-level risk policy for ETF rotation orders.

    The policy reads ``RotationConfig`` / ``RotationState`` via provider
    callables, so it always sees the latest instance even after
    ``RotationEngine.update_config`` or state reloads.
    """

    #: Declarative schema consumed by ``StrategyRiskSettingsPanel``.
    #: Keep this list ordered the way you want fields to appear in the UI.
    _CONFIG_SCHEMA: Tuple[RiskConfigField, ...] = (
        RiskConfigField(
            name="enable_risk_check",
            label="启用策略风控",
            type="bool",
            default=True,
        help="关闭后，ETF 轮动实盘所有订单跳过本策略的 policy 检查（账户级闸仍生效）",
        ),
        RiskConfigField(
            name="trading_start",
            label="交易开始时间",
            type="time",
            default="09:30",
            help="早于此时间的订单会被拦截（早盘集合竞价可设 09:30）",
            depends_on="enable_risk_check",
        ),
        RiskConfigField(
            name="trading_end",
            label="交易截止时间",
            type="time",
            default="14:57",
            help="晚于此时间的订单会被拦截，留 3 分钟 buffer 避免尾盘流动性差",
            depends_on="enable_risk_check",
        ),
        RiskConfigField(
            name="max_trades_per_day",
            label="每日最大交易次数",
            type="int",
            default=2,
            min_value=0,
            max_value=50,
            step=1,
            suffix=" 次",
            help="当日买+卖达到此次数后新订单会被拦截；0 表示不限制",
            depends_on="enable_risk_check",
        ),
        RiskConfigField(
            name="min_hold_days",
            label="最少持有天数",
            type="int",
            default=0,
            min_value=0,
            max_value=120,
            step=1,
            suffix=" 天",
            help="持有不满此天数时卖单会被拦截；0 表示不限制",
            depends_on="enable_risk_check",
        ),
        RiskConfigField(
            name="max_single_loss_pct",
            label="单笔最大亏损告警",
            type="float",
            default=15.0,
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            decimals=1,
            suffix=" %",
            help="亏损超过此阈值的卖单会触发 warn（不拦截，兼容止损退出）",
            depends_on="enable_risk_check",
        ),
    )

    def __init__(
        self,
        *,
        strategy_id: str,
        config_provider: Callable[[], Any],
        state_provider: Callable[[], Any],
        config_saver: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.strategy_id = str(strategy_id or "").strip() or "etf_rotation"
        self._config_provider = config_provider
        self._state_provider = state_provider
        self._config_saver = config_saver

    # ------------------------------------------------------------------
    #  Declarative config contract (consumed by StrategyRiskSettingsPanel)
    # ------------------------------------------------------------------

    def config_schema(self) -> List[RiskConfigField]:
        return list(self._CONFIG_SCHEMA)

    def get_config(self) -> Dict[str, Any]:
        try:
            cfg = self._config_provider()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ETFRotationRiskPolicy.get_config 读取配置失败: %s", exc)
            return {f.name: f.default for f in self._CONFIG_SCHEMA}
        return {f.name: getattr(cfg, f.name, f.default) for f in self._CONFIG_SCHEMA}

    def apply_config(self, values: Dict[str, Any]) -> None:
        """Persist updated values back into the owning ``RotationConfig``.

        Requires a ``config_saver`` injected by the engine; otherwise raises
        so the UI surfaces the misconfiguration instead of silently dropping
        user input.
        """
        if self._config_saver is None:
            raise RuntimeError(
                "ETFRotationRiskPolicy.apply_config 需要 config_saver 回调（应由 RotationEngine 注入）"
            )
        clean: Dict[str, Any] = {}
        for f in self._CONFIG_SCHEMA:
            if f.name not in values:
                continue
            clean[f.name] = f.from_display(values[f.name])
        self._config_saver(clean)

    def evaluate(
        self,
        request: Any,
        context: StrategyRiskContext,
    ) -> RiskPolicyDecision:
        try:
            config = self._config_provider()
            state = self._state_provider()
        except Exception as exc:
            logger.exception(
                "ETFRotationRiskPolicy 获取配置/状态失败 strategy_id=%s",
                self.strategy_id,
            )
            return RiskPolicyDecision.block(f"ETF 风控读取配置失败: {exc}")

        if not getattr(config, "enable_risk_check", True):
            return RiskPolicyDecision.approve("ETF 风控已禁用")

        now = context.now or datetime.now()

        ok, msg = check_trading_time(config, now)
        if not ok:
            return RiskPolicyDecision.block(msg, metadata={"rule": "trading_time"})

        ok, msg = check_daily_trades(config, state)
        if not ok:
            return RiskPolicyDecision.block(msg, metadata={"rule": "daily_trades"})

        is_sell = int(getattr(request, "order_type", 0) or 0) == SELL_ORDER_TYPE
        if is_sell:
            ok, msg = check_hold_days(config, state, now=now)
            if not ok:
                return RiskPolicyDecision.block(msg, metadata={"rule": "min_hold_days"})

            try:
                price = float(getattr(request, "price", 0.0) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            if price > 0:
                within_limit, msg = check_loss_limit(config, state, price)
                if not within_limit:
                    return RiskPolicyDecision.warn(
                        msg,
                        metadata={"rule": "single_loss_limit"},
                    )

        return RiskPolicyDecision.approve("ETF 风控通过")
