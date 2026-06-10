from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from common.io_utils import atomic_write_json

from common.agent.agent_context_service import BrokerContext
from trading_app.services.execution.portfolio_risk_service import PortfolioRiskService
from trading_app.services.ai.trade_decision_models import (
    RiskCheckItem,
    RiskCheckResult,
    TradeAction,
    TradeDecision,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "min_confidence": 0.6,
    "max_single_position_pct": 0.30,
    "max_total_position_pct": 0.90,
    "max_stop_loss_pct": 0.10,
    "max_risk_score_for_buy": 0.80,
    "block_st_stocks": False,
    "warn_st_stocks": True,
    "block_limit_up_buy": True,
    "block_limit_down_sell": True,
    "limit_up_pct": 0.098,
    "limit_down_pct": -0.098,
}

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_guard_config.json"


class RiskGuardService:
    """Rule-based risk assessment engine for trade decisions."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self._config_path: Path = Path(config_path) if config_path else _CONFIG_PATH
        self._load_config(self._config_path)
        self._portfolio_risk = PortfolioRiskService()

    def _load_config(self, path: Path) -> None:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                self.config.update(user_cfg)
        except Exception as exc:
            logger.warning("Failed to load risk guard config: %s", exc)

    @property
    def config_path(self) -> Path:
        return self._config_path

    def update_config(self, **overrides: Any) -> Dict[str, Any]:
        """合并修改内存中的配置，并持久化到 :attr:`config_path`。

        只允许覆盖 ``DEFAULT_CONFIG`` 中已声明的 key，其他字段忽略 + warning
        日志。持久化失败会向上抛出，由调用方（通常是 UI 保存按钮）展示错误。
        """
        clean: Dict[str, Any] = {}
        for key, value in (overrides or {}).items():
            if key not in DEFAULT_CONFIG:
                logger.warning("RiskGuardService.update_config 忽略未知字段: %s", key)
                continue
            clean[key] = value
        if not clean:
            return dict(self.config)
        self.config.update(clean)
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._config_path, dict(self.config))
        except Exception as exc:
            logger.error("写入 risk guard config 失败: %s", exc, exc_info=True)
            raise
        logger.info(
            "RiskGuardService 配置已更新并保存: fields=%s path=%s",
            list(clean.keys()),
            self._config_path,
        )
        return dict(self.config)

    def evaluate(
        self,
        decision: TradeDecision,
        broker: Optional[BrokerContext] = None,
    ) -> RiskCheckResult:
        checks: list[RiskCheckItem] = []
        warnings: list[str] = []
        blocked: list[str] = []

        self._check_confidence(decision, checks, warnings, blocked)
        self._check_stop_loss(decision, checks, warnings, blocked)
        self._check_risk_score(decision, checks, warnings, blocked)
        self._check_st_stock(decision, checks, warnings, blocked)
        self._check_limit_price(decision, checks, warnings, blocked)

        if broker and broker.connected:
            self._check_single_position(decision, broker, checks, warnings, blocked)
            self._check_total_position(decision, broker, checks, warnings, blocked)

            portfolio_items = self._portfolio_risk.check(decision, broker)
            for item in portfolio_items:
                checks.append(item)
                if item.level == "warn" and not item.passed:
                    warnings.append(item.message)
                elif item.level == "block" and not item.passed:
                    blocked.append(item.message)

        passed = len(blocked) == 0
        level = self._calc_risk_level(decision, blocked, warnings)

        return RiskCheckResult(
            passed=passed,
            checks=checks,
            overall_risk_level=level,
            warnings=warnings,
            blocked_reasons=blocked,
        )

    def _check_confidence(self, d: TradeDecision, checks, warnings, blocked):
        threshold = self.config["min_confidence"]
        ok = d.confidence >= threshold
        checks.append(RiskCheckItem(
            name="置信度检查",
            passed=ok,
            level="block" if not ok else "info",
            message=f"置信度 {d.confidence:.0%}{'≥' if ok else '<'}{threshold:.0%}",
        ))
        if not ok and d.is_actionable:
            blocked.append(f"置信度 {d.confidence:.0%} 低于阈值 {threshold:.0%}")

    def _check_stop_loss(self, d: TradeDecision, checks, warnings, blocked):
        if d.current_price <= 0 or d.stop_loss_price <= 0:
            checks.append(RiskCheckItem(name="止损幅度检查", passed=True, message="无止损价或无现价，跳过"))
            return
        loss_pct = abs(d.stop_loss_price - d.current_price) / d.current_price
        threshold = self.config["max_stop_loss_pct"]
        ok = loss_pct <= threshold
        checks.append(RiskCheckItem(
            name="止损幅度检查",
            passed=ok,
            level="warn" if not ok else "info",
            message=f"止损幅度 {loss_pct:.1%}{'≤' if ok else '>'}{threshold:.0%}",
        ))
        if not ok:
            warnings.append(f"止损幅度 {loss_pct:.1%} 超过 {threshold:.0%}，风险较大")

    def _check_risk_score(self, d: TradeDecision, checks, warnings, blocked):
        threshold = self.config["max_risk_score_for_buy"]
        is_buy = d.action in (TradeAction.BUY.value, TradeAction.ADD.value)
        if not is_buy:
            checks.append(RiskCheckItem(name="风险评分检查", passed=True, message="非买入操作，跳过"))
            return
        ok = d.risk_score <= threshold
        checks.append(RiskCheckItem(
            name="风险评分检查",
            passed=ok,
            level="block" if not ok else "info",
            message=f"风险评分 {d.risk_score:.2f}{'≤' if ok else '>'}{threshold:.2f}",
        ))
        if not ok:
            blocked.append(f"风险评分 {d.risk_score:.2f} 过高，不宜买入")

    def _check_st_stock(self, d: TradeDecision, checks, warnings, blocked):
        is_st = "ST" in d.symbol_name.upper() or "st" in d.symbol_name.lower()
        if not is_st:
            checks.append(RiskCheckItem(name="ST股检查", passed=True, message="非ST股"))
            return
        if self.config.get("block_st_stocks"):
            checks.append(RiskCheckItem(name="ST股检查", passed=False, level="block", message="ST股禁止交易"))
            blocked.append("标的为ST股，已被风控规则禁止")
        elif self.config.get("warn_st_stocks"):
            checks.append(RiskCheckItem(name="ST股检查", passed=True, level="warn", message="ST股，请注意风险"))
            warnings.append("标的为ST股，请务必注意退市风险")
        else:
            checks.append(RiskCheckItem(name="ST股检查", passed=True, message="ST股（未启用检查）"))

    def _check_limit_price(self, d: TradeDecision, checks, warnings, blocked):
        if d.current_price <= 0:
            checks.append(RiskCheckItem(name="涨跌停检查", passed=True, message="无现价数据，跳过"))
            return
        previous_close = float(
            getattr(d, "previous_close", 0.0)
            or getattr(d, "latest_close", 0.0)
            or 0.0
        )
        if previous_close <= 0:
            checks.append(RiskCheckItem(
                name="涨跌停检查",
                passed=True,
                level="info",
                message="缺少昨收参考，涨跌停下单前由券商实时校验",
            ))
            return

        change_pct = (float(d.current_price or 0.0) - previous_close) / previous_close
        is_buy = d.action in (TradeAction.BUY.value, TradeAction.ADD.value)
        is_sell = d.action in (TradeAction.SELL.value, TradeAction.REDUCE.value)
        limit_up_pct = float(self.config.get("limit_up_pct", DEFAULT_CONFIG["limit_up_pct"]) or 0.0)
        limit_down_pct = float(self.config.get("limit_down_pct", DEFAULT_CONFIG["limit_down_pct"]) or 0.0)
        if is_buy and self.config.get("block_limit_up_buy", True) and change_pct >= limit_up_pct:
            message = f"当前涨幅 {change_pct:.1%}≥涨停阈值 {limit_up_pct:.1%}，禁止买入"
            checks.append(RiskCheckItem(name="涨跌停检查", passed=False, level="block", message=message))
            blocked.append(message)
            return
        if is_sell and self.config.get("block_limit_down_sell", True) and change_pct <= limit_down_pct:
            message = f"当前跌幅 {change_pct:.1%}≤跌停阈值 {limit_down_pct:.1%}，禁止卖出"
            checks.append(RiskCheckItem(name="涨跌停检查", passed=False, level="block", message=message))
            blocked.append(message)
            return

        checks.append(RiskCheckItem(
            name="涨跌停检查",
            passed=True,
            level="info",
            message=f"当前涨跌幅 {change_pct:.1%}，未触发涨跌停禁令",
        ))

    def _check_single_position(self, d: TradeDecision, broker: BrokerContext, checks, warnings, blocked):
        is_buy = d.action in (TradeAction.BUY.value, TradeAction.ADD.value)
        if not is_buy or broker.total_asset <= 0:
            checks.append(RiskCheckItem(name="单票仓位检查", passed=True, message="非买入或无账户信息，跳过"))
            return
        threshold = self.config["max_single_position_pct"]
        projected = d.position_pct
        ok = projected <= threshold
        checks.append(RiskCheckItem(
            name="单票仓位检查",
            passed=ok,
            level="block" if not ok else "info",
            message=f"建议仓位 {projected:.0%}{'≤' if ok else '>'}{threshold:.0%}",
        ))
        if not ok:
            blocked.append(f"单票仓位 {projected:.0%} 超过上限 {threshold:.0%}")

    def _check_total_position(self, d: TradeDecision, broker: BrokerContext, checks, warnings, blocked):
        is_buy = d.action in (TradeAction.BUY.value, TradeAction.ADD.value)
        if not is_buy or broker.total_asset <= 0:
            checks.append(RiskCheckItem(name="总仓位检查", passed=True, message="非买入或无账户信息，跳过"))
            return
        threshold = self.config["max_total_position_pct"]
        used_pct = 1.0 - (broker.available_cash / broker.total_asset) if broker.total_asset > 0 else 1.0
        ok = used_pct <= threshold
        checks.append(RiskCheckItem(
            name="总仓位检查",
            passed=ok,
            level="block" if not ok else "info",
            message=f"当前仓位 {used_pct:.0%}{'≤' if ok else '>'}{threshold:.0%}",
        ))
        if not ok:
            blocked.append(f"总仓位 {used_pct:.0%} 已超过上限 {threshold:.0%}，不宜继续买入")

    @staticmethod
    def _calc_risk_level(d: TradeDecision, blocked: list, warnings: list) -> str:
        if blocked:
            return "critical"
        if d.risk_score > 0.7 or len(warnings) >= 2:
            return "high"
        if d.risk_score > 0.4 or len(warnings) >= 1:
            return "medium"
        return "low"
