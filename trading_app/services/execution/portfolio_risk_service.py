"""Portfolio-level risk checks: industry concentration, correlation, diversification."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from common.agent.agent_context_service import BrokerContext
from trading_app.services.ai.trade_decision_models import (
    RiskCheckItem,
    TradeAction,
    TradeDecision,
)

logger = logging.getLogger(__name__)

_STOCKLIST_PATH = Path(__file__).resolve().parents[3] / "stocklist" / "stocklist.csv"
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

_DEFAULT_CONFIG = {
    "max_single_industry_pct": 0.40,
    "max_single_industry_count": 3,
    "max_correlation_warn": 0.75,
    "min_diversification_count": 3,
}


class _IndustryMapper:
    """Lightweight stock → industry mapper from stocklist.csv."""

    _instance: Optional["_IndustryMapper"] = None

    def __init__(self):
        self._map: Dict[str, str] = {}
        self._loaded = False

    @classmethod
    def get(cls) -> "_IndustryMapper":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if not _STOCKLIST_PATH.exists():
            logger.warning("stocklist.csv not found at %s", _STOCKLIST_PATH)
            return
        try:
            df = pd.read_csv(_STOCKLIST_PATH, dtype=str, usecols=["symbol", "industry"])
            for _, row in df.iterrows():
                code = str(row.get("symbol", "")).strip()
                ind = str(row.get("industry", "")).strip()
                if code and ind:
                    self._map[code] = ind
            logger.info("IndustryMapper loaded %d stock-industry pairs", len(self._map))
        except Exception as exc:
            logger.warning("Failed to load stocklist.csv: %s", exc)

    def lookup(self, code: str) -> str:
        self._ensure_loaded()
        clean = code.split(".")[0].lstrip("0") if "." in code else code
        result = self._map.get(code, "")
        if not result:
            result = self._map.get(code.zfill(6), "")
        return result or "未知"


class PortfolioRiskService:
    """Evaluate portfolio-level risk for a proposed trade."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(_DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self._mapper = _IndustryMapper.get()

    def check(
        self,
        decision: TradeDecision,
        broker: Optional[BrokerContext] = None,
    ) -> List[RiskCheckItem]:
        items: List[RiskCheckItem] = []
        is_buy = decision.action in (TradeAction.BUY.value, TradeAction.ADD.value)
        if not is_buy or not broker or not broker.connected:
            return items

        positions = broker.top_positions or []
        if not positions:
            return items

        items.extend(self._check_industry_concentration(decision, positions, broker))
        items.extend(self._check_correlation(decision, positions))
        items.extend(self._check_diversification(decision, positions))
        return items

    def get_portfolio_industry_summary(
        self, broker: Optional[BrokerContext] = None,
    ) -> Dict[str, Any]:
        """Return a summary of current portfolio industry exposure."""
        if not broker or not broker.connected:
            return {}
        positions = broker.top_positions or []
        total_mv = sum(float(p.get("market_value", 0) or 0) for p in positions)
        if total_mv <= 0:
            return {}

        industry_mv: Dict[str, float] = defaultdict(float)
        industry_codes: Dict[str, List[str]] = defaultdict(list)
        for p in positions:
            code = str(p.get("stock_code", ""))
            mv = float(p.get("market_value", 0) or 0)
            ind = self._mapper.lookup(code)
            industry_mv[ind] += mv
            industry_codes[ind].append(code)

        breakdown = {}
        for ind, mv in sorted(industry_mv.items(), key=lambda x: -x[1]):
            breakdown[ind] = {
                "market_value": round(mv, 2),
                "pct": round(mv / total_mv * 100, 2),
                "count": len(industry_codes[ind]),
                "codes": industry_codes[ind],
            }
        return {"total_market_value": round(total_mv, 2), "industries": breakdown}

    def _check_industry_concentration(
        self, decision: TradeDecision, positions: List[Dict], broker: BrokerContext,
    ) -> List[RiskCheckItem]:
        items: List[RiskCheckItem] = []
        target_ind = self._mapper.lookup(decision.symbol_code)

        total_mv = broker.total_asset if broker.total_asset > 0 else 1.0
        ind_mv: Dict[str, float] = defaultdict(float)
        ind_count: Dict[str, int] = Counter()

        for p in positions:
            code = str(p.get("stock_code", ""))
            mv = float(p.get("market_value", 0) or 0)
            ind = self._mapper.lookup(code)
            ind_mv[ind] += mv
            ind_count[ind] += 1

        projected_mv = float(decision.current_price * decision.position_pct * total_mv) if decision.current_price > 0 else 0
        ind_mv[target_ind] += projected_mv
        ind_count[target_ind] += 1

        pct = ind_mv[target_ind] / total_mv if total_mv > 0 else 0
        max_pct = self.config["max_single_industry_pct"]
        pct_ok = pct <= max_pct
        items.append(RiskCheckItem(
            name="行业集中度",
            passed=pct_ok,
            level="warn" if not pct_ok else "info",
            message=(
                f"买入后 [{target_ind}] 行业占比 {pct:.0%}"
                f"{'≤' if pct_ok else '>'}{max_pct:.0%}"
            ),
        ))

        max_count = self.config["max_single_industry_count"]
        cnt = ind_count[target_ind]
        cnt_ok = cnt <= max_count
        items.append(RiskCheckItem(
            name="同行业持仓数",
            passed=cnt_ok,
            level="warn" if not cnt_ok else "info",
            message=(
                f"[{target_ind}] 行业将有 {cnt} 只持仓"
                f"{'≤' if cnt_ok else '>'}{max_count} 只上限"
            ),
        ))
        return items

    def _check_correlation(
        self, decision: TradeDecision, positions: List[Dict],
    ) -> List[RiskCheckItem]:
        items: List[RiskCheckItem] = []
        target_code = decision.symbol_code

        pos_codes = [str(p.get("stock_code", "")) for p in positions if str(p.get("stock_code", ""))]
        if not pos_codes:
            return items

        all_codes = [target_code] + pos_codes[:10]
        returns_map: Dict[str, pd.Series] = {}
        for code in all_codes:
            pq = _DATA_DIR / f"{code}.parquet"
            if not pq.exists():
                continue
            try:
                df = pd.read_parquet(pq, columns=["close"])
                if len(df) < 30:
                    continue
                rets = df["close"].pct_change().dropna().tail(60)
                returns_map[code] = rets.reset_index(drop=True)
            except Exception:
                continue

        if target_code not in returns_map:
            return items

        target_ret = returns_map[target_code]
        high_corr_pairs: List[Tuple[str, float]] = []
        threshold = self.config["max_correlation_warn"]

        for code in pos_codes:
            if code not in returns_map:
                continue
            other = returns_map[code]
            min_len = min(len(target_ret), len(other))
            if min_len < 20:
                continue
            corr = float(np.corrcoef(
                target_ret.values[-min_len:], other.values[-min_len:]
            )[0, 1])
            if corr > threshold:
                high_corr_pairs.append((code, round(corr, 2)))

        if high_corr_pairs:
            pair_str = ", ".join(f"{c}(r={v})" for c, v in high_corr_pairs[:3])
            items.append(RiskCheckItem(
                name="持仓相关性",
                passed=False,
                level="warn",
                message=f"与现有持仓高相关(>{threshold}): {pair_str}",
            ))
        else:
            items.append(RiskCheckItem(
                name="持仓相关性",
                passed=True,
                level="info",
                message="与现有持仓相关性正常",
            ))
        return items

    def _check_diversification(
        self, decision: TradeDecision, positions: List[Dict],
    ) -> List[RiskCheckItem]:
        items: List[RiskCheckItem] = []
        industries: set[str] = set()
        for p in positions:
            code = str(p.get("stock_code", ""))
            industries.add(self._mapper.lookup(code))
        industries.add(self._mapper.lookup(decision.symbol_code))
        industries.discard("未知")

        min_count = self.config["min_diversification_count"]
        ok = len(industries) >= min_count
        items.append(RiskCheckItem(
            name="行业分散度",
            passed=ok,
            level="info" if ok else "warn",
            message=f"覆盖 {len(industries)} 个行业{'≥' if ok else '<'}{min_count} (建议最低)",
        ))
        return items
