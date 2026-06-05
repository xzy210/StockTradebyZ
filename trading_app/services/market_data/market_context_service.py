"""Market environment context service.

Collects macro market data (major indices, market breadth, sentiment proxies)
for injection into the AI agent's decision prompt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context
from trading_app.services.market_data.quote_service import QuoteData, get_quote_service, to_xt_code

logger = logging.getLogger(__name__)

_INDEX_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "index"

_MAJOR_INDICES = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000300": "沪深300",
    "000905": "中证500",
}


@dataclass
class IndexSnapshot:
    code: str
    name: str
    close: float = 0.0
    change_pct: float = 0.0
    ma5: float = 0.0
    ma20: float = 0.0
    trend: str = ""  # "up" / "down" / "sideways"
    vol_ratio: float = 0.0


@dataclass
class MarketContextSnapshot:
    date: str = ""
    indices: List[IndexSnapshot] = field(default_factory=list)
    breadth_up: int = 0
    breadth_down: int = 0
    breadth_flat: int = 0
    sentiment_tag: str = ""
    summary_text: str = ""
    daily_bar_as_of: str = ""
    realtime_as_of: str = ""
    breadth_as_of: str = ""

    def to_prompt_lines(self) -> List[str]:
        lines = [f"市场环境快照 ({self.date}):"]
        if self.daily_bar_as_of or self.realtime_as_of:
            lines.append(
                f"  数据时点: 日线截至 {self.daily_bar_as_of or '-'} / "
                f"实时截至 {self.realtime_as_of or '-'}"
            )
        for idx in self.indices:
            arrow = "↑" if idx.change_pct > 0 else ("↓" if idx.change_pct < 0 else "→")
            trend_label = {"up": "多头排列", "down": "空头排列"}.get(idx.trend, "震荡")
            lines.append(
                f"  {idx.name}: {idx.close:.2f} ({idx.change_pct:+.2f}% {arrow}) "
                f"MA5={idx.ma5:.2f} MA20={idx.ma20:.2f} [{trend_label}] "
                f"量比={idx.vol_ratio:.2f}"
            )
        if self.breadth_up or self.breadth_down:
            total = self.breadth_up + self.breadth_down + self.breadth_flat
            breadth_suffix = f" / 截至 {self.breadth_as_of}" if self.breadth_as_of else ""
            lines.append(
                f"  涨跌家数: 上涨{self.breadth_up} / 下跌{self.breadth_down} / "
                f"平盘{self.breadth_flat} (共{total}){breadth_suffix}"
            )
        if self.sentiment_tag:
            lines.append(f"  市场情绪: {self.sentiment_tag}")
        if self.summary_text:
            lines.append(f"  综合评估: {self.summary_text}")
        return lines


class MarketContextService:
    """Builds a market-level environment snapshot from local index parquet data."""

    def __init__(self, index_dir: Optional[Path] = None):
        self._index_dir = index_dir or _INDEX_DATA_DIR

    def build_snapshot(
        self,
        run_context: DecisionRunContext | Dict[str, Any] | None = None,
    ) -> MarketContextSnapshot:
        resolved_run_context = self._coerce_run_context(run_context)
        snap = MarketContextSnapshot()
        indices: List[IndexSnapshot] = []
        realtime_quotes = self._fetch_realtime_quotes(resolved_run_context)

        for code, name in _MAJOR_INDICES.items():
            idx_snap = self._load_index_snapshot(
                code,
                name,
                realtime_quote=realtime_quotes.get(code),
            )
            if idx_snap:
                indices.append(idx_snap)

        snap.indices = indices
        if indices:
            snap.daily_bar_as_of = self._get_latest_daily_date(indices)
            snap.breadth_as_of = snap.daily_bar_as_of
            snap.realtime_as_of = self._resolve_realtime_as_of(realtime_quotes)
            snap.date = snap.realtime_as_of[:10] if snap.realtime_as_of else snap.daily_bar_as_of

        self._compute_breadth(snap)
        self._compute_sentiment(snap)
        self._compose_summary(snap)
        return snap

    def _load_index_snapshot(
        self,
        code: str,
        name: str,
        realtime_quote: Optional[QuoteData] = None,
    ) -> Optional[IndexSnapshot]:
        parquet_path = self._index_dir / f"{code}.parquet"
        if not parquet_path.exists():
            return None
        try:
            df = pd.read_parquet(parquet_path)
            if df.empty or len(df) < 20:
                return None
            df = df.sort_values("date").reset_index(drop=True)
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma20"] = df["close"].rolling(20).mean()
            df["vol_ma5"] = df["volume"].rolling(5).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            close = float(last["close"])
            prev_close = float(prev["close"])
            volume_value = float(last["volume"])
            if realtime_quote is not None and float(getattr(realtime_quote, "last_price", 0.0) or 0.0) > 0:
                close = float(realtime_quote.last_price)
                prev_close = float(getattr(realtime_quote, "prev_close", 0.0) or prev_close)
                volume_value = float(getattr(realtime_quote, "volume", 0.0) or 0.0) / 100.0
            change_pct = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
            ma5 = float(last["ma5"]) if pd.notna(last["ma5"]) else close
            ma20 = float(last["ma20"]) if pd.notna(last["ma20"]) else close

            vol_ma5 = float(last["vol_ma5"]) if pd.notna(last["vol_ma5"]) else 1.0
            vol_ratio = volume_value / vol_ma5 if vol_ma5 > 0 else 1.0

            if close > ma5 > ma20:
                trend = "up"
            elif close < ma5 < ma20:
                trend = "down"
            else:
                trend = "sideways"

            return IndexSnapshot(
                code=code, name=name, close=round(close, 2),
                change_pct=round(change_pct, 2),
                ma5=round(ma5, 2), ma20=round(ma20, 2),
                trend=trend, vol_ratio=round(vol_ratio, 2),
            )
        except Exception as exc:
            logger.warning("Failed to load index %s: %s", code, exc)
            return None

    @staticmethod
    def _get_latest_daily_date(indices: List[IndexSnapshot]) -> str:
        try:
            path = _INDEX_DATA_DIR / f"{indices[0].code}.parquet"
            df = pd.read_parquet(path)
            return str(df["date"].max())[:10]
        except Exception:
            return ""

    @staticmethod
    def _resolve_realtime_as_of(quotes: Dict[str, QuoteData]) -> str:
        timestamps = []
        for quote in quotes.values():
            timestamp = getattr(quote, "source_time", None) or getattr(quote, "timestamp", None)
            if timestamp is not None:
                timestamps.append(timestamp)
        if not timestamps:
            return ""
        return max(timestamps).strftime("%Y-%m-%d %H:%M")

    def _compute_breadth(self, snap: MarketContextSnapshot):
        stock_dir = self._index_dir.parent
        up = down = flat = 0
        try:
            parquet_files = list(stock_dir.glob("*.parquet"))
            for pf in parquet_files:
                try:
                    df = pd.read_parquet(pf, columns=["close"])
                    if len(df) < 2:
                        continue
                    last_close = float(df["close"].iloc[-1])
                    prev_close = float(df["close"].iloc[-2])
                    if last_close > prev_close * 1.001:
                        up += 1
                    elif last_close < prev_close * 0.999:
                        down += 1
                    else:
                        flat += 1
                except Exception:
                    continue
        except Exception:
            pass
        snap.breadth_up = up
        snap.breadth_down = down
        snap.breadth_flat = flat

    @staticmethod
    def _coerce_run_context(
        run_context: DecisionRunContext | Dict[str, Any] | None,
    ) -> DecisionRunContext:
        if isinstance(run_context, DecisionRunContext):
            return run_context
        if isinstance(run_context, dict):
            return DecisionRunContext.from_dict(run_context)
        return build_decision_run_context(prefer_realtime=True)

    @staticmethod
    def _fetch_realtime_quotes(run_context: DecisionRunContext) -> Dict[str, QuoteData]:
        if not run_context.should_try_realtime:
            return {}
        try:
            quote_service = get_quote_service()
            xt_codes = [to_xt_code(code, is_index=True) for code in _MAJOR_INDICES.keys()]
            quote_service.refresh_quotes(xt_codes)
            quotes: Dict[str, QuoteData] = {}
            for code in _MAJOR_INDICES.keys():
                quote = quote_service.get_quote(to_xt_code(code, is_index=True))
                if (
                    quote is not None
                    and float(getattr(quote, "last_price", 0.0) or 0.0) > 0
                    and bool(getattr(quote, "is_fresh", False))
                ):
                    quotes[code] = quote
            return quotes
        except Exception as exc:
            logger.debug("获取大盘实时快照失败: %s", exc)
            return {}

    @staticmethod
    def _compute_sentiment(snap: MarketContextSnapshot):
        idx_changes = [i.change_pct for i in snap.indices]
        avg_change = sum(idx_changes) / len(idx_changes) if idx_changes else 0

        total = snap.breadth_up + snap.breadth_down + snap.breadth_flat
        up_ratio = snap.breadth_up / total if total > 0 else 0.5

        up_indices = sum(1 for c in idx_changes if c > 0.3)
        down_indices = sum(1 for c in idx_changes if c < -0.3)

        score = 0
        if avg_change > 1.0:
            score += 2
        elif avg_change > 0.3:
            score += 1
        elif avg_change < -1.0:
            score -= 2
        elif avg_change < -0.3:
            score -= 1

        if up_ratio > 0.65:
            score += 1
        elif up_ratio < 0.35:
            score -= 1

        if up_indices >= 3:
            score += 1
        if down_indices >= 3:
            score -= 1

        vol_ratios = [i.vol_ratio for i in snap.indices if i.vol_ratio > 0]
        avg_vol = sum(vol_ratios) / len(vol_ratios) if vol_ratios else 1.0
        if avg_vol > 1.3:
            score += 1 if avg_change > 0 else -1

        labels = {
            -4: "极度恐慌", -3: "恐慌", -2: "偏空",
            -1: "谨慎", 0: "中性", 1: "偏多",
            2: "乐观", 3: "亢奋", 4: "极度亢奋",
        }
        clamped = max(-4, min(4, score))
        snap.sentiment_tag = labels.get(clamped, "中性")

    @staticmethod
    def _compose_summary(snap: MarketContextSnapshot):
        parts = []
        up_trends = sum(1 for i in snap.indices if i.trend == "up")
        down_trends = sum(1 for i in snap.indices if i.trend == "down")
        total_idx = len(snap.indices)

        if up_trends == total_idx:
            parts.append("所有主要指数均处于多头排列")
        elif down_trends == total_idx:
            parts.append("所有主要指数均处于空头排列")
        elif up_trends > down_trends:
            parts.append(f"{up_trends}/{total_idx} 个指数多头排列，市场偏强")
        elif down_trends > up_trends:
            parts.append(f"{down_trends}/{total_idx} 个指数空头排列，市场偏弱")
        else:
            parts.append("指数多空分化，市场方向不明")

        total = snap.breadth_up + snap.breadth_down + snap.breadth_flat
        if total > 0:
            up_ratio = snap.breadth_up / total
            if up_ratio > 0.7:
                parts.append("个股普涨")
            elif up_ratio < 0.3:
                parts.append("个股普跌")
            elif up_ratio > 0.55:
                parts.append("多数个股上涨")
            elif up_ratio < 0.45:
                parts.append("多数个股下跌")

        vol_ratios = [i.vol_ratio for i in snap.indices if i.vol_ratio > 0]
        avg_vol = sum(vol_ratios) / len(vol_ratios) if vol_ratios else 1.0
        if avg_vol > 1.3:
            parts.append("成交量放大")
        elif avg_vol < 0.7:
            parts.append("成交缩量")

        snap.summary_text = "，".join(parts) if parts else "市场表现平淡"
