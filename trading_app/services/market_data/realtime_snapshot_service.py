from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from common.data_portal import get_data_portal

from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context
from trading_app.services.market_data.quote_service import get_quote_service, to_xt_code

logger = logging.getLogger(__name__)


def load_symbol_view(
    code: str,
    *,
    asset_type: str,
    data_dir: str,
    run_context: DecisionRunContext | None = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    run_ctx = run_context or build_decision_run_context(prefer_realtime=False)
    plain_code = str(code or "").split(".", 1)[0].strip().upper()
    daily_df = get_data_portal().get_daily_bars(
        plain_code,
        data_dir=data_dir,
        asset_type="etf" if asset_type == "ETF" else "stock",
        use_cache=bool(use_cache and not run_ctx.prefer_realtime),
    )
    metadata: Dict[str, Any] = {
        "daily_bar_as_of": "",
        "realtime_as_of": "",
        "has_realtime": False,
        "realtime_price": 0.0,
        "realtime_change_pct": 0.0,
    }
    if daily_df is None or daily_df.empty:
        return daily_df, metadata

    frame = daily_df.sort_values("date").reset_index(drop=True).copy()
    last_daily_date = pd.Timestamp(frame["date"].iloc[-1]).strftime("%Y-%m-%d")
    metadata["daily_bar_as_of"] = last_daily_date
    if not run_ctx.should_try_realtime:
        return frame, metadata

    overlay = _build_realtime_overlay(plain_code, frame, run_ctx)
    if not overlay:
        return frame, metadata

    view_df = _apply_realtime_overlay(frame, overlay)
    metadata.update({
        "daily_bar_as_of": last_daily_date,
        "realtime_as_of": overlay.get("realtime_as_of", ""),
        "has_realtime": True,
        "realtime_price": float(overlay.get("close", 0.0) or 0.0),
        "realtime_change_pct": float(overlay.get("change_pct", 0.0) or 0.0),
    })
    return view_df, metadata


def _build_realtime_overlay(
    code: str,
    daily_df: pd.DataFrame,
    run_context: DecisionRunContext,
) -> Dict[str, Any]:
    try:
        quote_service = get_quote_service()
        xt_code = to_xt_code(code)
        quote_service.refresh_quotes([xt_code])
        quote = quote_service.get_quote(xt_code)
        if (
            quote is None
            or float(getattr(quote, "last_price", 0.0) or 0.0) <= 0
            or not bool(getattr(quote, "is_fresh", False))
        ):
            return {}

        quote_timestamp = getattr(quote, "source_time", None) or getattr(quote, "timestamp", None)
        open_price = float(getattr(quote, "open_price", 0.0) or 0.0)
        high_price = float(getattr(quote, "high_price", 0.0) or 0.0)
        low_price = float(getattr(quote, "low_price", 0.0) or 0.0)
        last_price = float(getattr(quote, "last_price", 0.0) or 0.0)
        prev_close = float(getattr(quote, "prev_close", 0.0) or 0.0)
        volume = float(getattr(quote, "volume", 0.0) or 0.0) / 100.0
        realtime_as_of = quote_timestamp.strftime("%Y-%m-%d %H:%M") if quote_timestamp else ""
        if prev_close <= 0 and len(daily_df) >= 1:
            prev_close = float(pd.to_numeric(daily_df["close"], errors="coerce").iloc[-1] or 0.0)
        if last_price <= 0:
            return {}
        high_price = max(high_price, open_price, last_price)
        low_price = min(value for value in [low_price, open_price, last_price] if value > 0)
        change_pct = (last_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        return {
            "date": run_context.trading_day,
            "open": open_price or last_price,
            "high": high_price,
            "low": low_price or last_price,
            "close": last_price,
            "volume": max(volume, 0.0),
            "realtime_as_of": realtime_as_of,
            "change_pct": change_pct,
        }
    except Exception as exc:
        logger.debug("%s 构建实时叠加行情失败: %s", code, exc)
        return {}


def _apply_realtime_overlay(daily_df: pd.DataFrame, overlay: Dict[str, Any]) -> pd.DataFrame:
    trade_date = pd.to_datetime(str(overlay.get("date", "") or ""), errors="coerce")
    if pd.isna(trade_date):
        return daily_df
    frame = daily_df.copy()
    frame = frame[pd.to_datetime(frame["date"], errors="coerce") != trade_date].copy()
    row = pd.DataFrame([{
        "date": trade_date,
        "open": float(overlay.get("open", 0.0) or 0.0),
        "high": float(overlay.get("high", 0.0) or 0.0),
        "low": float(overlay.get("low", 0.0) or 0.0),
        "close": float(overlay.get("close", 0.0) or 0.0),
        "volume": float(overlay.get("volume", 0.0) or 0.0),
    }])
    frame = pd.concat([frame, row], ignore_index=True, sort=False)
    return frame.sort_values("date").reset_index(drop=True)
