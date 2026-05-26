"""Lightweight factor panel construction and native factor analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from common.data_portal import get_data_portal


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class FactorPanelBuildResult:
    panel: pd.DataFrame
    success_count: int
    fail_count: int
    missing_codes: list[str]


class FactorPanelBuilder:
    """Build a date-code factor panel from per-symbol factor CSV files."""

    def __init__(self, data_dir: str | Path, factors_dir: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.factors_dir = Path(factors_dir) if factors_dir else self.data_dir / "factors"

    @staticmethod
    def _plain_code(code: str) -> str:
        value = str(code or "").strip()
        return value.split(".", 1)[0] if "." in value else value

    def build_panel(
        self,
        stock_codes: Iterable[str],
        factor_names: Iterable[str],
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> FactorPanelBuildResult:
        factor_names = [str(name) for name in factor_names if str(name or "").strip()]
        frames: list[pd.DataFrame] = []
        missing_codes: list[str] = []
        stock_codes = [self._plain_code(code) for code in stock_codes]
        total = len(stock_codes)

        for index, code in enumerate(stock_codes, start=1):
            if progress_callback:
                progress_callback(index, total, code)

            factor_file = self.factors_dir / f"{code}.csv"
            if not factor_file.exists():
                missing_codes.append(code)
                continue

            try:
                factor_df = pd.read_csv(factor_file)
                if factor_df.empty or "date" not in factor_df.columns:
                    missing_codes.append(code)
                    continue
                factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce")
                factor_df = factor_df.dropna(subset=["date"])
                available_factors = [name for name in factor_names if name in factor_df.columns]
                if not available_factors:
                    missing_codes.append(code)
                    continue

                price_df = get_data_portal().get_daily_bars(
                    code,
                    data_dir=self.data_dir,
                    start=start_date,
                    end=end_date,
                    asset_type="stock",
                    use_cache=False,
                )
                if price_df is None or price_df.empty or "close" not in price_df.columns:
                    missing_codes.append(code)
                    continue

                price_df = price_df[["date", "close"]].copy()
                price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
                price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
                price_df = price_df.dropna(subset=["date", "close"])

                if start_date:
                    factor_df = factor_df[factor_df["date"] >= pd.to_datetime(start_date)]
                if end_date:
                    factor_df = factor_df[factor_df["date"] <= pd.to_datetime(end_date)]

                merged = pd.merge(
                    factor_df[["date"] + available_factors],
                    price_df,
                    on="date",
                    how="inner",
                )
                if merged.empty:
                    missing_codes.append(code)
                    continue
                merged["code"] = code
                frames.append(merged[["date", "code", "close"] + available_factors])
            except Exception:
                missing_codes.append(code)

        if not frames:
            return FactorPanelBuildResult(
                panel=pd.DataFrame(columns=["date", "code", "close"] + factor_names),
                success_count=0,
                fail_count=total,
                missing_codes=missing_codes,
            )

        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
        return FactorPanelBuildResult(
            panel=panel,
            success_count=len(frames),
            fail_count=total - len(frames),
            missing_codes=missing_codes,
        )


class FactorAnalysisService:
    """Native Alphalens-style factor evaluation for one factor."""

    def analyze(
        self,
        panel: pd.DataFrame,
        factor_name: str,
        *,
        forward_period: int = 5,
        quantiles: int = 5,
    ) -> dict:
        if panel is None or panel.empty:
            raise ValueError("因子面板为空")
        if factor_name not in panel.columns:
            raise ValueError(f"因子列不存在: {factor_name}")
        if "date" not in panel.columns or "code" not in panel.columns or "close" not in panel.columns:
            raise ValueError("因子面板必须包含 date, code, close 列")

        data = panel[["date", "code", "close", factor_name]].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data[factor_name] = pd.to_numeric(data[factor_name], errors="coerce")
        data = data.dropna(subset=["date", "code", "close", factor_name])
        data = data.sort_values(["code", "date"]).reset_index(drop=True)
        data["forward_return"] = data.groupby("code")["close"].shift(-int(forward_period)) / data["close"] - 1.0
        data = data.dropna(subset=["forward_return"]).copy()
        if data.empty:
            raise ValueError("无法计算未来收益，请检查日期范围或 forward_period")

        data["quantile"] = data.groupby("date", group_keys=False)[factor_name].apply(
            lambda s: self._assign_quantiles(s, quantiles)
        )
        data = data.dropna(subset=["quantile"]).copy()
        data["quantile"] = data["quantile"].astype(int)

        ic_by_date = self._compute_ic_by_date(data, factor_name)
        quantile_returns = self._compute_quantile_returns(data)
        long_short_by_date = self._compute_long_short_by_date(data, quantiles)

        summary = self._build_summary(data, ic_by_date, quantile_returns, long_short_by_date)
        return {
            "factor_name": factor_name,
            "forward_period": int(forward_period),
            "quantiles": int(quantiles),
            "summary": summary,
            "ic_by_date": ic_by_date,
            "quantile_returns": quantile_returns,
            "long_short_by_date": long_short_by_date,
            "analyzed_rows": int(len(data)),
        }

    @staticmethod
    def _assign_quantiles(series: pd.Series, quantiles: int) -> pd.Series:
        valid = series.dropna()
        result = pd.Series(np.nan, index=series.index)
        if len(valid) < max(2, quantiles):
            return result
        try:
            result.loc[valid.index] = pd.qcut(valid.rank(method="first"), quantiles, labels=False) + 1
        except ValueError:
            return result
        return result

    @staticmethod
    def _compute_ic_by_date(data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        rows = []
        for date, group in data.groupby("date"):
            if len(group) < 2:
                continue
            ic = group[factor_name].corr(group["forward_return"], method="pearson")
            rank_ic = group[factor_name].corr(group["forward_return"], method="spearman")
            rows.append({"date": date, "ic": ic, "rank_ic": rank_ic, "count": len(group)})
        return pd.DataFrame(rows)

    @staticmethod
    def _compute_quantile_returns(data: pd.DataFrame) -> pd.DataFrame:
        grouped = data.groupby(["date", "quantile"])["forward_return"].mean().reset_index()
        return grouped.groupby("quantile")["forward_return"].agg(["mean", "std", "count"]).reset_index()

    @staticmethod
    def _compute_long_short_by_date(data: pd.DataFrame, quantiles: int) -> pd.DataFrame:
        daily_quantile = data.groupby(["date", "quantile"])["forward_return"].mean().unstack()
        if 1 not in daily_quantile.columns or quantiles not in daily_quantile.columns:
            return pd.DataFrame(columns=["date", "long_return", "short_return", "long_short_return"])
        result = pd.DataFrame({
            "date": daily_quantile.index,
            "long_return": daily_quantile[quantiles].values,
            "short_return": daily_quantile[1].values,
        })
        result["long_short_return"] = result["long_return"] - result["short_return"]
        return result.dropna().reset_index(drop=True)

    @staticmethod
    def _build_summary(
        data: pd.DataFrame,
        ic_by_date: pd.DataFrame,
        quantile_returns: pd.DataFrame,
        long_short_by_date: pd.DataFrame,
    ) -> dict:
        ic_mean = float(ic_by_date["ic"].mean()) if not ic_by_date.empty else np.nan
        ic_std = float(ic_by_date["ic"].std()) if not ic_by_date.empty else np.nan
        rank_ic_mean = float(ic_by_date["rank_ic"].mean()) if not ic_by_date.empty else np.nan
        rank_ic_std = float(ic_by_date["rank_ic"].std()) if not ic_by_date.empty else np.nan
        long_short_mean = (
            float(long_short_by_date["long_short_return"].mean())
            if not long_short_by_date.empty else np.nan
        )
        long_short_win_rate = (
            float((long_short_by_date["long_short_return"] > 0).mean())
            if not long_short_by_date.empty else np.nan
        )
        return {
            "rows": int(len(data)),
            "dates": int(data["date"].nunique()),
            "symbols": int(data["code"].nunique()),
            "coverage": float(data["forward_return"].notna().mean()),
            "ic_mean": ic_mean,
            "ic_ir": float(ic_mean / ic_std) if ic_std and not np.isnan(ic_std) else np.nan,
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_ir": float(rank_ic_mean / rank_ic_std) if rank_ic_std and not np.isnan(rank_ic_std) else np.nan,
            "long_short_mean": long_short_mean,
            "long_short_win_rate": long_short_win_rate,
            "quantile_return_spread": (
                float(quantile_returns["mean"].iloc[-1] - quantile_returns["mean"].iloc[0])
                if len(quantile_returns) >= 2 else np.nan
            ),
        }
