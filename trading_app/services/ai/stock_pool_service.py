from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context

from common.data_portal import get_data_portal
from common.io_utils import atomic_write_json

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stock_pool_config.json"
_SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "daily_candidate_pool.json"
_DATA_DIR = _PROJECT_ROOT / "data"
_INDEX_DIR = _DATA_DIR / "index"
_STOCKLIST_DIR = _PROJECT_ROOT / "stocklist"
_MASTER_STOCKLIST_PATH = _STOCKLIST_DIR / "stocklist.csv"


@dataclass
class StockPoolConfig:
    enabled: bool = True
    pool_name: str = "中证500趋势精选池"
    universe_file: str = "中证500成分股_股票列表.csv"
    benchmark_index_code: str = "000905"
    min_listing_days: int = 250
    min_price: float = 5.0
    min_avg_turnover: float = 200000000.0
    exclude_st: bool = True
    exclude_chinext: bool = True
    exclude_star_market: bool = True
    exclude_bse: bool = True
    max_candidates: int = 30
    ai_review_limit: int = 10
    factor_weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockPoolConfig":
        payload = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StockPoolService:
    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        snapshot_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        stocklist_dir: Optional[Path] = None,
    ) -> None:
        self.config_path = config_path or _CONFIG_PATH
        self.snapshot_path = snapshot_path or _SNAPSHOT_PATH
        self.data_dir = data_dir or _DATA_DIR
        self.stocklist_dir = stocklist_dir or _STOCKLIST_DIR
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()
        self._master_stocklist = self._load_master_stocklist()

    def _load_config(self) -> StockPoolConfig:
        if not self.config_path.exists():
            cfg = StockPoolConfig()
            atomic_write_json(self.config_path, cfg.to_dict())
            return cfg
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return StockPoolConfig.from_dict(json.load(f) or {})
        except Exception as exc:
            logger.warning("读取股票池配置失败，使用默认配置: %s", exc)
            return StockPoolConfig()

    def get_config(self) -> StockPoolConfig:
        self._config = self._load_config()
        return self._config

    def get_snapshot(self) -> Dict[str, Any]:
        if not self.snapshot_path.exists():
            return {}
        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as exc:
            logger.warning("读取候选池快照失败: %s", exc)
            return {}

    def refresh_candidate_pool(
        self,
        *,
        force: bool = False,
        run_context: DecisionRunContext | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        cfg = self.get_config()
        resolved_run_context = self._coerce_run_context(run_context)
        today = resolved_run_context.trading_day or datetime.now().strftime("%Y-%m-%d")
        existing = self.get_snapshot()
        if (
            not force
            and existing.get("trade_date") == today
            and existing.get("data_as_of") == resolved_run_context.daily_bar_as_of
            and existing.get("items")
        ):
            return existing

        universe = self._load_universe(cfg.universe_file)
        if universe.empty:
            snapshot = {
                "trade_date": today,
                "snapshot_date": today,
                "data_as_of": resolved_run_context.daily_bar_as_of,
                "realtime_overlay_as_of": resolved_run_context.realtime_as_of,
                "pool_name": cfg.pool_name,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "universe_size": 0,
                "filtered_count": 0,
                "items": [],
                "error": f"股票池文件为空: {cfg.universe_file}",
            }
            atomic_write_json(self.snapshot_path, snapshot)
            return snapshot

        benchmark_metrics = self._load_benchmark_metrics(cfg.benchmark_index_code)
        candidates: List[Dict[str, Any]] = []
        filtered_count = 0
        total = len(universe)

        for _, row in universe.iterrows():
            code = str(row.get("ts_code", "") or row.get("code", "") or "").strip().upper()
            plain_code = code.split(".", 1)[0] if "." in code else code
            if not plain_code:
                continue
            name = str(row.get("name", "") or self._master_name(plain_code) or plain_code).strip()
            industry = str(row.get("industry", "") or self._master_industry(plain_code) or "").strip()
            candidate = self._score_symbol(
                code=plain_code,
                ts_code=code if "." in code else self._to_ts_code(plain_code),
                name=name,
                industry=industry,
                cfg=cfg,
                benchmark_metrics=benchmark_metrics,
            )
            if candidate is None:
                continue
            filtered_count += 1
            candidates.append(candidate)

        candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        selected = candidates[: max(int(cfg.max_candidates or 0), 1)]
        for idx, item in enumerate(selected, start=1):
            item["rank"] = idx

        snapshot = {
            "trade_date": today,
            "snapshot_date": today,
            "data_as_of": resolved_run_context.daily_bar_as_of,
            "realtime_overlay_as_of": resolved_run_context.realtime_as_of,
            "pool_name": cfg.pool_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "universe_file": cfg.universe_file,
            "universe_size": total,
            "filtered_count": filtered_count,
            "benchmark_index_code": cfg.benchmark_index_code,
            "items": selected,
        }
        atomic_write_json(self.snapshot_path, snapshot)
        return snapshot

    def get_candidate_items(
        self,
        *,
        refresh: bool = False,
        limit: Optional[int] = None,
        run_context: DecisionRunContext | Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        snapshot = self.refresh_candidate_pool(force=refresh, run_context=run_context)
        items = list(snapshot.get("items", []) or [])
        if limit is None or limit <= 0:
            limit = int(self.get_config().ai_review_limit or 10)
        return items[:limit]

    def get_candidate_codes(
        self,
        *,
        refresh: bool = False,
        limit: Optional[int] = None,
        run_context: DecisionRunContext | Dict[str, Any] | None = None,
    ) -> List[str]:
        return [
            str(item.get("code", ""))
            for item in self.get_candidate_items(refresh=refresh, limit=limit, run_context=run_context)
            if item.get("code")
        ]

    def _load_master_stocklist(self) -> pd.DataFrame:
        if not _MASTER_STOCKLIST_PATH.exists():
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "industry"])
        try:
            df = pd.read_csv(_MASTER_STOCKLIST_PATH)
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            if "ts_code" in df.columns:
                df["ts_code"] = df["ts_code"].astype(str).str.upper()
            return df
        except Exception as exc:
            logger.warning("读取主 stocklist 失败: %s", exc)
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "industry"])

    def _load_universe(self, universe_file: str) -> pd.DataFrame:
        path = self.stocklist_dir / universe_file
        if not path.exists():
            logger.warning("股票池文件不存在: %s", path)
            return pd.DataFrame(columns=["ts_code", "name", "industry"])
        try:
            df = pd.read_csv(path)
            if "ts_code" not in df.columns:
                raise ValueError("missing ts_code header")
        except Exception:
            try:
                df = pd.read_csv(path, header=None, names=["ts_code", "name"])
            except Exception as exc:
                logger.warning("读取股票池文件失败: %s", exc)
                return pd.DataFrame(columns=["ts_code", "name", "industry"])

        df["ts_code"] = df["ts_code"].astype(str).str.upper().str.strip()
        df["symbol"] = df["ts_code"].str.split(".").str[0].str.zfill(6)
        if "name" not in df.columns:
            df["name"] = df["symbol"].map(lambda code: self._master_name(code) or code)
        df["industry"] = df.get("industry", "").astype(str) if "industry" in df.columns else ""
        if self._master_stocklist is not None and not self._master_stocklist.empty:
            master = self._master_stocklist[["symbol", "name", "industry"]].drop_duplicates("symbol")
            df = df.merge(master, on="symbol", how="left", suffixes=("", "_master"))
            df["name"] = df["name"].where(df["name"].astype(str).str.len() > 0, df["name_master"])
            df["industry"] = df["industry"].where(df["industry"].astype(str).str.len() > 0, df["industry_master"])
            drop_cols = [c for c in ["name_master", "industry_master"] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
        return df[["ts_code", "symbol", "name", "industry"]].drop_duplicates("symbol").reset_index(drop=True)

    def _load_benchmark_metrics(self, index_code: str) -> Dict[str, float]:
        path = _INDEX_DIR / f"{index_code}.parquet"
        if not path.exists():
            return {}
        try:
            df = pd.read_parquet(path, columns=["date", "close"])
            frame = df.sort_values("date").reset_index(drop=True)
            if len(frame) < 61:
                return {}
            close = frame["close"].astype(float)
            return {
                "return_20": float(close.iloc[-1] / close.iloc[-21] - 1.0),
                "return_60": float(close.iloc[-1] / close.iloc[-61] - 1.0),
            }
        except Exception as exc:
            logger.warning("读取基准指数失败: %s", exc)
            return {}

    def _score_symbol(
        self,
        *,
        code: str,
        ts_code: str,
        name: str,
        industry: str,
        cfg: StockPoolConfig,
        benchmark_metrics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        if cfg.exclude_st and "ST" in name.upper():
            return None
        if cfg.exclude_chinext and self._is_chinext(code):
            return None
        if cfg.exclude_star_market and self._is_star_market(code):
            return None
        if cfg.exclude_bse and self._is_bse(code):
            return None
        df = get_data_portal().get_daily_bars(
            code,
            data_dir=self.data_dir,
            asset_type="stock",
            use_cache=False,
        )
        if df is None or df.empty:
            return None
        frame = df.sort_values("date").reset_index(drop=True).copy()
        if len(frame) < max(int(cfg.min_listing_days or 0), 70):
            return None
        close = frame["close"].astype(float)
        volume = frame["volume"].astype(float).fillna(0.0)
        latest_close = float(close.iloc[-1])
        if latest_close < float(cfg.min_price or 0.0):
            return None

        # 本地日线 volume 口径为“手”，这里换算成股后再近似成交额。
        recent_turnover = float((close.tail(20) * volume.tail(20) * 100.0).mean())
        if recent_turnover < float(cfg.min_avg_turnover or 0.0):
            return None

        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        ret60 = float(close.iloc[-1] / close.iloc[-61] - 1.0)
        ma60 = float(close.tail(60).mean())
        trend_strength = float(close.iloc[-1] / ma60 - 1.0) if ma60 > 0 else 0.0
        recent20 = close.tail(20).reset_index(drop=True)
        rolling_max = recent20.cummax()
        drawdown_20 = float((recent20 / rolling_max - 1.0).min())
        vol20 = float(volume.tail(20).mean())
        vol5 = float(volume.tail(5).mean())
        volume_ratio = float(vol5 / vol20) if vol20 > 0 else 0.0
        excess_return_20 = ret20 - float(benchmark_metrics.get("return_20", 0.0) or 0.0)

        weights = dict(cfg.factor_weights or {})
        score = 100.0 * (
            float(weights.get("momentum_20", 0.0)) * ret20
            + float(weights.get("momentum_60", 0.0)) * ret60
            + float(weights.get("trend_strength", 0.0)) * trend_strength
            + float(weights.get("excess_return_20", 0.0)) * excess_return_20
            + float(weights.get("volume_ratio", 0.0)) * max(min(volume_ratio - 1.0, 1.5), -0.5)
            - float(weights.get("drawdown_20", 0.0)) * abs(min(drawdown_20, 0.0))
        )

        reasons: List[str] = []
        if ret20 > 0.08:
            reasons.append(f"20日涨幅 {ret20:.1%}")
        if ret60 > 0.15:
            reasons.append(f"60日涨幅 {ret60:.1%}")
        if trend_strength > 0:
            reasons.append(f"站上60日线 {trend_strength:.1%}")
        if excess_return_20 > 0:
            reasons.append(f"相对中证500超额 {excess_return_20:.1%}")
        if volume_ratio > 1.1:
            reasons.append(f"量能放大 {volume_ratio:.2f}x")
        if not reasons:
            reasons.append("量化初筛通过")

        return {
            "code": code,
            "ts_code": ts_code,
            "name": name or code,
            "industry": industry or "",
            "score": round(score, 4),
            "latest_close": round(latest_close, 3),
            "avg_turnover_20": round(recent_turnover, 2),
            "factors": {
                "momentum_20": round(ret20, 6),
                "momentum_60": round(ret60, 6),
                "trend_strength": round(trend_strength, 6),
                "excess_return_20": round(excess_return_20, 6),
                "volume_ratio": round(volume_ratio, 6),
                "drawdown_20": round(drawdown_20, 6),
            },
            "reasons": reasons[:4],
            "latest_date": str(frame["date"].iloc[-1])[:10],
        }

    def _master_name(self, code: str) -> str:
        if self._master_stocklist.empty:
            return ""
        matched = self._master_stocklist.loc[self._master_stocklist["symbol"] == str(code).zfill(6), "name"]
        return str(matched.iloc[0]) if not matched.empty else ""

    def _master_industry(self, code: str) -> str:
        if self._master_stocklist.empty or "industry" not in self._master_stocklist.columns:
            return ""
        matched = self._master_stocklist.loc[self._master_stocklist["symbol"] == str(code).zfill(6), "industry"]
        return str(matched.iloc[0]) if not matched.empty else ""

    @staticmethod
    def _is_chinext(code: str) -> bool:
        plain = str(code).zfill(6)
        return plain.startswith(("300", "301"))

    @staticmethod
    def _is_star_market(code: str) -> bool:
        plain = str(code).zfill(6)
        return plain.startswith(("688", "689"))

    @staticmethod
    def _is_bse(code: str) -> bool:
        plain = str(code).zfill(6)
        return plain.startswith(("4", "8"))

    @staticmethod
    def _to_ts_code(code: str) -> str:
        plain = str(code).zfill(6)
        suffix = ".SH" if plain.startswith(("5", "6", "9")) else ".SZ"
        return f"{plain}{suffix}"

    @staticmethod
    def _coerce_run_context(
        run_context: DecisionRunContext | Dict[str, Any] | None,
    ) -> DecisionRunContext:
        if isinstance(run_context, DecisionRunContext):
            return run_context
        if isinstance(run_context, dict):
            return DecisionRunContext.from_dict(run_context)
        return build_decision_run_context(prefer_realtime=True)


_stock_pool_service: Optional[StockPoolService] = None


def get_stock_pool_service() -> StockPoolService:
    global _stock_pool_service
    if _stock_pool_service is None:
        _stock_pool_service = StockPoolService()
    return _stock_pool_service
