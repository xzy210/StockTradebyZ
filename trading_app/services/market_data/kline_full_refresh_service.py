"""End-of-day full K-line refresh service.

Synchronously refreshes all stock, ETF, and index K-line data using
xtquant/miniQMT with ``full_update=True`` so that forward-adjusted
(前复权) prices are recalculated from scratch.

Designed to be called from a background QThread (e.g. _EndOfDayWorker),
NOT from the main/GUI thread.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from common.daily_update_policy import get_daily_update_policy
from common.data_portal import get_data_portal
from common.kline_update_engine import (
    check_xtquant_ready,
    run_batched_updates,
    run_xtquant_daily_history_precheck,
    update_rotation_etf_pool,
)
from trading_app.services.market_data.data_update_result import DataUpdateResult

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"
_STOCKLIST_PATH = _PROJECT_ROOT / "stocklist" / "stocklist.csv"
_ETF_CONFIG_PATH = _PROJECT_ROOT / "trading_app" / "config" / "etf_list.json"
_ROTATION_DATA_DIR = _PROJECT_ROOT / "live_rotation" / "data"
_STOCK_FRESHNESS_PROBE_CODES = ("000001", "600000", "000333", "300750", "600519")

StatusCallback = Callable[[str], None]


def _noop_status(_msg: str) -> None:
    pass


class KlineFullRefreshService:
    """Coordinate a full (前复权) K-line refresh for stocks, ETFs and indices.

    All public methods are **blocking** – call them from a worker thread.
    """

    def __init__(
        self,
        *,
        data_dir: Path = _DATA_DIR,
        stocklist_path: Path = _STOCKLIST_PATH,
        etf_config_path: Path = _ETF_CONFIG_PATH,
        rotation_data_dir: Path = _ROTATION_DATA_DIR,
        rotation_etf_pool: Optional[List[str]] = None,
        max_workers: int = 4,
        start_date: str = "20190101",
    ) -> None:
        self.data_dir = data_dir
        self.stocklist_path = stocklist_path
        self.etf_config_path = etf_config_path
        self.rotation_data_dir = rotation_data_dir
        self.rotation_etf_pool = list(rotation_etf_pool or [])
        self.max_workers = max_workers
        self.start_date = start_date

    def run_full_refresh(
        self,
        status_cb: Optional[StatusCallback] = None,
    ) -> Tuple[bool, str]:
        """Run full-overwrite refresh for stocks + ETFs + indices.

        Returns ``(overall_success, summary_message)``.
        """
        return self.run_full_refresh_result(status_cb).to_legacy_tuple()

    def run_full_refresh_result(
        self,
        status_cb: Optional[StatusCallback] = None,
    ) -> DataUpdateResult:
        """Run full-overwrite refresh and return a structured update result."""
        cb = status_cb or _noop_status
        t0 = time.time()
        results: Dict[str, Tuple[bool, str]] = {}

        ok, msg = self._check_xtquant(cb)
        if not ok:
            return DataUpdateResult(ok=False, message=msg, details={"xtquant": msg})

        ok, msg = self._check_xtquant_daily_history(cb)
        if not ok:
            return DataUpdateResult(ok=False, message=msg, details={"daily_history": msg})

        results["stock"] = self._refresh_stocks(cb)
        results["etf"] = self._refresh_etfs(cb)
        results["index"] = self._refresh_indices(cb)
        results["rotation_etf"] = self._refresh_rotation_etfs(cb)

        elapsed = time.time() - t0
        failed = [k for k, (ok, _) in results.items() if not ok]
        parts = [f"{k}: {msg}" for k, (_, msg) in results.items()]
        summary = " | ".join(parts) + f" ({elapsed:.0f}s)"
        details = {k: msg for k, (_, msg) in results.items()}
        failed_codes = list(failed)
        if failed:
            cb(f"⚠ 部分数据刷新失败: {', '.join(failed)}")
            return DataUpdateResult(
                ok=False,
                failed_codes=failed_codes,
                message=summary,
                details=details,
            )
        cache_result = get_data_portal().refresh_loaded_caches(data_dir=self.data_dir)
        if cache_result.refreshed:
            summary = f"{summary} | cache: 股票{cache_result.stock_count} ETF{cache_result.etf_count}"
        cb(f"✅ 全量K线刷新完成 ({elapsed:.0f}s)")
        return DataUpdateResult(
            ok=True,
            updated_stocks=self._extract_success_count(results.get("stock", (True, ""))[1]),
            updated_etfs=self._extract_success_count(results.get("etf", (True, ""))[1]),
            updated_indices=self._extract_success_count(results.get("index", (True, ""))[1]),
            updated_rotation_etfs=self._extract_success_count(results.get("rotation_etf", (True, ""))[1]),
            cache_refreshed=cache_result.refreshed,
            cache_refreshed_stocks=cache_result.stock_count,
            cache_refreshed_etfs=cache_result.etf_count,
            message=summary,
            details=details,
        )

    @staticmethod
    def _extract_success_count(message: str) -> int:
        import re

        match = re.search(r"(\d+)\s*/\s*(\d+)", message or "")
        if match:
            return int(match.group(1))
        match = re.search(r"成功\s*(\d+)", message or "")
        if match:
            return int(match.group(1))
        return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_xtquant(cb: StatusCallback) -> Tuple[bool, str]:
        cb("检查 miniQMT 连接...")
        ok, msg = check_xtquant_ready()
        if ok:
            cb("miniQMT 连接正常")
            return True, "ok"
        return False, msg

    @staticmethod
    def _check_xtquant_daily_history(cb: StatusCallback) -> Tuple[bool, str]:
        cb("验证 miniQMT 是否可完整获取最近10天历史K线...")
        ok, msg = run_xtquant_daily_history_precheck(
            action_hint="请确认 miniQMT 已正常运行并登录后再执行全量K线刷新。",
            require_recent_daily_history=True,
            recent_calendar_days=10,
        )
        if ok:
            cb(msg)
        return ok, msg

    def _validate_stock_outputs(self, codes: List[str]) -> List[str]:
        code_set = {str(code).strip().upper().split(".", 1)[0] for code in codes}
        check_codes = [code for code in _STOCK_FRESHNESS_PROBE_CODES if code in code_set]
        return get_daily_update_policy().validate_daily_outputs(
            check_codes,
            asset_type="stock",
            data_dir=self.data_dir,
        )

    def _validate_etf_outputs(self, codes: List[str]) -> List[str]:
        return get_daily_update_policy().validate_daily_outputs(
            codes,
            asset_type="etf",
            data_dir=self.data_dir,
        )

    def _validate_index_outputs(self, indices: List[dict]) -> List[str]:
        return get_daily_update_policy().validate_daily_outputs(
            [str(idx.get("code", "") or "") for idx in indices],
            asset_type="index",
            data_dir=self.data_dir,
        )

    def _validate_rotation_etf_outputs(self, codes: List[str]) -> List[str]:
        stale_items: List[str] = []
        policy = get_daily_update_policy()
        for code in codes:
            fresh, info = policy.check_daily_freshness(
                code,
                asset_type="etf",
                data_dir=self.rotation_data_dir,
                data_path=self.rotation_data_dir / f"{code}.parquet",
            )
            if not fresh:
                stale_items.append(f"{code}: {info}")
        return stale_items

    @staticmethod
    def _format_stale_items(label: str, stale_items: List[str], unit: str) -> str:
        return get_daily_update_policy().format_stale_items(label, stale_items, unit)

    def _refresh_stocks(self, cb: StatusCallback) -> Tuple[bool, str]:
        from scripts import fetch_kline_xtquant

        codes = self._load_stock_codes()
        if not codes:
            return True, "无股票需更新"

        total = len(codes)
        cb(f"🔄 全量刷新 {total} 只股票K线...")
        end_date = get_daily_update_policy().resolve_fetch_window(asset_type="stock", full_update=True).end_date
        self.data_dir.mkdir(parents=True, exist_ok=True)

        summary = run_batched_updates(
            codes,
            lambda code: fetch_kline_xtquant.fetch_one_full(code, self.start_date, end_date, self.data_dir, "1d"),
            max_workers=self.max_workers,
            batch_size=50,
            progress_cb=lambda current, total, _code, _msg: cb(f"股票K线: {current}/{total}") if current % 20 == 0 or current == total else None,
        )

        msg = f"股票 {summary.success}/{total} 成功"
        if summary.failed_items:
            msg += f" ({len(summary.failed_items)} 失败)"
            return False, msg
        stale_items = self._validate_stock_outputs(codes)
        if stale_items:
            return False, self._format_stale_items("股票", stale_items, "只")
        return True, msg

    def _refresh_etfs(self, cb: StatusCallback) -> Tuple[bool, str]:
        from scripts import fetch_kline_xtquant

        codes = self._load_etf_codes()
        if not codes:
            return True, "无ETF需更新"

        total = len(codes)
        cb(f"🔄 全量刷新 {total} 只ETF K线...")
        end_date = get_daily_update_policy().resolve_fetch_window(asset_type="etf", full_update=True).end_date
        (self.data_dir / "etf").mkdir(parents=True, exist_ok=True)

        summary = run_batched_updates(
            codes,
            lambda code: fetch_kline_xtquant.fetch_etf_one_full(code, self.start_date, end_date, self.data_dir, "1d"),
            max_workers=self.max_workers,
            batch_size=50,
            progress_cb=lambda current, total, _code, _msg: cb(f"ETF K线: {current}/{total}") if current % 10 == 0 or current == total else None,
        )

        msg = f"ETF {summary.success}/{total} 成功"
        if summary.failed_items:
            msg += f" ({len(summary.failed_items)} 失败)"
            return False, msg
        stale_items = self._validate_etf_outputs(codes)
        if stale_items:
            stale_msg = self._format_stale_items("ETF", stale_items, "只")
            logger.warning("全市场ETF刷新后存在历史/退市/特殊ETF未达到最新交易日，按非关键提示处理: %s", stale_msg)
            cb(f"⚠ {stale_msg}，已按非关键全市场ETF提示处理")
            return True, f"{msg}；{stale_msg}（非轮动ETF提示，不阻断日终）"
        return True, msg

    def _refresh_indices(self, cb: StatusCallback) -> Tuple[bool, str]:
        from scripts import fetch_kline_xtquant

        indices = fetch_kline_xtquant.load_index_codes_from_config(None)
        if not indices:
            return True, "无指数需更新"

        total = len(indices)
        cb(f"🔄 全量刷新 {total} 个指数K线...")
        end_date = get_daily_update_policy().resolve_fetch_window(asset_type="index", full_update=True).end_date
        (self.data_dir / "index").mkdir(parents=True, exist_ok=True)

        summary = run_batched_updates(
            indices,
            lambda idx: fetch_kline_xtquant.fetch_index_one_full(
                idx["code"], "19900101", end_date,
                self.data_dir, "1d", idx.get("exchange"),
            ),
            max_workers=self.max_workers,
            batch_size=50,
            item_label=lambda idx: str(idx.get("code", "") or ""),
        )

        msg = f"指数 {summary.success}/{total} 成功"
        if summary.failed_items:
            msg += f" ({len(summary.failed_items)} 失败)"
            cb(msg)
            return False, msg
        stale_items = self._validate_index_outputs(indices)
        if stale_items:
            msg = self._format_stale_items("指数", stale_items, "个")
            cb(msg)
            return False, msg
        cb(msg)
        return True, msg

    def _refresh_rotation_etfs(self, cb: StatusCallback) -> Tuple[bool, str]:
        codes = self.rotation_etf_pool
        if not codes:
            return True, "无轮动ETF需更新"

        cb(f"🔄 全量刷新 {len(codes)} 只轮动ETF K线...")
        try:
            s, t, errs = update_rotation_etf_pool(
                codes,
                self.rotation_data_dir,
                full=True,
                progress_cb=lambda cur, tot, code, _msg: cb(f"轮动ETF: {cur}/{tot} ({code})"),
            )
            if errs:
                for e in errs:
                    logger.warning("轮动ETF更新错误: %s", e)
                return False, f"轮动ETF {s}/{t} 成功 ({len(errs)} 错误)"
            stale_items = self._validate_rotation_etf_outputs(codes)
            if stale_items:
                return False, self._format_stale_items("轮动ETF", stale_items, "只")
            return True, f"轮动ETF {s}/{t} 成功"
        except Exception as exc:
            logger.exception("轮动ETF全量刷新异常")
            return False, f"轮动ETF异常: {exc}"

    # ------------------------------------------------------------------
    # Code list loaders
    # ------------------------------------------------------------------

    def _load_stock_codes(self) -> List[str]:
        try:
            from scripts.fetch_kline_xtquant import load_codes_from_stocklist

            if self.stocklist_path.exists():
                return load_codes_from_stocklist(self.stocklist_path)
        except Exception as exc:
            logger.warning("加载 stocklist 失败: %s", exc)
        return []

    def _load_etf_codes(self) -> List[str]:
        """Load ETF codes from config or by scanning existing parquet files."""
        try:
            from scripts.fetch_kline_xtquant import load_etf_codes_from_config

            if self.etf_config_path.exists():
                codes = load_etf_codes_from_config(self.etf_config_path)
                if codes:
                    return codes
        except Exception:
            pass

        etf_dir = self.data_dir / "etf"
        if etf_dir.is_dir():
            return sorted(
                p.stem for p in etf_dir.glob("*.parquet") if p.stem.isdigit()
            )
        return []
