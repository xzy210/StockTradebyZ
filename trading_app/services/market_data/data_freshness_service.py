"""Data freshness check & auto-update before AI tasks.

Checks whether local parquet files contain today's (or latest trading day's) data.
If stale, tests xtquant connectivity via reconnect, then triggers DataUpdateThread
for the required stock codes + indices.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from common.daily_update_policy import get_daily_update_policy
from common.data_portal import get_data_portal
from common.market_data_policy import is_etf_like_code, normalize_symbol_code
from common.xtquant_data_health import (
    FreshnessCheckResult,
    XtquantFreshnessReport,
    check_batch_freshness,
    check_parquet_freshness,
    evaluate_xtquant_data_freshness,
    test_xtquant_data_freshness,
)
from trading_app.services.market_data.data_update_result import DataUpdateResult

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"
_STOCKLIST_PATH = _PROJECT_ROOT / "stocklist" / "stocklist.csv"


def _normalize_symbol_code(code: str) -> str:
    return normalize_symbol_code(code)


def _is_etf_like_code(code: str) -> bool:
    return is_etf_like_code(code)


def _dedupe_codes(codes: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for code in codes:
        normalized = _normalize_symbol_code(code)
        if not normalized:
            continue
        normalized = normalized.zfill(6)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _refresh_loaded_market_data_caches(stock_codes: List[str], etf_codes: List[str]) -> Tuple[int, int]:
    """Reload in-memory caches after parquet freshness is confirmed."""
    try:
        result = get_data_portal().refresh_loaded_caches(
            data_dir=_DATA_DIR,
            stock_codes=_dedupe_codes(stock_codes),
            etf_codes=_dedupe_codes(etf_codes),
        )
    except Exception as exc:
        logger.warning("刷新行情缓存失败，跳过缓存刷新: %s", exc)
        return 0, 0

    if result.refreshed:
        logger.info("已刷新内存行情缓存: 股票 %d 只, ETF %d 只", result.stock_count, result.etf_count)
    return result.stock_count, result.etf_count


class DataFreshnessGuard(QObject):
    """Pre-flight check before AI scheduled tasks.

    Signals:
        check_started: emitted when freshness check begins
        update_needed: (stale_count, message) emitted when data needs updating
        update_progress: (current, total, msg) forwarded from DataUpdateThread
        update_finished: (success, message) when everything is ready
        xtquant_failed: (message) when xtquant reconnect test fails
    """

    check_started = pyqtSignal()
    update_needed = pyqtSignal(int, str)
    update_progress = pyqtSignal(int, int, str)
    update_finished = pyqtSignal(bool, str)
    result_signal = pyqtSignal(object)  # DataUpdateResult
    status_notice = pyqtSignal(str, str)
    xtquant_failed = pyqtSignal(str)
    freshness_test_done = pyqtSignal(object, list, list, list, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._update_thread = None
        self._etf_thread = None
        self._pending_callback: Optional[Callable] = None
        self._stale_codes: List[str] = []
        self._stale_etf_codes: List[str] = []
        self._stale_index_codes: List[str] = []
        self._checked_stock_codes: List[str] = []
        self._checked_etf_codes: List[str] = []
        self._pending_notice_message: str = ""
        self.last_result = DataUpdateResult(ok=False, message="尚未执行")
        self.freshness_test_done.connect(self._on_freshness_test_done)

    def _finish_update(self, result: DataUpdateResult):
        self.last_result = result
        self.result_signal.emit(result)
        self.update_finished.emit(*result.to_legacy_tuple())

    def ensure_fresh_then_run(
        self,
        codes: List[str],
        callback: Callable,
        *,
        include_indices: bool = True,
        prefer_realtime: bool = False,
        require_minute_freshness: bool = False,
    ):
        """Check freshness for `codes`; if stale, update then call `callback`.

        If data is already fresh, `callback` is called immediately.
        """
        stock_like_codes = [code for code in codes if not _is_etf_like_code(code)]
        etf_like_codes = [code for code in codes if _is_etf_like_code(code)]
        self._checked_stock_codes = _dedupe_codes(stock_like_codes)
        self._checked_etf_codes = _dedupe_codes(etf_like_codes)
        logger.info(
            "开始校验任务数据新鲜度: 股票 %d 只, ETF %d 只, 包含指数=%s",
            len(stock_like_codes),
            len(etf_like_codes),
            include_indices,
        )
        self.check_started.emit()
        self.status_notice.emit("info", "开始执行数据新鲜度检查")
        self._pending_callback = callback

        policy = get_daily_update_policy()
        stale_stock_codes = policy.find_stale_symbols(stock_like_codes, asset_type="stock", data_dir=_DATA_DIR)
        stale_etf_codes = policy.find_stale_symbols(etf_like_codes, asset_type="etf", data_dir=_DATA_DIR)

        stale_index_codes = []
        if include_indices:
            stale_index_codes = policy.find_stale_symbols(
                ["000001", "399001", "399006", "000300", "000905"],
                asset_type="index",
                data_dir=_DATA_DIR,
            )

        total_stale = len(stale_stock_codes) + len(stale_etf_codes) + len(stale_index_codes)
        need_intraday_test = bool(prefer_realtime and policy.is_intraday_check_window())
        if total_stale == 0 and not need_intraday_test:
            logger.info("All data is fresh, proceeding directly")
            refreshed_stocks, refreshed_etfs = _refresh_loaded_market_data_caches(self._checked_stock_codes, self._checked_etf_codes)
            self.status_notice.emit("success", "日线完整性已通过，本地数据已是最新")
            self._finish_update(DataUpdateResult(
                ok=True,
                cache_refreshed=bool(refreshed_stocks or refreshed_etfs),
                cache_refreshed_stocks=refreshed_stocks,
                cache_refreshed_etfs=refreshed_etfs,
                message="数据已是最新",
            ))
            self._pending_notice_message = ""
            QTimer.singleShot(0, callback)
            return

        self._stale_codes = stale_stock_codes
        self._stale_etf_codes = stale_etf_codes
        if total_stale > 0:
            logger.info(
                "发现数据待更新: 股票 %d 只, ETF %d 只, 指数 %d 个",
                len(stale_stock_codes),
                len(stale_etf_codes),
                len(stale_index_codes),
            )
            self.update_needed.emit(
                total_stale,
                f"发现 {len(stale_stock_codes)} 只股票 + {len(stale_etf_codes)} 只ETF + {len(stale_index_codes)} 个指数数据需更新"
            )
            self.status_notice.emit("info", "检测到本地日线未更新，准备执行增量更新")
        elif need_intraday_test:
            logger.info("日线已最新，继续校验盘中实时行情链路")

        # Run the (potentially slow) xtquant freshness test in a background
        # thread so the UI remains responsive.
        def _bg_test():
            logger.info("开始执行 xtquant 新鲜度拉取测试")
            try:
                report = evaluate_xtquant_data_freshness(
                    require_minute_freshness=require_minute_freshness
                )
            except Exception as exc:
                logger.exception("xtquant 新鲜度拉取测试异常")
                report = XtquantFreshnessReport(
                    False,
                    [FreshnessCheckResult("数据链路", False, f"xtquant 新鲜度测试异常: {exc}")],
                )
            logger.info("xtquant 新鲜度拉取测试结束: success=%s, %s", report.ok, report.summary)
            self.freshness_test_done.emit(
                report,
                stale_stock_codes,
                stale_etf_codes,
                stale_index_codes,
                callback,
            )

        threading.Thread(target=_bg_test, daemon=True).start()

    def _on_freshness_test_done(
        self,
        report: object,
        stale_stock_codes: List[str],
        stale_etf_codes: List[str],
        stale_index_codes: List[str],
        callback: Callable,
    ):
        """Called on main thread after background freshness test finishes."""
        if not isinstance(report, XtquantFreshnessReport):
            report = XtquantFreshnessReport(
                False,
                [FreshnessCheckResult("数据链路", False, f"未知的新鲜度检查结果: {report}")],
            )
        self._pending_notice_message = report.summary
        if not report.ok:
            logger.warning("xtquant freshness test failed: %s", report.summary)
            self.status_notice.emit("error", report.summary)
            self.xtquant_failed.emit(report.summary)
            self._finish_update(DataUpdateResult(ok=False, message=report.summary))
            self._pending_notice_message = ""
            return
        if report.has_warning:
            self.status_notice.emit("warning", report.summary)
        else:
            self.status_notice.emit("success", report.summary)

        if not stale_stock_codes and not stale_etf_codes and not stale_index_codes:
            logger.info("xtquant OK, intraday realtime path is ready")
            refreshed_stocks, refreshed_etfs = _refresh_loaded_market_data_caches(self._checked_stock_codes, self._checked_etf_codes)
            self._finish_update(DataUpdateResult(
                ok=True,
                cache_refreshed=bool(refreshed_stocks or refreshed_etfs),
                cache_refreshed_stocks=refreshed_stocks,
                cache_refreshed_etfs=refreshed_etfs,
                message=report.summary,
            ))
            self._pending_callback = None
            self._pending_notice_message = ""
            QTimer.singleShot(0, callback)
            return

        total = len(stale_stock_codes) + len(stale_etf_codes) + len(stale_index_codes)
        logger.info("xtquant OK, starting data update for %d items", total)
        self._start_update(stale_stock_codes, stale_etf_codes, stale_index_codes, callback)

    def _start_update(
        self,
        stock_codes: List[str],
        etf_codes: List[str],
        index_codes: List[str],
        callback: Callable,
    ):
        from trading_app.data_updater import DataUpdateThread, ETFUpdateThread, IndexUpdateThread

        self._pending_callback = callback
        self._stale_codes = list(stock_codes)
        self._stale_etf_codes = list(etf_codes)
        self._stale_index_codes = list(index_codes)
        self._stock_update_done = not bool(stock_codes)
        self._etf_update_done = not bool(etf_codes)
        self._index_update_done = not bool(index_codes)
        self._stock_update_success = not bool(stock_codes)
        self._etf_update_success = not bool(etf_codes)
        self._index_update_success = not bool(index_codes)
        self._update_errors: List[str] = []

        if stock_codes:
            self._stock_thread = DataUpdateThread(
                data_dir=str(_DATA_DIR),
                stocklist_path=str(_STOCKLIST_PATH),
                codes=stock_codes,
                data_source="xtquant",
                period="1d",
                max_workers=4,
            )
            self._stock_thread.progress_updated.connect(
                lambda c, t, m: self.update_progress.emit(c, t, f"[股票] {m}")
            )
            self._stock_thread.finished_signal.connect(self._on_stock_update_done)
            self._stock_thread.start()

        if etf_codes:
            logger.info("启动ETF更新线程: %d 只ETF", len(etf_codes))
            self._etf_thread = ETFUpdateThread(
                data_dir=str(_DATA_DIR),
                codes=etf_codes,
                full_update=False,
            )
            self._etf_thread.progress_updated.connect(
                lambda c, t, m: self.update_progress.emit(c, t, f"[ETF] {m}")
            )
            self._etf_thread.finished_signal.connect(self._on_etf_update_done)
            self._etf_thread.start()

        if index_codes:
            logger.info("启动指数更新线程: %d 个指数", len(index_codes))
            self._index_thread = IndexUpdateThread(
                data_dir=str(_DATA_DIR),
                index_codes=[
                    {
                        "code": code,
                        "exchange": "SZ" if str(code).startswith("399") else "SH",
                    }
                    for code in index_codes
                ],
                full_update=False,
            )
            self._index_thread.progress_updated.connect(
                lambda c, t, m: self.update_progress.emit(c, t, f"[指数] {m}")
            )
            self._index_thread.finished_signal.connect(self._on_index_update_done)
            self._index_thread.start()

        if self._stock_update_done and self._etf_update_done and self._index_update_done:
            self._all_done()

    def _on_stock_update_done(self, success: bool, msg: str):
        logger.info("Stock update finished: success=%s, %s", success, msg)
        self._stock_update_done = True
        self._stock_update_success = bool(success)
        if not success and msg:
            self._update_errors.append(f"股票更新失败: {msg}")
        if self._etf_update_done and self._index_update_done:
            self._all_done()

    def _on_etf_update_done(self, success: bool, msg: str):
        logger.info("ETF update finished: success=%s, %s", success, msg)
        self._etf_update_done = True
        self._etf_update_success = bool(success)
        if not success and msg:
            self._update_errors.append(f"ETF更新失败: {msg}")
        if self._stock_update_done and self._index_update_done:
            self._all_done()

    def _on_index_update_done(self, success: bool, msg: str):
        logger.info("Index update finished: success=%s, %s", success, msg)
        self._index_update_done = True
        self._index_update_success = bool(success)
        if not success and msg:
            self._update_errors.append(f"指数更新失败: {msg}")
        if self._stock_update_done and self._etf_update_done:
            self._all_done()

    def _all_done(self):
        if not self._stock_update_success or not self._etf_update_success or not self._index_update_success:
            message = "；".join(self._update_errors) if self._update_errors else "数据更新失败"
            self.status_notice.emit("error", message)
            self._finish_update(DataUpdateResult(ok=False, message=message, failed_codes=list(self._update_errors)))
            self._pending_callback = None
            self._pending_notice_message = ""
            return

        policy = get_daily_update_policy()
        remaining_stock_codes = policy.find_stale_symbols(
            getattr(self, "_stale_codes", []),
            asset_type="stock",
            data_dir=_DATA_DIR,
        )
        remaining_index_codes = policy.find_stale_symbols(
            getattr(self, "_stale_index_codes", []),
            asset_type="index",
            data_dir=_DATA_DIR,
        )
        remaining_etf_codes = policy.find_stale_symbols(
            getattr(self, "_stale_etf_codes", []),
            asset_type="etf",
            data_dir=_DATA_DIR,
        )
        if remaining_stock_codes or remaining_etf_codes or remaining_index_codes:
            parts = []
            if remaining_stock_codes:
                preview = ", ".join(remaining_stock_codes[:5])
                if len(remaining_stock_codes) > 5:
                    preview += f" 等{len(remaining_stock_codes)}只股票"
                parts.append(f"股票数据仍未就绪: {preview}")
            if remaining_etf_codes:
                preview = ", ".join(remaining_etf_codes[:5])
                if len(remaining_etf_codes) > 5:
                    preview += f" 等{len(remaining_etf_codes)}只ETF"
                parts.append(f"ETF数据仍未就绪: {preview}")
            if remaining_index_codes:
                preview = ", ".join(remaining_index_codes[:5])
                if len(remaining_index_codes) > 5:
                    preview += f" 等{len(remaining_index_codes)}个指数"
                parts.append(f"指数数据仍未就绪: {preview}")
            message = "；".join(parts)
            self.status_notice.emit("error", message)
            self._finish_update(DataUpdateResult(
                ok=False,
                stale_codes=list(remaining_stock_codes + remaining_etf_codes + remaining_index_codes),
                message=message,
            ))
            self._pending_callback = None
            self._pending_notice_message = ""
            return

        final_message = "数据更新完成，开始 AI 分析"
        refreshed_stocks, refreshed_etfs = _refresh_loaded_market_data_caches(
            self._checked_stock_codes,
            self._checked_etf_codes,
        )
        if refreshed_stocks or refreshed_etfs:
            final_message = f"{final_message}；已刷新内存缓存 股票{refreshed_stocks}只 ETF{refreshed_etfs}只"
        if self._pending_notice_message:
            final_message = f"{final_message}；{self._pending_notice_message}"
        self.status_notice.emit("success", final_message)
        self._finish_update(DataUpdateResult(
            ok=True,
            updated_stocks=len(getattr(self, "_stale_codes", [])),
            updated_etfs=len(getattr(self, "_stale_etf_codes", [])),
            updated_indices=len(getattr(self, "_stale_index_codes", [])),
            cache_refreshed=bool(refreshed_stocks or refreshed_etfs),
            cache_refreshed_stocks=refreshed_stocks,
            cache_refreshed_etfs=refreshed_etfs,
            message=final_message,
        ))
        cb = self._pending_callback
        self._pending_callback = None
        self._pending_notice_message = ""
        if cb:
            QTimer.singleShot(500, cb)
