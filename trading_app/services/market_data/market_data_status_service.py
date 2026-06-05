from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional

from common.data_portal import get_data_portal
from common.market_data_policy import latest_expected_trading_day, normalize_symbol_code
from trading_app.services.market_data.market_data_gateway import get_market_data_gateway

logger = logging.getLogger(__name__)

_DEFAULT_STOCK_PROBES = ("000001", "600000", "000333", "300750", "600519")
_DEFAULT_INDEX_PROBES = ("000001", "399001", "399006", "000300", "000905")
_DEFAULT_REALTIME_PROBES = ("159915", "510300", "000001")


@dataclass(frozen=True)
class MarketDataCheck:
    name: str
    ok: bool
    message: str
    required: bool = True


@dataclass(frozen=True)
class MarketDataStatus:
    xtdata_available: bool
    realtime_available: bool
    realtime_fresh: bool
    daily_fresh: bool
    minute_fresh: bool
    cache_synced: bool
    latest_expected_trading_day: date
    checked_at: datetime
    blocking_reason: str = ""
    checks: List[MarketDataCheck] = field(default_factory=list)

    @property
    def can_run_live_strategy(self) -> bool:
        return (
            self.xtdata_available
            and self.realtime_available
            and self.realtime_fresh
            and self.daily_fresh
            and self.minute_fresh
            and self.cache_synced
            and not self.blocking_reason
        )

    @property
    def summary(self) -> str:
        if self.can_run_live_strategy:
            return "行情数据状态正常，可以执行实盘策略"
        if self.blocking_reason:
            return self.blocking_reason
        failed = [check for check in self.checks if not check.ok and check.required]
        if failed:
            return "；".join(f"{check.name}: {check.message}" for check in failed[:3])
        warnings = [check for check in self.checks if not check.ok]
        if warnings:
            return "；".join(f"{check.name}: {check.message}" for check in warnings[:3])
        return "行情数据状态未知"


class MarketDataStatusService:
    """Read-only health center for market-data availability and freshness."""

    def __init__(self) -> None:
        self._last_status: Optional[MarketDataStatus] = None

    @property
    def last_status(self) -> Optional[MarketDataStatus]:
        return self._last_status

    def check_status(
        self,
        *,
        stock_codes: Optional[Iterable[str]] = None,
        etf_codes: Optional[Iterable[str]] = None,
        index_codes: Optional[Iterable[str]] = None,
        realtime_probe_codes: Optional[Iterable[str]] = None,
        require_minute_freshness: bool = False,
        stock_data_dir: Optional[Path] = None,
        etf_data_dir: Optional[Path] = None,
        index_data_dir: Optional[Path] = None,
    ) -> MarketDataStatus:
        checked_at = datetime.now()
        expected_day = latest_expected_trading_day(checked_at)
        checks: List[MarketDataCheck] = []

        realtime_codes = _DEFAULT_REALTIME_PROBES if realtime_probe_codes is None else realtime_probe_codes
        stock_probe_codes = _DEFAULT_STOCK_PROBES if stock_codes is None else stock_codes
        etf_probe_codes = () if etf_codes is None else etf_codes
        index_probe_codes = _DEFAULT_INDEX_PROBES if index_codes is None else index_codes

        xtdata_available, realtime_available, realtime_fresh, realtime_message = self._check_realtime(
            realtime_codes
        )
        checks.append(MarketDataCheck("miniQMT行情接口", xtdata_available, realtime_message))
        checks.append(MarketDataCheck("实时tick可用性", realtime_available, realtime_message))
        checks.append(MarketDataCheck("实时tick freshness", realtime_fresh, realtime_message))

        daily_fresh, daily_message = self._check_daily_freshness(
            stock_probe_codes,
            etf_probe_codes,
            index_probe_codes,
            stock_data_dir=stock_data_dir,
            etf_data_dir=etf_data_dir,
            index_data_dir=index_data_dir,
        )
        checks.append(MarketDataCheck("日线parquet freshness", daily_fresh, daily_message))

        minute_fresh, minute_message = self._check_minute_freshness(require_minute_freshness)
        checks.append(MarketDataCheck("分钟线freshness", minute_fresh, minute_message, required=require_minute_freshness))

        cache_synced, cache_message = self._check_cache_synced(daily_fresh)
        checks.append(MarketDataCheck("内存缓存同步", cache_synced, cache_message))

        blocking_reason = ""
        for check in checks:
            if check.required and not check.ok:
                blocking_reason = f"{check.name}: {check.message}"
                break

        status = MarketDataStatus(
            xtdata_available=xtdata_available,
            realtime_available=realtime_available,
            realtime_fresh=realtime_fresh,
            daily_fresh=daily_fresh,
            minute_fresh=minute_fresh,
            cache_synced=cache_synced,
            latest_expected_trading_day=expected_day,
            checked_at=checked_at,
            blocking_reason=blocking_reason,
            checks=checks,
        )
        self._last_status = status
        return status

    def _check_realtime(self, probe_codes: Iterable[str]) -> tuple[bool, bool, bool, str]:
        gateway = get_market_data_gateway()
        messages: List[str] = []
        saw_tick = False
        saw_fresh = False
        xtdata_available = True
        for code in probe_codes:
            try:
                snapshot = gateway.get_realtime_snapshot(str(code), require_fresh=True)
            except ImportError as exc:
                return False, False, False, f"xtquant不可用: {exc}"
            except Exception as exc:
                xtdata_available = False
                messages.append(f"{code}: {exc}")
                continue
            if snapshot.price > 0:
                saw_tick = True
            if snapshot.price > 0 and snapshot.is_fresh:
                saw_fresh = True
                return True, True, True, f"{snapshot.xt_code} 最新价 {snapshot.price:.3f}"
            messages.append(f"{snapshot.xt_code}: {snapshot.message}")
        if not xtdata_available:
            return False, saw_tick, saw_fresh, "；".join(messages[:3]) or "实时行情接口异常"
        if saw_tick:
            return True, True, False, "；".join(messages[:3]) or "实时tick不新鲜"
        return True, False, False, "；".join(messages[:3]) or "未获取到有效实时tick"

    def _check_daily_freshness(
        self,
        stock_codes: Iterable[str],
        etf_codes: Iterable[str],
        index_codes: Iterable[str],
        *,
        stock_data_dir: Optional[Path] = None,
        etf_data_dir: Optional[Path] = None,
        index_data_dir: Optional[Path] = None,
    ) -> tuple[bool, str]:
        portal = get_data_portal()
        stale_items: List[str] = []
        checked = 0
        for code in stock_codes:
            normalized = normalize_symbol_code(str(code)).zfill(6)
            status = portal.get_daily_metadata(normalized, asset_type="stock", data_dir=stock_data_dir)
            checked += 1
            if not status.is_fresh:
                stale_items.append(f"{normalized}: {portal.format_daily_status_message(status)}")
        for code in etf_codes:
            normalized = normalize_symbol_code(str(code)).zfill(6)
            status = portal.get_daily_metadata(normalized, asset_type="etf", data_dir=etf_data_dir)
            checked += 1
            if not status.is_fresh:
                stale_items.append(f"{normalized}: {portal.format_daily_status_message(status)}")
        for code in index_codes:
            normalized = normalize_symbol_code(str(code)).zfill(6)
            status = portal.get_daily_metadata(normalized, asset_type="index", data_dir=index_data_dir)
            checked += 1
            if not status.is_fresh:
                stale_items.append(f"index/{normalized}: {portal.format_daily_status_message(status)}")
        if stale_items:
            return False, "；".join(stale_items[:5])
        return True, f"已检查 {checked} 个parquet文件"

    def _check_minute_freshness(self, required: bool) -> tuple[bool, str]:
        if not required:
            return True, "未要求分钟线强校验"
        try:
            from common.xtquant_data_health import evaluate_xtquant_data_freshness

            report = evaluate_xtquant_data_freshness(require_minute_freshness=True)
        except Exception as exc:
            logger.exception("分钟线freshness检查异常")
            return False, f"分钟线检查异常: {exc}"
        return report.ok, report.summary

    def _check_cache_synced(self, daily_fresh: bool) -> tuple[bool, str]:
        try:
            from common.data_portal import get_etf_cache, get_stock_cache
        except Exception as exc:
            return True, f"缓存管理器不可用，按未加载处理: {exc}"
        loaded_caches: List[str] = []
        try:
            if get_stock_cache().is_loaded():
                loaded_caches.append("股票")
        except Exception as exc:
            return False, f"股票缓存状态检查失败: {exc}"
        try:
            if get_etf_cache().is_loaded():
                loaded_caches.append("ETF")
        except Exception as exc:
            return False, f"ETF缓存状态检查失败: {exc}"
        if not loaded_caches:
            return True, "内存缓存尚未加载"
        if not daily_fresh:
            return False, f"{'/'.join(loaded_caches)}缓存已加载但parquet未新鲜"
        return True, f"{'/'.join(loaded_caches)}缓存与parquet freshness一致"


_market_data_status_service: Optional[MarketDataStatusService] = None


def get_market_data_status_service() -> MarketDataStatusService:
    global _market_data_status_service
    if _market_data_status_service is None:
        _market_data_status_service = MarketDataStatusService()
    return _market_data_status_service
