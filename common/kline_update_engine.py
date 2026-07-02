"""Shared Qt-free K-line update execution helpers."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd

from common.daily_update_policy import get_daily_update_policy
from common.data_portal import get_data_portal

logger = logging.getLogger(__name__)

T = TypeVar("T")
ProgressCallback = Callable[[int, int, object, str], None]
StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class BatchUpdateSummary:
    total: int
    success: int = 0
    failed_items: List[str] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.failed_items)

    @property
    def ok(self) -> bool:
        return not self.failed_items

    @property
    def failed_codes(self) -> List[str]:
        return [item.split(":", 1)[0] for item in self.failed_items]


def run_batched_updates(
    items: Sequence[T],
    update_one: Callable[[T], None],
    *,
    max_workers: int = 4,
    batch_size: int = 50,
    should_stop: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    success_message: Optional[Callable[[T, int, int], str]] = None,
    failure_message: Optional[Callable[[T, Exception], str]] = None,
    item_label: Optional[Callable[[T], str]] = None,
) -> BatchUpdateSummary:
    """Run update tasks in bounded batches and collect failures."""
    total = len(items)
    completed = 0
    success = 0
    failed_items: List[str] = []
    should_stop = should_stop or (lambda: False)
    item_label = item_label or (lambda item: str(item))

    for start in range(0, total, batch_size):
        if should_stop():
            break
        batch = list(items[start : start + batch_size])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(update_one, item): item for item in batch}
            for future in as_completed(futures):
                if should_stop():
                    executor.shutdown(wait=False)
                    return BatchUpdateSummary(total=total, success=success, failed_items=failed_items)
                item = futures[future]
                completed += 1
                try:
                    future.result()
                    success += 1
                    msg = success_message(item, completed, total) if success_message else f"已更新 {item} ({completed}/{total})"
                except Exception as exc:
                    msg = failure_message(item, exc) if failure_message else f"更新 {item_label(item)} 失败: {exc}"
                    failed_items.append(f"{item_label(item)}: {exc}")
                    logger.warning(msg)
                if progress_cb:
                    progress_cb(completed, total, item, msg)

    return BatchUpdateSummary(total=total, success=success, failed_items=failed_items)


def format_failed_update_message(label: str, failed_items: List[str], unit: str) -> str:
    preview = "；".join(failed_items[:8])
    suffix = f"；另有 {len(failed_items) - 8} {unit}" if len(failed_items) > 8 else ""
    return f"部分{label}更新失败: {preview}{suffix}"


def validate_daily_outputs(
    codes: Iterable[str],
    *,
    asset_type: str,
    data_dir: Path,
) -> List[str]:
    return get_daily_update_policy().validate_daily_outputs(
        codes,
        asset_type=asset_type,
        data_dir=data_dir,
    )


def check_xtquant_ready() -> Tuple[bool, str]:
    try:
        from scripts import fetch_kline_xtquant
    except ImportError:
        return False, "fetch_kline_xtquant 导入失败"

    if not fetch_kline_xtquant.check_xtquant_available():
        return False, "xtquant 未安装"

    connected, msg = fetch_kline_xtquant.check_connection()
    if not connected:
        return False, f"miniQMT 连接失败: {msg}"
    return True, msg or "miniQMT 连接正常"


def run_xtquant_daily_history_precheck(
    *,
    action_hint: str,
    require_recent_daily_history: bool = False,
    recent_calendar_days: int = 10,
) -> Tuple[bool, str]:
    from common.xtquant_data_health import test_xtquant_data_freshness

    result = get_daily_update_policy().run_daily_history_precheck(
        lambda: test_xtquant_data_freshness(
            require_minute_freshness=False,
            require_recent_daily_history=require_recent_daily_history,
            recent_calendar_days=recent_calendar_days,
        ),
        action_hint=action_hint,
    )
    return result.ok, result.message


def update_rotation_single_etf(
    code: str,
    data_dir: Path,
    start: str = "20190101",
    full: bool = False,
) -> Tuple[bool, str]:
    """Update one live_rotation ETF daily parquet in an independent directory."""
    try:
        from scripts.fetch_kline_xtquant import fetch_etf_kline, validate
    except ImportError:
        return False, "无法导入 fetch_kline_xtquant"

    data_dir = Path(data_dir)
    if data_dir.name == "data" and data_dir.parent.name != "live_rotation":
        pq = data_dir / "etf" / f"{code}.parquet"
    else:
        pq = data_dir / f"{code}.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True)

    window = get_daily_update_policy().resolve_fetch_window(
        asset_type="etf",
        default_start=start,
        full_update=full,
        local_path=pq,
    )
    end = window.end_date
    incremental_start = window.start_date
    existing_df = None

    if not full and pq.exists():
        try:
            existing_df = pd.read_parquet(pq)
        except Exception:
            existing_df = None

    for attempt in range(1, 4):
        try:
            new_df = fetch_etf_kline(code, incremental_start, end, "1d")

            if new_df.empty:
                if existing_df is not None and not existing_df.empty:
                    fresh, last_str = get_daily_update_policy().check_daily_freshness(
                        code,
                        asset_type="etf",
                        data_dir=data_dir,
                        data_path=pq,
                    )
                    if fresh:
                        return True, "无新数据"
                    return False, f"无新数据且本地仍未达到最新交易日，最新 {last_str or '未知'}"
                return False, "无法获取数据"

            if existing_df is not None and not existing_df.empty:
                merged = pd.concat([existing_df, new_df], ignore_index=True)
                new_df = merged.drop_duplicates(subset="date", keep="last")

            new_df = validate(new_df, "1d")
            new_df = new_df.sort_values("date").reset_index(drop=True)
            new_df.to_parquet(pq, index=False)
            get_data_portal().write_parquet_sidecar(
                pq,
                symbol=code,
                asset_type="etf",
                frequency="1d",
                data_source="xtquant",
                provider_symbol=code,
                update_mode="full" if full else "incremental",
                fetch_start=incremental_start,
                fetch_end=end,
                source_start=start,
                source_end=end,
                extra={"storage_scope": "live_rotation"},
            )
            fresh, last_str = get_daily_update_policy().check_daily_freshness(
                code,
                asset_type="etf",
                data_dir=data_dir,
                data_path=pq,
            )
            if not fresh:
                return False, f"更新后仍未达到最新交易日，最新 {last_str or '未知'}"
            return True, f"{len(new_df)} 条"

        except Exception as exc:
            if attempt < 3:
                time.sleep(1)
            else:
                return False, f"3次重试失败: {exc}"

    return False, "未知错误"


def update_rotation_etf_pool(
    codes: List[str],
    data_dir: Path,
    *,
    full: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[int, int, List[str]]:
    """Synchronously update a live_rotation ETF pool."""
    ok, msg = check_xtquant_ready()
    if not ok:
        if "xtquant 未安装" in msg:
            return 0, len(codes), ["xtquant 未安装"]
        return 0, len(codes), [msg]

    history_ok, history_msg = run_xtquant_daily_history_precheck(
        action_hint="请确认 miniQMT 已正常运行并登录后再更新/执行ETF轮动实盘。",
    )
    if not history_ok:
        return 0, len(codes), [history_msg]

    success = 0
    errors: List[str] = []
    total = len(codes)
    for index, code in enumerate(codes, start=1):
        ok, item_msg = update_rotation_single_etf(code, data_dir, full=full)
        if ok:
            success += 1
        else:
            errors.append(f"{code}: {item_msg}")
        if progress_cb:
            progress_cb(index, total, code, item_msg)
    return success, total, errors


__all__ = [
    "BatchUpdateSummary",
    "ProgressCallback",
    "StatusCallback",
    "check_xtquant_ready",
    "format_failed_update_message",
    "run_batched_updates",
    "run_xtquant_daily_history_precheck",
    "update_rotation_etf_pool",
    "update_rotation_single_etf",
    "validate_daily_outputs",
]
