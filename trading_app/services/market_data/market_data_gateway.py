from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from trading_app.services.market_data.market_data_policy import (
    can_use_daily_fallback,
    evaluate_tick_freshness,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketPriceSnapshot:
    code: str
    xt_code: str
    price: float
    source: str
    source_time: Optional[datetime] = None
    received_time: Optional[datetime] = None
    age_seconds: Optional[float] = None
    is_fresh: bool = False
    message: str = ""


def to_xt_code(code: str, *, is_index: bool = False) -> str:
    if "." in str(code or ""):
        return str(code).strip().upper()

    normalized = str(code or "").strip().upper().zfill(6)
    if is_index:
        return f"{normalized}.SZ" if normalized.startswith("399") else f"{normalized}.SH"
    if normalized.startswith("920"):
        return f"{normalized}.BJ"
    if normalized.startswith(("60", "68", "5", "9")):
        return f"{normalized}.SH"
    if normalized.startswith(("4", "8")):
        return f"{normalized}.BJ"
    return f"{normalized}.SZ"


class MarketDataGateway:
    """Centralized gateway for miniQMT market-data reads used by live paths."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def _import_xtdata(self):
        from xtquant import xtdata

        return xtdata

    def is_available(self) -> bool:
        try:
            self._import_xtdata()
            return True
        except ImportError:
            return False

    def check_connection(self) -> tuple[bool, str]:
        try:
            result = self.get_full_tick(["000001.SZ"])
        except ImportError:
            return False, "xtquant 未安装"
        except Exception as exc:
            return False, f"启动失败: {exc}"
        if result is None:
            return False, "xtquant 返回空数据，请确认 miniQMT 已启动并连接"
        return True, "行情服务已启动"

    def get_full_tick(self, xt_codes: List[str]) -> Dict[str, dict]:
        codes = [str(code or "").strip().upper() for code in (xt_codes or []) if code]
        if not codes:
            return {}
        with self._lock:
            xtdata = self._import_xtdata()
            result = xtdata.get_full_tick(codes)
        return result if isinstance(result, dict) else {}

    def get_realtime_snapshot(
        self,
        code: str,
        *,
        is_index: bool = False,
        require_fresh: bool = True,
        allow_missing_time_outside_session_with_volume: bool = True,
    ) -> MarketPriceSnapshot:
        xt_code = to_xt_code(code, is_index=is_index)
        received_time = datetime.now()
        try:
            tick_map = self.get_full_tick([xt_code])
            tick = tick_map.get(xt_code)
        except Exception as exc:
            logger.warning("get_full_tick failed for %s: %s", xt_code, exc)
            return MarketPriceSnapshot(
                code=str(code),
                xt_code=xt_code,
                price=0.0,
                source="tick",
                received_time=received_time,
                is_fresh=False,
                message=f"get_full_tick failed: {exc}",
            )

        if not isinstance(tick, dict):
            return MarketPriceSnapshot(
                code=str(code),
                xt_code=xt_code,
                price=0.0,
                source="tick",
                received_time=received_time,
                is_fresh=False,
                message="tick not returned",
            )

        price = float(tick.get("lastPrice") or 0.0)
        freshness = evaluate_tick_freshness(
            tick,
            received_time,
            allow_missing_time_outside_session_with_volume=allow_missing_time_outside_session_with_volume,
        )
        is_fresh = freshness.is_fresh
        if require_fresh and not is_fresh:
            message = freshness.reason or "tick is stale"
        elif price <= 0:
            message = "invalid latest price"
            is_fresh = False
        else:
            message = "fresh realtime tick" if freshness.is_fresh else (freshness.reason or "tick accepted")
        return MarketPriceSnapshot(
            code=str(code),
            xt_code=xt_code,
            price=price if price > 0 else 0.0,
            source="tick",
            source_time= freshness.source_time,
            received_time=received_time,
            age_seconds=freshness.age_seconds,
            is_fresh=is_fresh and price > 0,
            message=message,
        )

    def get_latest_daily_close(self, code: str, *, is_index: bool = False) -> MarketPriceSnapshot:
        xt_code = to_xt_code(code, is_index=is_index)
        received_time = datetime.now()
        try:
            with self._lock:
                xtdata = self._import_xtdata()
                data = xtdata.get_market_data(
                    ["close"],
                    [xt_code],
                    period="1d",
                    count=1,
                    dividend_type="front",
                )
            if data and "close" in data:
                arr = data["close"].get(xt_code)
                if arr is not None and len(arr) > 0:
                    price = float(arr.iloc[-1] if hasattr(arr, "iloc") else arr[-1])
                    if price > 0:
                        return MarketPriceSnapshot(
                            code=str(code),
                            xt_code=xt_code,
                            price=price,
                            source="daily_close",
                            received_time=received_time,
                            is_fresh=True,
                            message="daily close fallback",
                        )
        except Exception as exc:
            logger.warning("get_market_data daily close failed for %s: %s", xt_code, exc)
            return MarketPriceSnapshot(
                code=str(code),
                xt_code=xt_code,
                price=0.0,
                source="daily_close",
                received_time=received_time,
                is_fresh=False,
                message=f"daily close failed: {exc}",
            )
        return MarketPriceSnapshot(
            code=str(code),
            xt_code=xt_code,
            price=0.0,
            source="daily_close",
            received_time=received_time,
            is_fresh=False,
            message="daily close not available",
        )

    def get_price_snapshot(
        self,
        code: str,
        *,
        is_index: bool = False,
        allow_daily_fallback: bool = True,
        require_fresh: bool = True,
    ) -> MarketPriceSnapshot:
        snapshot = self.get_realtime_snapshot(
            code,
            is_index=is_index,
            require_fresh=require_fresh,
        )
        if snapshot.price > 0 and (snapshot.is_fresh or not require_fresh):
            return snapshot
        if not allow_daily_fallback or not can_use_daily_fallback(datetime.now()):
            return MarketPriceSnapshot(
                code=str(code),
                xt_code=snapshot.xt_code,
                price=0.0,
                source="none",
                received_time=snapshot.received_time,
                is_fresh=False,
                message="fresh realtime tick unavailable and daily fallback is not allowed",
            )
        return self.get_latest_daily_close(code, is_index=is_index)


_market_data_gateway: Optional[MarketDataGateway] = None


def get_market_data_gateway() -> MarketDataGateway:
    global _market_data_gateway
    if _market_data_gateway is None:
        _market_data_gateway = MarketDataGateway()
    return _market_data_gateway
