from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.services.market_data.data_update_result import DataUpdateResult
from trading_app.services.market_data.market_data_gateway import MarketDataGateway, to_xt_code
from trading_app.services.market_data.market_data_policy import evaluate_tick_freshness, parse_tick_datetime


class FakeSeries:
    def __init__(self, values):
        self._values = list(values)
        self.iloc = self

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


class FakeXtData:
    def __init__(self, tick):
        self.tick = tick
        self.full_tick_calls = []
        self.market_data_calls = []

    def get_full_tick(self, codes):
        self.full_tick_calls.append(list(codes))
        return {code: dict(self.tick) for code in codes}

    def get_market_data(self, fields, codes, period="1d", count=1, dividend_type="front"):
        self.market_data_calls.append((fields, codes, period, count, dividend_type))
        return {"close": {codes[0]: FakeSeries([3.21])}}


class FakeGateway(MarketDataGateway):
    def __init__(self, fake_xtdata):
        super().__init__()
        self.fake_xtdata = fake_xtdata

    def _import_xtdata(self):
        return self.fake_xtdata


def case_parse_tick_datetime() -> None:
    assert parse_tick_datetime("2026-04-25 14:30:01") == datetime(2026, 4, 25, 14, 30, 1)
    dt = datetime(2026, 4, 25, 14, 30, 1)
    assert parse_tick_datetime(int(dt.timestamp())) == dt
    assert parse_tick_datetime(int(dt.timestamp() * 1000)) == dt
    assert parse_tick_datetime("") is None
    print("[parse_tick_datetime] OK")


def case_evaluate_tick_freshness() -> None:
    now = datetime(2026, 4, 24, 14, 30, 0)
    fresh_tick = {"lastPrice": 10.0, "time": int((now - timedelta(seconds=30)).timestamp())}
    stale_tick = {"lastPrice": 10.0, "time": int((now - timedelta(seconds=120)).timestamp())}
    old_day_tick = {"lastPrice": 10.0, "time": int((now - timedelta(days=1)).timestamp())}

    fresh = evaluate_tick_freshness(fresh_tick, now)
    stale = evaluate_tick_freshness(stale_tick, now)
    old_day = evaluate_tick_freshness(old_day_tick, now)

    assert fresh.is_fresh, fresh
    assert not stale.is_fresh, stale
    assert not old_day.is_fresh, old_day
    print("[evaluate_tick_freshness] OK")


def case_gateway_snapshot() -> None:
    now = datetime.now()
    fake_xtdata = FakeXtData({"lastPrice": 2.34, "time": int(now.timestamp()), "volume": 1000})
    gateway = FakeGateway(fake_xtdata)
    snapshot = gateway.get_price_snapshot("510300", allow_daily_fallback=False, require_fresh=True)

    assert snapshot.price == 2.34, snapshot
    assert snapshot.is_fresh, snapshot
    assert snapshot.xt_code == "510300.SH", snapshot
    assert fake_xtdata.full_tick_calls == [["510300.SH"]]
    assert to_xt_code("399001", is_index=True) == "399001.SZ"
    assert to_xt_code("920002") == "920002.BJ"
    assert to_xt_code("900901") == "900901.SH"
    print("[gateway_snapshot] OK")


def case_xtquant_symbol_helpers() -> None:
    from scripts.fetch_kline_xtquant import _to_xt_code, load_etf_codes_from_config

    assert _to_xt_code("920002") == "920002.BJ"
    assert _to_xt_code("900901") == "900901.SH"

    config = {
        "categories": [
            {
                "name": "demo",
                "etfs": [
                    {"code": "515730", "name": "家居家电ETF", "exchange": "SH"},
                    {"code": "562070", "name": "沪深300指增ETF", "exchange": "SH"},
                    {"code": "515733", "name": "家居家电", "exchange": "SH"},
                    {"code": "520870", "name": "巴西ETF", "exchange": "SH"},
                    {"code": "520873", "name": "巴西ETF", "exchange": "SH"},
                    {"code": "513724", "name": "认购款", "exchange": "SH"},
                    {"code": "588430", "name": "N科创创业人工智能ETF工银", "exchange": "SH"},
                    {"code": "589133", "name": "科创芯易", "exchange": "SH"},
                    {"code": "159949", "name": "创业板50ETF", "exchange": "SZ"},
                ],
            },
            {
                "name": "demo_cross_category",
                "etfs": [
                    {"code": "562073", "name": "300ETF增", "exchange": "SH"},
                ],
            },
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "etf_list.json"
        path.write_text(__import__("json").dumps(config, ensure_ascii=False), encoding="utf-8")
        codes = load_etf_codes_from_config(path)
    assert codes == ["515730", "562070", "520870", "159949"], codes
    print("[xtquant_symbol_helpers] OK")


def case_gateway_daily_fallback() -> None:
    now = datetime.now()
    fake_xtdata = FakeXtData({"lastPrice": 2.34, "time": int((now - timedelta(days=1)).timestamp()), "volume": 1000})
    gateway = FakeGateway(fake_xtdata)
    snapshot = gateway.get_latest_daily_close("510300")

    assert snapshot.price == 3.21, snapshot
    assert snapshot.source == "daily_close", snapshot
    assert snapshot.is_fresh, snapshot
    print("[gateway_daily_fallback] OK")


def case_data_update_result() -> None:
    result = DataUpdateResult(
        ok=True,
        updated_stocks=2,
        updated_etfs=1,
        cache_refreshed=True,
        cache_refreshed_stocks=2,
        cache_refreshed_etfs=1,
    )
    assert result.to_legacy_tuple()[0] is True
    assert "股票2只" in result.summary
    assert "ETF1只" in result.summary
    assert "缓存 股票2 ETF1" in result.summary
    print("[data_update_result] OK")


def case_full_market_etf_stale_is_non_blocking() -> None:
    import trading_app.services.market_data.kline_full_refresh_service as refresh_module
    from trading_app.services.market_data.kline_full_refresh_service import KlineFullRefreshService

    class FakeBatchSummary:
        success = 1
        failed_items = []

    original_run_batched_updates = refresh_module.run_batched_updates
    refresh_module.run_batched_updates = lambda *args, **kwargs: FakeBatchSummary()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            service = KlineFullRefreshService(data_dir=Path(tmp))
            service._load_etf_codes = lambda: ["159702"]
            service._validate_etf_outputs = lambda codes: ["159702: 2022-11-18"]
            messages: list[str] = []
            ok, msg = service._refresh_etfs(messages.append)
        assert ok, msg
        assert "非轮动ETF提示" in msg, msg
        assert any("非关键全市场ETF" in item for item in messages), messages
    finally:
        refresh_module.run_batched_updates = original_run_batched_updates
    print("[full_market_etf_stale_is_non_blocking] OK")


def main() -> None:
    case_parse_tick_datetime()
    case_evaluate_tick_freshness()
    case_gateway_snapshot()
    case_xtquant_symbol_helpers()
    case_gateway_daily_fallback()
    case_data_update_result()
    case_full_market_etf_stale_is_non_blocking()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
