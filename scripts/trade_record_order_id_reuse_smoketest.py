"""Smoke test: miniQMT 委托号跨日复用时，回填与去重不能互相打架。

复现 2026-08-25 实盘中枢卡死：
  - 7 月 3 日黄金 ETF 买入委托号 940572673
  - 8 月 19 日黄金 ETF 再次买入，券商复用同一委托号
  - 去重按委托号忽略交易日，删掉 8 月 19 日回填记录
  - 本地委托回填再写入，形成死循环

Run::

    conda run -n stock python scripts/trade_record_order_id_reuse_smoketest.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.services.execution.trade_record_service import (
    OrderRecord,
    TradeRecordService,
)


ORDER_ID = 940572673
CODE = "518880"
STRATEGY_ID = "etf_rotation"
VIRTUAL_ACCOUNT_ID = "va_etf_rotation"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count_trades(service: TradeRecordService) -> int:
    conn = service._get_connection()  # noqa: SLF001
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM trades WHERE stock_code = ? AND broker_order_id = ?",
            (CODE, ORDER_ID),
        )
        return int(cursor.fetchone()[0] or 0)
    finally:
        conn.close()


def _trade_dates(service: TradeRecordService) -> list[str]:
    conn = service._get_connection()  # noqa: SLF001
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT trade_date FROM trades
            WHERE stock_code = ? AND broker_order_id = ?
            ORDER BY trade_date ASC
            """,
            (CODE, ORDER_ID),
        )
        return [str(row[0]) for row in cursor.fetchall()]
    finally:
        conn.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="trade_order_id_reuse_") as tmpdir:
        tmp = Path(tmpdir)
        service = TradeRecordService()
        service.db_path = tmp / "trade_records.db"
        service._init_database()  # noqa: SLF001

        first = service.add_record(
            stock_code=CODE,
            stock_name="黄金ETF",
            direction="buy",
            price=8.663,
            volume=3300,
            broker_order_id=ORDER_ID,
            trade_date="2026-07-03",
            source="live_strategy_center",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            remark="委托号:940572673",
            emit_signals=False,
        )
        _assert(first is not None, "应写入 7 月 3 日成交")

        later_order = OrderRecord(
            broker_order_id=ORDER_ID,
            stock_code=CODE,
            stock_name="黄金ETF",
            direction="buy",
            price=8.958,
            order_volume=3100,
            executed_price=8.957,
            executed_volume=3100,
            order_status_code=56,
            status="filled",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            created_at="2026-08-19 14:45:05",
            updated_at="2026-08-20 06:29:59",
        )

        added = service.sync_from_order_records([later_order])
        _assert(added == 1, f"8 月 19 日成交应回填 1 条，实际 {added}")
        _assert(_count_trades(service) == 2, "跨日复用委托号应保留两笔成交")
        _assert(_trade_dates(service) == ["2026-07-03", "2026-08-19"], f"成交日期异常: {_trade_dates(service)}")
        print("[backfill_cross_day] OK")

        changed = service.dedupe_trade_records_by_broker_order()
        _assert(changed == 0, f"跨日成交不应被按委托号删掉，实际 changed={changed}")
        _assert(_count_trades(service) == 2, "去重后仍应保留两笔跨日成交")
        print("[dedupe_keeps_cross_day] OK")

        for i in range(5):
            added = service.sync_from_order_records([later_order])
            changed = service.dedupe_trade_records_by_broker_order()
            _assert(added == 0, f"第 {i + 1} 次回填不应再新增，实际 added={added}")
            _assert(changed == 0, f"第 {i + 1} 次去重不应再改动，实际 changed={changed}")
            _assert(_count_trades(service) == 2, "循环回填/去重后记录数应稳定为 2")
        print("[loop_stable] OK")

        extra = service.add_record(
            stock_code=CODE,
            stock_name="黄金ETF",
            direction="buy",
            price=8.950,
            volume=3100,
            broker_order_id=ORDER_ID,
            trade_date="2026-08-19",
            source="broker_sync",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            remark="委托号:940572673 重复回填",
            emit_signals=False,
        )
        _assert(extra is not None and extra.id != 0, "同日重复记录应先能写入")
        _assert(_count_trades(service) == 3, "同日重复写入后应有 3 条")
        changed = service.dedupe_trade_records_by_broker_order()
        _assert(changed > 0, "同一交易日的重复成交应被清理")
        _assert(_count_trades(service) == 2, "同日去重后仍应只留跨日两笔")
        _assert(_trade_dates(service) == ["2026-07-03", "2026-08-19"], "同日去重不应删掉另一天的成交")
        print("[same_day_dedupe] OK")

    print("trade_record_order_id_reuse_smoketest: ALL PASSED")


if __name__ == "__main__":
    main()
