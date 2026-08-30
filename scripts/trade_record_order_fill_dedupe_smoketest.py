"""Smoke test: 委托号成交与成交号回填不能重复记账。

复现 2026-08-30 虚报持仓：
  - 7 月 31 日 AI 下单写入 601168 买入 100 股，备注 委托号:2014314497
  - 次日券商同步带回同一笔成交，委托号为 0，备注 成交号:70364083
  - 账本回放后持仓变成 200，对账显示虚报 100 股

Run::

    conda run -n stock python scripts/trade_record_order_fill_dedupe_smoketest.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.services.execution.trade_record_service import TradeRecordService


STRATEGY_ID = "ai_trade_decision_center"
VIRTUAL_ACCOUNT_ID = "va_ai_trade_decision_center"
CODE = "601168"
ORDER_ID = 2014314497
TRADE_ID = 70364083


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(service: TradeRecordService, code: str = CODE) -> int:
    conn = service._get_connection()  # noqa: SLF001
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades WHERE stock_code = ?", (code,))
        return int(cursor.fetchone()[0] or 0)
    finally:
        conn.close()


def _row(service: TradeRecordService, code: str = CODE) -> dict:
    conn = service._get_connection()  # noqa: SLF001
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, broker_order_id, remark, volume, stock_name
            FROM trades WHERE stock_code = ? ORDER BY id
            """,
            (code,),
        )
        fetched = cursor.fetchall()
        _assert(len(fetched) == 1, f"{code} 应只剩 1 条，实际 {len(fetched)}")
        return dict(fetched[0])
    finally:
        conn.close()


def _insert_raw(service: TradeRecordService, **kwargs) -> None:
    conn = sqlite3.connect(str(service.db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (
                trade_id, broker_order_id, stock_code, stock_name, direction,
                price, volume, amount, commission, stamp_tax, transfer_fee,
                trade_date, source, strategy_id, virtual_account_id, intent_id,
                remark, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kwargs.get("trade_id") or f"{kwargs['stock_code']}_{kwargs.get('remark', '')}_{kwargs.get('created_at', '')}",
                kwargs.get("broker_order_id", 0),
                kwargs["stock_code"],
                kwargs.get("stock_name", kwargs["stock_code"]),
                kwargs.get("direction", "buy"),
                kwargs.get("price", 38.51),
                kwargs.get("volume", 100),
                kwargs.get("amount", 3851.0),
                kwargs.get("commission", 5.0),
                kwargs.get("stamp_tax", 0.0),
                kwargs.get("transfer_fee", 0.04),
                kwargs.get("trade_date", "2026-07-31"),
                kwargs.get("source", "broker_sync"),
                kwargs.get("strategy_id", STRATEGY_ID),
                kwargs.get("virtual_account_id", VIRTUAL_ACCOUNT_ID),
                kwargs.get("intent_id", ""),
                kwargs.get("remark", ""),
                kwargs.get("created_at", "2026-07-31 14:28:41"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="trade_order_fill_dedupe_") as tmpdir:
        service = TradeRecordService()
        service.db_path = Path(tmpdir) / "trade_records.db"
        service._init_database()  # noqa: SLF001

        first = service.add_record(
            stock_code=CODE,
            stock_name="西部矿业",
            direction="buy",
            price=38.51,
            volume=100,
            broker_order_id=ORDER_ID,
            trade_date="2026-07-31",
            source="ai_agent",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            remark=f"委托号:{ORDER_ID}",
            emit_signals=False,
        )
        _assert(first is not None, "应写入委托成交")

        second = service.add_record(
            stock_code=CODE,
            stock_name=CODE,
            direction="buy",
            price=38.51,
            volume=100,
            broker_order_id=0,
            trade_date="2026-07-31",
            source="broker_sync",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            remark=f"成交号:{TRADE_ID}",
            emit_signals=False,
        )
        _assert(second is not None, "成交号回填应合并到原记录")
        _assert(int(second.id or 0) == int(first.id or 0), "回填不应新增第二条")
        _assert(_count(service) == 1, "委托号+成交号回填后仍应只有 1 条")
        merged = _row(service)
        _assert(f"委托号:{ORDER_ID}" in str(merged["remark"]), f"应保留委托号: {merged['remark']}")
        _assert(f"成交号:{TRADE_ID}" in str(merged["remark"]), f"应补上成交号: {merged['remark']}")
        _assert(int(merged["broker_order_id"] or 0) == ORDER_ID, "合并后应保留委托号")
        print("[add_record_merge] OK")

        leftover_code = "600000"
        _insert_raw(
            service,
            stock_code=leftover_code,
            stock_name="浦发银行",
            broker_order_id=ORDER_ID,
            remark=f"委托号:{ORDER_ID}",
            source="ai_agent",
            created_at="2026-07-31 14:28:41",
        )
        _insert_raw(
            service,
            stock_code=leftover_code,
            stock_name=leftover_code,
            broker_order_id=0,
            remark=f"成交号:{TRADE_ID}",
            source="broker_sync",
            created_at="2026-08-01 14:28:21",
        )
        _assert(_count(service, leftover_code) == 2, "历史重复记录应先能写入 2 条")
        changed = service.dedupe_trade_records_by_broker_order()
        _assert(changed > 0, "历史委托/成交重复应被清理")
        leftover = _row(service, leftover_code)
        _assert(_count(service, leftover_code) == 1, "去重后应只留 1 条")
        _assert(f"委托号:{ORDER_ID}" in str(leftover["remark"]), f"去重后应保留委托号: {leftover['remark']}")
        _assert(f"成交号:{TRADE_ID}" in str(leftover["remark"]), f"去重后应保留成交号: {leftover['remark']}")
        print("[dedupe_leftover_pair] OK")

        twin_code = "600519"
        first_twin = service.add_record(
            stock_code=twin_code,
            stock_name="贵州茅台",
            direction="buy",
            price=1500.0,
            volume=100,
            broker_order_id=111,
            trade_date="2026-07-31",
            source="ai_agent",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            remark="委托号:111",
            emit_signals=False,
        )
        second_twin = service.add_record(
            stock_code=twin_code,
            stock_name="贵州茅台",
            direction="buy",
            price=1500.0,
            volume=100,
            broker_order_id=222,
            trade_date="2026-07-31",
            source="ai_agent",
            strategy_id=STRATEGY_ID,
            virtual_account_id=VIRTUAL_ACCOUNT_ID,
            remark="委托号:222",
            emit_signals=False,
        )
        _assert(first_twin is not None and second_twin is not None, "两笔真实同价买入都应写入")
        _assert(int(first_twin.id or 0) != int(second_twin.id or 0), "两笔真实委托不应被合并")
        _assert(_count(service, twin_code) == 2, "两笔真实同价买入应保留 2 条")
        changed = service.dedupe_trade_records_by_broker_order()
        _assert(_count(service, twin_code) == 2, f"两笔真实委托去重后仍应是 2 条，changed={changed}")
        print("[keep_two_real_orders] OK")

    print("trade_record_order_fill_dedupe_smoketest: ALL PASSED")


if __name__ == "__main__":
    main()
