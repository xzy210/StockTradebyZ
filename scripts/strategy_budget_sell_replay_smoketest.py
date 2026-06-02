"""Smoke test for strategy budget sell replay.

Run::

    conda run -n stock python scripts/strategy_budget_sell_replay_smoketest.py
"""
from __future__ import annotations

import sys
import sqlite3
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import trading_app.services.execution.trade_record_service as trade_record_module
from trading_app.services.strategy.strategy_budget_service import StrategyBudgetService
from trading_app.services.strategy_constants import AI_STOCK_STRATEGY_ID, AI_STOCK_VIRTUAL_ACCOUNT_ID
from trading_app.services.trade_record_service import TradeDirection, TradeRecordService


STRATEGY_ID = AI_STOCK_STRATEGY_ID
VIRTUAL_ACCOUNT_ID = AI_STOCK_VIRTUAL_ACCOUNT_ID
CODE = "002436"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="strategy_budget_sell_replay_") as tmpdir:
        tmp = Path(tmpdir)
        trade_service = TradeRecordService()
        trade_service.db_path = tmp / "trade_records.db"
        trade_service._init_database()  # noqa: SLF001
        trade_service._init_pnl_table()  # noqa: SLF001
        trade_service._init_position_snapshot_table()  # noqa: SLF001
        trade_service._init_strategy_snapshot_tables()  # noqa: SLF001

        old_trade_service = trade_record_module._trade_record_service
        trade_record_module._trade_record_service = trade_service
        try:
            budget = StrategyBudgetService(
                config_path=tmp / "strategy_budget_config.json",
                state_path=tmp / "strategy_budget_state.json",
            )
            budget.upsert_strategy_config(
                strategy_id=STRATEGY_ID,
                strategy_name="AI实盘决策",
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
                capital_limit=100000.0,
            )

            trade_service.add_record(
                stock_code=CODE,
                stock_name="兴森科技",
                direction=TradeDirection.BUY.value,
                price=10.0,
                volume=1000,
                trade_date="2026-05-20",
                strategy_id=STRATEGY_ID,
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
                commission=0.0,
                stamp_tax=0.0,
                transfer_fee=0.0,
            )
            trade_service.add_record(
                stock_code=CODE,
                stock_name="兴森科技",
                direction=TradeDirection.BUY.value,
                price=12.0,
                volume=1000,
                trade_date="2026-05-30",
                strategy_id=STRATEGY_ID,
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
                commission=0.0,
                stamp_tax=0.0,
                transfer_fee=0.0,
            )
            trade_service.add_record(
                stock_code=CODE,
                stock_name="兴森科技",
                direction=TradeDirection.SELL.value,
                price=14.0,
                volume=500,
                trade_date="2026-06-01",
                strategy_id=STRATEGY_ID,
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
                commission=0.0,
                stamp_tax=0.0,
                transfer_fee=0.0,
            )

            state = budget.rebuild_strategy_state_from_trade_records(
                STRATEGY_ID,
                strategy_name="AI实盘决策",
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
            )
            position = state.get_positions()[CODE]
            _assert(position.quantity == 1500, "卖出后剩余持仓数量应为 1500")
            _assert(abs(position.avg_cost - 11.0) < 0.0001, "卖出后不应改变持仓成本价")
            _assert(abs(state.realized_pnl - 1500.0) < 0.0001, "卖出收益应进入已实现盈亏")

            budget.sync_strategy_positions(
                strategy_id=STRATEGY_ID,
                strategy_name="AI实盘决策",
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
                positions=[
                    {
                        "stock_code": CODE,
                        "volume": 1500,
                        "open_price": round(22000.0 / 1500, 4),
                    }
                ],
            )
            synced = budget.get_strategy_state_record(
                STRATEGY_ID,
                strategy_name="AI实盘决策",
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
            )
            synced_position = synced.get_positions()[CODE]
            _assert(abs(synced_position.avg_cost - 11.0) < 0.0001, "同步券商持仓不应覆盖本地成本价")

            synced.positions[CODE]["avg_cost"] = 31.3904
            synced.realized_pnl = 0.0
            budget.save_strategy_state_record(synced)
            account = budget.build_account_snapshot(
                STRATEGY_ID,
                strategy_name="AI实盘决策",
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
                live_positions=[
                    {
                        "stock_code": CODE,
                        "volume": 1500,
                        "market_value": 21000.0,
                    }
                ],
            )
            refreshed = budget.get_strategy_state_record(
                STRATEGY_ID,
                strategy_name="AI实盘决策",
                virtual_account_id=VIRTUAL_ACCOUNT_ID,
            )
            refreshed_position = refreshed.get_positions()[CODE]
            _assert(abs(refreshed_position.avg_cost - 11.0) < 0.0001, "账户快照应自动修复被污染的成本价")
            _assert(abs(float(account.get("realized_pnl", 0.0) or 0.0) - 1500.0) < 0.0001, "账户快照应使用回放后的已实现盈亏")
            conn = sqlite3.connect(trade_service.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT cost_price, cost_amount, open_price
                FROM strategy_position_snapshots
                WHERE strategy_id = ? AND stock_code = ?
                ORDER BY snapshot_date DESC, id DESC
                LIMIT 1
                """,
                (STRATEGY_ID, CODE),
            ).fetchone()
            conn.close()
            _assert(row is not None, "数据库应记录策略持仓成本快照")
            _assert(abs(float(row["cost_price"] or 0.0) - 11.0) < 0.0001, "数据库 cost_price 应来自账本成本")
            _assert(abs(float(row["cost_amount"] or 0.0) - 16500.0) < 0.0001, "数据库 cost_amount 应等于账本成本金额")
            _assert(abs(float(row["open_price"] or 0.0) - 11.0) < 0.0001, "兼容字段 open_price 应等于账本成本")
            print("ALL_PASSED")
        finally:
            trade_record_module._trade_record_service = old_trade_service


if __name__ == "__main__":
    main()
