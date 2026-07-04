"""Smoke test for unmanaged ownership reconcile flow.

Scenarios:
  1. broker 持仓落入 unmanaged -> 自动补登记 unmanaged 归属
  2. 股票不再属于 unmanaged   -> 自动释放 unmanaged 归属

Run::

    conda run -n stock python scripts/unmanaged_reconcile_smoketest.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.services.strategy.strategy_budget_service import (
    StrategyBudgetService,
    StrategyPositionState,
)
from trading_app.services.strategy.strategy_constants import (
    OWNER_TYPE_UNMANAGED,
    UNMANAGED_STRATEGY_ID,
)
from trading_app.services.strategy.strategy_registry_service import StrategyRegistryService
import trading_app.services.strategy.strategy_registry_service as registry_module

TEST_CODE = "600816"


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="unmanaged_smoketest_") as tmpdir:
        tmp = Path(tmpdir)
        budget = StrategyBudgetService(
            config_path=tmp / "strategy_budget_config.json",
            state_path=tmp / "strategy_budget_state.json",
        )
        registry = StrategyRegistryService(path=tmp / "strategy_symbol_ownership.json")

        old_registry_singleton = registry_module._strategy_registry_service
        registry_module._strategy_registry_service = registry
        try:
            summary = budget.reconcile_unmanaged_with_broker(
                broker_cash=100000.0,
                broker_positions=[
                    {"stock_code": TEST_CODE, "volume": 6000, "open_price": 2.55},
                ],
            )
            owner = registry.get_owner(TEST_CODE)
            _assert(owner is not None, f"{TEST_CODE} 应该被登记归属到 unmanaged")
            _assert(owner.strategy_id == UNMANAGED_STRATEGY_ID, f"{TEST_CODE} 应归属于 unmanaged")
            _assert(owner.owner_type == OWNER_TYPE_UNMANAGED, f"{TEST_CODE} 归属类型应为 unmanaged")
            _assert(TEST_CODE in summary.get("claimed_unmanaged_codes", []), "应记录补登记摘要")
            unmanaged_state = budget._states[UNMANAGED_STRATEGY_ID]  # noqa: SLF001
            _assert(TEST_CODE in unmanaged_state.get_positions(), f"unmanaged 应持有 {TEST_CODE}")
            print("[claim_unmanaged] OK")

            budget.upsert_strategy_config(
                strategy_id="managed_live_case",
                strategy_name="ManagedLiveCase",
                virtual_account_id="va_managed_live_case",
                capital_limit=20000.0,
                enabled=True,
                is_test=False,
                hidden=False,
                is_unmanaged=False,
            )
            managed_state = budget.get_strategy_state_record(
                "managed_live_case",
                strategy_name="ManagedLiveCase",
                virtual_account_id="va_managed_live_case",
            )
            managed_state.positions = {
                TEST_CODE: StrategyPositionState(
                    symbol_code=TEST_CODE,
                    quantity=6000,
                    avg_cost=2.55,
                ).to_dict()
            }
            budget._states["managed_live_case"] = managed_state  # noqa: SLF001
            budget._save_states()  # noqa: SLF001

            summary = budget.reconcile_unmanaged_with_broker(
                broker_cash=100000.0,
                broker_positions=[
                    {"stock_code": TEST_CODE, "volume": 6000, "open_price": 2.55},
                ],
            )
            owner = registry.get_owner(TEST_CODE)
            _assert(owner is None, f"{TEST_CODE} 不再属于 unmanaged 时，应自动释放 unmanaged 归属")
            _assert(
                TEST_CODE in summary.get("released_unmanaged_codes", []),
                "应记录自动迁出摘要",
            )
            unmanaged_state = budget._states[UNMANAGED_STRATEGY_ID]  # noqa: SLF001
            _assert(TEST_CODE not in unmanaged_state.get_positions(), f"unmanaged 不应再持有 {TEST_CODE}")
            print("[release_unmanaged] OK")

            managed_state.cash_balance = 200.0
            budget._states["managed_live_case"] = managed_state  # noqa: SLF001
            budget._save_states()  # noqa: SLF001
            budget_logger = logging.getLogger("trading_app.services.strategy.strategy_budget_service")
            handler = _ListHandler()
            old_level = budget_logger.level
            budget_logger.setLevel(logging.WARNING)
            budget_logger.addHandler(handler)
            try:
                for _ in range(2):
                    summary = budget.reconcile_unmanaged_with_broker(
                        broker_cash=100.0,
                        broker_positions=[
                            {"stock_code": TEST_CODE, "volume": 5900, "open_price": 2.55},
                        ],
                    )
            finally:
                budget_logger.removeHandler(handler)
                budget_logger.setLevel(old_level)
            messages = [record.getMessage() for record in handler.records]
            _assert(summary.get("cash_shortfall") == -100.0, "应返回现金短缺摘要")
            _assert(summary.get("position_shortfalls"), "应返回持仓短缺摘要")
            _assert(
                sum("已认领现金超过券商实际现金" in message for message in messages) == 1,
                "重复现金短缺 warning 应被冷却",
            )
            _assert(
                sum("策略声明持仓超过券商实际" in message for message in messages) == 1,
                "重复持仓短缺 warning 应被冷却",
            )
            print("[warning_cooldown] OK")

            print("ALL_PASSED")
        finally:
            registry_module._strategy_registry_service = old_registry_singleton


if __name__ == "__main__":
    main()
