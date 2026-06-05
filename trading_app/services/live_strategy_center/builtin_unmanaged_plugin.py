from __future__ import annotations

from typing import Any, Callable

from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service

from .strategy_plugin import LiveStrategyPlugin, LiveStrategyPortfolioProvider, LiveStrategyTaskSpec


class UnmanagedSystemAccountPortfolioPlugin:
    """Portfolio provider callbacks for the built-in unmanaged system account."""

    def __init__(self) -> None:
        self.spec = get_strategy_spec_service().unmanaged()

    def create_provider(self, *, order: int = 90) -> LiveStrategyPortfolioProvider:
        return LiveStrategyPortfolioProvider(
            strategy_id=self.spec.strategy_id,
            strategy_name=self.spec.strategy_name,
            account_row_provider=self.build_account_row,
            position_rows_provider=self.build_position_rows,
            order=order,
        )

    def build_account_row(self, service: Any, broker_live_positions: list[dict] | None = None) -> dict:
        try:
            account = service.strategy_budget.build_account_snapshot(
                self.spec.strategy_id,
                strategy_name=self.spec.strategy_name,
                virtual_account_id=self.spec.virtual_account_id,
                live_positions=broker_live_positions or None,
            )
        except Exception:
            return {}
        if not account:
            return {}
        account["strategy_name"] = account.get("strategy_name") or self.spec.strategy_name
        account["is_unmanaged"] = True
        return account

    def build_position_rows(self, service: Any, broker_live_positions: list[dict] | None = None) -> list[dict]:
        try:
            rows = service.strategy_budget.get_positions_view(
                self.spec.strategy_id,
                strategy_name=self.spec.strategy_name,
                virtual_account_id=self.spec.virtual_account_id,
                live_positions=broker_live_positions or None,
            )
        except Exception:
            return []
        for row in rows:
            row["strategy_name"] = self.spec.strategy_name
            row["is_unmanaged"] = True
        return rows


def create_unmanaged_system_account_provider(*, order: int = 90) -> LiveStrategyPortfolioProvider:
    return UnmanagedSystemAccountPortfolioPlugin().create_provider(order=order)


UNMANAGED_REVIEW_PLUGIN_NAME = "未管理持仓"
UNMANAGED_REVIEW_TASK_TYPE = "review"
UNMANAGED_REVIEW_TASK_KEY = "daily_unmanaged_position_scan"


def create_unmanaged_position_review_plugin(
    widget: object,
    *,
    tab_key: str,
    task_provider: Callable[[], dict],
    run_scan_action: Callable[[], Any],
    order: int = 20,
) -> LiveStrategyPlugin:
    spec = get_strategy_spec_service().unmanaged()
    return LiveStrategyPlugin(
        plugin_id=spec.plugin_id,
        plugin_name=UNMANAGED_REVIEW_PLUGIN_NAME,
        widget=widget,
        tab_key=tab_key,
        tab_title=spec.plugin_tab_title or "未管理持仓",
        task_specs=(
            LiveStrategyTaskSpec(
                task_key=UNMANAGED_REVIEW_TASK_KEY,
                task_type=UNMANAGED_REVIEW_TASK_TYPE,
                title="未管理持仓 AI 巡检",
                provider=task_provider,
                strategy_id=spec.strategy_id,
                strategy_name=UNMANAGED_REVIEW_PLUGIN_NAME,
                actions={"立即执行": run_scan_action},
                order=10,
            ),
        ),
        portfolio_providers=(
            create_unmanaged_system_account_provider(order=90),
        ),
        metadata=spec.to_plugin_metadata(),
        order=order,
    )
