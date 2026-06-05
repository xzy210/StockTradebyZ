from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.execution_contract import OrderExecutionReport, OrderIntent, RebalanceIntent, StrategySignal, TargetPortfolio
from common.strategy_spec import StrategySpec
from trading_app.services.live_strategy_center.strategy_adapter import PanelLiveStrategyAdapter
from trading_app.services.live_strategy_center.hub_controller import LiveStrategyHubController
from trading_app.services.live_strategy_center.strategy_plugin import LiveStrategyPlugin
from trading_app.services.strategy.strategy_budget_service import StrategyBudgetService
from trading_app.services.strategy.strategy_registry_service import StrategyRegistryService, get_strategy_registry_service
from trading_app.services.strategy import strategy_spec_service
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    research_spec = StrategySpec(
        strategy_id="research_smoke",
        strategy_name="Research Smoke",
        universe=["510300.SH", " 510880 ", ""],
        metadata={"source": "strategy_app"},
    )
    _assert(research_spec.virtual_account_id == "va_research_smoke", "common StrategySpec virtual account fallback mismatch")
    _assert(research_spec.universe == ["510300", "510880"], "common StrategySpec universe normalization mismatch")
    _assert(not hasattr(strategy_spec_service, "StrategySpec"), "legacy StrategySpec export should be removed from live service")

    spec_service = get_strategy_spec_service()
    ai_spec = spec_service.ai_stock()
    etf_spec = spec_service.etf_rotation()
    unmanaged_spec = spec_service.unmanaged()

    _assert(ai_spec.strategy_id, "AI strategy_id should not be empty")
    _assert(ai_spec.virtual_account_id, "AI virtual_account_id should not be empty")
    _assert(etf_spec.strategy_id == "etf_rotation", "ETF live strategy_id should align to common etf_rotation id")
    _assert(etf_spec.virtual_account_id == "va_etf_rotation", "ETF virtual_account_id should align to common etf_rotation account")
    _assert(unmanaged_spec.is_unmanaged, "unmanaged spec should be marked as unmanaged")
    _assert(unmanaged_spec.enabled is False, "unmanaged spec should not be tradable")

    from strategy_app.strategies import create_strategy, get_all_strategies, normalize_strategy_id
    from strategy_app.strategies.etf_three_factor_momentum_strategy_fast import ETFThreeFactorMomentumStrategyFast

    unified_registry = get_strategy_registry_service()
    _assert(not hasattr(__import__("strategy_app.strategies", fromlist=["STRATEGIES"]), "STRATEGIES"), "STRATEGIES registry export should be removed")
    _assert(normalize_strategy_id("etf_rotation") == "etf_rotation", "ETF strategy_id should normalize to itself")
    _assert(unified_registry.get_strategy_class("etf_rotation") is ETFThreeFactorMomentumStrategyFast, "unified registry class mismatch")
    _assert(create_strategy("etf_rotation").strategy_id == etf_spec.strategy_id, "research ETF strategy_id should match live spec")
    _assert(unified_registry.create_strategy("etf_rotation").strategy_id == etf_spec.strategy_id, "unified registry should create ETF strategy")
    _assert(get_all_strategies()["etf_rotation"] == ETFThreeFactorMomentumStrategyFast.spec.strategy_name, "strategy labels should come from registry spec")
    _assert(unified_registry.get_strategy_labels()["etf_rotation"] == ETFThreeFactorMomentumStrategyFast.spec.strategy_name, "unified labels should come from registry spec")

    with tempfile.TemporaryDirectory(prefix="strategy_spec_smoketest_") as tmpdir:
        tmp = Path(tmpdir)
        budget = StrategyBudgetService(
            config_path=tmp / "strategy_budget_config.json",
            state_path=tmp / "strategy_budget_state.json",
        )
        ai_snapshot = budget.get_strategy_snapshot(
            ai_spec.strategy_id,
            strategy_name=ai_spec.strategy_name,
            virtual_account_id=ai_spec.virtual_account_id,
        )
        etf_snapshot = budget.get_strategy_snapshot(
            etf_spec.strategy_id,
            strategy_name=etf_spec.strategy_name,
            virtual_account_id=etf_spec.virtual_account_id,
        )
        unmanaged_snapshot = budget.get_strategy_snapshot(
            unmanaged_spec.strategy_id,
            strategy_name=unmanaged_spec.strategy_name,
            virtual_account_id=unmanaged_spec.virtual_account_id,
        )
        _assert(ai_snapshot["virtual_account_id"] == ai_spec.virtual_account_id, "AI budget identity mismatch")
        _assert(etf_snapshot["strategy_name"] == etf_spec.strategy_name, "ETF budget identity mismatch")
        _assert(unmanaged_snapshot["is_unmanaged"], "unmanaged budget config mismatch")

        registry = StrategyRegistryService(path=tmp / "strategy_symbol_ownership.json")
        for code in etf_spec.universe:
            owner = registry.get_owner(code)
            _assert(owner is not None, f"{code} should have default ETF ownership")
            _assert(owner.strategy_id == etf_spec.strategy_id, f"{code} owner strategy_id mismatch")
            _assert(owner.virtual_account_id == etf_spec.virtual_account_id, f"{code} owner virtual_account_id mismatch")
            _assert(owner.owner_type == etf_spec.owner_type, f"{code} owner_type mismatch")

    plugin = LiveStrategyPlugin(
        plugin_id=ai_spec.plugin_id,
        plugin_name=ai_spec.plugin_name,
        tab_key=ai_spec.plugin_tab_key,
        tab_title=ai_spec.plugin_tab_title,
        metadata=ai_spec.to_plugin_metadata(),
    )
    _assert(plugin.metadata["strategy_id"] == ai_spec.strategy_id, "plugin metadata strategy_id mismatch")
    _assert(plugin.metadata["virtual_account_id"] == ai_spec.virtual_account_id, "plugin metadata virtual_account_id mismatch")

    class SignalPanel:
        def __init__(self) -> None:
            self.executed: list[StrategySignal] = []

        def generate_live_signals(self, payload=None):
            return [
                StrategySignal(
                    symbol="510880",
                    action="buy",
                    target_quantity=100,
                    price=1.23,
                    reason=str((payload or {}).get("reason", "adapter smoke")),
                )
            ]

        def execute_live_signals(self, signals, *, execution_service=None, stock_name_map=None):
            self.executed = list(signals or [])
            return execution_service.execute_signals(signals, stock_name_map=stock_name_map or {})

    class FakeExecutionService:
        def __init__(self) -> None:
            self.received: list[StrategySignal] = []
            self.received_order_intents: list[OrderIntent] = []
            self.received_rebalance_intent: RebalanceIntent | None = None

        def execute_signals(self, signals, *, stock_name_map=None):
            self.received = list(signals or [])
            return [
                OrderExecutionReport(
                    intent=None,
                    accepted=True,
                    status="submitted",
                    message="adapter smoke executed",
                    execution_mode="live",
                )
            ]

        def execute_order_intents(self, intents, *, stock_name_map=None):
            self.received_order_intents = list(intents or [])
            return [
                OrderExecutionReport(
                    intent=self.received_order_intents[0],
                    accepted=True,
                    status="submitted",
                    message="order intents executed",
                    execution_mode="live",
                )
            ]

        def execute_rebalance_intent(self, rebalance_intent, *, stock_name_map=None):
            self.received_rebalance_intent = rebalance_intent
            self.received_order_intents = list(rebalance_intent.order_intents or [])
            return [
                OrderExecutionReport(
                    intent=self.received_order_intents[0],
                    accepted=True,
                    status="submitted",
                    message="rebalance intent executed",
                    execution_mode="live",
                )
            ]

    signal_panel = SignalPanel()
    adapter = PanelLiveStrategyAdapter.from_panel(
        signal_panel,
        strategy_id=etf_spec.strategy_id,
        strategy_name=etf_spec.strategy_name,
        virtual_account_id=etf_spec.virtual_account_id,
    )
    signals = adapter.generate_live_signals({"reason": "center smoke"})
    _assert(len(signals) == 1, "adapter should generate one signal")
    _assert(signals[0].strategy_id == etf_spec.strategy_id, "adapter should inject strategy_id")
    _assert(signals[0].metadata["virtual_account_id"] == etf_spec.virtual_account_id, "adapter should inject virtual account")
    fake_execution = FakeExecutionService()
    reports = adapter.execute_live_signals(signals, execution_service=fake_execution)
    _assert(len(reports) == 1 and reports[0].accepted, "adapter execution report mismatch")
    _assert(fake_execution.received[0].metadata["source"] == "live_strategy_center", "adapter source metadata mismatch")
    _assert(signal_panel.executed[0].metadata["trigger"] == "strategy_center", "adapter should delegate panel execution")

    class RebalancePanel:
        def __init__(self) -> None:
            self.executed_rebalance: RebalanceIntent | None = None

        def generate_live_rebalance_intent(self, payload=None):
            target = TargetPortfolio.single_asset(
                symbol="510880",
                weight=1.0,
                reason=str((payload or {}).get("reason", "rebalance smoke")),
            )
            return RebalanceIntent(
                target_portfolio=target,
                order_intents=(
                    OrderIntent(
                        symbol="510880",
                        side="buy",
                        quantity=100,
                        price=1.23,
                        reason=target.reason,
                    ),
                ),
                reason=target.reason,
            )

        def execute_live_rebalance_intent(self, rebalance_intent, *, execution_service=None, stock_name_map=None):
            self.executed_rebalance = rebalance_intent
            return execution_service.execute_rebalance_intent(rebalance_intent, stock_name_map=stock_name_map or {})

    rebalance_panel = RebalancePanel()
    rebalance_adapter = PanelLiveStrategyAdapter.from_panel(
        rebalance_panel,
        strategy_id=etf_spec.strategy_id,
        strategy_name=etf_spec.strategy_name,
        virtual_account_id=etf_spec.virtual_account_id,
    )
    rebalance_intent = rebalance_adapter.generate_live_rebalance_intent({"reason": "native rebalance"})
    _assert(isinstance(rebalance_intent, RebalanceIntent), "adapter should generate native rebalance intent")
    _assert(rebalance_intent.target_portfolio.strategy_id == etf_spec.strategy_id, "rebalance target should receive strategy_id")
    _assert(rebalance_intent.order_intents[0].virtual_account_id == etf_spec.virtual_account_id, "order intent should receive virtual account")
    native_execution = FakeExecutionService()
    native_reports = rebalance_adapter.execute_live_rebalance_intent(rebalance_intent, execution_service=native_execution)
    _assert(len(native_reports) == 1 and native_reports[0].accepted, "native rebalance execution report mismatch")
    _assert(native_execution.received_rebalance_intent is not None, "execution service should receive rebalance intent")
    _assert(rebalance_panel.executed_rebalance is not None, "adapter should delegate native rebalance execution")

    class FakeTaskService:
        def register_task(self, *args, **kwargs):
            return None

    class FakeHubStateService:
        def refresh_state(self):
            return None

    controller = LiveStrategyHubController(
        task_service=FakeTaskService(),
        hub_state_service=FakeHubStateService(),
        eod_service=None,
        strategy_adapters=[rebalance_adapter],
    )
    controller_execution = FakeExecutionService()
    controller_reports = controller.execute_strategy_signals(
        etf_spec.strategy_id,
        payload={"reason": "controller native"},
        execution_service=controller_execution,
    )
    _assert(len(controller_reports) == 1 and controller_reports[0].accepted, "controller should execute native rebalance intent")
    _assert(controller_execution.received_rebalance_intent is not None, "controller should prefer rebalance intent over signals")

    class OrderIntentPanel:
        def generate_live_order_intents(self, payload=None):
            return [
                OrderIntent(
                    symbol="159949",
                    side="sell",
                    quantity=200,
                    price=2.34,
                    reason=str((payload or {}).get("reason", "order smoke")),
                )
            ]

    order_adapter = PanelLiveStrategyAdapter.from_panel(
        OrderIntentPanel(),
        strategy_id=etf_spec.strategy_id,
        strategy_name=etf_spec.strategy_name,
        virtual_account_id=etf_spec.virtual_account_id,
    )
    order_execution = FakeExecutionService()
    order_reports = order_adapter.execute_live_order_intents(
        order_adapter.generate_live_order_intents({"reason": "native orders"}),
        execution_service=order_execution,
    )
    _assert(len(order_reports) == 1 and order_reports[0].accepted, "native order intent execution report mismatch")
    _assert(order_execution.received_order_intents[0].strategy_id == etf_spec.strategy_id, "order intent should receive strategy identity")

    print("STRATEGY_SPEC_SMOKE= ok")


if __name__ == "__main__":
    main()
