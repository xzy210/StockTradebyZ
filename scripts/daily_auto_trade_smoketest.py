from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.agent.agent_context_service import BrokerContext
from trading_app.services.ai.auto_trade_config_service import AutoTradeConfig
from trading_app.services.ai.daily_auto_trade_service import DailyAutoTradeService
from trading_app.services.ops.live_strategy_end_of_day_service import LiveStrategyEndOfDayService, StrategyEndOfDayResult
from trading_app.services.ai.trade_decision_models import RiskCheckResult, TradeDecision
from trading_app.services.execution.trade_execution_service import TradeExecutionService


class FakeBroker:
    def __init__(self):
        self.is_connected = True
        self.order_calls = 0

    def query_stock_asset(self):
        return SimpleNamespace(cash=500000.0, total_asset=1000000.0)

    def query_stock_positions(self):
        return []

    def query_stock_orders(self):
        return []

    def order_stock(self, **kwargs):
        self.order_calls += 1
        raise RuntimeError("shadow 模式不应触发真实下单")


class FakePortfolioService:
    def __init__(self):
        self.calls = 0

    def finalize_day_snapshots(self, *, remark: str = ""):
        self.calls += 1
        return [
            {
                "strategy_id": "demo_strategy",
                "strategy_name": "Demo Strategy",
                "success": True,
                "remark": remark,
            }
        ]


def _assert_unified_end_of_day_flow() -> None:
    broker = FakeBroker()
    service = DailyAutoTradeService(broker_service=broker)
    state_path = PROJECT_ROOT / "trading_app" / "data" / f"daily_auto_trade_state_smoketest_{int(time.time() * 1000)}.json"
    service._state_path = state_path
    service._reconcile_timer.stop()
    service.config_service.get_config = lambda: AutoTradeConfig(auto_reconcile_enabled=True)
    service.run_end_of_day_reconcile = lambda *, slot="manual", snapshot_date=None: (True, "共享日终对账完成，委托 1，成交 1")

    portfolio = FakePortfolioService()
    eod = LiveStrategyEndOfDayService(
        daily_auto_trade=service,
        portfolio_service=portfolio,
        rotation_etf_pool=[],
    )
    eod._run_kline_full_refresh = lambda: (True, "K线刷新完成")
    automation_events = []
    eod.set_automation_controls(
        pause=lambda: automation_events.append("pause") or "中心自动化已暂停",
        resume=lambda: automation_events.append("resume") or "中心自动化已恢复",
    )
    eod.register_strategy(
        "demo_strategy",
        "Demo Strategy",
        lambda snapshot_date: StrategyEndOfDayResult(
            strategy_id="demo_strategy",
            strategy_name="Demo Strategy",
            success=True,
            message=f"策略日终完成 {snapshot_date}",
        ),
    )

    try:
        success, message, payload = eod.run_manual_cycle()
        assert success, message
        assert payload["workflow_version"] == "unified_v1"
        assert payload["pause_automation"]["success"]
        assert payload["shared_reconcile"]["success"]
        assert payload["portfolio_snapshots"]["success"]
        assert payload["portfolio_snapshots"]["details"]["snapshot_count"] == 1
        assert payload["kline_refresh"]["success"]
        assert payload["next_day_unlock"]["success"]
        assert payload["strategy_results"]["demo_strategy"]["success"]
        assert portfolio.calls == 1
        assert automation_events == ["pause", "resume"]
        cycle_state = service.get_day_state_section(eod._CYCLE_STATE_KEY)
        assert cycle_state["status"] == "completed"
        phases = cycle_state["phases"]
        for key in eod._UNIFIED_PHASE_KEYS:
            assert phases[key]["status"] == "completed", key
    finally:
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass


def main():
    broker = FakeBroker()
    service = DailyAutoTradeService(broker_service=broker)
    execution_service = TradeExecutionService(broker)
    execution_service.config_service.get_config = lambda: AutoTradeConfig(
        manual_orders_enabled=True,
        auto_trade_mode="shadow",
        require_trading_time=False,
        duplicate_window_seconds=1,
        status_poll_seconds=1.0,
        status_poll_interval_seconds=0.2,
        max_new_positions_per_day=2,
        max_buy_orders_per_day=2,
        max_sell_orders_per_day=2,
        reserve_cash_pct=0.2,
        max_intraday_failures=2,
        max_daily_loss_pct=0.5,
        auto_reconcile_enabled=False,
        reconcile_time="15:10",
        reconcile_retry_time="15:20",
    )
    service.execution_service = execution_service
    service._check_previous_snapshot_guard = lambda: ""
    task_id = f"test_task_{int(time.time())}"

    ok, msg = service.begin_task(task_id, {"name": "测试任务", "auto_execute": True})
    print("BEGIN=", ok, msg)

    decision = TradeDecision(
        action="buy",
        symbol_code="000001.SZ",
        symbol_name="平安银行",
        confidence=0.82,
        current_price=10.0,
        position_pct=0.1,
        risk_score=0.2,
    )
    risk_result = RiskCheckResult(
        passed=True,
        checks=[],
        overall_risk_level="low",
        warnings=[],
        blocked_reasons=[],
    )

    captured = []
    service.cycle_finished.connect(lambda task_id, success, message, summary: captured.append((task_id, success, message, summary)))
    service.handle_scan_results(
        task_id,
        {"name": "测试任务", "auto_execute": True},
        [{
            "decision": decision,
            "risk_result": risk_result,
            "symbol_code": "000001.SZ",
            "symbol_name": "平安银行",
            "decision_record_id": "demo123",
        }],
        BrokerContext(
            connected=True,
            total_asset=1000000.0,
            available_cash=500000.0,
            position_count=0,
            top_positions=[],
        ),
    )

    task_id, success, message, summary = captured[0]
    print("CYCLE=", task_id, success, message)
    print("EXECUTED=", summary.get("executed"))
    print("BROKER_ORDER_CALLS=", broker.order_calls)

    _assert_unified_end_of_day_flow()
    print("UNIFIED_EOD= ok")


if __name__ == "__main__":
    main()
