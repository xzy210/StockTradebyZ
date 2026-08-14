from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from live_rotation.config import RotationConfig
from live_rotation.reconciler import StartupReconciler
from live_rotation.state_manager import RotationState
from live_rotation.trade_executor import TradeExecutor


class FakePos:
    def __init__(self, code: str, quantity: int, avg_cost: float) -> None:
        self.symbol_code = code
        self.quantity = quantity
        self.avg_cost = avg_cost


class FakeBudget:
    def __init__(self, positions: dict) -> None:
        self._positions = positions

    def get_strategy_state_record(self, strategy_id: str, **kwargs):
        return SimpleNamespace(get_positions=lambda: dict(self._positions))


class FakeStateManager:
    def __init__(self, state: RotationState, budget: FakeBudget | None = None) -> None:
        self.state = state
        self.budget_service = budget
        self.saved = 0
        self.clear_count = 0

    def save(self) -> None:
        self.saved += 1

    def update_holding(self, code, name, score, price, quantity) -> None:
        self.state.current_holding = code
        self.state.current_holding_name = name
        self.state.current_score = score
        self.state.buy_price = price
        self.state.buy_quantity = quantity
        self.save()

    def sync_holding(self, code, *, name="", quantity=0, price=0.0) -> None:
        if not code or int(quantity or 0) <= 0:
            self.clear_holding()
            return
        same = str(self.state.current_holding or "") == str(code)
        self.state.current_holding = code
        if name:
            self.state.current_holding_name = name
        self.state.buy_quantity = int(quantity)
        if price > 0:
            self.state.buy_price = float(price)
        if not same:
            self.state.current_score = 0.0
        self.save()

    def clear_holding(self) -> None:
        self.state.current_holding = None
        self.state.current_holding_name = ""
        self.state.buy_price = 0.0
        self.state.buy_quantity = 0
        self.clear_count += 1
        self.save()


class FakeExecutor(TradeExecutor):
    def __init__(self, positions: dict | None = None, connected: bool = True) -> None:
        self.positions = positions or {}
        self.connected = connected

    def is_connected(self) -> bool:
        return bool(self.connected)

    def get_current_price(self, code: str) -> float:
        return 0.0

    def query_position(self, code: str):
        pos = self.positions.get(code) or {}
        return int(pos.get("quantity", 0) or 0), float(pos.get("avg_price", 0.0) or 0.0)

    def query_sellable_position(self, code: str):
        return self.query_position(code)

    def query_all_positions(self):
        rows = []
        for code, pos in self.positions.items():
            qty = int((pos or {}).get("quantity", 0) or 0)
            if qty <= 0:
                continue
            rows.append(
                {
                    "code": f"{code}.SH",
                    "quantity": qty,
                    "cost": float((pos or {}).get("avg_price", 0.0) or 0.0),
                    "name": str((pos or {}).get("name", "") or ""),
                }
            )
        return rows


class FakeEngine:
    def __init__(self, state: RotationState, state_mgr: FakeStateManager, executor: FakeExecutor) -> None:
        self.state = state
        self.state_mgr = state_mgr
        self.executor = executor
        self.config = RotationConfig(etf_pool=["510880", "159949", "513100", "518880"])
        self._etf_name_map = {"510880": "红利ETF"}

    def _etf_strategy_identity(self):
        return ("etf_rotation", "ETF轮动实盘", "va_etf_rotation")


def _run(engine: FakeEngine) -> str:
    reconciler = StartupReconciler()
    reconciler._is_claimable = lambda code, strategy_id, pool, already_ours=False: (
        already_ours or code in pool
    )
    return reconciler.reconcile(engine, source="test")


def main() -> None:
    # 1) 运行态空仓，主账本已有红利ETF -> 认领，避免误发 BUY
    state = RotationState(last_signal="BUY")
    budget = FakeBudget({"510880": FakePos("510880", 8400, 3.31)})
    state_mgr = FakeStateManager(state, budget)
    engine = FakeEngine(state, state_mgr, FakeExecutor(connected=False))
    action = _run(engine)
    assert action == "adopted_existing_position", action
    assert state.current_holding == "510880"
    assert state.current_holding_name == "红利ETF"
    assert state.buy_quantity == 8400
    assert abs(state.buy_price - 3.31) < 1e-6
    assert state.last_signal == ""
    print("[adopt_from_budget] OK")

    # 2) 运行态空仓，主账本为空，券商有池内持仓 -> 认领
    state = RotationState()
    state_mgr = FakeStateManager(state, FakeBudget({}))
    executor = FakeExecutor(
        positions={"510880": {"quantity": 8400, "avg_price": 3.31, "name": "红利ETF"}},
        connected=True,
    )
    engine = FakeEngine(state, state_mgr, executor)
    action = _run(engine)
    assert action == "adopted_existing_position", action
    assert state.current_holding == "510880"
    assert state.buy_quantity == 8400
    print("[adopt_from_broker] OK")

    # 3) 已持仓且数量一致 -> 无需改动
    state = RotationState(current_holding="510880", buy_quantity=8400, buy_price=3.31)
    budget = FakeBudget({"510880": FakePos("510880", 8400, 3.31)})
    state_mgr = FakeStateManager(state, budget)
    engine = FakeEngine(state, state_mgr, FakeExecutor({"510880": {"quantity": 8400, "avg_price": 3.31}}))
    action = _run(engine)
    assert action == "position_consistent", action
    print("[position_consistent] OK")

    # 4) 运行态空仓，券商未连接且账本也空 -> no_position
    state = RotationState()
    state_mgr = FakeStateManager(state, FakeBudget({}))
    engine = FakeEngine(state, state_mgr, FakeExecutor(connected=False))
    action = _run(engine)
    assert action == "no_position", action
    assert state.current_holding is None
    print("[no_position] OK")

    # 5) 运行态有仓，券商已连接但该标的与账本都没有 -> 清空
    state = RotationState(current_holding="159949", buy_quantity=1000, buy_price=1.0)
    state_mgr = FakeStateManager(state, FakeBudget({}))
    engine = FakeEngine(state, state_mgr, FakeExecutor(positions={}, connected=True))
    action = _run(engine)
    assert action == "cleared_missing_position", action
    assert state.current_holding is None
    assert state_mgr.clear_count == 1
    print("[cleared_missing_position] OK")

    print("rotation reconciler smoketest passed")


if __name__ == "__main__":
    main()
