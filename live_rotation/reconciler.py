from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from trading_app.services.strategy.strategy_constants import normalize_symbol_code

logger = logging.getLogger(__name__)


@dataclass
class ReconcileResult:
    action: str
    position_adjusted: bool = False
    cash_adjusted: bool = False  # 保留字段兼容历史调用方，现金由主账本维护，始终为 False
    qty_before: int = 0
    qty_after: int = 0
    price_before: float = 0.0
    price_after: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0

    def __str__(self) -> str:
        parts = [self.action]
        if self.position_adjusted:
            parts.append(
                f"持仓: {self.qty_before}股@{self.price_before:.3f}"
                f" -> {self.qty_after}股@{self.price_after:.3f}"
            )
        return " | ".join(parts)


@dataclass
class _HoldingCandidate:
    code: str
    quantity: int
    cost: float = 0.0
    name: str = ""
    from_budget: bool = False
    from_broker: bool = False


class StartupReconciler:
    """Reconciles persisted runtime state with broker / budget positions.

    注意：现金余额统一由 ``StrategyBudgetService`` 主账本维护（commit_buy/sell 实时扣加），
    本对账器负责对齐 **运行态 current_holding** 与 **持仓数量 / 成本**。
    主账本或券商已有 ETF 池内仓位时，即使运行态显示空仓也会认领，避免误发 BUY。
    """

    def reconcile(self, engine, source: str = "startup") -> str:
        result = self._reconcile_position(engine, source=source)
        return result.action

    def reconcile_end_of_day(self, engine) -> ReconcileResult:
        return self._reconcile_position(engine, source="eod")

    def _reconcile_position(self, engine, source: str = "") -> ReconcileResult:
        from .trade_executor import SimulatedExecutor

        state = engine.state
        executor = engine.executor
        old_qty = int(getattr(state, "buy_quantity", 0) or 0)
        old_price = float(getattr(state, "buy_price", 0.0) or 0.0)
        old_holding = normalize_symbol_code(str(getattr(state, "current_holding", "") or ""))

        if isinstance(executor, SimulatedExecutor):
            return ReconcileResult(action="no_position")

        broker_connected = bool(executor.is_connected())
        candidates = self._collect_candidates(engine, broker_connected=broker_connected)

        if old_holding and old_holding in candidates:
            target = candidates[old_holding]
        elif candidates:
            target = self._pick_candidate(candidates)
        elif old_holding and broker_connected:
            logger.warning(
                "[%s] 对账发现持仓丢失，清空本地状态: %s",
                source, old_holding,
            )
            engine.state_mgr.clear_holding()
            return ReconcileResult(
                action="cleared_missing_position",
                position_adjusted=True,
                qty_before=old_qty, qty_after=0,
                price_before=old_price, price_after=0.0,
            )
        elif old_holding:
            return ReconcileResult(action="broker_disconnected")
        else:
            return ReconcileResult(action="no_position")

        name = target.name or self._holding_name(engine, target.code)
        cost = self._resolve_cost(state, old_holding, target)
        adopted = old_holding != target.code
        qty_changed = int(target.quantity) != old_qty
        price_changed = cost > 0 and abs(cost - old_price) > 1e-6

        if not adopted and not qty_changed and not price_changed:
            return ReconcileResult(action="position_consistent")

        self._apply_holding(engine, target.code, name=name, quantity=int(target.quantity), price=cost)
        if adopted and str(getattr(state, "last_signal", "") or "") in {"BUY", "SELL_ALL"}:
            state.last_signal = ""
            engine.state_mgr.save()
        action = "adopted_existing_position" if adopted else "updated_existing_position"
        logger.info(
            "[%s] 对账%s: %s qty=%s cost=%.4f budget=%s broker=%s",
            source,
            "认领已有持仓" if adopted else "更新持仓数量/成本",
            target.code,
            target.quantity,
            cost,
            target.from_budget,
            target.from_broker,
        )
        return ReconcileResult(
            action=action,
            position_adjusted=True,
            qty_before=old_qty, qty_after=int(target.quantity),
            price_before=old_price, price_after=cost if cost > 0 else old_price,
        )

    def _collect_candidates(self, engine, *, broker_connected: bool) -> Dict[str, _HoldingCandidate]:
        pool = self._etf_pool(engine)
        strategy_id = self._strategy_id(engine)
        candidates: Dict[str, _HoldingCandidate] = {}

        for item in self._budget_positions(engine):
            candidates[item.code] = item

        if not broker_connected:
            return candidates

        for item in self._broker_positions(engine):
            code = item.code
            if not self._is_claimable(code, strategy_id, pool, already_ours=code in candidates):
                continue
            existing = candidates.get(code)
            if existing is None:
                candidates[code] = item
                continue
            existing.from_broker = True
            existing.quantity = int(item.quantity)
            if not existing.name and item.name:
                existing.name = item.name
            if existing.cost <= 0 and item.cost > 0:
                existing.cost = item.cost
        return candidates

    @staticmethod
    def _pick_candidate(candidates: Dict[str, _HoldingCandidate]) -> _HoldingCandidate:
        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                0 if item.from_budget else 1,
                -int(item.quantity or 0),
                item.code,
            ),
        )
        return ranked[0]

    @staticmethod
    def _resolve_cost(state, old_holding: str, target: _HoldingCandidate) -> float:
        if target.cost > 0 and target.from_budget:
            return float(target.cost)
        if old_holding == target.code and float(getattr(state, "buy_price", 0.0) or 0.0) > 0:
            return float(state.buy_price)
        if target.cost > 0:
            return float(target.cost)
        return float(getattr(state, "buy_price", 0.0) or 0.0)

    @staticmethod
    def _apply_holding(engine, code: str, *, name: str, quantity: int, price: float) -> None:
        state_mgr = engine.state_mgr
        sync_holding = getattr(state_mgr, "sync_holding", None)
        if callable(sync_holding):
            sync_holding(code, name=name, quantity=quantity, price=price)
            return
        state_mgr.update_holding(
            code,
            name,
            float(getattr(engine.state, "current_score", 0.0) or 0.0),
            price,
            quantity,
        )

    @staticmethod
    def _etf_pool(engine) -> set[str]:
        config = getattr(engine, "config", None)
        return {
            normalize_symbol_code(code)
            for code in (getattr(config, "etf_pool", None) or [])
            if normalize_symbol_code(code)
        }

    @staticmethod
    def _strategy_id(engine) -> str:
        identity_fn = getattr(engine, "_etf_strategy_identity", None)
        if callable(identity_fn):
            try:
                return str(identity_fn()[0] or "").strip() or "etf_rotation"
            except Exception:
                pass
        config = getattr(engine, "config", None)
        return str(getattr(config, "strategy_id", "") or "etf_rotation").strip() or "etf_rotation"

    @staticmethod
    def _holding_name(engine, code: str) -> str:
        name_map = getattr(engine, "_etf_name_map", None) or {}
        name = str(name_map.get(code, "") or "")
        if name:
            return name
        code_name_fn = getattr(engine, "_code_name", None)
        if callable(code_name_fn):
            try:
                label = str(code_name_fn(code) or "")
                if label.startswith(f"{code}(") and label.endswith(")"):
                    return label[len(code) + 1:-1]
                if label and label != code:
                    return label
            except Exception:
                pass
        return str(getattr(engine.state, "current_holding_name", "") or "")

    def _budget_positions(self, engine) -> List[_HoldingCandidate]:
        state_mgr = getattr(engine, "state_mgr", None)
        budget = getattr(state_mgr, "budget_service", None)
        if budget is None:
            return []
        try:
            identity_fn = getattr(engine, "_etf_strategy_identity", None)
            if callable(identity_fn):
                strategy_id, strategy_name, virtual_account_id = identity_fn()
            else:
                strategy_id = self._strategy_id(engine)
                strategy_name = ""
                virtual_account_id = ""
            record = budget.get_strategy_state_record(
                strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                real_total_asset=0.0,
            )
        except Exception as exc:
            logger.warning("读取 ETF 主账本持仓失败，跳过账本认领: %s", exc)
            return []

        rows: List[_HoldingCandidate] = []
        for code, pos in (record.get_positions() or {}).items():
            normalized = normalize_symbol_code(getattr(pos, "symbol_code", "") or code)
            quantity = int(getattr(pos, "quantity", 0) or 0)
            if not normalized or quantity <= 0:
                continue
            rows.append(
                _HoldingCandidate(
                    code=normalized,
                    quantity=quantity,
                    cost=float(getattr(pos, "avg_cost", 0.0) or 0.0),
                    from_budget=True,
                )
            )
        return rows

    def _broker_positions(self, engine) -> List[_HoldingCandidate]:
        executor = engine.executor
        rows: List[_HoldingCandidate] = []
        seen: set[str] = set()
        raw_rows = []
        query_all = getattr(executor, "query_all_positions", None)
        if callable(query_all):
            try:
                raw_rows = list(query_all() or [])
            except Exception as exc:
                logger.warning("查询券商全部持仓失败: %s", exc)
                raw_rows = []
        for item in raw_rows:
            parsed = self._parse_position_row(item)
            if parsed is None or parsed.code in seen:
                continue
            seen.add(parsed.code)
            rows.append(parsed)

        if rows:
            return rows

        for code in sorted(self._etf_pool(engine)):
            try:
                quantity, cost = executor.query_position(code)
            except Exception as exc:
                logger.warning("查询券商持仓失败 %s: %s", code, exc)
                continue
            quantity = int(quantity or 0)
            if quantity <= 0:
                continue
            rows.append(
                _HoldingCandidate(
                    code=code,
                    quantity=quantity,
                    cost=float(cost or 0.0),
                    from_broker=True,
                )
            )
        return rows

    @staticmethod
    def _parse_position_row(item) -> Optional[_HoldingCandidate]:
        if isinstance(item, _HoldingCandidate):
            item.from_broker = True
            return item
        if not isinstance(item, dict):
            return None
        code = normalize_symbol_code(str(item.get("code") or item.get("stock_code") or item.get("symbol_code") or ""))
        quantity = int(item.get("quantity") or item.get("volume") or 0)
        if not code or quantity <= 0:
            return None
        return _HoldingCandidate(
            code=code,
            quantity=quantity,
            cost=float(item.get("cost") or item.get("avg_cost") or item.get("open_price") or 0.0),
            name=str(item.get("name") or item.get("stock_name") or ""),
            from_broker=True,
        )

    @staticmethod
    def _is_claimable(code: str, strategy_id: str, pool: set[str], *, already_ours: bool) -> bool:
        if already_ours:
            return True
        try:
            from trading_app.services.strategy.strategy_registry_service import get_strategy_registry_service

            owner = get_strategy_registry_service().get_owner(code)
        except Exception:
            owner = None
        if owner is not None and bool(getattr(owner, "enabled", False)):
            return str(getattr(owner, "strategy_id", "") or "").strip() == strategy_id
        return bool(not pool or code in pool)
