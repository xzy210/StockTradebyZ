# -*- coding: utf-8 -*-
"""Helpers for backtest/live execution contract smoke tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from trading_app.services.strategy.strategy_risk import StrategyRiskRegistry
from trading_app.services.execution.trade_execution_service import TradeExecutionService
from trading_app.services.execution.risk_guard_service import RiskGuardService


class BrokerStub:
    def __init__(self, *, cash: float, total_asset: float, positions: dict[str, Any] | None = None) -> None:
        self.is_connected = True
        self.cash = float(cash)
        self.total_asset = float(total_asset)
        self.positions = dict(positions or {})

    def query_asset(self):
        return SimpleNamespace(
            cash=self.cash,
            available_cash=self.cash,
            total_asset=self.total_asset,
        )

    def query_position(self, symbol: str = ""):
        plain = _plain_code(symbol)
        if not plain:
            return list(self.positions.values())
        return self.positions.get(plain)


def create_backtest_live_gateway_factory(
    *,
    broker_cash: float | None = None,
    broker_total_asset: float | None = None,
    positions: dict[str, Any] | None = None,
    risk_config: dict[str, Any] | None = None,
) -> Callable[[Any, Any], TradeExecutionService]:
    """Build a deterministic ``TradeExecutionService`` for contract tests."""

    def factory(_context, config) -> TradeExecutionService:
        initial_cash = float(getattr(config, "initial_cash", 0.0) or 0.0)
        cash = float(broker_cash if broker_cash is not None else initial_cash)
        total_asset = float(broker_total_asset if broker_total_asset is not None else max(cash, initial_cash))

        service = TradeExecutionService.__new__(TradeExecutionService)
        service.broker_service = None
        service.broker = BrokerStub(cash=cash, total_asset=total_asset, positions=positions)
        service.risk_guard = RiskGuardService(config_path=Path("__backtest_live_contract_missing__.json"))
        service.risk_guard.config.update(
            {
                "max_single_position_pct": 1.0,
                "max_total_position_pct": 1.0,
                **dict(risk_config or {}),
            }
        )
        service.strategy_risk_registry = StrategyRiskRegistry()
        service._ai_stock_policy = None
        service._recent_fingerprints = {}
        service.pending_order_lifecycles = {}
        service._event_storage = None
        return service

    return factory


def _plain_code(code: str) -> str:
    value = str(code or "").strip().upper()
    return value.split(".")[0] if "." in value else value
