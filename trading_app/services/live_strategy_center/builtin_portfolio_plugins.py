from __future__ import annotations

from typing import Any

from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service

from .strategy_plugin import LiveStrategyPortfolioProvider


class AIStockPortfolioPlugin:
    """Portfolio provider callbacks for the built-in AI stock strategy."""

    def __init__(self, panel: object) -> None:
        self.panel = panel
        self.spec = get_strategy_spec_service().ai_stock()

    def create_provider(self, *, order: int = 10) -> LiveStrategyPortfolioProvider:
        return LiveStrategyPortfolioProvider(
            strategy_id=self.spec.strategy_id,
            strategy_name=self.spec.strategy_name,
            account_row_provider=self.build_account_row,
            position_rows_provider=self.build_position_rows,
            finalize_day_provider=self.build_finalize_day_provider,
            name_resolver=self.resolve_symbol_name,
            order=order,
        )

    def build_account_row(self, service: Any, _broker_live_positions: list[dict] | None = None) -> dict:
        account_panel = getattr(self.panel, "account_panel", None)
        total_asset_ref = 0.0
        live_positions: list[dict] = []
        if account_panel is not None:
            try:
                broker_ctx = account_panel.get_broker_context()
                total_asset_ref = float(getattr(broker_ctx, "total_asset", 0.0) or 0.0)
            except Exception:
                total_asset_ref = 0.0
            try:
                for item in (account_panel.get_live_positions() or []):
                    code = str(item.get("code", "") or "")
                    if not code:
                        continue
                    live_positions.append({
                        "code": code,
                        "stock_code": code,
                        "volume": int(item.get("volume", 0) or 0),
                        "market_value": float(item.get("market_value", 0.0) or 0.0),
                        "name": str(item.get("name", "") or ""),
                    })
            except Exception:
                live_positions = []
        account = service.strategy_budget.build_account_snapshot(
            self.spec.strategy_id,
            strategy_name=self.spec.strategy_name,
            virtual_account_id=self.spec.virtual_account_id,
            real_total_asset=total_asset_ref,
            live_positions=live_positions or None,
        )
        account["strategy_name"] = account.get("strategy_name") or self.spec.strategy_name
        return account

    def build_position_rows(self, service: Any, _broker_live_positions: list[dict] | None = None) -> list[dict]:
        account_panel = getattr(self.panel, "account_panel", None)
        if account_panel is None:
            return []
        try:
            live = [
                {
                    "stock_code": item.get("code", "") or "",
                    "market_value": float(item.get("market_value", 0.0) or 0.0),
                    "volume": int(item.get("volume", 0) or 0),
                    "name": item.get("name", "") or "",
                }
                for item in (account_panel.get_live_positions() or [])
            ]
            rows = service.strategy_budget.get_positions_view(
                self.spec.strategy_id,
                strategy_name=self.spec.strategy_name,
                virtual_account_id=self.spec.virtual_account_id,
                live_positions=live,
            )
            for row in rows:
                row["strategy_name"] = self.spec.strategy_name
            return rows
        except Exception:
            return []

    def build_finalize_day_provider(self, _service: Any, remark: str) -> dict:
        account_panel = getattr(self.panel, "account_panel", None)
        if account_panel is None:
            return {}
        try:
            live = account_panel.get_live_positions() or []
        except Exception:
            return {}
        return {
            "live_positions": [
                {
                    "stock_code": item.get("code", "") or "",
                    "market_value": float(item.get("market_value", 0.0) or 0.0),
                    "volume": int(item.get("volume", 0) or 0),
                    "name": item.get("name", "") or "",
                }
                for item in live
            ],
            "remark": remark,
        }

    def resolve_symbol_name(self, code: str) -> str:
        lookup = getattr(self.panel, "lookup_symbol_name", None)
        if callable(lookup):
            try:
                return str(lookup(code) or "").strip()
            except Exception:
                return ""
        return ""


class ETFRotationPortfolioPlugin:
    """Portfolio provider callbacks for the built-in ETF rotation strategy."""

    def __init__(self, panel: object, adapter: object) -> None:
        self.panel = panel
        self.adapter = adapter

    def create_provider(self, *, order: int = 10) -> LiveStrategyPortfolioProvider:
        return LiveStrategyPortfolioProvider(
            strategy_id=str(getattr(self.adapter, "strategy_id", "") or "").strip(),
            strategy_name=str(getattr(self.adapter, "strategy_name", "") or "").strip(),
            account_row_provider=self.build_account_row,
            position_rows_provider=self.build_position_rows,
            finalize_day_provider=self.build_finalize_day_provider,
            name_resolver=self.resolve_symbol_name,
            order=order,
        )

    def build_account_row(self, service: Any, _broker_live_positions: list[dict] | None = None) -> dict:
        strategy_id = str(getattr(self.adapter, "strategy_id", "") or "").strip()
        if not strategy_id:
            return {}
        strategy_name = str(getattr(self.adapter, "strategy_name", "") or strategy_id).strip()
        virtual_account_id = str(getattr(self.adapter, "virtual_account_id", "") or "").strip()
        market_value = 0.0
        get_account_view = getattr(self.panel, "_get_etf_strategy_account_view", None)
        summary = self.get_engine_status()
        if callable(get_account_view):
            try:
                account_view = dict(get_account_view(summary) or {})
                market_value = float(account_view.get("market_value", 0.0) or 0.0)
            except Exception:
                market_value = 0.0
        snapshot_kwargs = {
            "strategy_name": strategy_name,
            "virtual_account_id": virtual_account_id,
        }
        if market_value > 0:
            snapshot_kwargs["market_value_override"] = market_value
        account = service.strategy_budget.build_account_snapshot(strategy_id, **snapshot_kwargs)
        account["strategy_name"] = account.get("strategy_name") or strategy_name
        return account

    def build_position_rows(self, service: Any, _broker_live_positions: list[dict] | None = None) -> list[dict]:
        strategy_id = str(getattr(self.adapter, "strategy_id", "") or "").strip()
        if not strategy_id:
            return []
        strategy_name = str(getattr(self.adapter, "strategy_name", "") or strategy_id).strip()
        virtual_account_id = str(getattr(self.adapter, "virtual_account_id", "") or "").strip()
        try:
            rows = service.strategy_budget.get_positions_view(
                strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                spot_prices=self.get_spot_prices() or None,
            )
            for row in rows:
                row["strategy_name"] = strategy_name
            return rows
        except Exception:
            return []

    def build_finalize_day_provider(self, _service: Any, remark: str) -> dict:
        provider: dict[str, object] = {"remark": remark}
        spot_prices = self.get_spot_prices()
        if spot_prices:
            provider["spot_prices"] = spot_prices
        return provider

    def get_engine_status(self) -> dict:
        engine = getattr(self.panel, "engine", None)
        get_status = getattr(engine, "get_status_summary", None)
        if callable(get_status):
            try:
                return dict(get_status() or {})
            except Exception:
                return {}
        get_adapter_status = getattr(self.adapter, "get_status_summary", None)
        if callable(get_adapter_status):
            try:
                return dict(get_adapter_status() or {})
            except Exception:
                return {}
        return {}

    def get_spot_prices(self) -> dict[str, float]:
        summary = self.get_engine_status()
        holding = str(summary.get("holding", "") or "")
        if not holding:
            return {}
        current_price = float(summary.get("current_price", 0.0) or 0.0)
        price_source = str(summary.get("price_source", "") or "")
        if current_price <= 0 or price_source == "buy_price":
            return {}
        return {holding: current_price}

    def resolve_symbol_name(self, code: str) -> str:
        normalized = str(code or "").strip()
        if not normalized:
            return ""
        for attr_name in ("_ui_etf_name_map",):
            name_map = getattr(self.panel, attr_name, None)
            if isinstance(name_map, dict):
                resolved = name_map.get(normalized) or name_map.get(normalized.split(".")[0])
                if resolved:
                    return str(resolved).strip()
        engine = getattr(self.panel, "engine", None)
        name_map = getattr(engine, "_etf_name_map", None)
        if isinstance(name_map, dict):
            resolved = name_map.get(normalized) or name_map.get(normalized.split(".")[0])
            if resolved:
                return str(resolved).strip()
        return ""


def create_ai_stock_portfolio_provider(panel: object, *, order: int = 10) -> LiveStrategyPortfolioProvider:
    return AIStockPortfolioPlugin(panel).create_provider(order=order)


def create_etf_rotation_portfolio_provider(
    panel: object,
    adapter: object,
    *,
    order: int = 10,
) -> LiveStrategyPortfolioProvider:
    return ETFRotationPortfolioPlugin(panel, adapter).create_provider(order=order)
