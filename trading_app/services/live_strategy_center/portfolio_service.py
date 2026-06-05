from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional

from common.broker_session_service import get_broker_session_service
from trading_app.services.live_strategy_center.strategy_plugin import LiveStrategyPortfolioProvider
from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service
from trading_app.services.execution.trade_record_service import get_trade_record_service


class LiveStrategyPortfolioService:
    """Build portfolio/performance snapshots for the live strategy center.

    The service is the single data collection point for the performance widget:
    broker live positions, unmanaged reconciliation, strategy account rows,
    position rows, trade summary and daily trade statistics.
    """

    def __init__(
        self,
        *,
        strategy_adapters: Optional[Iterable[object]] = None,
        portfolio_providers: Optional[Iterable[LiveStrategyPortfolioProvider]] = None,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
        broker_service=None,
        strategy_budget=None,
        trade_service=None,
    ) -> None:
        self.strategy_adapters = list(strategy_adapters or [])
        self.portfolio_providers = list(portfolio_providers or [])
        self.symbol_name_resolver = symbol_name_resolver
        self.strategy_spec_service = get_strategy_spec_service()
        self.unmanaged_strategy_id = self.strategy_spec_service.unmanaged().strategy_id
        self._broker_service = broker_service or get_broker_session_service()
        self.strategy_budget = strategy_budget or get_strategy_budget_service()
        self.trade_service = trade_service or get_trade_record_service()

    @property
    def broker_service(self):
        return self._broker_service

    def refresh_snapshot(self) -> dict:
        broker_live_positions = self.fetch_broker_live_positions()
        reconcile_summary = self.reconcile_unmanaged_from_broker(broker_live_positions)
        strategy_rows = self.build_strategy_rows(broker_live_positions)
        active_rows = [row for row in strategy_rows if not bool(row.get("is_unmanaged", False))]
        active_ids = {
            str(row.get("strategy_id", "") or "").strip()
            for row in strategy_rows
            if row.get("strategy_id")
        }
        return {
            "broker_connected": bool(getattr(self._broker_service, "is_connected", False)),
            "broker_live_positions": broker_live_positions,
            "reconcile_summary": reconcile_summary,
            "strategy_rows": strategy_rows,
            "portfolio_totals": self.build_portfolio_totals(strategy_rows, reconcile_summary),
            "summary_metrics": self.build_summary_metrics(active_rows),
            "position_rows": self.build_position_rows(strategy_rows, broker_live_positions),
            "daily_rows": self.build_daily_rows(active_ids),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def active_strategy_ids(self) -> set[str]:
        ids = {
            str(provider.strategy_id or "").strip()
            for provider in self.portfolio_providers
            if bool(getattr(provider, "enabled", True))
            and str(provider.strategy_id or "").strip()
            and str(provider.strategy_id or "").strip() != self.unmanaged_strategy_id
        }
        for adapter in self.strategy_adapters:
            strategy_id = str(getattr(adapter, "strategy_id", "") or "").strip()
            if strategy_id and strategy_id != self.unmanaged_strategy_id:
                ids.add(strategy_id)
        return ids

    def fetch_broker_live_positions(self) -> list[dict]:
        if not getattr(self._broker_service, "is_connected", False):
            return []
        try:
            raw = self._broker_service.query_stock_positions() or []
        except Exception:
            return []
        result: list[dict] = []
        for pos in raw:
            try:
                code = str(getattr(pos, "stock_code", "") or "")
                volume = int(getattr(pos, "volume", 0) or 0)
                if not code or volume <= 0:
                    continue
                result.append({
                    "stock_code": code,
                    "volume": volume,
                    "market_value": float(getattr(pos, "market_value", 0.0) or 0.0),
                    "name": str(getattr(pos, "stock_name", "") or ""),
                    "open_price": float(getattr(pos, "open_price", 0.0) or 0.0),
                })
            except Exception:
                continue
        return result

    def reconcile_unmanaged_from_broker(self, broker_live_positions: list[dict]) -> dict | None:
        if not getattr(self._broker_service, "is_connected", False):
            return None
        try:
            asset = self._broker_service.query_stock_asset()
            broker_cash = float(getattr(asset, "cash", 0.0) or 0.0)
        except Exception:
            return None
        broker_positions = [
            {
                "stock_code": item.get("stock_code", ""),
                "volume": item.get("volume", 0),
                "open_price": item.get("open_price", 0.0),
            }
            for item in (broker_live_positions or [])
        ]
        try:
            return self.strategy_budget.reconcile_unmanaged_with_broker(
                broker_cash=broker_cash,
                broker_positions=broker_positions,
            )
        except Exception:
            return None

    def build_strategy_rows(self, broker_live_positions: list[dict] | None = None) -> list[dict]:
        rows: list[dict] = []
        for provider in self.portfolio_providers:
            if not bool(getattr(provider, "enabled", True)) or provider.account_row_provider is None:
                continue
            try:
                row = dict(provider.account_row_provider(self, broker_live_positions) or {})
            except Exception:
                row = {}
            if row:
                row.setdefault("strategy_id", provider.strategy_id)
                row.setdefault("strategy_name", provider.strategy_name or provider.strategy_id)
                rows.append(row)
        return rows

    def build_summary_metrics(self, rows: list[dict]) -> dict:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        active_ids = sorted(
            {str(item.get("strategy_id", "") or "").strip() for item in rows if item.get("strategy_id")}
        )
        period = self.trade_service.get_period_stats(
            strategy_ids=active_ids or None,
            start_date=start_date,
        )
        win_stats = self.trade_service.get_win_rate_stats(
            strategy_ids=active_ids or None,
            start_date=start_date,
        )
        return {
            "total_trades": int(period.get("total_trades", 0) or 0),
            "buy_count": int(period.get("buy_count", 0) or 0),
            "sell_count": int(period.get("sell_count", 0) or 0),
            "closed_count": int(win_stats.get("closed_count", 0) or 0),
            "win_rate": float(win_stats.get("win_rate", 0.0) or 0.0) * 100.0,
            "total_pnl": round(sum(float(item.get("total_pnl", 0.0) or 0.0) for item in rows), 2),
        }

    def build_position_rows(
        self,
        rows: list[dict],
        broker_live_positions: list[dict] | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        for provider in self.portfolio_providers:
            if not bool(getattr(provider, "enabled", True)) or provider.position_rows_provider is None:
                continue
            try:
                provider_rows = list(provider.position_rows_provider(self, broker_live_positions) or [])
            except Exception:
                provider_rows = []
            for row in provider_rows:
                row.setdefault("strategy_id", provider.strategy_id)
                row.setdefault("strategy_name", provider.strategy_name or provider.strategy_id)
            results.extend(provider_rows)

        strategy_order: dict[str, int] = {}
        for idx, row in enumerate(rows):
            strategy_id = str(row.get("strategy_id", "") or "").strip()
            if strategy_id and strategy_id not in strategy_order:
                strategy_order[strategy_id] = idx
        results.sort(
            key=lambda row: (
                strategy_order.get(str(row.get("strategy_id", "") or "").strip(), len(strategy_order)),
                -float(row.get("market_value", 0.0) or 0.0),
                str(row.get("stock_code", "") or ""),
            )
        )
        for row in results:
            if not str(row.get("stock_name", "") or "").strip():
                row["stock_name"] = self.resolve_position_name(row)
        return results

    def build_daily_rows(self, active_ids: set[str]) -> list[dict]:
        start_date = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
        daily = self.trade_service.get_daily_stats(
            strategy_ids=sorted(active_ids) if active_ids else None,
            start_date=start_date,
        )
        return [
            {
                "trade_date": str(item.get("trade_date", "") or ""),
                "trade_count": int(item.get("total_trades", 0) or 0),
                "buy_amount": float(item.get("buy_amount", 0.0) or 0.0),
                "sell_amount": float(item.get("sell_amount", 0.0) or 0.0),
                "net_inflow": float(item.get("net_inflow", 0.0) or 0.0),
            }
            for item in daily
        ]

    def build_finalize_day_providers(self, *, remark: str = "日终统一快照") -> dict[str, dict]:
        providers: dict[str, dict] = {}
        for provider in self.portfolio_providers:
            if not bool(getattr(provider, "enabled", True)) or provider.finalize_day_provider is None:
                continue
            try:
                payload = dict(provider.finalize_day_provider(self, remark) or {})
            except Exception:
                payload = {}
            if payload:
                providers[provider.strategy_id] = payload
        return providers

    def finalize_day_snapshots(self, *, remark: str = "日终统一快照"):
        providers = self.build_finalize_day_providers(remark=remark)
        return self.strategy_budget.finalize_day(providers=providers, remark=remark)

    def build_portfolio_totals(self, rows: list[dict], reconcile_summary: dict | None = None) -> dict:
        portfolio_market_value = sum(float(row.get("market_value", 0.0) or 0.0) for row in rows)
        portfolio_capital_limit = sum(float(row.get("capital_limit", 0.0) or 0.0) for row in rows)
        portfolio_realized_pnl = sum(float(row.get("realized_pnl", 0.0) or 0.0) for row in rows)
        portfolio_unrealized_pnl = sum(float(row.get("unrealized_pnl", 0.0) or 0.0) for row in rows)
        portfolio_total_pnl = sum(float(row.get("total_pnl", 0.0) or 0.0) for row in rows)
        portfolio_cash = sum(
            max(float(row.get("cash_balance", 0.0) or 0.0) - float(row.get("reserved_cash", 0.0) or 0.0), 0.0)
            for row in rows
        )
        diff = self.build_broker_diff(reconcile_summary)
        return {
            "total_asset": portfolio_cash + portfolio_market_value,
            "cash": portfolio_cash,
            "market_value": portfolio_market_value,
            "capital_limit": portfolio_capital_limit,
            "realized_pnl": portfolio_realized_pnl,
            "unrealized_pnl": portfolio_unrealized_pnl,
            "total_pnl": portfolio_total_pnl,
            "broker_diff_text": diff.get("text", ""),
            "broker_diff_color": diff.get("color", "#888"),
            "broker_diff_tooltip": diff.get("tooltip", ""),
        }

    def build_broker_diff(self, reconcile_summary: dict | None = None) -> dict:
        if not getattr(self._broker_service, "is_connected", False):
            return {"text": "券商未连接", "color": "#888", "tooltip": ""}
        if not reconcile_summary:
            return {"text": "对账数据缺失", "color": "#eab308", "tooltip": ""}

        cash_shortfall = float(reconcile_summary.get("cash_shortfall", 0.0) or 0.0)
        pos_shortfalls = list(reconcile_summary.get("position_shortfalls", []) or [])
        untracked = list(reconcile_summary.get("untracked_broker_codes", []) or [])
        broker_cash = float(reconcile_summary.get("broker_cash", 0.0) or 0.0)
        unmanaged_cash = float(reconcile_summary.get("unmanaged_cash", 0.0) or 0.0)

        if cash_shortfall < -1.0 or pos_shortfalls:
            parts = []
            if cash_shortfall < -1.0:
                parts.append(f"活跃策略虚报现金 ¥{abs(cash_shortfall):,.2f}")
            if pos_shortfalls:
                codes_preview = ",".join(
                    f"{item['stock_code']}×{item['shortfall']}" for item in pos_shortfalls[:3]
                )
                more = "…" if len(pos_shortfalls) > 3 else ""
                parts.append(f"虚报持仓 {len(pos_shortfalls)}项({codes_preview}{more})")
            tooltip_lines = ["账本漂移（reconcile 之前的真实差）："]
            if cash_shortfall < -1.0:
                tooltip_lines.append(f"  现金: 活跃策略声明 > 券商实际 {abs(cash_shortfall):,.2f} 元")
            for item in pos_shortfalls:
                tooltip_lines.append(
                    f"  持仓 {item['stock_code']}: 声明 {item['claimed']} / 券商 {item['broker']} / 差 {item['shortfall']}"
                )
            return {
                "text": "异常: " + "；".join(parts),
                "color": "#d9534f",
                "tooltip": "\n".join(tooltip_lines),
            }

        text = f"一致 (券商现金 ¥{broker_cash:,.2f}，未管理吸收 ¥{unmanaged_cash:,.2f})"
        tooltip = ""
        if untracked:
            text += f" · 巡检{len(untracked)}只"
            tooltip = "券商有但无策略声明（理论由未管理吸收）：\n  " + ", ".join(untracked)
        return {"text": text, "color": "#16a34a", "tooltip": tooltip}

    def resolve_position_name(self, item: dict) -> str:
        name = str(item.get("stock_name", "") or item.get("name", "") or "").strip()
        if name:
            return name
        code = str(item.get("stock_code", "") or "").strip()
        if not code:
            return ""
        if callable(self.symbol_name_resolver):
            try:
                resolved = self.symbol_name_resolver(code)
                if resolved:
                    return str(resolved).strip()
            except Exception:
                pass
        for provider in self.portfolio_providers:
            resolver = getattr(provider, "name_resolver", None)
            if callable(resolver):
                try:
                    resolved = resolver(code)
                    if resolved:
                        return str(resolved).strip()
                except Exception:
                    pass
        for adapter in self.strategy_adapters:
            panel = getattr(adapter, "widget", None)
            lookup = getattr(panel, "lookup_symbol_name", None)
            if callable(lookup):
                try:
                    resolved = lookup(code)
                    if resolved:
                        return str(resolved).strip()
                except Exception:
                    pass
        return ""