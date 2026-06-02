from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from common.broker_session_service import get_broker_session_service
except ImportError:
    from common.broker_session_service import get_broker_session_service

from trading_app.services.market_data_gateway import get_market_data_gateway
from trading_app.services.strategy_budget_service import get_strategy_budget_service
from trading_app.services.strategy_registry_service import get_strategy_registry_service
from trading_app.services.trade_record_service import OrderRecord, TradeRecord, get_trade_record_service
from trading_app.services.strategy_constants import AI_STOCK_STRATEGY_ID, normalize_symbol_code

logger = logging.getLogger(__name__)


@dataclass
class StrategyTradeViewContext:
    strategy_id: str
    strategy_name: str = ""
    virtual_account_id: str = ""


class StrategyTradeViewService:
    def __init__(self) -> None:
        self.trade_service = get_trade_record_service()
        self.strategy_budget = get_strategy_budget_service()
        self.strategy_registry = get_strategy_registry_service()
        self.broker = get_broker_session_service()
        self._last_sync_at: Dict[str, datetime] = {}

    @staticmethod
    def _today_bounds() -> tuple[str, str, str]:
        today = datetime.now().strftime("%Y-%m-%d")
        return today, f"{today} 00:00:00", f"{today} 23:59:59"

    @staticmethod
    def _dict_value(item: Dict[str, Any], *keys: str, default: Any = "") -> Any:
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_trade_direction(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"buy", "b", "买入"}:
            return "buy"
        if text in {"sell", "s", "sell_all", "卖出"}:
            return "sell"
        return ""

    def _trade_record_from_budget_history_item(
        self,
        ctx: StrategyTradeViewContext,
        item: Dict[str, Any],
    ) -> Optional[TradeRecord]:
        if not isinstance(item, dict):
            return None

        code = normalize_symbol_code(str(self._dict_value(item, "stock_code", "code", "symbol_code")))
        direction = self._normalize_trade_direction(
            self._dict_value(item, "direction", "action", "trade_direction")
        )
        price = self._safe_float(self._dict_value(item, "price", "filled_price", "ordered_price"))
        volume = self._safe_int(
            self._dict_value(item, "volume", "quantity", "filled_qty", "ordered_qty")
        )
        amount = self._safe_float(self._dict_value(item, "amount"))
        if amount <= 0 and price > 0 and volume > 0:
            amount = round(price * volume, 2)

        trade_date = str(self._dict_value(item, "trade_date", "date")).strip()
        if not trade_date:
            trade_date = str(self._dict_value(item, "created_at")).strip().split(" ")[0]
        if len(trade_date) > 10:
            trade_date = trade_date[:10]

        if not code or not direction or price <= 0 or volume <= 0 or not trade_date:
            return None

        trade_time = str(self._dict_value(item, "time")).strip()
        created_at = str(self._dict_value(item, "created_at")).strip()
        if not created_at:
            created_at = f"{trade_date} {trade_time or '00:00:00'}"
        stock_name = str(self._dict_value(item, "stock_name", "name", default=code)).strip()
        source = str(
            self._dict_value(item, "source_display", "source", default=ctx.strategy_name or ctx.strategy_id)
        ).strip()

        return TradeRecord(
            trade_id=str(self._dict_value(item, "trade_id")),
            broker_order_id=self._safe_int(self._dict_value(item, "broker_order_id"), -1),
            stock_code=code,
            stock_name=self.trade_service.normalize_stock_name(code, stock_name),
            direction=direction,
            price=price,
            volume=volume,
            amount=round(amount, 2),
            commission=max(self._safe_float(self._dict_value(item, "commission")), 0.0),
            stamp_tax=max(self._safe_float(self._dict_value(item, "stamp_tax")), 0.0),
            transfer_fee=max(self._safe_float(self._dict_value(item, "transfer_fee")), 0.0),
            trade_date=trade_date,
            source=source or ctx.strategy_name or ctx.strategy_id,
            strategy_id=ctx.strategy_id,
            virtual_account_id=ctx.virtual_account_id,
            intent_id=str(self._dict_value(item, "intent_id")),
            remark=str(self._dict_value(item, "remark", "reason")),
            created_at=created_at,
        )

    def _normalize_context(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
    ) -> StrategyTradeViewContext:
        return StrategyTradeViewContext(
            strategy_id=(strategy_id or "").strip(),
            strategy_name=(strategy_name or "").strip(),
            virtual_account_id=(virtual_account_id or "").strip(),
        )

    def _belongs_to_context(self, ctx: StrategyTradeViewContext, stock_code: str) -> bool:
        code = normalize_symbol_code(stock_code)
        if not code or not ctx.strategy_id:
            return False
        owner = self.strategy_registry.get_owner(code)
        if owner and owner.enabled:
            return owner.strategy_id == ctx.strategy_id
        return False

    def _filter_orders_for_context(self, orders: List[Any], ctx: StrategyTradeViewContext) -> List[Any]:
        local_order_ids = self._local_broker_order_ids_for_context(ctx)
        return [
            order
            for order in list(orders or [])
            if (
                self._belongs_to_context(ctx, getattr(order, "stock_code", ""))
                or int(getattr(order, "order_id", 0) or 0) in local_order_ids
            )
        ]

    def _filter_trades_for_context(self, trades: List[Any], ctx: StrategyTradeViewContext) -> List[Any]:
        local_order_ids = self._local_broker_order_ids_for_context(ctx)
        return [
            trade
            for trade in list(trades or [])
            if (
                self._belongs_to_context(ctx, getattr(trade, "stock_code", ""))
                or int(getattr(trade, "order_id", 0) or 0) in local_order_ids
            )
        ]

    def _local_broker_order_ids_for_context(self, ctx: StrategyTradeViewContext) -> set[int]:
        if not ctx.strategy_id:
            return set()
        try:
            records = self.trade_service.get_order_records(
                strategy_id=ctx.strategy_id,
                virtual_account_id=ctx.virtual_account_id,
                limit=5000,
            )
        except Exception:
            return set()
        result: set[int] = set()
        for rec in records or []:
            try:
                order_id = int(getattr(rec, "broker_order_id", 0) or 0)
            except Exception:
                order_id = 0
            if order_id > 0:
                result.add(order_id)
        return result

    def _rebuild_all_strategy_states(self) -> None:
        seen: set[str] = set()
        for snapshot in self.strategy_budget.list_strategy_snapshots():
            strategy_id = str(snapshot.get("strategy_id", "") or "")
            if not strategy_id or strategy_id in seen:
                continue
            seen.add(strategy_id)
            try:
                self.strategy_budget.rebuild_strategy_state_from_trade_records(
                    strategy_id,
                    strategy_name=str(snapshot.get("strategy_name", "") or ""),
                    virtual_account_id=str(snapshot.get("virtual_account_id", "") or ""),
                    real_total_asset=0.0,
                )
            except Exception:
                continue

    def _backfill_trades_from_local_orders(self, ctx: StrategyTradeViewContext) -> int:
        if not ctx.strategy_id:
            return 0
        try:
            local_orders = self.trade_service.get_order_records(
                strategy_id=ctx.strategy_id,
                virtual_account_id=ctx.virtual_account_id,
                limit=5000,
            )
            added = self.trade_service.sync_from_order_records(local_orders)
            if added > 0:
                self.strategy_budget.rebuild_strategy_state_from_trade_records(
                    ctx.strategy_id,
                    strategy_name=ctx.strategy_name,
                    virtual_account_id=ctx.virtual_account_id,
                    real_total_asset=0.0,
                )
            return added
        except Exception:
            return 0

    def sync_strategy_broker_records(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
    ) -> None:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id or not self.broker.is_connected:
            return
        last_sync = self._last_sync_at.get(ctx.strategy_id)
        now = datetime.now()
        if last_sync is not None and (now - last_sync).total_seconds() < 5:
            return
        self._last_sync_at[ctx.strategy_id] = now
        inferred_trades_synced = 0
        local_order_record_trades_synced = 0
        broker_trades_synced = 0
        corrected_records = 0
        deduped_records = 0
        rebuilt_all = False
        needs_rebuild = False
        try:
            deduped_records = self.trade_service.dedupe_trade_records_by_broker_order()
            needs_rebuild = deduped_records > 0
        except Exception:
            deduped_records = 0
        try:
            corrected_records = self.trade_service.realign_broker_sync_records_by_ownership()
            needs_rebuild = needs_rebuild or corrected_records > 0
        except Exception:
            corrected_records = 0
        try:
            orders = self._filter_orders_for_context(
                self.broker.query_stock_orders_safe(timeout_seconds=4.0) or [],
                ctx,
            )
            self.trade_service.sync_order_records_from_orders(orders)
            inferred_trades_synced = self.trade_service.sync_from_orders(
                orders,
                strategy_id=ctx.strategy_id,
                virtual_account_id=ctx.virtual_account_id,
            )
        except Exception:
            pass
        try:
            local_orders = self.trade_service.get_order_records(
                strategy_id=ctx.strategy_id,
                virtual_account_id=ctx.virtual_account_id,
                limit=5000,
            )
            local_order_record_trades_synced = self.trade_service.sync_from_order_records(local_orders)
        except Exception:
            pass
        if needs_rebuild:
            self._rebuild_all_strategy_states()
            rebuilt_all = True
        elif inferred_trades_synced > 0 or local_order_record_trades_synced > 0 or broker_trades_synced > 0:
            try:
                self.strategy_budget.rebuild_strategy_state_from_trade_records(
                    ctx.strategy_id,
                    strategy_name=ctx.strategy_name,
                    virtual_account_id=ctx.virtual_account_id,
                    real_total_asset=0.0,
                )
            except Exception:
                pass

    def get_strategy_positions(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
    ) -> List[Dict[str, Any]]:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id:
            return []
        state = self.strategy_budget.get_strategy_state_record(
            ctx.strategy_id,
            strategy_name=ctx.strategy_name,
            virtual_account_id=ctx.virtual_account_id,
            real_total_asset=0.0,
        )
        budget_positions = state.get_positions()
        runtime_state = dict(getattr(state, "runtime_state", {}) or {})
        name_map = self._build_name_map(ctx)
        broker_position_map = self._build_broker_position_map()
        all_codes = sorted(set(budget_positions.keys()))
        rows: List[Dict[str, Any]] = []
        for code in all_codes:
            budget_pos = budget_positions.get(code)
            quantity = int(getattr(budget_pos, "quantity", 0) or 0)
            if quantity <= 0:
                continue
            broker_pos = broker_position_map.get(normalize_symbol_code(code), {})
            broker_can_use = int(broker_pos.get("can_use_volume", 0) or 0)
            can_use = min(quantity, broker_can_use) if broker_can_use > 0 else quantity
            avg_cost = float(getattr(budget_pos, "avg_cost", 0.0) or 0.0)
            stock_name = name_map.get(code) or str(broker_pos.get("stock_name", "") or "")
            realtime_price = self._fetch_realtime_price(code)
            broker_price = float(broker_pos.get("current_price", 0.0) or 0.0)
            display_price = self._fetch_display_price(code)
            if not stock_name:
                fallback_name, _ = self._resolve_local_symbol_snapshot(state, code, runtime_state)
                stock_name = fallback_name
            current_price = next(
                (price for price in (realtime_price, broker_price, display_price, avg_cost) if price > 0),
                0.0,
            )
            broker_market_value = float(broker_pos.get("market_value", 0.0) or 0.0)
            market_value = (
                round(current_price * quantity, 2)
                if current_price > 0 else
                round(avg_cost * quantity, 2)
            )
            if broker_market_value > 0 and broker_price > 0 and realtime_price <= 0:
                market_value = round(broker_market_value, 2)
            pnl = round((current_price - avg_cost) * quantity, 2) if avg_cost > 0 and current_price > 0 else 0.0
            pnl_pct = round((current_price - avg_cost) / avg_cost * 100, 2) if avg_cost > 0 and current_price > 0 else 0.0
            rows.append(
                {
                    "stock_code": code,
                    "stock_name": stock_name or code,
                    "volume": quantity,
                    "can_use_volume": can_use,
                    "avg_cost": round(avg_cost, 4),
                    "current_price": round(current_price, 4),
                    "market_value": round(market_value, 2),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                }
            )
        return rows

    def get_today_orders(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        limit: int = 200,
    ) -> List[Any]:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id:
            return []
        today, start_time, end_time = self._today_bounds()
        records = self.trade_service.get_order_records(
            start_time=start_time,
            end_time=end_time,
            strategy_id=ctx.strategy_id,
            virtual_account_id=ctx.virtual_account_id,
            limit=limit,
        )
        if records:
            return records
        state = self.strategy_budget.get_strategy_state_record(
            ctx.strategy_id,
            strategy_name=ctx.strategy_name,
            virtual_account_id=ctx.virtual_account_id,
            real_total_asset=0.0,
        )
        fallback: List[OrderRecord] = []
        for item in list(getattr(state, "order_records", []) or []):
            if str(item.get("date", "") or "") != today:
                continue
            try:
                fallback.append(OrderRecord.from_dict(item))
            except Exception:
                continue
        return list(reversed(fallback))[:limit]

    def get_today_trades(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        limit: int = 200,
    ) -> List[Any]:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id:
            return []
        self._backfill_trades_from_local_orders(ctx)
        today, _, _ = self._today_bounds()
        records = self.trade_service.get_records(
            start_date=today,
            end_date=today,
            strategy_id=ctx.strategy_id,
            virtual_account_id=ctx.virtual_account_id,
            limit=limit,
        )
        if records:
            return records
        state = self.strategy_budget.get_strategy_state_record(
            ctx.strategy_id,
            strategy_name=ctx.strategy_name,
            virtual_account_id=ctx.virtual_account_id,
            real_total_asset=0.0,
        )
        fallback: List[TradeRecord] = []
        for item in list(getattr(state, "trade_history", []) or []):
            rec = self._trade_record_from_budget_history_item(ctx, item)
            if rec is None or rec.trade_date != today:
                continue
            fallback.append(rec)
        return list(reversed(fallback))[:limit]

    def get_trade_history(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        limit: int = 200,
    ) -> List[Any]:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id:
            return []
        self._backfill_trades_from_local_orders(ctx)
        records = self.trade_service.get_records(
            strategy_id=ctx.strategy_id,
            virtual_account_id=ctx.virtual_account_id,
            limit=limit,
        )
        if records:
            return records
        state = self.strategy_budget.get_strategy_state_record(
            ctx.strategy_id,
            strategy_name=ctx.strategy_name,
            virtual_account_id=ctx.virtual_account_id,
            real_total_asset=0.0,
        )
        fallback: List[TradeRecord] = []
        for item in list(getattr(state, "trade_history", []) or []):
            rec = self._trade_record_from_budget_history_item(ctx, item)
            if rec is None:
                continue
            fallback.append(rec)
        return list(reversed(fallback))[:limit]

    def get_capital_ledger(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id:
            return []
        state = self.strategy_budget.get_strategy_state_record(
            ctx.strategy_id,
            strategy_name=ctx.strategy_name,
            virtual_account_id=ctx.virtual_account_id,
            real_total_asset=0.0,
        )
        entries: List[Dict[str, Any]] = []
        for item in list(getattr(state, "capital_ledger", []) or []):
            if not isinstance(item, dict):
                continue
            entries.append(dict(item))
        return list(reversed(entries))[:limit]

    def get_equity_curve(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        limit: int = 365,
    ) -> List[Dict[str, Any]]:
        ctx = self._normalize_context(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        if not ctx.strategy_id:
            return []

        snapshots = self.trade_service.get_strategy_daily_pnl_snapshots(
            strategy_id=ctx.strategy_id,
            limit=limit,
        )
        rows: List[Dict[str, Any]] = []
        if snapshots:
            for snap in snapshots:
                rows.append(
                    {
                        "date": snap.snapshot_date,
                        "total_asset": round(float(snap.total_asset or 0.0), 2),
                        "cash": round(float(snap.cash or 0.0), 2),
                        "market_value": round(float(snap.market_value or 0.0), 2),
                        "daily_return_pct": 0.0,
                        "cumulative_return_pct": 0.0,
                    }
                )
            rows = self._override_ai_runtime_equity_row(ctx, rows)
            return self._recalculate_equity_metrics(rows)

        state = self.strategy_budget.get_strategy_state_record(
            ctx.strategy_id,
            strategy_name=ctx.strategy_name,
            virtual_account_id=ctx.virtual_account_id,
            real_total_asset=0.0,
        )
        equity_dict = dict(getattr(state, "daily_equity", {}) or {})
        if not equity_dict:
            return rows
        dates = sorted(equity_dict.keys())[-limit:]
        for date in dates:
            total_asset = round(float(equity_dict.get(date, 0.0) or 0.0), 2)
            rows.append(
                {
                    "date": date,
                    "total_asset": total_asset,
                    "cash": 0.0,
                    "market_value": 0.0,
                    "daily_return_pct": 0.0,
                    "cumulative_return_pct": 0.0,
                }
            )
        rows = self._override_ai_runtime_equity_row(ctx, rows)
        return self._recalculate_equity_metrics(rows)

    def _override_ai_runtime_equity_row(
        self,
        ctx: StrategyTradeViewContext,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if ctx.strategy_id != AI_STOCK_STRATEGY_ID:
            return rows
        today = datetime.now().strftime("%Y-%m-%d")
        runtime_row = self._build_ai_runtime_equity_row(ctx, today)
        if runtime_row is None:
            return rows
        updated = list(rows or [])
        if updated and str(updated[-1].get("date", "") or "") == today:
            updated[-1] = runtime_row
        else:
            updated.append(runtime_row)
        updated.sort(key=lambda item: str(item.get("date", "") or ""))
        return updated

    def _build_ai_runtime_equity_row(
        self,
        ctx: StrategyTradeViewContext,
        snapshot_date: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            snapshot = self.strategy_budget.get_strategy_snapshot(
                ctx.strategy_id,
                strategy_name=ctx.strategy_name,
                virtual_account_id=ctx.virtual_account_id,
                real_total_asset=0.0,
            )
            state = self.strategy_budget.get_strategy_state_record(
                ctx.strategy_id,
                strategy_name=ctx.strategy_name,
                virtual_account_id=ctx.virtual_account_id,
                real_total_asset=0.0,
            )
            positions = self.get_strategy_positions(
                ctx.strategy_id,
                strategy_name=ctx.strategy_name,
                virtual_account_id=ctx.virtual_account_id,
            )
            market_value = round(
                sum(float(item.get("market_value", 0.0) or 0.0) for item in (positions or [])),
                2,
            )
            capital_limit = float(snapshot.get("capital_limit", 0.0) or 0.0)
            invested_cost = round(
                sum(
                    float(getattr(pos, "avg_cost", 0.0) or 0.0) * int(getattr(pos, "quantity", 0) or 0)
                    for pos in state.get_positions().values()
                ),
                2,
            )
            realized_pnl = float(getattr(state, "realized_pnl", 0.0) or 0.0)
            cash = round(max(capital_limit + realized_pnl - invested_cost, 0.0), 2)
            return {
                "date": snapshot_date,
                "total_asset": round(cash + market_value, 2),
                "cash": cash,
                "market_value": market_value,
                "daily_return_pct": 0.0,
                "cumulative_return_pct": 0.0,
            }
        except Exception:
            return None

    @staticmethod
    def _recalculate_equity_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        recalculated: List[Dict[str, Any]] = []
        first_asset = float(rows[0].get("total_asset", 0.0) or 0.0)
        prev_asset = 0.0
        for item in rows:
            row = dict(item)
            total_asset = float(row.get("total_asset", 0.0) or 0.0)
            row["daily_return_pct"] = (
                round((total_asset - prev_asset) / prev_asset * 100, 2)
                if prev_asset > 0 else 0.0
            )
            row["cumulative_return_pct"] = (
                round((total_asset - first_asset) / first_asset * 100, 2)
                if first_asset > 0 else 0.0
            )
            recalculated.append(row)
            prev_asset = total_asset
        return recalculated

    def _fetch_realtime_price(self, code: str) -> float:
        try:
            from .quote_service import get_quote_service, to_xt_code
            quote = get_quote_service().get_quote(code)
            if (
                quote
                and float(getattr(quote, "last_price", 0) or 0) > 0
                and bool(getattr(quote, "is_fresh", False))
            ):
                return float(quote.last_price)
        except Exception:
            pass
        try:
            broker = self.broker
            if broker.is_connected:
                snapshot = get_market_data_gateway().get_price_snapshot(
                    code,
                    allow_daily_fallback=False,
                    require_fresh=True,
                )
                if snapshot.price > 0 and snapshot.is_fresh:
                    return float(snapshot.price)
        except Exception:
            pass
        return 0.0

    def _fetch_display_price(self, code: str) -> float:
        try:
            from .quote_service import get_quote_service
            quote = get_quote_service().get_quote(code)
            last_price = float(getattr(quote, "last_price", 0) or 0) if quote else 0.0
            if last_price > 0:
                return last_price
        except Exception:
            pass
        try:
            snapshot = get_market_data_gateway().get_price_snapshot(
                code,
                allow_daily_fallback=True,
                require_fresh=False,
            )
            if snapshot.price > 0:
                return float(snapshot.price)
        except Exception:
            pass
        return 0.0

    def _build_broker_position_map(self) -> Dict[str, Dict[str, Any]]:
        if not getattr(self.broker, "is_connected", False):
            return {}
        try:
            positions = self.broker.query_stock_positions() or []
        except Exception:
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for pos in positions:
            try:
                code = normalize_symbol_code(str(getattr(pos, "stock_code", "") or ""))
                volume = int(getattr(pos, "volume", 0) or 0)
                if not code or volume <= 0:
                    continue
                market_value = float(getattr(pos, "market_value", 0.0) or 0.0)
                current_price = round(market_value / volume, 4) if market_value > 0 else 0.0
                result[code] = {
                    "stock_name": str(getattr(pos, "stock_name", "") or ""),
                    "volume": volume,
                    "can_use_volume": int(getattr(pos, "can_use_volume", 0) or 0),
                    "open_price": float(getattr(pos, "open_price", 0.0) or 0.0),
                    "market_value": market_value,
                    "current_price": current_price,
                }
            except Exception:
                continue
        return result

    def _build_name_map(self, ctx: StrategyTradeViewContext) -> Dict[str, str]:
        """Build code→name mapping from trade records in SQLite."""
        result: Dict[str, str] = {}
        try:
            records = self.trade_service.get_records(
                strategy_id=ctx.strategy_id,
                virtual_account_id=ctx.virtual_account_id,
                limit=2000,
            )
            for rec in records or []:
                code = normalize_symbol_code(getattr(rec, "stock_code", "") or "")
                name = str(getattr(rec, "stock_name", "") or "").strip()
                if code and name:
                    result[code] = name
        except Exception:
            pass
        return result

    @staticmethod
    def _resolve_local_symbol_snapshot(state: Any, code: str, runtime_state: Dict[str, Any]) -> tuple[str, float]:
        normalized = normalize_symbol_code(code)
        if runtime_state.get("current_holding") == normalized:
            return (
                str(runtime_state.get("current_holding_name", "") or normalized),
                0.0,
            )
        for item in reversed(list(getattr(state, "trade_history", []) or [])):
            item_code = normalize_symbol_code(str(item.get("code", "") or ""))
            if item_code != normalized:
                continue
            return (
                str(item.get("name", "") or normalized),
                float(item.get("price", 0.0) or 0.0),
            )
        for item in reversed(list(getattr(state, "order_records", []) or [])):
            item_code = normalize_symbol_code(str(item.get("code", "") or ""))
            if item_code != normalized:
                continue
            return (
                str(item.get("name", "") or normalized),
                float(item.get("filled_price", 0.0) or item.get("ordered_price", 0.0) or 0.0),
            )
        return normalized, 0.0


_strategy_trade_view_service: Optional[StrategyTradeViewService] = None


def get_strategy_trade_view_service() -> StrategyTradeViewService:
    global _strategy_trade_view_service
    if _strategy_trade_view_service is None:
        _strategy_trade_view_service = StrategyTradeViewService()
    return _strategy_trade_view_service
