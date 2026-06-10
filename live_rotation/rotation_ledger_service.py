"""
ETF rotation ledger service.

This module owns ETF rotation order records, capital ledger entries, unified
strategy budget synchronization, and cash/equity calculations. RotationEngine
keeps compatibility wrappers while delegating ledger responsibilities here.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional, Tuple

from .config import RotationConfig
from .order_state_machine import OrderStatus, resolve_order_status
from .state_manager import CapitalLedgerEntry, OrderRecord, RotationState, StateManager
from .trade_executor import SimulatedExecutor, TradeExecutor

logger = logging.getLogger(__name__)


class RotationLedgerService:
    """Manage ETF rotation ledger, order records, and budget sync."""

    ACTIVE_ORDER_STATUSES = {"pending_submit", "pending_fill", "submitted"}

    def __init__(
        self,
        *,
        config: RotationConfig,
        state: RotationState,
        state_mgr: StateManager,
        executor: TradeExecutor,
        strategy_identity_fn: Callable[[], Tuple[str, str, str]],
        code_name_map_fn: Optional[Callable[[str], str]] = None,
        logger_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.state = state
        self.state_mgr = state_mgr
        self.executor = executor
        self.strategy_identity_fn = strategy_identity_fn
        self.code_name_map_fn = code_name_map_fn or (lambda code: "")
        self.logger_fn = logger_fn or (lambda message: None)

    def update_context(
        self,
        *,
        config: RotationConfig,
        state: RotationState,
        executor: TradeExecutor,
    ) -> None:
        """Refresh mutable config, state, and executor references."""
        self.config = config
        self.state = state
        self.executor = executor

    def add_capital_entry(
        self,
        action: str,
        code: str = "",
        name: str = "",
        amount: float = 0.0,
        commission: float = 0.0,
        fee_source: str = "",
    ) -> None:
        """Append one capital ledger entry."""
        if isinstance(self.executor, SimulatedExecutor):
            return
        if not self.config.use_dedicated_capital:
            return
        now = datetime.now()
        entry = CapitalLedgerEntry(
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            action=action,
            code=code,
            name=name,
            amount=amount,
            commission=commission,
            balance=round(self.ledger_available_cash(), 2),
            fee_source=fee_source,
        )
        self.state_mgr.add_capital_entry(entry)

    def record_daily_equity(self) -> None:
        """Record today's equity snapshot."""
        try:
            equity = self.total_asset()
            if equity > 0:
                self.state_mgr.record_daily_equity(equity)
        except Exception as e:
            logger.debug(f"记录净值快照失败: {e}")

    def add_order_record(
        self,
        order_id: int,
        action: str,
        code: str,
        ordered_qty: int,
        ordered_price: float,
        reason: str = "",
        status: str = OrderStatus.PENDING_FILL,
    ) -> OrderRecord:
        """Create and save an order record."""
        name = self.code_name_map_fn(code)
        now = datetime.now()
        record = OrderRecord(
            order_id=order_id,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            action=action,
            code=code,
            name=name,
            ordered_qty=ordered_qty,
            ordered_price=ordered_price,
            status=status,
            reason=reason,
        )
        self.state_mgr.add_order_record(record)
        return record

    def update_order_record(self, order_id: int, fill: dict, pnl: float = 0.0) -> None:
        """Update order record fields from fill confirmation result."""
        filled_qty = fill.get('filled_qty', 0)
        filled_price = fill.get('filled_price', 0.0)
        commission = fill.get('commission', -1.0)
        status = resolve_order_status(fill)
        self.state_mgr.update_order_record(
            order_id,
            filled_qty=filled_qty,
            filled_price=filled_price,
            commission=commission,
            status=status,
            pnl=pnl,
        )

    def active_order_records(self) -> list[dict]:
        """Return local ETF orders that are not in a terminal state yet."""
        active = []
        for record in list(self.state.order_records or []):
            status = str(record.get("status", "") or "").strip().lower()
            if status in self.ACTIVE_ORDER_STATUSES:
                active.append(dict(record))
        return active

    def has_active_order_records(self) -> bool:
        """Whether this strategy still has submitted/pending local ETF orders."""
        return bool(self.active_order_records())

    def resolve_trade_fees(
        self,
        *,
        direction: str,
        amount: float,
        stock_code: str,
        actual_commission: float = -1.0,
    ) -> dict:
        """Read global trade fee config and override commission when broker returns it."""
        try:
            from trading_app.services.execution.trade_record_service import get_trade_record_service

            fees = dict(
                get_trade_record_service().estimate_trade_fees(
                    direction=direction,
                    amount=amount,
                    stock_code=stock_code,
                )
                or {}
            )
        except Exception:
            fees = {
                "commission": 0.0,
                "stamp_tax": 0.0,
                "transfer_fee": 0.0,
                "total_fee": 0.0,
            }
        if float(actual_commission or -1.0) >= 0:
            fees["commission"] = round(float(actual_commission or 0.0), 2)
        fees["commission"] = round(float(fees.get("commission", 0.0) or 0.0), 2)
        fees["stamp_tax"] = round(float(fees.get("stamp_tax", 0.0) or 0.0), 2)
        fees["transfer_fee"] = round(float(fees.get("transfer_fee", 0.0) or 0.0), 2)
        fees["total_fee"] = round(
            float(fees["commission"]) + float(fees["stamp_tax"]) + float(fees["transfer_fee"]),
            2,
        )
        return fees

    def sync_unified_ledger_on_buy(
        self,
        *,
        code: str,
        name: str,
        price: float,
        volume: int,
        commission: float,
        stamp_tax: float,
        transfer_fee: float,
        broker_order_id: int,
        reason: str,
    ) -> None:
        """Sync successful ETF buy into trade records and strategy budget."""
        if isinstance(self.executor, SimulatedExecutor):
            return
        if price <= 0 or volume <= 0:
            return
        strategy_id, strategy_name, virtual_account_id = self.strategy_identity_fn()
        try:
            from trading_app.services.execution.trade_record_service import get_trade_record_service
            from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

            get_trade_record_service().add_record(
                stock_code=code,
                stock_name=name or code,
                direction="buy",
                price=float(price),
                volume=int(volume),
                broker_order_id=int(broker_order_id or -1),
                source="etf_rotation",
                strategy_id=strategy_id,
                virtual_account_id=virtual_account_id,
                remark=reason or "",
                commission=round(float(commission or 0.0), 2),
                stamp_tax=round(float(stamp_tax or 0.0), 2),
                transfer_fee=round(float(transfer_fee or 0.0), 2),
            )
            get_strategy_budget_service().commit_buy(
                strategy_id=strategy_id,
                symbol_code=code,
                price=float(price),
                volume=int(volume),
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                commission=round(float(commission or 0.0), 2),
                stamp_tax=round(float(stamp_tax or 0.0), 2),
                transfer_fee=round(float(transfer_fee or 0.0), 2),
            )
        except Exception as exc:
            logger.error(f"ETF 同步成交记录/主账本失败（买入）: {exc}")

    def sync_unified_ledger_on_sell(
        self,
        *,
        code: str,
        name: str,
        price: float,
        volume: int,
        commission: float,
        stamp_tax: float,
        transfer_fee: float,
        broker_order_id: int,
        reason: str,
    ) -> None:
        """Sync successful ETF sell into trade records and strategy budget."""
        if isinstance(self.executor, SimulatedExecutor):
            return
        if price <= 0 or volume <= 0:
            return
        strategy_id, strategy_name, virtual_account_id = self.strategy_identity_fn()
        try:
            from trading_app.services.execution.trade_record_service import get_trade_record_service
            from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

            get_trade_record_service().add_record(
                stock_code=code,
                stock_name=name or code,
                direction="sell",
                price=float(price),
                volume=int(volume),
                broker_order_id=int(broker_order_id or -1),
                source="etf_rotation",
                strategy_id=strategy_id,
                virtual_account_id=virtual_account_id,
                remark=reason or "",
                commission=round(float(commission or 0.0), 2),
                stamp_tax=round(float(stamp_tax or 0.0), 2),
                transfer_fee=round(float(transfer_fee or 0.0), 2),
            )
            get_strategy_budget_service().commit_sell(
                strategy_id=strategy_id,
                symbol_code=code,
                price=float(price),
                volume=int(volume),
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                commission=round(float(commission or 0.0), 2),
                stamp_tax=round(float(stamp_tax or 0.0), 2),
                transfer_fee=round(float(transfer_fee or 0.0), 2),
            )
        except Exception as exc:
            logger.error(f"ETF 同步成交记录/主账本失败（卖出）: {exc}")

    def available_cash(self) -> float:
        """Return strategy available cash."""
        if isinstance(self.executor, SimulatedExecutor):
            return self.executor.cash
        return self.ledger_available_cash()

    def ledger_available_cash(self) -> float:
        """Read current available cash from the strategy budget ledger."""
        try:
            from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

            strategy_id, strategy_name, virtual_account_id = self.strategy_identity_fn()
            budget = get_strategy_budget_service().get_available_budget(
                strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
            )
            return float(budget.get("available", 0.0) or 0.0)
        except Exception as e:
            logger.error(f"查询可用资金失败（账本层）: {e}")
            return 0.0

    def init_dedicated_capital(self) -> None:
        """Ensure the strategy budget ledger has this strategy's capital limit."""
        if isinstance(self.executor, SimulatedExecutor):
            return
        if not self.config.use_dedicated_capital:
            return
        try:
            from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

            strategy_id, strategy_name, virtual_account_id = self.strategy_identity_fn()
            get_strategy_budget_service().upsert_strategy_config(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                capital_limit=float(self.config.dedicated_capital or 0.0),
            )
        except Exception as exc:
            logger.error(f"主账本启动资金初始化失败: {exc}")

    def reset_dedicated_capital(self, new_capital: Optional[float] = None) -> None:
        """Reset this strategy's capital limit and realized budget state."""
        cap = float(new_capital if new_capital is not None else self.config.dedicated_capital)
        try:
            from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

            strategy_id, strategy_name, virtual_account_id = self.strategy_identity_fn()
            service = get_strategy_budget_service()
            service.upsert_strategy_config(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                capital_limit=cap,
            )
            service.reset_strategy_account(strategy_id)
        except Exception as exc:
            logger.error(f"主账本重置失败: {exc}")

        today = datetime.now().strftime("%Y-%m-%d")
        holding_value = 0.0
        if self.state.current_holding and self.state.buy_quantity > 0:
            price = self.executor.get_current_price(self.state.current_holding)
            if price <= 0:
                price = self.state.buy_price
            if price > 0:
                holding_value = price * self.state.buy_quantity
        self.state.daily_equity[today] = round(cap + holding_value, 2)
        self.state_mgr.save()
        self.logger_fn(f"💰 专用资金账本已重置为: {cap:,.0f} 元")
        self.add_capital_entry("手动重置", amount=cap)

    def clear_analytics_data(self) -> None:
        """Clear historical analysis data while keeping positions and budget balance."""
        self.state.trade_history = []
        self.state.order_records = []
        self.state.capital_ledger = []
        self.state.daily_equity = {}
        self.state.total_pnl = 0.0
        self.state_mgr.save()
        self.logger_fn("🗑 历史分析数据已全部清空")

    def total_asset(self) -> float:
        """Calculate strategy total asset."""
        if self.config.use_dedicated_capital:
            equity = self.ledger_available_cash()
            if self.state.current_holding and self.state.buy_quantity > 0:
                price = self.executor.get_current_price(self.state.current_holding)
                if price <= 0:
                    price = self.state.buy_price
                if price > 0:
                    equity += price * self.state.buy_quantity
            return equity

        if isinstance(self.executor, SimulatedExecutor):
            cash = self.executor.cash
            position_value = 0.0
            if self.state.current_holding:
                price = self.executor.get_current_price(self.state.current_holding)
                if price > 0:
                    position_value = price * self.state.buy_quantity
            return cash + position_value

        try:
            if hasattr(self.executor, 'query_total_asset'):
                value = float(self.executor.query_total_asset())
                if value > 0:
                    return value
        except Exception as e:
            logger.error(f"查询总资产失败: {e}")

        cash = self.available_cash()
        if self.state.current_holding and self.state.buy_quantity > 0:
            price = self.executor.get_current_price(self.state.current_holding)
            if price <= 0:
                price = self.state.buy_price
            if price > 0:
                return cash + price * self.state.buy_quantity
        return cash
