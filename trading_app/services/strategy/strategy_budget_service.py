from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.io_utils import atomic_write_json

from trading_app.services.strategy_constants import (
    AI_STOCK_STRATEGY_ID,
    OWNER_TYPE_UNMANAGED,
    UNMANAGED_STRATEGY_ID,
    UNMANAGED_STRATEGY_NAME,
    UNMANAGED_VIRTUAL_ACCOUNT_ID,
    normalize_symbol_code,
)
from trading_app.services.strategy_spec_service import get_strategy_spec_service

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "strategy_budget_config.json"
_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "strategy_budget_state.json"


@dataclass
class StrategyBudgetConfig:
    strategy_id: str
    strategy_name: str = ""
    virtual_account_id: str = ""
    capital_limit: float = 0.0
    enabled: bool = True
    is_test: bool = False
    hidden: bool = False
    is_unmanaged: bool = False  # 标记：未管理账户（承载券商未认领的现金/持仓），不允许下单
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.strategy_id = (self.strategy_id or "").strip()
        self.strategy_name = (self.strategy_name or "").strip()
        self.virtual_account_id = (self.virtual_account_id or "").strip()
        self.capital_limit = float(self.capital_limit or 0.0)
        self.enabled = bool(self.enabled)
        self.is_test = bool(self.is_test)
        self.hidden = bool(self.hidden)
        self.is_unmanaged = bool(self.is_unmanaged)
        if not self.updated_at:
            self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyBudgetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyPositionState:
    symbol_code: str
    quantity: int = 0
    avg_cost: float = 0.0
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.symbol_code = normalize_symbol_code(self.symbol_code)
        self.quantity = int(self.quantity or 0)
        self.avg_cost = float(self.avg_cost or 0.0)
        if not self.updated_at:
            self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyPositionState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyBudgetState:
    strategy_id: str
    strategy_name: str = ""
    virtual_account_id: str = ""
    capital_limit: float = 0.0
    cash_balance: float = 0.0
    reserved_cash: float = 0.0
    realized_pnl: float = 0.0
    positions: Dict[str, dict] = field(default_factory=dict)
    reservations: Dict[str, float] = field(default_factory=dict)
    runtime_state: Dict[str, object] = field(default_factory=dict)
    trade_history: List[dict] = field(default_factory=list)
    capital_ledger: List[dict] = field(default_factory=list)
    daily_equity: Dict[str, float] = field(default_factory=dict)
    order_records: List[dict] = field(default_factory=list)
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.strategy_id = (self.strategy_id or "").strip()
        self.strategy_name = (self.strategy_name or "").strip()
        self.virtual_account_id = (self.virtual_account_id or "").strip()
        self.capital_limit = float(self.capital_limit or 0.0)
        self.cash_balance = float(self.cash_balance or 0.0)
        self.reserved_cash = float(self.reserved_cash or 0.0)
        self.realized_pnl = float(self.realized_pnl or 0.0)
        self.positions = {
            normalize_symbol_code(code): StrategyPositionState.from_dict(value).to_dict()
            for code, value in (self.positions or {}).items()
            if normalize_symbol_code(code)
        }
        self.reservations = {
            str(key): float(value or 0.0)
            for key, value in (self.reservations or {}).items()
            if str(key)
        }
        self.runtime_state = dict(self.runtime_state or {})
        self.trade_history = list(self.trade_history or [])
        self.capital_ledger = list(self.capital_ledger or [])
        self.daily_equity = {
            str(key): round(float(value or 0.0), 2)
            for key, value in (self.daily_equity or {}).items()
            if str(key)
        }
        self.order_records = list(self.order_records or [])
        if not self.updated_at:
            self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return asdict(self)

    def get_positions(self) -> Dict[str, StrategyPositionState]:
        return {
            code: StrategyPositionState.from_dict(data)
            for code, data in (self.positions or {}).items()
        }

    def available_cash(self) -> float:
        return max(float(self.cash_balance or 0.0) - float(self.reserved_cash or 0.0), 0.0)

    def invested_market_value(self) -> float:
        return round(
            sum(max(pos.quantity, 0) * max(pos.avg_cost, 0.0) for pos in self.get_positions().values()),
            2,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyBudgetState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StrategyBudgetService:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ):
        self.config_path = config_path or _CONFIG_PATH
        self.state_path = state_path or _STATE_PATH
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, StrategyBudgetConfig] = {}
        self._states: Dict[str, StrategyBudgetState] = {}
        self._load()

    def _load(self) -> None:
        self._configs = self._load_configs()
        self._states = self._load_states()
        config_changed = self._ensure_default_configs()
        config_changed = self._migrate_strategy_visibility_flags() or config_changed
        if config_changed or not self.config_path.exists():
            self._save_configs()
        if not self.state_path.exists():
            self._save_states()

    @staticmethod
    def _looks_like_test_strategy(strategy_id: str, strategy_name: str = "") -> bool:
        text = f"{strategy_id} {strategy_name}".strip().lower()
        if not text:
            return False
        keywords = (
            "smoke",
            "test",
            "debug",
            "demo",
            "tmp",
            "temp",
            "sandbox",
            "mock",
        )
        return any(keyword in text for keyword in keywords)

    def _migrate_strategy_visibility_flags(self) -> bool:
        changed = False
        builtin_strategy_ids = {
            spec.strategy_id
            for spec in get_strategy_spec_service().builtin_specs()
        }
        for strategy_id, cfg in self._configs.items():
            if strategy_id in builtin_strategy_ids:
                continue
            looks_like_test = self._looks_like_test_strategy(strategy_id, cfg.strategy_name)
            updated = False
            if looks_like_test and not cfg.is_test:
                cfg.is_test = True
                updated = True
            if looks_like_test and not cfg.hidden:
                cfg.hidden = True
                updated = True
            if updated:
                cfg.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
        return changed

    def _load_configs(self) -> Dict[str, StrategyBudgetConfig]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
        except Exception as exc:
            logger.warning("读取策略预算配置失败，使用空配置重建: %s", exc)
            return {}
        items = raw.get("strategies", {}) if isinstance(raw, dict) else {}
        if isinstance(items, list):
            iterable = items
        else:
            iterable = items.values()
        result: Dict[str, StrategyBudgetConfig] = {}
        for item in iterable:
            try:
                cfg = StrategyBudgetConfig.from_dict(item or {})
            except Exception:
                continue
            if cfg.strategy_id:
                result[cfg.strategy_id] = cfg
        return result

    def _load_states(self) -> Dict[str, StrategyBudgetState]:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
        except Exception as exc:
            logger.warning("读取策略预算状态失败，使用空状态重建: %s", exc)
            return {}
        items = raw.get("strategies", {}) if isinstance(raw, dict) else {}
        result: Dict[str, StrategyBudgetState] = {}
        if isinstance(items, list):
            iterable = items
        else:
            iterable = items.values()
        for item in iterable:
            try:
                state = StrategyBudgetState.from_dict(item or {})
            except Exception:
                continue
            if state.strategy_id:
                result[state.strategy_id] = state
        return result

    def _save_configs(self) -> None:
        payload = {
            "strategies": {
                strategy_id: cfg.to_dict()
                for strategy_id, cfg in sorted(self._configs.items(), key=lambda item: item[0])
            }
        }
        atomic_write_json(self.config_path, payload)

    def _save_states(self) -> None:
        payload = {
            "strategies": {
                strategy_id: state.to_dict()
                for strategy_id, state in sorted(self._states.items(), key=lambda item: item[0])
            }
        }
        atomic_write_json(self.state_path, payload)

    def _ensure_default_configs(self) -> bool:
        changed = False
        for spec in get_strategy_spec_service().builtin_specs():
            cfg = self._configs.get(spec.strategy_id)
            if cfg is None:
                self._configs[spec.strategy_id] = StrategyBudgetConfig(**spec.to_budget_kwargs())
                changed = True
                continue
            updated = False
            if not cfg.strategy_name and spec.strategy_name:
                cfg.strategy_name = spec.strategy_name
                updated = True
            if not cfg.virtual_account_id and spec.virtual_account_id:
                cfg.virtual_account_id = spec.virtual_account_id
                updated = True
            if cfg.capital_limit <= 0 and spec.capital_limit > 0:
                cfg.capital_limit = spec.capital_limit
                updated = True
            if cfg.enabled != spec.enabled and spec.is_unmanaged:
                cfg.enabled = spec.enabled
                updated = True
            if cfg.is_unmanaged != spec.is_unmanaged:
                cfg.is_unmanaged = spec.is_unmanaged
                updated = True
            if updated:
                cfg.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
        return changed

    def _effective_capital_limit(self, strategy_id: str, real_total_asset: float = 0.0) -> float:
        cfg = self._configs.get(strategy_id)
        if cfg is None:
            return 0.0
        explicit = float(cfg.capital_limit or 0.0)
        if explicit > 0:
            return explicit
        total_asset = float(real_total_asset or 0.0)
        if total_asset <= 0:
            return 0.0
        reserved_by_others = sum(
            max(float(item.capital_limit or 0.0), 0.0)
            for sid, item in self._configs.items()
            if sid != strategy_id and bool(item.enabled)
        )
        return max(total_asset - reserved_by_others, 0.0)

    def _ensure_strategy(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
    ) -> StrategyBudgetState:
        strategy_id = (strategy_id or "").strip()
        if not strategy_id:
            raise ValueError("strategy_id 不能为空")

        cfg = self._configs.get(strategy_id)
        if cfg is None:
            looks_like_test = self._looks_like_test_strategy(strategy_id, strategy_name)
            cfg = StrategyBudgetConfig(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                capital_limit=0.0,
                enabled=True,
                is_test=looks_like_test,
                hidden=looks_like_test,
            )
            self._configs[strategy_id] = cfg
            self._save_configs()
        else:
            updated = False
            if strategy_name and not cfg.strategy_name:
                cfg.strategy_name = strategy_name
                updated = True
            if virtual_account_id and not cfg.virtual_account_id:
                cfg.virtual_account_id = virtual_account_id
                updated = True
            if updated:
                cfg.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._save_configs()

        state = self._states.get(strategy_id)
        initial_capital = self._effective_capital_limit(strategy_id, real_total_asset=real_total_asset)
        if state is None:
            state = StrategyBudgetState(
                strategy_id=strategy_id,
                strategy_name=strategy_name or cfg.strategy_name,
                virtual_account_id=virtual_account_id or cfg.virtual_account_id,
                capital_limit=initial_capital,
                cash_balance=initial_capital,
                reserved_cash=0.0,
                realized_pnl=0.0,
            )
            self._states[strategy_id] = state
            self._save_states()
            return state

        updated = False
        if strategy_name and not state.strategy_name:
            state.strategy_name = strategy_name
            updated = True
        if virtual_account_id and not state.virtual_account_id:
            state.virtual_account_id = virtual_account_id
            updated = True
        if state.capital_limit <= 0 and initial_capital > 0 and not state.positions and state.cash_balance <= 0:
            state.capital_limit = initial_capital
            state.cash_balance = initial_capital
            updated = True
        if updated:
            state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_states()
        return state

    def get_strategy_snapshot(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
    ) -> dict:
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._rehydrate_from_trade_records_if_needed(
            state,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._refresh_accounting_from_trade_records_if_needed(
            state,
            virtual_account_id=virtual_account_id,
        )
        cfg = self._configs.get(strategy_id)
        return {
            "strategy_id": state.strategy_id,
            "strategy_name": state.strategy_name,
            "virtual_account_id": state.virtual_account_id,
            "capital_limit": round(float(state.capital_limit or 0.0), 2),
            "cash_balance": round(float(state.cash_balance or 0.0), 2),
            "reserved_cash": round(float(state.reserved_cash or 0.0), 2),
            "available_cash": round(state.available_cash(), 2),
            "invested_market_value": round(state.invested_market_value(), 2),
            "position_count": len([pos for pos in state.get_positions().values() if pos.quantity > 0]),
            "realized_pnl": round(float(state.realized_pnl or 0.0), 2),
            "enabled": bool(getattr(cfg, "enabled", True)),
            "is_test": bool(getattr(cfg, "is_test", False)),
            "hidden": bool(getattr(cfg, "hidden", False)),
            "is_unmanaged": bool(getattr(cfg, "is_unmanaged", False)),
        }

    @staticmethod
    def _extract_code(entry: dict) -> str:
        for key in ("symbol_code", "stock_code", "code"):
            value = entry.get(key) if isinstance(entry, dict) else None
            if value:
                return normalize_symbol_code(str(value))
        return ""

    @staticmethod
    def _extract_volume(entry: dict) -> int:
        for key in ("quantity", "volume", "can_use_volume"):
            value = entry.get(key) if isinstance(entry, dict) else None
            if value:
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    continue
        return 0

    def _build_live_map(
        self,
        live_positions: Optional[List[dict]],
    ) -> Dict[str, dict]:
        """把外部传入的 live_positions 归一化为 {normalized_code: entry}。
        支持的字段别名：code/stock_code/symbol_code、volume/quantity、market_value。
        """
        result: Dict[str, dict] = {}
        for item in live_positions or []:
            if not isinstance(item, dict):
                continue
            code = self._extract_code(item)
            if not code:
                continue
            market_value = float(item.get("market_value", 0.0) or 0.0)
            volume = self._extract_volume(item)
            result[code] = {
                "market_value": market_value,
                "volume": volume,
                "name": str(item.get("name", "") or item.get("stock_name", "") or ""),
            }
        return result

    def finalize_day(
        self,
        snapshot_date: Optional[str] = None,
        *,
        providers: Optional[Dict[str, dict]] = None,
        include_hidden: bool = False,
        include_test: bool = False,
        remark: str = "",
    ) -> List[Dict[str, object]]:
        """策略日终快照固化入口（统一收口）。

        职责：
          - 以主账本为口径，为每个活跃策略组装 build_account_snapshot + get_positions_view
            + 当日成交统计，一次性落盘到 trade_record_service 的三张快照表。
          - 成为 EOD 对账 / PnL 曲线 / 回测数据等的"**唯一数据源**"。

        Args:
            snapshot_date: 快照日期 YYYY-MM-DD（默认今天）
            providers: 策略粒度的行情/cash/上限覆盖，形如：
                {
                    strategy_id: {
                        "live_positions": [...],
                        "spot_prices": {code: price, ...},
                        "cash_override": float,
                        "capital_limit_override": float,
                        "remark": str,
                    },
                    ...
                }
                未提供行情的策略会以"持仓成本价"兜底（浮盈=0）。
            include_hidden / include_test: 是否包括 hidden / is_test 策略，默认均为 False
            remark: 统一写入备注（provider 里的 remark 优先）

        Returns:
            每个策略保存结果的摘要列表。
        """
        try:
            from .trade_record_service import get_trade_record_service
        except ImportError:  # 被其它包作为 top-level 导入时
            from trading_app.services.trade_record_service import get_trade_record_service  # type: ignore

        trade_service = get_trade_record_service()
        snapshot_date = str(snapshot_date or datetime.now().strftime("%Y-%m-%d"))
        providers = providers or {}

        target_ids: List[str] = []
        seen: set = set()
        for sid, cfg in self._configs.items():
            if not sid or sid in seen:
                continue
            if not include_test and bool(getattr(cfg, "is_test", False)):
                continue
            if not include_hidden and bool(getattr(cfg, "hidden", False)):
                continue
            seen.add(sid)
            target_ids.append(sid)
        for sid in providers.keys():
            if sid and sid not in seen:
                seen.add(sid)
                target_ids.append(sid)

        results: List[Dict[str, object]] = []
        positions_by_strategy: Dict[str, Dict[str, object]] = {}

        for strategy_id in target_ids:
            provider = dict(providers.get(strategy_id) or {})
            cfg = self._configs.get(strategy_id)
            strategy_name = str(getattr(cfg, "strategy_name", "") or "")
            virtual_account_id = str(getattr(cfg, "virtual_account_id", "") or "")
            spot_prices = provider.get("spot_prices") or None
            live_positions = provider.get("live_positions") or None
            cash_override = provider.get("cash_override")
            capital_limit_override = provider.get("capital_limit_override")
            strategy_remark = str(provider.get("remark", "") or remark or "")

            try:
                account = self.build_account_snapshot(
                    strategy_id,
                    strategy_name=strategy_name,
                    virtual_account_id=virtual_account_id,
                    spot_prices=spot_prices,
                    live_positions=live_positions,
                    cash_override=cash_override,
                    capital_limit_override=capital_limit_override,
                )
                positions_view = self.get_positions_view(
                    strategy_id,
                    strategy_name=strategy_name,
                    virtual_account_id=virtual_account_id,
                    spot_prices=spot_prices,
                    live_positions=live_positions,
                )
                period_stats = trade_service.get_period_stats(
                    strategy_ids=strategy_id,
                    start_date=snapshot_date,
                    end_date=snapshot_date,
                )

                pnl_snapshot = trade_service.save_strategy_daily_pnl_snapshot(
                    snapshot_date=snapshot_date,
                    strategy_id=strategy_id,
                    strategy_name=str(account.get("strategy_name") or strategy_name),
                    virtual_account_id=str(account.get("virtual_account_id") or virtual_account_id),
                    total_asset=float(account.get("total_asset", 0.0) or 0.0),
                    cash=float(account.get("available_cash", 0.0) or 0.0),
                    market_value=float(account.get("market_value", 0.0) or 0.0),
                    position_count=int(account.get("position_count", 0) or 0),
                    capital_limit=float(account.get("capital_limit", 0.0) or 0.0),
                    invested_cost=float(account.get("invested_cost", 0.0) or 0.0),
                    realized_pnl=float(account.get("realized_pnl", 0.0) or 0.0),
                    unrealized_pnl=float(account.get("unrealized_pnl", 0.0) or 0.0),
                    total_pnl=float(account.get("total_pnl", 0.0) or 0.0),
                    remark=strategy_remark,
                )
                trade_service.save_strategy_daily_trade_summary(
                    snapshot_date=snapshot_date,
                    strategy_id=strategy_id,
                    strategy_name=str(account.get("strategy_name") or strategy_name),
                    virtual_account_id=str(account.get("virtual_account_id") or virtual_account_id),
                    trade_count=int(period_stats.get("total_trades", 0) or 0),
                    buy_count=int(period_stats.get("buy_count", 0) or 0),
                    sell_count=int(period_stats.get("sell_count", 0) or 0),
                    total_buy_amount=float(period_stats.get("buy_amount", 0.0) or 0.0),
                    total_sell_amount=float(period_stats.get("sell_amount", 0.0) or 0.0),
                    total_commission=float(period_stats.get("total_fee", 0.0) or 0.0),
                    remark=strategy_remark,
                )
                positions_by_strategy[strategy_id] = {
                    "strategy_name": str(account.get("strategy_name") or strategy_name),
                    "virtual_account_id": str(account.get("virtual_account_id") or virtual_account_id),
                    "positions": [
                        {
                            "stock_code": p.get("stock_code", ""),
                            "stock_name": p.get("stock_name", "") or p.get("stock_code", ""),
                            "volume": int(p.get("quantity", 0) or 0),
                            "can_use_volume": int(p.get("quantity", 0) or 0),
                            "open_price": float(p.get("avg_cost", 0.0) or 0.0),
                            "cost_price": float(p.get("avg_cost", 0.0) or 0.0),
                            "cost_amount": float(p.get("cost_amount", 0.0) or 0.0),
                            "market_value": float(p.get("market_value", 0.0) or 0.0),
                        }
                        for p in positions_view
                    ],
                }

                results.append(
                    {
                        "strategy_id": strategy_id,
                        "strategy_name": str(account.get("strategy_name") or strategy_name),
                        "snapshot_date": snapshot_date,
                        "saved": pnl_snapshot is not None,
                        "total_asset": float(account.get("total_asset", 0.0) or 0.0),
                        "total_pnl": float(account.get("total_pnl", 0.0) or 0.0),
                        "realized_pnl": float(account.get("realized_pnl", 0.0) or 0.0),
                        "unrealized_pnl": float(account.get("unrealized_pnl", 0.0) or 0.0),
                        "position_count": int(account.get("position_count", 0) or 0),
                        "trade_count": int(period_stats.get("total_trades", 0) or 0),
                    }
                )
            except Exception as exc:
                logger.warning("finalize_day 策略 %s 失败: %s", strategy_id, exc)
                results.append(
                    {
                        "strategy_id": strategy_id,
                        "snapshot_date": snapshot_date,
                        "saved": False,
                        "error": str(exc),
                    }
                )

        if positions_by_strategy:
            try:
                trade_service.save_strategy_position_snapshots(
                    snapshot_date, positions_by_strategy
                )
            except Exception as exc:
                logger.warning("finalize_day 保存持仓快照失败: %s", exc)

        return results

    def get_positions_view(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        spot_prices: Optional[Dict[str, float]] = None,
        live_positions: Optional[List[dict]] = None,
        include_zero: bool = False,
    ) -> List[Dict[str, object]]:
        """统一的"策略持仓明细"视图（主账本口径）。

        数量与均价来自主账本（state.positions），价格来源按优先级：
          1. spot_prices[code]   —— 调用方直接传入的当前价
          2. live_positions 中匹配到的 market_value（直接使用，不再反推价）
          3. avg_cost            —— 退化兜底，浮盈为 0

        Args:
            strategy_id: 策略标识
            spot_prices: {code -> 当前价}，code 会自动 normalize
            live_positions: 券商查回来的持仓 [{code, market_value, volume?}, ...]，
                字段名允许 code/stock_code/symbol_code、volume/quantity、market_value
            include_zero: 是否包括 quantity<=0 的历史仓位（默认过滤）

        Returns:
            [{
                strategy_id, stock_code, stock_name,
                quantity, avg_cost, cost_amount,
                current_price, market_value,
                unrealized_pnl, unrealized_pnl_pct,
                weight_in_strategy,
                has_live_price,
            }, ...]，按 market_value 倒序
        """
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._rehydrate_from_trade_records_if_needed(
            state,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._refresh_accounting_from_trade_records_if_needed(
            state,
            virtual_account_id=virtual_account_id,
        )

        price_map: Dict[str, float] = {}
        for code, price in (spot_prices or {}).items():
            normalized = normalize_symbol_code(str(code))
            if not normalized:
                continue
            try:
                price_map[normalized] = float(price or 0.0)
            except (TypeError, ValueError):
                continue
        live_map = self._build_live_map(live_positions)

        rows: List[Dict[str, object]] = []
        for code, position in state.get_positions().items():
            quantity = int(position.quantity or 0)
            if quantity <= 0 and not include_zero:
                continue
            normalized = normalize_symbol_code(code)
            avg_cost = float(position.avg_cost or 0.0)
            cost_amount = round(quantity * avg_cost, 2)

            current_price = 0.0
            market_value = 0.0
            has_live_price = False
            if normalized in price_map and price_map[normalized] > 0:
                current_price = price_map[normalized]
                market_value = round(quantity * current_price, 2)
                has_live_price = True
            elif normalized in live_map and live_map[normalized]["market_value"] > 0:
                live_entry = live_map[normalized]
                live_market_value = float(live_entry.get("market_value", 0.0) or 0.0)
                live_volume = int(live_entry.get("volume", 0) or 0)
                if live_volume > 0:
                    current_price = round(live_market_value / live_volume, 4)
                    market_value = round(quantity * current_price, 2)
                else:
                    market_value = round(live_market_value, 2)
                    current_price = round(market_value / quantity, 4) if quantity > 0 else 0.0
                has_live_price = True
            else:
                current_price = avg_cost
                market_value = cost_amount

            unrealized_pnl = round(market_value - cost_amount, 2)
            unrealized_pnl_pct = round(unrealized_pnl / cost_amount * 100.0, 4) if cost_amount > 0 else 0.0
            stock_name = ""
            if normalized in live_map:
                stock_name = live_map[normalized].get("name", "") or ""

            rows.append(
                {
                    "strategy_id": state.strategy_id,
                    "stock_code": normalized,
                    "stock_name": stock_name,
                    "quantity": quantity,
                    "avg_cost": round(avg_cost, 4),
                    "cost_amount": cost_amount,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "has_live_price": has_live_price,
                }
            )

        total_market_value = sum(float(r["market_value"] or 0.0) for r in rows)
        for row in rows:
            row["weight_in_strategy"] = (
                round(float(row["market_value"]) / total_market_value, 4)
                if total_market_value > 0
                else 0.0
            )
        rows.sort(key=lambda r: float(r.get("market_value", 0.0) or 0.0), reverse=True)
        return rows

    def build_account_snapshot(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        live_positions: Optional[List[dict]] = None,
        spot_prices: Optional[Dict[str, float]] = None,
        market_value_override: Optional[float] = None,
        cash_override: Optional[float] = None,
        capital_limit_override: Optional[float] = None,
    ) -> dict:
        """统一账户收益快照（主账本口径）。

        作为 AI实盘决策 / ETF轮动实盘 / 实盘收益的唯一数据组装点，统一以下字段：
          - capital_limit / realized_pnl / invested_cost / available_cash 来自 strategy_budget 主账本
          - market_value 由调用方提供（券商实时或行情价 × 数量）；若无则退化为持仓成本
          - unrealized_pnl = market_value - invested_cost
          - total_pnl      = realized_pnl + unrealized_pnl
          - total_asset    = available_cash + market_value

        Args:
            live_positions: [{"market_value": x}, ...] 用于计算实时市值
            market_value_override: 直接指定市值（优先级高于 live_positions）
            cash_override: 用 dedicated cash 等外部口径覆盖可用现金（ETF 专用资金模式）
            capital_limit_override: 覆盖资金上限（ETF engine 的 dedicated_capital）
        """
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._rehydrate_from_trade_records_if_needed(
            state,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._refresh_accounting_from_trade_records_if_needed(
            state,
            virtual_account_id=virtual_account_id,
        )
        cfg = self._configs.get(strategy_id)
        is_unmanaged = bool(getattr(cfg, "is_unmanaged", False))

        invested_cost = round(state.invested_market_value(), 2)
        realized_pnl = round(float(state.realized_pnl or 0.0), 2)
        capital_limit = round(
            float(capital_limit_override) if capital_limit_override is not None else float(state.capital_limit or 0.0),
            2,
        )

        if market_value_override is not None:
            market_value = round(float(market_value_override or 0.0), 2)
        elif live_positions is not None or spot_prices is not None:
            positions_view = self.get_positions_view(
                strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                real_total_asset=real_total_asset,
                spot_prices=spot_prices,
                live_positions=live_positions,
            )
            if positions_view and any(r.get("has_live_price") for r in positions_view):
                market_value = round(
                    sum(float(r.get("market_value", 0.0) or 0.0) for r in positions_view),
                    2,
                )
            elif live_positions and not is_unmanaged:
                # 主账本尚未记录对应持仓（例如 AI 首次接管前手动持有的股票），
                # 退化为直接按 live_positions 的 market_value 求和，保证不低估
                market_value = round(
                    sum(float((p or {}).get("market_value", 0.0) or 0.0) for p in live_positions),
                    2,
                )
            else:
                market_value = invested_cost
        else:
            market_value = invested_cost

        if cash_override is not None:
            available_cash = round(float(cash_override or 0.0), 2)
        elif is_unmanaged:
            # 未管理账户的现金来自券商对账（cash_balance 即真实券商里未认领的余额），
            # 不适用 "capital_limit + realized_pnl - invested_cost" 的启动资金公式。
            available_cash = round(
                max(float(state.cash_balance or 0.0) - float(state.reserved_cash or 0.0), 0.0),
                2,
            )
        else:
            available_cash = round(max(capital_limit + realized_pnl - invested_cost, 0.0), 2)

        unrealized_pnl = round(market_value - invested_cost, 2)
        total_pnl = round(realized_pnl + unrealized_pnl, 2)
        total_asset = round(available_cash + market_value, 2)
        position_count = len([pos for pos in state.get_positions().values() if pos.quantity > 0])

        return {
            "strategy_id": state.strategy_id,
            "strategy_name": state.strategy_name,
            "virtual_account_id": state.virtual_account_id,
            "capital_limit": capital_limit,
            "cash_balance": round(float(state.cash_balance or 0.0), 2),
            "reserved_cash": round(float(state.reserved_cash or 0.0), 2),
            "available_cash": available_cash,
            "invested_cost": invested_cost,
            "invested_market_value": invested_cost,
            "market_value": market_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_asset": total_asset,
            "position_count": position_count,
            "enabled": bool(getattr(cfg, "enabled", True)),
            "is_test": bool(getattr(cfg, "is_test", False)),
            "hidden": bool(getattr(cfg, "hidden", False)),
            "is_unmanaged": is_unmanaged,
        }

    def get_available_budget(
        self,
        strategy_id: str,
        *,
        reserve_pct: float = 0.0,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        capital_limit_override: Optional[float] = None,
    ) -> dict:
        """下单前可用额度（主账本口径，**只守账本层**）。

        作为 AI / ETF / 风控等所有下单前"剩余可用资金"估算的**唯一入口**。
        仅以主账本余额推导，不查询券商实时现金——即"假设虚拟账户严格隔离、
        各策略只能动用自己启动资金 + 已实现盈亏的钱"。

        计算：
            budget_remaining = max(capital_limit + realized_pnl - invested_cost - reserved_cash, 0)
            reserve_amount   = max(capital_limit * reserve_pct, 0)
            available        = max(budget_remaining - reserve_amount, 0)

        Args:
            reserve_pct: 保底比例（相对 capital_limit），不想下单时保留的现金
            capital_limit_override: 覆盖资金上限（例如 ETF 专用资金模式下的 dedicated_capital）

        Returns:
            {
                strategy_id, enabled,
                capital_limit, realized_pnl, invested_cost, reserved_cash,
                budget_remaining,
                reserve_pct, reserve_amount,
                available,
                binding_constraint: "disabled" | "budget" | "reserve" | "ok",
            }
        """
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._rehydrate_from_trade_records_if_needed(
            state,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        cfg = self._configs.get(strategy_id)

        capital_limit = round(
            float(capital_limit_override)
            if capital_limit_override is not None
            else float(state.capital_limit or 0.0),
            2,
        )
        invested_cost = round(state.invested_market_value(), 2)
        realized_pnl = round(float(state.realized_pnl or 0.0), 2)
        reserved_cash = round(float(state.reserved_cash or 0.0), 2)

        reserve_pct = max(float(reserve_pct or 0.0), 0.0)
        reserve_amount = round(max(capital_limit * reserve_pct, 0.0), 2)

        budget_remaining = round(
            max(capital_limit + realized_pnl - invested_cost - reserved_cash, 0.0),
            2,
        )
        available = round(max(budget_remaining - reserve_amount, 0.0), 2)

        enabled = bool(getattr(cfg, "enabled", True))
        if not enabled:
            available = 0.0
            binding_constraint = "disabled"
        elif budget_remaining <= 0:
            binding_constraint = "budget"
        elif reserve_amount > 0 and budget_remaining - reserve_amount < budget_remaining:
            binding_constraint = "reserve" if available < budget_remaining else "ok"
        else:
            binding_constraint = "ok"

        return {
            "strategy_id": state.strategy_id,
            "enabled": enabled,
            "capital_limit": capital_limit,
            "realized_pnl": realized_pnl,
            "invested_cost": invested_cost,
            "reserved_cash": reserved_cash,
            "budget_remaining": budget_remaining,
            "reserve_pct": reserve_pct,
            "reserve_amount": reserve_amount,
            "available": available,
            "binding_constraint": binding_constraint,
        }

    def _build_trade_record_replay(
        self,
        state: StrategyBudgetState,
        *,
        virtual_account_id: str = "",
    ) -> Optional[dict]:
        strategy_id = (state.strategy_id or "").strip()
        if not strategy_id:
            return None
        try:
            from trading_app.services.trade_record_service import TradeDirection, get_trade_record_service

            records = get_trade_record_service().get_records(
                strategy_id=strategy_id,
                virtual_account_id=virtual_account_id or state.virtual_account_id,
                limit=5000,
            )
        except Exception as exc:
            logger.debug("从成交记录回放策略账本失败: %s", exc)
            return None
        if not records:
            return None

        positions: Dict[str, StrategyPositionState] = {}
        capital_limit = float(state.capital_limit or 0.0)
        cash_balance = capital_limit
        realized_pnl = 0.0
        trade_history: List[dict] = []

        ordered_records = sorted(
            list(records or []),
            key=lambda item: (
                str(getattr(item, "trade_date", "") or ""),
                str(getattr(item, "created_at", "") or ""),
                int(getattr(item, "id", 0) or 0),
            ),
        )
        for rec in ordered_records:
            code = normalize_symbol_code(getattr(rec, "stock_code", "") or "")
            if not code:
                continue
            volume = int(getattr(rec, "volume", 0) or 0)
            price = float(getattr(rec, "price", 0.0) or 0.0)
            amount = float(getattr(rec, "amount", 0.0) or 0.0)
            if volume <= 0 or price <= 0:
                continue
            if amount <= 0:
                amount = round(price * volume, 2)
            total_fee = round(
                float(getattr(rec, "commission", 0.0) or 0.0)
                + float(getattr(rec, "stamp_tax", 0.0) or 0.0)
                + float(getattr(rec, "transfer_fee", 0.0) or 0.0),
                2,
            )
            position = positions.get(code) or StrategyPositionState(symbol_code=code)
            direction = str(getattr(rec, "direction", "") or "").lower()
            if direction == TradeDirection.BUY.value:
                total_qty = int(position.quantity or 0) + volume
                total_cost = float(position.quantity or 0) * float(position.avg_cost or 0.0) + amount
                position.quantity = total_qty
                position.avg_cost = round(total_cost / total_qty, 4) if total_qty > 0 else 0.0
                cash_balance = round(cash_balance - amount - total_fee, 2)
            else:
                sold_qty = min(volume, max(int(position.quantity or 0), 0))
                if sold_qty > 0:
                    realized_pnl = round(realized_pnl + (price - float(position.avg_cost or 0.0)) * sold_qty - total_fee, 2)
                remaining_qty = max(int(position.quantity or 0) - volume, 0)
                if remaining_qty > 0:
                    position.quantity = remaining_qty
                else:
                    position.quantity = 0
                cash_balance = round(cash_balance + amount - total_fee, 2)
            position.updated_at = str(getattr(rec, "created_at", "") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if position.quantity > 0:
                positions[code] = position
            else:
                positions.pop(code, None)
            trade_date = self._normalize_trade_date(
                getattr(rec, "trade_date", ""),
                fallback=str(getattr(rec, "created_at", "") or ""),
            )
            trade_history.append(
                {
                    "date": trade_date,
                    "time": str(getattr(rec, "created_at", "") or "").split(" ")[-1],
                    "action": "BUY" if direction == TradeDirection.BUY.value else "SELL",
                    "code": code,
                    "name": str(getattr(rec, "stock_name", "") or code),
                    "price": price,
                    "quantity": volume,
                    "amount": amount,
                    "reason": str(getattr(rec, "remark", "") or ""),
                    "broker_order_id": int(getattr(rec, "broker_order_id", -1) or -1),
                    "success": True,
                    "error_msg": "",
                    "pnl": round(realized_pnl, 2) if direction != TradeDirection.BUY.value else 0.0,
                }
            )

        return {
            "positions": {code: pos.to_dict() for code, pos in positions.items()},
            "cash_balance": round(max(cash_balance, 0.0), 2),
            "realized_pnl": round(realized_pnl, 2),
            "trade_history": trade_history,
        }

    def _refresh_accounting_from_trade_records_if_needed(
        self,
        state: StrategyBudgetState,
        *,
        virtual_account_id: str = "",
    ) -> StrategyBudgetState:
        """AI 股票策略以成交记录作为成本和已实现盈亏的最终口径。"""
        strategy_id = (state.strategy_id or "").strip()
        if strategy_id != AI_STOCK_STRATEGY_ID:
            return state

        replayed = self._build_trade_record_replay(
            state,
            virtual_account_id=virtual_account_id or state.virtual_account_id,
        )
        if not replayed:
            return state

        replayed_positions = dict(replayed.get("positions") or {})
        replayed_cash = round(float(replayed.get("cash_balance", 0.0) or 0.0), 2)
        replayed_realized = round(float(replayed.get("realized_pnl", 0.0) or 0.0), 2)
        replayed_history = list(replayed.get("trade_history") or [])

        current_cash = round(float(state.cash_balance or 0.0), 2)
        current_realized = round(float(state.realized_pnl or 0.0), 2)
        if (
            dict(state.positions or {}) == replayed_positions
            and abs(current_cash - replayed_cash) < 0.01
            and abs(current_realized - replayed_realized) < 0.01
            and len(state.trade_history or []) == len(replayed_history)
        ):
            return state

        state.positions = replayed_positions
        state.cash_balance = replayed_cash
        state.realized_pnl = replayed_realized
        state.trade_history = replayed_history
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._states[strategy_id] = state
        self._save_states()
        logger.info(
            "已按成交记录刷新 AI 策略账本: positions=%d realized_pnl=%.2f",
            len(replayed_positions),
            replayed_realized,
        )
        self._save_current_position_cost_snapshot(state)
        return state

    def _save_current_position_cost_snapshot(self, state: StrategyBudgetState) -> None:
        """把当前策略持仓成本落到 SQLite 快照表。

        策略持仓成本只取本地账本 avg_cost；没有实时行情时 market_value
        使用成本金额占位，避免借用券商 open_price。
        """
        strategy_id = (state.strategy_id or "").strip()
        if not strategy_id:
            return
        positions = []
        for code, pos in state.get_positions().items():
            quantity = int(pos.quantity or 0)
            if quantity <= 0:
                continue
            cost_price = float(pos.avg_cost or 0.0)
            cost_amount = round(cost_price * quantity, 2)
            positions.append(
                {
                    "stock_code": code,
                    "stock_name": code,
                    "volume": quantity,
                    "can_use_volume": quantity,
                    "open_price": cost_price,
                    "cost_price": cost_price,
                    "cost_amount": cost_amount,
                    "market_value": cost_amount,
                }
            )
        try:
            from trading_app.services.trade_record_service import get_trade_record_service

            get_trade_record_service().save_strategy_position_snapshots(
                datetime.now().strftime("%Y-%m-%d"),
                {
                    strategy_id: {
                        "strategy_name": state.strategy_name,
                        "virtual_account_id": state.virtual_account_id,
                        "positions": positions,
                    }
                },
            )
        except Exception as exc:
            logger.debug("保存策略持仓成本快照失败: %s", exc)

    def _rehydrate_from_trade_records_if_needed(
        self,
        state: StrategyBudgetState,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        force: bool = False,
    ) -> StrategyBudgetState:
        if not force and any(pos.quantity > 0 for pos in state.get_positions().values()):
            return state
        if not force and (state.trade_history or state.order_records):
            return state
        strategy_id = (state.strategy_id or "").strip()
        if not strategy_id:
            return state

        rebuilt = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name or state.strategy_name,
            virtual_account_id=virtual_account_id or state.virtual_account_id,
            real_total_asset=real_total_asset,
        )
        replayed = self._build_trade_record_replay(
            rebuilt,
            virtual_account_id=virtual_account_id,
        )
        if not replayed:
            return state

        rebuilt.positions = dict(replayed.get("positions") or {})
        rebuilt.cash_balance = float(replayed.get("cash_balance", 0.0) or 0.0)
        rebuilt.realized_pnl = float(replayed.get("realized_pnl", 0.0) or 0.0)
        rebuilt.trade_history = list(replayed.get("trade_history") or [])
        rebuilt.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._states[strategy_id] = rebuilt
        self._save_states()
        logger.info("已从成交记录回放恢复策略账本: %s 持仓=%d 现金=%.2f", strategy_id, len(rebuilt.positions), rebuilt.cash_balance)
        return rebuilt

    def rebuild_strategy_state_from_trade_records(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
    ) -> StrategyBudgetState:
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        return self._rehydrate_from_trade_records_if_needed(
            self._refresh_accounting_from_trade_records_if_needed(
                state,
                virtual_account_id=virtual_account_id,
            ),
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
            force=True,
        )

    def _derive_position_costs_from_trade_records(
        self,
        *,
        strategy_id: str,
        virtual_account_id: str = "",
    ) -> Dict[str, float]:
        try:
            from trading_app.services.trade_record_service import TradeDirection, get_trade_record_service

            records = get_trade_record_service().get_records(
                strategy_id=strategy_id,
                virtual_account_id=virtual_account_id,
                limit=5000,
            )
        except Exception as exc:
            logger.debug("从成交记录推导持仓成本失败: %s", exc)
            return {}

        if not records:
            return {}

        positions: Dict[str, StrategyPositionState] = {}
        ordered_records = sorted(
            list(records or []),
            key=lambda item: (
                str(getattr(item, "trade_date", "") or ""),
                str(getattr(item, "created_at", "") or ""),
                int(getattr(item, "id", 0) or 0),
            ),
        )
        for rec in ordered_records:
            code = normalize_symbol_code(getattr(rec, "stock_code", "") or "")
            if not code:
                continue
            volume = int(getattr(rec, "volume", 0) or 0)
            price = float(getattr(rec, "price", 0.0) or 0.0)
            amount = float(getattr(rec, "amount", 0.0) or 0.0)
            if volume <= 0 or price <= 0:
                continue
            if amount <= 0:
                amount = round(price * volume, 2)

            position = positions.get(code) or StrategyPositionState(symbol_code=code)
            direction = str(getattr(rec, "direction", "") or "").lower()
            if direction == TradeDirection.BUY.value:
                total_qty = int(position.quantity or 0) + volume
                total_cost = float(position.quantity or 0) * float(position.avg_cost or 0.0) + amount
                position.quantity = total_qty
                position.avg_cost = round(total_cost / total_qty, 4) if total_qty > 0 else 0.0
                positions[code] = position
            elif direction == TradeDirection.SELL.value:
                remaining_qty = max(int(position.quantity or 0) - volume, 0)
                if remaining_qty > 0:
                    position.quantity = remaining_qty
                    positions[code] = position
                else:
                    positions.pop(code, None)

        return {
            code: round(float(pos.avg_cost or 0.0), 4)
            for code, pos in positions.items()
            if int(pos.quantity or 0) > 0 and float(pos.avg_cost or 0.0) > 0
        }

    @staticmethod
    def _normalize_trade_date(value: object, *, fallback: str = "") -> str:
        text = str(value or "").strip()
        for candidate in (text, str(fallback or "").strip().split(" ")[0]):
            if not candidate:
                continue
            try:
                parsed = datetime.strptime(candidate[:10], "%Y-%m-%d")
            except Exception:
                continue
            if parsed.year >= 2000:
                return parsed.strftime("%Y-%m-%d")
        return datetime.now().strftime("%Y-%m-%d")

    def upsert_strategy_config(
        self,
        *,
        strategy_id: str,
        strategy_name: str = "",
        virtual_account_id: str = "",
        capital_limit: Optional[float] = None,
        enabled: Optional[bool] = None,
        is_test: Optional[bool] = None,
        hidden: Optional[bool] = None,
        is_unmanaged: Optional[bool] = None,
    ) -> None:
        strategy_id = (strategy_id or "").strip()
        if not strategy_id:
            return
        cfg = self._configs.get(strategy_id)
        if cfg is None:
            auto_is_test = self._looks_like_test_strategy(strategy_id, strategy_name)
            cfg = StrategyBudgetConfig(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                capital_limit=float(capital_limit or 0.0),
                enabled=True if enabled is None else bool(enabled),
                is_test=auto_is_test if is_test is None else bool(is_test),
                hidden=auto_is_test if hidden is None else bool(hidden),
                is_unmanaged=False if is_unmanaged is None else bool(is_unmanaged),
            )
            self._configs[strategy_id] = cfg
            self._save_configs()
            return
        updated = False
        if strategy_name and cfg.strategy_name != strategy_name:
            cfg.strategy_name = strategy_name
            updated = True
        if virtual_account_id and cfg.virtual_account_id != virtual_account_id:
            cfg.virtual_account_id = virtual_account_id
            updated = True
        if capital_limit is not None and abs(float(cfg.capital_limit or 0.0) - float(capital_limit or 0.0)) > 1e-6:
            cfg.capital_limit = float(capital_limit or 0.0)
            updated = True
        if enabled is not None and cfg.enabled != bool(enabled):
            cfg.enabled = bool(enabled)
            updated = True
        if is_test is not None and cfg.is_test != bool(is_test):
            cfg.is_test = bool(is_test)
            updated = True
        if hidden is not None and cfg.hidden != bool(hidden):
            cfg.hidden = bool(hidden)
            updated = True
        if is_unmanaged is not None and cfg.is_unmanaged != bool(is_unmanaged):
            cfg.is_unmanaged = bool(is_unmanaged)
            updated = True
        if updated:
            cfg.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_configs()

    def reset_strategy_account(
        self,
        *,
        strategy_id: str,
        cash_balance: float,
        strategy_name: str = "",
        virtual_account_id: str = "",
        capital_limit: Optional[float] = None,
        preserve_positions: bool = True,
    ) -> None:
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=0.0,
        )
        state.cash_balance = round(float(cash_balance or 0.0), 2)
        if capital_limit is not None:
            state.capital_limit = round(float(capital_limit or 0.0), 2)
        if not preserve_positions:
            state.positions = {}
        state.reservations = {}
        state.reserved_cash = 0.0
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()

    def adjust_strategy_capital_limit(
        self,
        *,
        strategy_id: str,
        new_capital_limit: float,
        strategy_name: str = "",
        virtual_account_id: str = "",
    ) -> dict:
        """按新旧启动资金差额划拨现金，保留持仓、盈亏和冻结记录。

        用于实盘资金管理里的非重置修改：例如 ETF 从 1 万调到 3 万，
        主账本应同步表现为 capital_limit +2 万、cash_balance +2 万。
        unmanaged 账户会在下一次券商对账时自动少吸收这部分现金。
        """
        strategy_id = (strategy_id or "").strip()
        if not strategy_id:
            raise ValueError("strategy_id 不能为空")
        new_capital = round(float(new_capital_limit or 0.0), 2)
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=0.0,
        )
        state = self._rehydrate_from_trade_records_if_needed(
            state,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=0.0,
        )
        old_capital = round(float(state.capital_limit or 0.0), 2)
        delta = round(new_capital - old_capital, 2)
        if abs(delta) < 0.01:
            return {
                "strategy_id": strategy_id,
                "old_capital_limit": old_capital,
                "new_capital_limit": new_capital,
                "delta": 0.0,
                "cash_balance": round(float(state.cash_balance or 0.0), 2),
            }

        if delta < 0:
            available_cash = round(
                max(float(state.cash_balance or 0.0) - float(state.reserved_cash or 0.0), 0.0),
                2,
            )
            if available_cash + 1e-6 < abs(delta):
                raise ValueError(
                    f"策略可划出现金不足，需 {abs(delta):,.2f}，可用 {available_cash:,.2f}"
                )

        self.upsert_strategy_config(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            capital_limit=new_capital,
        )
        state.capital_limit = new_capital
        state.cash_balance = round(float(state.cash_balance or 0.0) + delta, 2)
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()
        return {
            "strategy_id": strategy_id,
            "old_capital_limit": old_capital,
            "new_capital_limit": new_capital,
            "delta": delta,
            "cash_balance": round(float(state.cash_balance or 0.0), 2),
        }

    def reserve_cash(
        self,
        *,
        strategy_id: str,
        intent_id: str,
        amount: float,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
    ) -> Tuple[bool, str]:
        amount = round(float(amount or 0.0), 2)
        if amount <= 0:
            return False, "预算冻结金额必须大于0"
        intent_id = (intent_id or "").strip()
        if not intent_id:
            return False, "预算冻结缺少 intent_id"
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        existing = float(state.reservations.get(intent_id, 0.0) or 0.0)
        if existing > 0:
            return True, ""
        available_cash = state.available_cash()
        if available_cash + 1e-6 < amount:
            return False, f"策略预算不足，需 {amount:,.2f}，可用 {available_cash:,.2f}"
        state.reservations[intent_id] = amount
        state.reserved_cash = round(float(state.reserved_cash or 0.0) + amount, 2)
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()
        return True, ""

    def release_reservation(self, *, strategy_id: str, intent_id: str) -> None:
        state = self._states.get((strategy_id or "").strip())
        if state is None:
            return
        amount = float(state.reservations.pop((intent_id or "").strip(), 0.0) or 0.0)
        if amount <= 0:
            return
        state.reserved_cash = round(max(float(state.reserved_cash or 0.0) - amount, 0.0), 2)
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()

    def commit_buy(
        self,
        *,
        strategy_id: str,
        symbol_code: str,
        price: float,
        volume: int,
        intent_id: str = "",
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        commission: float = 0.0,
        stamp_tax: float = 0.0,
        transfer_fee: float = 0.0,
    ) -> None:
        """买入提交主账本（扣现金 + 落持仓）。

        算法与 ``_rehydrate_from_trade_records_if_needed`` 严格一致：
            total_fee      = commission + stamp_tax + transfer_fee
            trade_amount   = price * volume
            cash_balance  -= trade_amount + total_fee
            avg_cost       = (旧成本 + trade_amount) / 新数量      # ★ 不把手续费摊入成本

        手续费只扣现金，不影响持仓 avg_cost——保证 invested_cost（=qty*avg_cost）
        只反映"真正买入股票花的钱"，unrealized_pnl 才是市值相对于买价的浮动。
        """
        price = float(price or 0.0)
        volume = int(volume or 0)
        if price <= 0 or volume <= 0:
            self.release_reservation(strategy_id=strategy_id, intent_id=intent_id)
            return
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        reserved_amount = float(state.reservations.pop((intent_id or "").strip(), 0.0) or 0.0)
        if reserved_amount > 0:
            state.reserved_cash = round(max(float(state.reserved_cash or 0.0) - reserved_amount, 0.0), 2)

        trade_amount = round(price * volume, 2)
        total_fee = round(
            max(float(commission or 0.0), 0.0)
            + max(float(stamp_tax or 0.0), 0.0)
            + max(float(transfer_fee or 0.0), 0.0),
            2,
        )
        state.cash_balance = round(
            max(float(state.cash_balance or 0.0) - trade_amount - total_fee, 0.0),
            2,
        )

        code = normalize_symbol_code(symbol_code)
        position = state.get_positions().get(code) or StrategyPositionState(symbol_code=code)
        total_qty = int(position.quantity or 0) + volume
        total_cost = float(position.quantity or 0) * float(position.avg_cost or 0.0) + trade_amount
        position.quantity = total_qty
        position.avg_cost = round(total_cost / total_qty, 4) if total_qty > 0 else 0.0
        position.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state.positions[code] = position.to_dict()
        state.updated_at = position.updated_at
        self._save_states()
        self._save_current_position_cost_snapshot(state)

    def commit_sell(
        self,
        *,
        strategy_id: str,
        symbol_code: str,
        price: float,
        volume: int,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        commission: float = 0.0,
        stamp_tax: float = 0.0,
        transfer_fee: float = 0.0,
    ) -> None:
        """卖出提交主账本（回现金 + 落 realized_pnl）。

        算法与 ``_rehydrate_from_trade_records_if_needed`` 严格一致：
            total_fee     = commission + stamp_tax + transfer_fee
            proceeds      = price * volume
            cash_balance += proceeds - total_fee
            realized_pnl += (price - avg_cost) * sold_qty - total_fee   # 仅当 sold_qty>0
        """
        price = float(price or 0.0)
        volume = int(volume or 0)
        if price <= 0 or volume <= 0:
            return
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        proceeds = round(price * volume, 2)
        total_fee = round(
            max(float(commission or 0.0), 0.0)
            + max(float(stamp_tax or 0.0), 0.0)
            + max(float(transfer_fee or 0.0), 0.0),
            2,
        )
        code = normalize_symbol_code(symbol_code)
        position = state.get_positions().get(code)
        if position:
            sold_qty = min(volume, max(position.quantity, 0))
            if sold_qty > 0:
                state.realized_pnl = round(
                    float(state.realized_pnl or 0.0)
                    + (price - position.avg_cost) * sold_qty
                    - total_fee,
                    2,
                )
            remaining_qty = max(position.quantity - sold_qty, 0)
            if remaining_qty <= 0:
                state.positions.pop(code, None)
            else:
                position.quantity = remaining_qty
                position.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state.positions[code] = position.to_dict()
        state.cash_balance = round(
            float(state.cash_balance or 0.0) + proceeds - total_fee,
            2,
        )
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()
        self._save_current_position_cost_snapshot(state)

    def sync_strategy_positions(
        self,
        *,
        strategy_id: str,
        positions: List[dict],
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
        clear_reservations: bool = True,
    ) -> None:
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        trade_cost_map = self._derive_position_costs_from_trade_records(
            strategy_id=strategy_id,
            virtual_account_id=virtual_account_id or state.virtual_account_id,
        )
        broker_codes: set[str] = set()
        new_positions: Dict[str, dict] = {}
        for item in positions or []:
            code = normalize_symbol_code(item.get("stock_code", ""))
            volume = int(item.get("volume", 0) or 0)
            if not code or volume <= 0:
                continue
            broker_codes.add(code)
            existing_position = state.get_positions().get(code)
            # 成本只允许来自本地账本或成交记录回放，不能用券商 open_price 兜底。
            avg_cost = float(getattr(existing_position, "avg_cost", 0.0) or 0.0)
            explicit_cost = float(item.get("cost_price", item.get("avg_cost", 0.0)) or 0.0)
            if avg_cost <= 0 and explicit_cost > 0:
                avg_cost = explicit_cost
            if code in trade_cost_map:
                avg_cost = trade_cost_map[code]
            new_positions[code] = StrategyPositionState(
                symbol_code=code,
                quantity=volume,
                avg_cost=avg_cost,
            ).to_dict()
        for code, pos in state.get_positions().items():
            if code in broker_codes or int(pos.quantity or 0) <= 0:
                continue
            if code in trade_cost_map:
                logger.info(
                    "sync_strategy_positions 保留本地持仓（券商未返回）: strategy=%s code=%s qty=%d",
                    strategy_id, code, pos.quantity,
                )
                new_positions[code] = pos.to_dict()
        state.positions = new_positions
        if clear_reservations:
            state.reservations = {}
            state.reserved_cash = 0.0
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()
        self._save_current_position_cost_snapshot(state)

    def list_strategy_snapshots(
        self,
        *,
        include_hidden: bool = True,
        include_test: bool = True,
    ) -> List[dict]:
        results: List[dict] = []
        for strategy_id in sorted(self._states.keys()):
            snapshot = self.get_strategy_snapshot(strategy_id)
            if not include_hidden and bool(snapshot.get("hidden", False)):
                continue
            if not include_test and bool(snapshot.get("is_test", False)):
                continue
            results.append(snapshot)
        return results

    def get_strategy_state_record(
        self,
        strategy_id: str,
        *,
        strategy_name: str = "",
        virtual_account_id: str = "",
        real_total_asset: float = 0.0,
    ) -> StrategyBudgetState:
        state = self._ensure_strategy(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        state = self._rehydrate_from_trade_records_if_needed(
            state,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            real_total_asset=real_total_asset,
        )
        return self._refresh_accounting_from_trade_records_if_needed(
            state,
            virtual_account_id=virtual_account_id,
        )

    def save_strategy_state_record(self, state: StrategyBudgetState) -> None:
        strategy_id = (getattr(state, "strategy_id", "") or "").strip()
        if not strategy_id:
            return
        state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._states[strategy_id] = state
        self._save_states()

    # ------------------------------------------------------------------
    #  全局账户聚合：所有策略（含 unmanaged）汇总视图
    # ------------------------------------------------------------------

    def get_portfolio_totals(
        self,
        *,
        include_unmanaged: bool = True,
        include_hidden: bool = False,
        include_test: bool = False,
        live_positions_by_strategy: Optional[Dict[str, List[dict]]] = None,
        spot_prices_by_strategy: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> dict:
        """主账本口径的账户全局聚合。

        聚合范围：所有非 hidden / 非 test 的策略 snapshot（默认包含 unmanaged）。
        返回字段：
            total_asset          Σ(strategy.total_asset)
            available_cash       Σ(strategy.available_cash)
            market_value         Σ(strategy.market_value)
            capital_limit        Σ(strategy.capital_limit)
            invested_cost        Σ(strategy.invested_cost)
            realized_pnl         Σ(strategy.realized_pnl)
            unrealized_pnl       Σ(strategy.unrealized_pnl)
            total_pnl            Σ(strategy.total_pnl)
            strategies           List[snapshot]（按 strategy_id 排序，包含 is_unmanaged 标志）
            managed_*            去掉 unmanaged 后的聚合，便于 UI 分开展示
            unmanaged_*          unmanaged 单独聚合
            updated_at           聚合时间戳
        """
        live_positions_by_strategy = live_positions_by_strategy or {}
        spot_prices_by_strategy = spot_prices_by_strategy or {}

        rows: List[dict] = []
        for sid in sorted(self._states.keys()):
            cfg = self._configs.get(sid)
            is_unmanaged = bool(getattr(cfg, "is_unmanaged", False)) if cfg is not None else False
            if not include_unmanaged and is_unmanaged:
                continue
            if not include_hidden and cfg is not None and bool(getattr(cfg, "hidden", False)):
                continue
            if not include_test and cfg is not None and bool(getattr(cfg, "is_test", False)):
                continue
            try:
                snapshot = self.build_account_snapshot(
                    sid,
                    live_positions=live_positions_by_strategy.get(sid),
                    spot_prices=spot_prices_by_strategy.get(sid),
                )
            except Exception as exc:
                logger.warning("get_portfolio_totals 聚合 %s 失败: %s", sid, exc)
                continue
            snapshot["is_unmanaged"] = bool(snapshot.get("is_unmanaged", False)) or is_unmanaged
            rows.append(snapshot)

        def _agg(items: List[dict]) -> dict:
            return {
                "total_asset": round(sum(float(i.get("total_asset", 0.0) or 0.0) for i in items), 2),
                "available_cash": round(sum(float(i.get("available_cash", 0.0) or 0.0) for i in items), 2),
                "market_value": round(sum(float(i.get("market_value", 0.0) or 0.0) for i in items), 2),
                "capital_limit": round(sum(float(i.get("capital_limit", 0.0) or 0.0) for i in items), 2),
                "invested_cost": round(sum(float(i.get("invested_cost", 0.0) or 0.0) for i in items), 2),
                "realized_pnl": round(sum(float(i.get("realized_pnl", 0.0) or 0.0) for i in items), 2),
                "unrealized_pnl": round(sum(float(i.get("unrealized_pnl", 0.0) or 0.0) for i in items), 2),
                "total_pnl": round(sum(float(i.get("total_pnl", 0.0) or 0.0) for i in items), 2),
                "strategies_count": len(items),
            }

        totals = _agg(rows)
        managed = _agg([r for r in rows if not bool(r.get("is_unmanaged", False))])
        unmanaged = _agg([r for r in rows if bool(r.get("is_unmanaged", False))])

        return {
            **totals,
            "strategies": rows,
            "managed_total_asset": managed["total_asset"],
            "managed_available_cash": managed["available_cash"],
            "managed_market_value": managed["market_value"],
            "managed_capital_limit": managed["capital_limit"],
            "managed_realized_pnl": managed["realized_pnl"],
            "managed_unrealized_pnl": managed["unrealized_pnl"],
            "managed_total_pnl": managed["total_pnl"],
            "managed_strategies_count": managed["strategies_count"],
            "unmanaged_total_asset": unmanaged["total_asset"],
            "unmanaged_available_cash": unmanaged["available_cash"],
            "unmanaged_market_value": unmanaged["market_value"],
            "unmanaged_strategies_count": unmanaged["strategies_count"],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ------------------------------------------------------------------
    #  未管理账户（unmanaged）：承载券商里未被任何策略认领的现金/持仓
    # ------------------------------------------------------------------

    def ensure_unmanaged_strategy(self) -> str:
        """幂等创建 unmanaged 虚拟账户（不允许下单，但参与账户聚合）。"""
        self.upsert_strategy_config(
            strategy_id=UNMANAGED_STRATEGY_ID,
            strategy_name=UNMANAGED_STRATEGY_NAME,
            virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
            capital_limit=0.0,
            enabled=False,       # 不允许下单
            is_test=False,
                hidden=False,        # 展示在实盘收益，提醒用户
            is_unmanaged=True,
        )
        self._ensure_strategy(
            UNMANAGED_STRATEGY_ID,
            strategy_name=UNMANAGED_STRATEGY_NAME,
            virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
            real_total_asset=0.0,
        )
        return UNMANAGED_STRATEGY_ID

    def reconcile_unmanaged_with_broker(
        self,
        *,
        broker_cash: float,
        broker_positions: List[dict],
    ) -> dict:
        """把券商里未被任何活跃策略认领的现金/持仓归入 unmanaged 虚拟账户。

        算法：
            claimed_cash       = Σ(非 unmanaged 策略的 cash_balance)
            claimed_qty[code]  = Σ(非 unmanaged 策略在该 code 上的持仓量)
            unmanaged_cash     = max(broker_cash - claimed_cash, 0)
            unmanaged_positions[code] =
                max(broker_qty[code] - claimed_qty[code], 0) 股，avg_cost 只取成交回放或既有账本成本

        不改动其它策略的字段；仅重写 unmanaged 策略的 cash_balance / positions。
        返回摘要 dict 用于日志和调试。
        """
        unmanaged_id = self.ensure_unmanaged_strategy()

        claimed_cash = 0.0
        claimed_qty: Dict[str, int] = {}
        for sid, st in self._states.items():
            if sid == unmanaged_id:
                continue
            cfg = self._configs.get(sid)
            if cfg is not None:
                # test / hidden 策略是沙箱/回放用途，不代表真实占用券商现金与持仓，
                # 不能计入 claimed；否则它们的 cash_balance 会把 unmanaged 额度吃光。
                if bool(getattr(cfg, "is_unmanaged", False)):
                    continue
                if bool(getattr(cfg, "is_test", False)):
                    continue
                if bool(getattr(cfg, "hidden", False)):
                    continue
            claimed_cash += float(getattr(st, "cash_balance", 0.0) or 0.0)
            for code, pos in st.get_positions().items():
                qty = int(getattr(pos, "quantity", 0) or 0)
                if qty <= 0 or not code:
                    continue
                claimed_qty[code] = claimed_qty.get(code, 0) + qty

        broker_qty: Dict[str, float] = {}
        for item in broker_positions or []:
            code = normalize_symbol_code(str(item.get("stock_code", "") or ""))
            volume = int(item.get("volume", 0) or 0)
            if not code or volume <= 0:
                continue
            broker_qty[code] = broker_qty.get(code, 0) + volume

        unmanaged_state = self._states[unmanaged_id]
        previous_unmanaged_positions = unmanaged_state.get_positions()
        replayed_unmanaged_positions = previous_unmanaged_positions
        # 未管理账户会高频对账；这里只静默补齐成交回放口径，
        # 不强制 rebuild 整个账本，避免在实盘中心持续刷 INFO 日志。
        replayed = self._build_trade_record_replay(
            unmanaged_state,
            virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
        )
        if replayed:
            unmanaged_state.realized_pnl = float(replayed.get("realized_pnl", 0.0) or 0.0)
            unmanaged_state.trade_history = list(replayed.get("trade_history") or [])
            replayed_unmanaged_positions = {
                code: StrategyPositionState.from_dict(data)
                for code, data in dict(replayed.get("positions") or {}).items()
            }
        new_positions: Dict[str, dict] = {}
        for code, total_qty in broker_qty.items():
            unclaimed = int(total_qty) - int(claimed_qty.get(code, 0))
            if unclaimed <= 0:
                continue
            derived_cost = float(getattr(replayed_unmanaged_positions.get(code), "avg_cost", 0.0) or 0.0)
            previous_cost = float(getattr(previous_unmanaged_positions.get(code), "avg_cost", 0.0) or 0.0)
            new_positions[code] = StrategyPositionState(
                symbol_code=code,
                quantity=unclaimed,
                avg_cost=derived_cost or previous_cost,
            ).to_dict()

        cash_diff = round(float(broker_cash or 0.0) - claimed_cash, 2)
        unmanaged_cash = round(max(cash_diff, 0.0), 2)
        if cash_diff < -1.0:
            logger.warning(
                "unmanaged 对账发现已认领现金超过券商实际现金: broker_cash=%.2f claimed=%.2f diff=%.2f",
                float(broker_cash or 0.0),
                claimed_cash,
                cash_diff,
            )

        # 持仓对账：活跃策略声明持有的股数超过券商实际持仓的情况（虚报持仓）
        position_shortfalls: List[Dict[str, object]] = []
        for code, claimed in claimed_qty.items():
            broker_have = int(broker_qty.get(code, 0) or 0)
            if int(claimed) > broker_have:
                position_shortfalls.append({
                    "stock_code": code,
                    "claimed": int(claimed),
                    "broker": broker_have,
                    "shortfall": int(claimed) - broker_have,
                })
                logger.warning(
                    "unmanaged 对账发现策略声明持仓超过券商实际: code=%s claimed=%d broker=%d",
                    code, int(claimed), broker_have,
                )
        # 券商有但任何策略（含 unmanaged）都没声明的持仓——理论上不会发生
        # （unmanaged 会吸收所有未认领量），保留一个计数指标用于巡检
        untracked_broker_codes: List[str] = []
        for code, qty in broker_qty.items():
            if int(qty) > 0 and int(claimed_qty.get(code, 0) or 0) == 0 and code not in new_positions:
                untracked_broker_codes.append(code)

        claimed_unmanaged_codes: List[str] = []
        released_unmanaged_codes: List[str] = []
        skipped_unmanaged_claims: List[str] = []
        try:
            from trading_app.services.strategy_registry_service import get_strategy_registry_service

            registry = get_strategy_registry_service()
            current_unmanaged_codes = {
                str(item.symbol_code or "").strip()
                for item in registry.list_symbols(strategy_id=UNMANAGED_STRATEGY_ID)
            }
            next_unmanaged_codes = {str(code or "").strip() for code in new_positions.keys()}
            for code in sorted(current_unmanaged_codes - next_unmanaged_codes):
                if registry.release_symbol(code, strategy_id=UNMANAGED_STRATEGY_ID):
                    released_unmanaged_codes.append(code)
            for code in new_positions.keys():
                owner = registry.get_owner(code)
                if owner is not None and bool(getattr(owner, "enabled", False)):
                    if str(getattr(owner, "strategy_id", "") or "").strip() == unmanaged_id:
                        continue
                    skipped_unmanaged_claims.append(code)
                    continue
                ok, _message, _owner = registry.claim_symbol(
                    code,
                    strategy_id=UNMANAGED_STRATEGY_ID,
                    strategy_name=UNMANAGED_STRATEGY_NAME,
                    virtual_account_id=UNMANAGED_VIRTUAL_ACCOUNT_ID,
                    owner_type=OWNER_TYPE_UNMANAGED,
                )
                if ok:
                    claimed_unmanaged_codes.append(code)
        except Exception:
            logger.debug("为未管理持仓补登记股票归属失败", exc_info=True)

        unmanaged_state.cash_balance = unmanaged_cash
        unmanaged_state.positions = new_positions
        unmanaged_state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_states()

        return {
            "broker_cash": round(float(broker_cash or 0.0), 2),
            "claimed_cash": round(claimed_cash, 2),
            "unmanaged_cash": unmanaged_cash,
            "cash_shortfall": round(min(cash_diff, 0.0), 2),
            "unmanaged_position_count": len(new_positions),
            "unmanaged_positions": list(new_positions.keys()),
            "claimed_unmanaged_codes": claimed_unmanaged_codes,
            "released_unmanaged_codes": released_unmanaged_codes,
            "skipped_unmanaged_claims": skipped_unmanaged_claims,
            "position_shortfalls": position_shortfalls,
            "untracked_broker_codes": untracked_broker_codes,
        }


_strategy_budget_service: Optional[StrategyBudgetService] = None


def get_strategy_budget_service() -> StrategyBudgetService:
    global _strategy_budget_service
    if _strategy_budget_service is None:
        _strategy_budget_service = StrategyBudgetService()
    return _strategy_budget_service
