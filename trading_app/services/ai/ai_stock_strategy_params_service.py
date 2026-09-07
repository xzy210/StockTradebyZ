from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from common.io_utils import atomic_write_json
from strategy_app.strategies.ai_stock_strategy_params import AIStockStrategyParams

from trading_app.services.ai.auto_trade_config_service import AutoTradeConfig
from trading_app.services.execution.risk_guard_service import DEFAULT_CONFIG as RISK_DEFAULT_CONFIG
from trading_app.services.live_strategy_center.storage import get_live_strategy_center_storage
from trading_app.services.ai.stock_pool_service import StockPoolConfig

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_AI_CONFIG_PATH = _CONFIG_DIR / "ai_config.json"
_RISK_CONFIG_PATH = _CONFIG_DIR / "risk_guard_config.json"
_STOCK_POOL_CONFIG_PATH = _CONFIG_DIR / "stock_pool_config.json"
_AUTO_TRADE_CONFIG_PATH = _CONFIG_DIR / "auto_trade_config.json"


class AIStockStrategyParamsService:
    """Bridge existing live AI config files to the shared parameter model."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self.config_dir = config_dir or _CONFIG_DIR

    def load_params(self) -> AIStockStrategyParams:
        ai_config = self._read_json(_AI_CONFIG_PATH)
        try:
            from trading_app.services.ai.kimi_compat import migrate_retired_kimi_config
            ai_config = migrate_retired_kimi_config(ai_config)
        except Exception:
            pass
        selected_model = str(ai_config.get("selected_model", "") or "").strip()
        model_configs = dict(ai_config.get("model_configs", {}) or {})
        selected_model_config = dict(model_configs.get(selected_model, {}) or {})

        risk_config = dict(RISK_DEFAULT_CONFIG)
        risk_config.update(self._read_json(_RISK_CONFIG_PATH))

        stock_pool = StockPoolConfig.from_dict(self._read_json(_STOCK_POOL_CONFIG_PATH)).to_dict()
        auto_trade = AutoTradeConfig.from_dict(self._read_json(_AUTO_TRADE_CONFIG_PATH)).to_dict()
        ai_schedule = self._load_schedule("daily_ai_strategy_cycle")
        unmanaged_schedule = self._load_schedule("daily_unmanaged_position_scan")

        return AIStockStrategyParams.from_mapping(
            {
                "model_name": selected_model,
                "model_base_url": str(selected_model_config.get("base_url", "") or ""),
                "system_prompt": str(ai_config.get("system_prompt", "") or ""),
                "candidate_pool_enabled": bool(stock_pool.get("enabled", True)),
                "candidate_pool_name": str(stock_pool.get("pool_name", "") or ""),
                "candidate_universe_file": str(stock_pool.get("universe_file", "") or ""),
                "benchmark_index_code": str(stock_pool.get("benchmark_index_code", "") or ""),
                "max_candidates": int(stock_pool.get("max_candidates", 30) or 30),
                "ai_review_limit": int(stock_pool.get("ai_review_limit", 10) or 10),
                **{key: risk_config.get(key) for key in RISK_DEFAULT_CONFIG},
                "execution_sequence": auto_trade.get("execution_sequence", "sell_first"),
                "buy_sizing_mode": auto_trade.get("buy_sizing_mode", "equal_slots"),
                "buy_value_per_order": auto_trade.get("buy_value_per_order", 5000.0),
                "buy_position_pct": auto_trade.get("buy_position_pct", 0.10),
                "sell_sizing_mode": auto_trade.get("sell_sizing_mode", "signal_driven"),
                "max_buy_orders_per_day": auto_trade.get("max_buy_orders_per_day", 2),
                "max_sell_orders_per_day": auto_trade.get("max_sell_orders_per_day", 6),
                "max_new_positions_per_day": auto_trade.get("max_new_positions_per_day", 2),
                "allow_open_new_position": auto_trade.get("allow_open_new_position", True),
                "allow_add_to_existing": auto_trade.get("allow_add_to_existing", True),
                "reserve_cash_pct": auto_trade.get("reserve_cash_pct", 0.20),
                "max_daily_loss_pct": auto_trade.get("max_daily_loss_pct", 0.02),
                "schedule_enabled": bool(ai_schedule.get("enabled", False)),
                "schedule_time": str(ai_schedule.get("scheduled_time", "") or "14:35"),
                "schedule_notify_on_complete": bool(ai_schedule.get("notify_on_complete", True)),
                "schedule_auto_execute": bool(ai_schedule.get("auto_execute", True)),
                "unmanaged_schedule_enabled": bool(unmanaged_schedule.get("enabled", False)),
                "unmanaged_schedule_time": str(unmanaged_schedule.get("scheduled_time", "") or "14:40"),
            }
        )

    def save_params(self, params: AIStockStrategyParams | dict) -> AIStockStrategyParams:
        model = params if isinstance(params, AIStockStrategyParams) else AIStockStrategyParams.from_mapping(params)
        self._write_ai_config(model)
        self._write_risk_config(model)
        self._write_stock_pool_config(model)
        self._write_auto_trade_config(model)
        return self.load_params()

    def _load_schedule(self, task_key: str) -> dict:
        try:
            return get_live_strategy_center_storage().get_task_schedule_config(task_key)
        except Exception as exc:
            logger.debug("读取 AI 调度参数失败: %s", exc)
            return {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                return dict(payload or {}) if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning("读取 AI 参数配置失败: %s path=%s", exc, path)
        return {}

    def _write_ai_config(self, params: AIStockStrategyParams) -> None:
        payload = self._read_json(_AI_CONFIG_PATH)
        payload["selected_model"] = params.model_name
        payload["system_prompt"] = params.system_prompt
        atomic_write_json(_AI_CONFIG_PATH, payload)

    def _write_risk_config(self, params: AIStockStrategyParams) -> None:
        payload = self._read_json(_RISK_CONFIG_PATH)
        for key in RISK_DEFAULT_CONFIG:
            if key in params.field_names():
                payload[key] = getattr(params, key)
        atomic_write_json(_RISK_CONFIG_PATH, payload)

    def _write_stock_pool_config(self, params: AIStockStrategyParams) -> None:
        payload = self._read_json(_STOCK_POOL_CONFIG_PATH)
        payload.update(
            {
                "enabled": params.candidate_pool_enabled,
                "pool_name": params.candidate_pool_name,
                "universe_file": params.candidate_universe_file,
                "benchmark_index_code": params.benchmark_index_code,
                "max_candidates": params.max_candidates,
                "ai_review_limit": params.ai_review_limit,
            }
        )
        atomic_write_json(_STOCK_POOL_CONFIG_PATH, payload)

    def _write_auto_trade_config(self, params: AIStockStrategyParams) -> None:
        payload = self._read_json(_AUTO_TRADE_CONFIG_PATH)
        payload.update(
            {
                "execution_sequence": params.execution_sequence,
                "buy_sizing_mode": params.buy_sizing_mode,
                "buy_value_per_order": params.buy_value_per_order,
                "buy_position_pct": params.buy_position_pct,
                "sell_sizing_mode": params.sell_sizing_mode,
                "max_buy_orders_per_day": params.max_buy_orders_per_day,
                "max_sell_orders_per_day": params.max_sell_orders_per_day,
                "max_new_positions_per_day": params.max_new_positions_per_day,
                "allow_open_new_position": params.allow_open_new_position,
                "allow_add_to_existing": params.allow_add_to_existing,
                "reserve_cash_pct": params.reserve_cash_pct,
                "max_daily_loss_pct": params.max_daily_loss_pct,
            }
        )
        atomic_write_json(_AUTO_TRADE_CONFIG_PATH, payload)


_params_service: Optional[AIStockStrategyParamsService] = None


def get_ai_stock_strategy_params_service() -> AIStockStrategyParamsService:
    global _params_service
    if _params_service is None:
        _params_service = AIStockStrategyParamsService()
    return _params_service


__all__ = [
    "AIStockStrategyParamsService",
    "get_ai_stock_strategy_params_service",
]
