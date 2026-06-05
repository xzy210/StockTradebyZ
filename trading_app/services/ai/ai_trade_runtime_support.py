from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from common.broker_session_service import get_broker_session_service
from common.data_portal import get_data_portal
from trading_app.indicators import attach_all_indicators
from common.agent.agent_evidence_service import TEMP_KLINE_PREFIX
from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context
from trading_app.services.market_data.realtime_snapshot_service import load_symbol_view
from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service
from trading_app.services.strategy.strategy_constants import (
    AI_STOCK_STRATEGY_ID,
    AI_STOCK_STRATEGY_NAME,
    AI_STOCK_VIRTUAL_ACCOUNT_ID,
)
from trading_app.widgets.kline_widget import KLineWidget


class AITradeRuntimeSupport:
    """Provide name resolution, runtime context and lightweight K-line snapshots."""

    def __init__(
        self,
        *,
        project_root: str | None = None,
        data_dir: str | None = None,
        stocklist_path: str | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3])
        self.data_dir = data_dir or self._resolve_data_dir()
        self.stocklist_path = stocklist_path or self._resolve_stocklist_path()
        portal = get_data_portal()
        self.name_map = portal.get_name_map(asset_type="stock", stocklist_path=self.stocklist_path)
        self.etf_name_map = portal.get_name_map(asset_type="etf")
        self.broker = get_broker_session_service()
        self.ma_windows = [5, 10, 20]
        self.show_volume = True
        self.show_macd = True
        self.show_kdj = False
        self.snapshot_visible_bars = 180
        self.snapshot_size = (1500, 920)
        self._snapshot_host: Optional[QWidget] = None
        self._snapshot_widget: Optional[KLineWidget] = None

    def shutdown(self) -> None:
        if self._snapshot_host is not None:
            self._snapshot_host.close()
            self._snapshot_host.deleteLater()
            self._snapshot_host = None
            self._snapshot_widget = None

    def build_agent_runtime_context(
        self,
        symbol_override: Dict[str, Any] | None = None,
        run_context: Dict[str, Any] | DecisionRunContext | None = None,
    ) -> Dict[str, Any]:
        symbol_override = dict(symbol_override or {})
        resolved_run_context = self._coerce_run_context(run_context)
        code = str(symbol_override.get("code", "") or "").strip()
        name = str(symbol_override.get("name", "") or "").strip()
        asset_type = str(symbol_override.get("asset_type", "") or self._infer_asset_type(code)).strip()
        return {
            "data_dir": self.data_dir,
            "symbol": self._build_symbol_context(
                code=code,
                name=name,
                asset_type=asset_type,
                run_context=resolved_run_context,
            ),
            "watchlist": self._build_watchlist_context(),
            "broker": self._build_broker_context(),
            "market_data": self._build_market_data_context(),
            "decision_run_context": resolved_run_context.to_dict(),
            "_agent_tool_hooks": self._build_agent_tool_hooks(resolved_run_context),
        }

    def lookup_symbol_name(self, code: str) -> str:
        candidates = self._symbol_candidates(code)
        for candidate in candidates:
            if candidate in self.name_map and self.name_map.get(candidate):
                return str(self.name_map.get(candidate))
            if candidate in self.etf_name_map and self.etf_name_map.get(candidate):
                return str(self.etf_name_map.get(candidate))
        return ""

    def _build_agent_tool_hooks(self, run_context: DecisionRunContext) -> Dict[str, Any]:
        return {
            "get_current_symbol_df": (
                lambda code="", asset_type="": self.get_symbol_dataframe(
                    code,
                    asset_type,
                    run_context=run_context,
                )
            ),
            "capture_current_kline_image": (
                lambda code="", asset_type="": self.capture_symbol_kline_image(
                    code,
                    asset_type,
                    run_context=run_context,
                )
            ),
        }

    def _build_symbol_context(
        self,
        *,
        code: str,
        name: str,
        asset_type: str,
        run_context: DecisionRunContext,
    ) -> Dict[str, Any]:
        resolved_name = name or self.lookup_symbol_name(code)
        if not code:
            return {
                "code": "",
                "name": "",
                "asset_type": "",
                "current_view": "",
                "latest_close": 0.0,
                "latest_change_pct": 0.0,
                "latest_volume": 0.0,
                "data_points": 0,
                "date_start": "",
                "date_end": "",
                "indicators": self._enabled_indicator_labels(),
            }

        df = self.get_symbol_dataframe(code, asset_type, run_context=run_context)
        latest_close = 0.0
        latest_change_pct = 0.0
        latest_volume = 0.0
        data_points = 0
        date_start = ""
        date_end = ""
        if df is not None and not df.empty:
            frame = df.sort_values("date").reset_index(drop=True)
            data_points = len(frame)
            latest = frame.iloc[-1]
            latest_close = float(latest.get("close", 0.0) or 0.0)
            latest_volume = float(latest.get("volume", 0.0) or 0.0)
            if data_points >= 2:
                prev_close = float(frame.iloc[-2].get("close", 0.0) or 0.0)
                if prev_close > 0:
                    latest_change_pct = (latest_close - prev_close) / prev_close * 100
            first_date = frame.iloc[0].get("date")
            last_date = latest.get("date")
            date_start = first_date.strftime("%Y-%m-%d") if hasattr(first_date, "strftime") else str(first_date)[:10]
            date_end = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)[:10]

        return {
            "code": code,
            "name": resolved_name,
            "asset_type": asset_type,
            "current_view": "etf" if asset_type == "ETF" else "stock",
            "latest_close": latest_close,
            "latest_change_pct": latest_change_pct,
            "latest_volume": latest_volume,
            "data_points": data_points,
            "date_start": date_start,
            "date_end": date_end,
            "indicators": self._enabled_indicator_labels(),
        }

    @staticmethod
    def _build_watchlist_context() -> Dict[str, Any]:
        return {
            "source_tab": "",
            "group_name": "",
            "visible_count": 0,
            "visible_codes": [],
            "visible_items": [],
        }

    def _build_broker_context(self) -> Dict[str, Any]:
        if not self.broker.is_connected:
            return {"connected": False}
        try:
            asset = self.broker.query_stock_asset()
            positions = self.broker.query_stock_positions() or []
            state_positions = get_strategy_budget_service().get_strategy_state_record(
                AI_STOCK_STRATEGY_ID,
                strategy_name=AI_STOCK_STRATEGY_NAME,
                virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            ).get_positions()
            top_positions = []
            for pos in positions[:5]:
                code = str(getattr(pos, "stock_code", "") or "")
                simple_code = self._plain_code(code)
                budget_pos = state_positions.get(simple_code)
                top_positions.append({
                    "code": simple_code,
                    "volume": int(getattr(pos, "volume", 0) or 0),
                    "cost_price": float(getattr(budget_pos, "avg_cost", 0.0) or 0.0),
                    "market_value": float(getattr(pos, "market_value", 0.0) or 0.0),
                })
            return {
                "connected": True,
                "account_id": str(getattr(asset, "account_id", "")),
                "total_asset": float(getattr(asset, "total_asset", 0.0) or 0.0),
                "available_cash": float(getattr(asset, "cash", 0.0) or 0.0),
                "position_count": len(positions),
                "top_positions": top_positions,
            }
        except Exception:
            return {"connected": False}

    def _build_market_data_context(self) -> Dict[str, Any]:
        config_path = self.project_root / "trading_app" / "config" / "scheduler_config.json"
        config: Dict[str, Any] = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        token = str(config.get("tushare_token", "") or os.environ.get("TUSHARE_TOKEN", "")).strip()
        return {
            "has_tushare_token": bool(token),
            "data_source": config.get("data_source", "xtquant"),
        }

    def get_symbol_dataframe(
        self,
        code: str = "",
        asset_type: str = "",
        run_context: Dict[str, Any] | DecisionRunContext | None = None,
    ) -> pd.DataFrame | None:
        plain_code = self._plain_code(code)
        resolved_asset_type = asset_type or self._infer_asset_type(plain_code)
        resolved_run_context = self._coerce_run_context(run_context)
        if not plain_code:
            return None
        df, _ = load_symbol_view(
            plain_code,
            asset_type=resolved_asset_type,
            data_dir=self.data_dir,
            run_context=resolved_run_context,
            use_cache=True,
        )
        if df is None or df.empty:
            return None
        return attach_all_indicators(
            df,
            ma_windows=self.ma_windows,
            include_macd=self.show_macd,
            include_kdj=self.show_kdj,
            include_bbi=False,
            vol_ma_window=5,
        )

    def capture_symbol_kline_image(
        self,
        code: str = "",
        asset_type: str = "",
        run_context: Dict[str, Any] | DecisionRunContext | None = None,
    ) -> str | None:
        plain_code = self._plain_code(code)
        resolved_asset_type = asset_type or self._infer_asset_type(plain_code)
        if not plain_code:
            return None
        df = self.get_symbol_dataframe(
            plain_code,
            resolved_asset_type,
            run_context=run_context,
        )
        if df is None or df.empty:
            return None
        widget = self._ensure_snapshot_widget()
        widget.set_indicators(
            show_volume=self.show_volume,
            show_macd=self.show_macd,
            show_kdj=self.show_kdj,
        )
        widget.set_ma_windows(self.ma_windows)
        widget.set_data(
            df,
            plain_code,
            self.lookup_symbol_name(plain_code) or plain_code,
            is_index=False,
        )
        self._configure_snapshot_view(widget, len(df))
        if self._snapshot_host is not None:
            self._snapshot_host.resize(*self.snapshot_size)
            self._snapshot_host.show()
        QApplication.processEvents()
        pixmap = widget.grab()
        if pixmap.isNull():
            fallback = QPixmap(widget.size())
            fallback.fill(Qt.GlobalColor.transparent)
            widget.render(fallback)
            pixmap = fallback
        if self._snapshot_host is not None:
            self._snapshot_host.hide()
        if pixmap.isNull():
            return None
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / f"{TEMP_KLINE_PREFIX}{plain_code}_{int(time.time())}.png"
        if not pixmap.save(str(file_path), "PNG"):
            return None
        return str(file_path)

    def _ensure_snapshot_widget(self) -> KLineWidget:
        if self._snapshot_widget is not None:
            return self._snapshot_widget
        self._snapshot_host = QWidget()
        try:
            self._snapshot_host.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        except Exception:
            pass
        self._snapshot_host.setWindowFlags(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
        )
        layout = QVBoxLayout(self._snapshot_host)
        layout.setContentsMargins(0, 0, 0, 0)
        self._snapshot_widget = KLineWidget()
        self._snapshot_widget.setMinimumSize(*self.snapshot_size)
        layout.addWidget(self._snapshot_widget)
        self._snapshot_host.resize(*self.snapshot_size)
        return self._snapshot_widget

    def _configure_snapshot_view(self, widget: KLineWidget, data_len: int) -> None:
        if data_len <= 0 or not hasattr(widget, "price_plot"):
            return
        visible_bars = min(max(90, self.snapshot_visible_bars), data_len)
        start = max(0, data_len - visible_bars)
        try:
            widget.price_plot.setXRange(start, data_len, padding=0.01)
            widget.update_y_range()
        except Exception:
            pass

    def _resolve_data_dir(self) -> str:
        candidates = [
            self.project_root / "data",
            Path("./data"),
            Path("../data"),
        ]
        for path in candidates:
            if path.exists():
                return str(path.resolve())
        return str(candidates[0].resolve())

    def _resolve_stocklist_path(self) -> str:
        candidates = [
            self.project_root / "stocklist" / "stocklist.csv",
            Path("./stocklist/stocklist.csv"),
            Path("../stocklist/stocklist.csv"),
        ]
        for path in candidates:
            if path.exists():
                return str(path.resolve())
        return str(candidates[0].resolve())

    def _infer_asset_type(self, code: str) -> str:
        plain_code = self._plain_code(code)
        if plain_code in self.etf_name_map:
            return "ETF"
        if plain_code.startswith(("51", "52", "56", "58", "15", "16", "18")):
            return "ETF"
        return "股票"

    @staticmethod
    def _plain_code(code: str) -> str:
        return str(code or "").split(".")[0].strip().upper()

    def _symbol_candidates(self, code: str) -> list[str]:
        plain_code = self._plain_code(code)
        candidates = [code, plain_code]
        if plain_code:
            if plain_code.startswith(("5", "6", "9")):
                candidates.append(f"{plain_code}.SH")
            elif plain_code.startswith(("0", "1", "2", "3")):
                candidates.append(f"{plain_code}.SZ")
        unique: list[str] = []
        for item in candidates:
            normalized = str(item or "").strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique

    def _enabled_indicator_labels(self) -> list[str]:
        indicators = []
        if self.show_volume:
            indicators.append("成交量")
        if self.show_macd:
            indicators.append("MACD")
        if self.show_kdj:
            indicators.append("KDJ")
        return indicators

    @staticmethod
    def _coerce_run_context(
        run_context: Dict[str, Any] | DecisionRunContext | None,
    ) -> DecisionRunContext:
        if isinstance(run_context, DecisionRunContext):
            return run_context
        if isinstance(run_context, dict):
            return DecisionRunContext.from_dict(run_context)
        return build_decision_run_context(prefer_realtime=True)
