from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import pandas as pd

from common.execution_contract import BUY, HOLD, SELL, StrategySignal
from common.strategy_spec import StrategySpec
from strategy_app.timing.inference import TimingModelPredictor, TimingPrediction

from .base_strategy import BaseStrategy


class TCNAttentionTimingStrategy(BaseStrategy):
    """TCN + Attention 三障碍方向时序策略。"""

    strategy_version: ClassVar[str] = "v1"
    spec: ClassVar[StrategySpec] = StrategySpec(
        strategy_id="tcn_attention_timing",
        strategy_name="TCN Attention 时序策略",
        asset_class="stock",
        frequency="5m/daily",
        plugin_name="TCN Attention 时序策略",
        metadata={"source": "timing", "strategy_version": strategy_version},
    )
    prefer_generate_signals: ClassVar[bool] = True

    def __init__(self):
        super().__init__()
        self.params.update(
            {
                "model_dir": "",
                "device": "auto",
                "up_threshold": 0.55,
                "down_threshold": 0.55,
                "direction_margin": 0.15,
                "target_percent": 0.5,
                "allow_buy": True,
                "sell_on_down": True,
                "frequency": "",
            }
        )
        self._predictor: Optional[TimingModelPredictor] = None
        self.last_prediction: Optional[TimingPrediction] = None
        self._backtest_signal_trace: list[dict[str, Any]] = []

    def check(self, code: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        prediction = self._predict(data)
        if prediction is None:
            return None
        return {
            "code": code,
            "signal": prediction.label,
            "score": prediction.confidence,
            **prediction.to_dict(),
        }

    def initialize_backtest(self, context, prepared) -> None:
        self._backtest_signal_trace = []
        self._ensure_predictor()

    def generate_signals(self, data: Any, context: Any = None) -> list[StrategySignal]:
        symbol = str((data or {}).get("primary_symbol") or (data or {}).get("code") or "")
        history = (data or {}).get("history_slice")
        if not symbol or history is None:
            return []

        prediction = self._predict(history)
        if prediction is None:
            return []
        self.last_prediction = prediction

        current_price = _current_price(data, symbol)
        position_qty = _position_quantity(context, symbol)
        metadata = {
            **prediction.to_dict(),
            "upper_price": prediction.upper_price,
            "lower_price": prediction.lower_price,
            "lookback": getattr(self._predictor, "lookback", 0),
            "horizon": getattr(getattr(self._predictor, "label_config", None), "horizon", 0),
            "frequency": getattr(self._predictor, "frequency", "") if self._predictor is not None else "",
        }

        is_buy = self._is_buy_signal(prediction) and position_qty <= 0 and bool(self.params.get("allow_buy", True))
        is_sell = self._is_sell_signal(prediction) and position_qty > 0 and bool(self.params.get("sell_on_down", True))
        if is_buy:
            self._record_signal_trace(data, symbol, current_price, position_qty, prediction, BUY, "达到看多交易门槛")
            return [
                StrategySignal(
                    symbol=symbol,
                    action=BUY,
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                    strength=prediction.p_up,
                    target_percent=float(self.params.get("target_percent") or 0.0),
                    price=current_price,
                    reason=f"TCN时序看多 p_up={prediction.p_up:.2%}",
                    timestamp=(data or {}).get("date"),
                    metadata=metadata,
                )
            ]

        if is_sell:
            self._record_signal_trace(data, symbol, current_price, position_qty, prediction, SELL, "达到看空交易门槛")
            return [
                StrategySignal(
                    symbol=symbol,
                    action=SELL,
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                    strength=prediction.p_down,
                    target_quantity=0,
                    price=current_price,
                    reason=f"TCN时序看空 p_down={prediction.p_down:.2%}",
                    timestamp=(data or {}).get("date"),
                    metadata=metadata,
                )
            ]

        self._record_signal_trace(data, symbol, current_price, position_qty, prediction, HOLD, "信号未达到交易门槛")
        return [
            StrategySignal(
                symbol=symbol,
                action=HOLD,
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                strength=prediction.confidence,
                price=current_price,
                reason="TCN时序信号未达到交易门槛",
                timestamp=(data or {}).get("date"),
                metadata=metadata,
            )
        ]

    def _predict(self, bars: pd.DataFrame) -> Optional[TimingPrediction]:
        self._ensure_predictor()
        return self._predictor.predict_latest(bars) if self._predictor is not None else None

    def _ensure_predictor(self) -> None:
        if self._predictor is not None:
            return
        model_dir = str(self.params.get("model_dir") or "").strip()
        if not model_dir:
            raise ValueError("TCN Attention 时序策略缺少 model_dir 参数")
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"模型目录不存在: {path}")
        self._predictor = TimingModelPredictor(path, device=str(self.params.get("device") or "auto"))
        expected_frequency = str(self.params.get("frequency") or "").strip()
        model_frequency = getattr(self._predictor, "frequency", "")
        if expected_frequency and model_frequency and expected_frequency != model_frequency:
            raise ValueError(f"模型周期({model_frequency})与回测周期({expected_frequency})不一致")

    def _is_buy_signal(self, prediction: TimingPrediction) -> bool:
        threshold = float(self.params.get("up_threshold") or 0.0)
        margin = float(self.params.get("direction_margin") or 0.0)
        return prediction.p_up >= threshold and (prediction.p_up - prediction.p_down) >= margin

    def _is_sell_signal(self, prediction: TimingPrediction) -> bool:
        threshold = float(self.params.get("down_threshold") or 0.0)
        margin = float(self.params.get("direction_margin") or 0.0)
        return prediction.p_down >= threshold and (prediction.p_down - prediction.p_up) >= margin

    def _record_signal_trace(
        self,
        data: Any,
        symbol: str,
        current_price: Optional[float],
        position_qty: int,
        prediction: TimingPrediction,
        action: str,
        reason: str,
    ) -> None:
        up_threshold = float(self.params.get("up_threshold") or 0.0)
        down_threshold = float(self.params.get("down_threshold") or 0.0)
        margin = float(self.params.get("direction_margin") or 0.0)
        self._backtest_signal_trace.append(
            {
                "date": (data or {}).get("date"),
                "symbol": symbol,
                "close": current_price,
                "position_qty": int(position_qty or 0),
                "action": action,
                "reason": reason,
                "label": prediction.label,
                "p_down": prediction.p_down,
                "p_flat": prediction.p_flat,
                "p_up": prediction.p_up,
                "confidence": prediction.confidence,
                "up_threshold": up_threshold,
                "down_threshold": down_threshold,
                "direction_margin": margin,
                "up_minus_down": prediction.p_up - prediction.p_down,
                "down_minus_up": prediction.p_down - prediction.p_up,
                "buy_threshold_hit": prediction.p_up >= up_threshold
                and (prediction.p_up - prediction.p_down) >= margin,
                "sell_threshold_hit": prediction.p_down >= down_threshold
                and (prediction.p_down - prediction.p_up) >= margin,
                "upper_price": prediction.upper_price,
                "lower_price": prediction.lower_price,
            }
        )

    def finalize_backtest_result(self, result: dict, **_kwargs) -> dict:
        result["timing_signal_trace"] = pd.DataFrame(self._backtest_signal_trace)
        return result


def _current_price(data: Any, symbol: str) -> Optional[float]:
    prices = (data or {}).get("prices") or {}
    value = prices.get(symbol) or prices.get(symbol.split(".", 1)[0])
    return float(value) if value is not None else None


def _position_quantity(context: Any, symbol: str) -> int:
    if context is None:
        return 0
    plain = symbol.split(".", 1)[0]
    position = getattr(context, "positions", {}).get(symbol) or getattr(context, "positions", {}).get(plain)
    return int(getattr(position, "quantity", 0) or 0) if position is not None else 0
