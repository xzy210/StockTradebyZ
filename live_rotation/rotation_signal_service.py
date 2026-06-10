"""
ETF rotation signal services.

This module contains the parts of the live rotation workflow that do not need
Qt signals or order execution: score calculation and rebalance decision making.
"""
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from .config import RotationConfig
from .rotation_data_service import RotationDataService
from .state_manager import RotationState

logger = logging.getLogger(__name__)


class RotationSignalService:
    """Load ETF data and calculate strategy scores."""

    def __init__(
        self,
        *,
        config: RotationConfig,
        data_dir: Path,
        strategy_provider,
        data_service: Optional[RotationDataService] = None,
        logger_fn: Optional[Callable[[str], None]] = None,
        code_name_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_service = data_service or RotationDataService(self.data_dir)
        self.strategy_provider = strategy_provider
        self.logger_fn = logger_fn or (lambda message: None)
        self.code_name_fn = code_name_fn or (lambda code: code)
        self._strategy = None

    def update_context(
        self,
        *,
        config: RotationConfig,
        data_dir: Optional[Path] = None,
        strategy_provider=None,
        reset_strategy: bool = False,
    ) -> None:
        """Refresh mutable dependencies after config or provider changes."""
        if config is not self.config:
            reset_strategy = True
        if strategy_provider is not None and strategy_provider is not self.strategy_provider:
            self.strategy_provider = strategy_provider
            reset_strategy = True
        self.config = config
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            self.data_service.update_context(data_dir=self.data_dir)
        if reset_strategy:
            self._strategy = None

    def reset_strategy(self) -> None:
        """Force the next score calculation to recreate the strategy instance."""
        self._strategy = None

    def get_strategy(self):
        """Create the configured strategy lazily."""
        if self._strategy is None:
            self._strategy = self.strategy_provider.create_strategy(
                self.config.strategy_id,
                self.config,
            )
        return self._strategy

    def calculate_scores(self) -> Dict[str, float]:
        """Load ETF parquet files and calculate all configured ETF scores."""
        strategy = self.get_strategy()

        all_data = {}
        for code in self.config.etf_pool:
            df = self.data_service.load_daily_bars(code)
            if df is not None and len(df) >= self.config.zscore_window:
                all_data[code] = df
                self.logger_fn(f"  ✓ {self.code_name_fn(code)}: {len(df)} 条数据")
            else:
                count = len(df) if df is not None else 0
                self.logger_fn(f"  ✗ {self.code_name_fn(code)}: 数据不足 ({count}条)")

        if len(all_data) < 2:
            self.logger_fn("可用ETF不足2只，无法计算轮动信号")
            return {}

        if hasattr(strategy, "score_data_view"):
            return strategy.score_data_view(all_data)
        return strategy.calculate_all_scores(all_data)


class RotationDecisionService:
    """Make rebalance decisions from calculated ETF scores and current state."""

    def __init__(
        self,
        *,
        config: RotationConfig,
        state: RotationState,
        code_name_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config
        self.state = state
        self.code_name_fn = code_name_fn or (lambda code: code)

    def update_context(self, *, config: RotationConfig, state: RotationState) -> None:
        """Refresh mutable config/state references."""
        self.config = config
        self.state = state

    def make_decision(self, scores: Dict[str, float]) -> Tuple[str, Optional[str], str]:
        """
        Decide the target rotation action.

        Returns:
            (signal, target_code, reason)
            signal: HOLD / SWITCH / SELL_ALL / BUY / NO_ACTION
        """
        if not scores:
            return "NO_ACTION", None, "没有可用得分"

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_code, top_score = sorted_scores[0]

        if self.config.enable_empty_position:
            all_below = all(s < self.config.empty_threshold for _, s in sorted_scores)
            if all_below:
                if self.state.current_holding:
                    return (
                        "SELL_ALL",
                        None,
                        f"所有ETF得分低于阈值({self.config.empty_threshold}), 最高={top_score:.4f}",
                    )
                return (
                    "NO_ACTION",
                    None,
                    f"空仓中，所有得分仍低于阈值 {self.config.empty_threshold}",
                )

        holding = self.state.current_holding
        holding_score = scores.get(holding) if holding else None

        if holding is None or holding_score is None:
            min_buy_score = float(getattr(self.config, "min_buy_score", 0.0) or 0.0)
            if top_score < min_buy_score:
                return (
                    "NO_ACTION",
                    None,
                    f"空仓中，最高分 {self.code_name_fn(top_code)}({top_score:.4f}) "
                    f"低于最低买入分数 {min_buy_score:.4f}",
                )
            return (
                "BUY",
                top_code,
                f"初始建仓，买入最优 {self.code_name_fn(top_code)} (得分={top_score:.4f})",
            )

        if top_code != holding:
            threshold = self.config.rebalance_threshold
            if top_score > 0 and holding_score > 0:
                if top_score > holding_score * threshold:
                    return (
                        "SWITCH",
                        top_code,
                        f"{self.code_name_fn(top_code)}({top_score:.4f}) > "
                        f"{self.code_name_fn(holding)}({holding_score:.4f}) × {threshold}",
                    )
            elif top_score > 0 >= holding_score:
                return (
                    "SWITCH",
                    top_code,
                    f"{self.code_name_fn(top_code)} 已转为正分({top_score:.4f})，"
                    f"当前持仓 {self.code_name_fn(holding)} 仍为负分({holding_score:.4f})",
                )
            elif top_score <= 0 and holding_score <= 0:
                return (
                    "HOLD",
                    None,
                    f"候选 {self.code_name_fn(top_code)}({top_score:.4f}) 与当前持仓 "
                    f"{self.code_name_fn(holding)}({holding_score:.4f}) 均处于负分区，"
                    f"不按倍率阈值切换",
                )

        return (
            "HOLD",
            None,
            f"继续持有 {self.code_name_fn(holding)} (得分={holding_score:.4f})",
        )
