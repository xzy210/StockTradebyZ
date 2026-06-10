"""
ETF轮动实盘 - 配置管理

管理策略参数、ETF池、交易参数等所有可配置项。
"""
import json
import logging
import warnings
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import ClassVar, Dict, List, Optional, Tuple

from common.io_utils import atomic_write_json
from strategy_app.strategies.etf_rotation_params import ETFRotationParams

logger = logging.getLogger(__name__)


@dataclass
class RotationConfig:
    """ETF轮动实盘配置"""

    # --- ETF标的池 ---
    etf_pool: List[str] = field(default_factory=lambda: [
        '510880', '159949', '513100', '518880'
    ])
    strategy_id: str = "etf_rotation"
    strategy_params: dict = field(default_factory=dict)

    # --- 策略参数（与回测保持一致）---
    factor_config: List[tuple] = field(default_factory=lambda: [
        ('bias_momentum_fast', 0.3),
        ('slope_momentum_fast', 0.3),
        ('efficiency_momentum_fast', 0.4),
    ])
    rebalance_threshold: float = 1.5
    momentum_window: int = 25
    zscore_window: int = 60
    empty_threshold: float = -0.5
    min_buy_score: float = 0.0          # 空仓/初始建仓时，最高分需达到此阈值才买入
    enable_empty_position: bool = True
    rebalance_period: int = 1            # 调仓周期（交易日）: 1=每日, 5=每周, 20=每月

    # --- 专用资金（资金隔离，始终启用）---
    use_dedicated_capital: bool = True     # 内部标志，始终为 True
    dedicated_capital: float = 100000.0   # 划拨给本策略的启动资金（元）

    # --- 交易参数 ---
    cash_ratio: float = 0.99            # 买入时使用的资金比例
    min_trade_amount: float = 1000.0    # 最小交易金额
    price_type: str = "market"          # market / limit
    limit_slip_pct: float = 0.1         # 限价单滑点百分比

    # --- 定时调度 ---
    auto_enabled: bool = False
    auto_signal_enabled: bool = True    # 到点后自动生成策略信号
    auto_execute_enabled: bool = False  # 到点生成信号后自动提交统一委托
    check_time: str = "14:50"           # 每日信号检查时间 (HH:MM)
    data_update_time: str = "14:30"     # 数据更新时间

    # --- 通知 ---
    notify_on_signal: bool = True       # 产生信号时通知
    notify_on_trade: bool = True        # 交易执行后通知
    notify_daily_report: bool = True    # 每日推送简报

    # --- 风控 ---
    max_trades_per_day: int = 2
    min_hold_days: int = 0              # 最少持有天数（0=不限制）
    max_single_loss_pct: float = 15.0   # 单笔最大亏损比例
    trading_start: str = "09:30"
    trading_end: str = "14:57"          # 留3分钟buffer
    enable_risk_check: bool = True

    # --- 风控（高级）---
    enable_trailing_stop: bool = True     # 移动止盈
    trailing_stop_pct: float = 0.08       # 持仓从最高价回撤此比例时触发止盈
    enable_drawdown_protection: bool = True  # 账户回撤保护
    max_drawdown_pct: float = 0.15        # 账户最大回撤比例
    drawdown_cooldown_days: int = 10      # 触发回撤保护后冷却天数

    def to_params(self) -> ETFRotationParams:
        """Return the shared ETF rotation parameter model."""
        payload = {
            'etf_pool': self.etf_pool,
            'factor_config': self.factor_config,
            'rebalance_threshold': self.rebalance_threshold,
            'momentum_window': self.momentum_window,
            'zscore_window': self.zscore_window,
            'empty_threshold': self.empty_threshold,
            'min_buy_score': self.min_buy_score,
            'enable_empty_position': self.enable_empty_position,
            'rebalance_period': self.rebalance_period,
            'enable_trailing_stop': self.enable_trailing_stop,
            'trailing_stop_pct': self.trailing_stop_pct,
            'enable_drawdown_protection': self.enable_drawdown_protection,
            'max_drawdown_pct': self.max_drawdown_pct,
            'drawdown_cooldown_days': self.drawdown_cooldown_days,
        }
        payload.update(self.strategy_params or {})
        return ETFRotationParams.from_mapping(payload)

    @classmethod
    def from_params(cls, params: ETFRotationParams | dict) -> 'RotationConfig':
        """Build a live config shell from the shared ETF rotation params."""
        model = params if isinstance(params, ETFRotationParams) else ETFRotationParams.from_mapping(params)
        values = model.to_dict()
        config = cls()
        for key, value in values.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.strategy_params = {}
        return config

    def to_strategy_params(self) -> dict:
        """Deprecated: use to_params() and pass ETFRotationParams through create_strategy."""
        warnings.warn(
            "RotationConfig.to_strategy_params() is deprecated; use to_params() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_params().to_dict()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'RotationConfig':
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        raw = dict(data or {})
        if "auto_signal_enabled" not in raw and "auto_execute" in raw:
            raw["auto_signal_enabled"] = bool(raw.get("auto_execute", True))
        if "auto_execute_enabled" not in raw:
            raw["auto_execute_enabled"] = False
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        # 历史版本里曾把手续费放在 ETF 独立配置中；现已统一迁移到
        # trading_app/config/trade_fee_config.json，这里直接忽略旧字段。
        strategy_payload = dict(filtered.get('strategy_params') or {})
        for key in ETFRotationParams.field_names():
            if key in filtered:
                strategy_payload.setdefault(key, filtered[key])
        for key in ('bias_weight', 'slope_weight', 'efficiency_weight'):
            if key in raw:
                strategy_payload.setdefault(key, raw[key])
        params = ETFRotationParams.from_mapping(strategy_payload)
        for key, value in params.to_dict().items():
            if key in valid_keys:
                filtered[key] = value
        filtered['strategy_params'] = {
            key: value
            for key, value in strategy_payload.items()
            if key not in ETFRotationParams.field_names()
            and key not in ('bias_weight', 'slope_weight', 'efficiency_weight')
        }
        return cls(**filtered)


class ConfigManager:
    """配置持久化管理"""

    CONFIG_FILE = "rotation_config.json"
    _CACHE: ClassVar[Dict[Path, Tuple[Tuple[int, int] | None, RotationConfig]]] = {}

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / self.CONFIG_FILE

    def load(self) -> RotationConfig:
        signature = self._file_signature()
        cached = self._CACHE.get(self.config_path)
        if cached is not None and cached[0] == signature:
            return deepcopy(cached[1])
        if signature is None:
            logger.info("配置文件不存在，使用默认配置")
            config = RotationConfig()
            self._CACHE[self.config_path] = (signature, deepcopy(config))
            return config
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = RotationConfig.from_dict(data)
            self._CACHE[self.config_path] = (signature, deepcopy(config))
            logger.info(f"已加载配置: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {e}，使用默认配置")
            return RotationConfig()

    def save(self, config: RotationConfig):
        try:
            atomic_write_json(self.config_path, config.to_dict())
            signature = self._file_signature()
            self._CACHE[self.config_path] = (signature, deepcopy(config))
            logger.info(f"配置已保存: {self.config_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    def _file_signature(self) -> Tuple[int, int] | None:
        try:
            stat = self.config_path.stat()
            return int(stat.st_mtime_ns), int(stat.st_size)
        except FileNotFoundError:
            return None
