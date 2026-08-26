# trade_record_service.py - 本地交易记录服务
"""
本地交易记录持久化存储服务

功能：
- 使用 SQLite 数据库存储所有交易记录
- 支持按日期、股票、策略等条件查询
- 提供统计分析功能（胜率、盈亏比等）
- 支持导出到 CSV/Excel
- 自动在下单成功后记录

数据表结构：
- trades: 交易记录主表
- daily_summary: 每日汇总表（自动计算）
"""

import sqlite3
import logging
import json
import re
import threading
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

# 设置日志
logger = logging.getLogger(__name__)

# 自动止损服务引用（延迟导入避免循环依赖）
_auto_stop_loss_service_getter = None

ETF_NAME_ALIASES: Dict[str, str] = {
    "510880": "红利ETF",
    "159949": "创业板50ETF",
    "513100": "纳指ETF",
    "518880": "黄金ETF",
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "512100": "中证1000ETF",
    "159901": "深证100ETF",
    "510050": "上证50ETF",
    "512010": "医药ETF",
    "512170": "医疗ETF",
    "515790": "光伏ETF",
    "516160": "新能源ETF",
    "516950": "基建50ETF",
    "516970": "基建ETF",
    "515180": "红利ETF基金",
    "159941": "纳指ETF(QDII)",
}

def set_auto_stop_loss_service_getter(getter):
    """设置自动止损服务的获取函数"""
    global _auto_stop_loss_service_getter
    _auto_stop_loss_service_getter = getter


class TradeDirection(Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


class TradeSource(Enum):
    """交易来源/策略"""
    MANUAL = "manual"              # 手动下单
    CONDITIONAL = "conditional"    # 条件单
    ETF_GRID = "etf_grid"         # ETF网格策略
    AI_AGENT = "ai_agent"         # AI智能交易
    BROKER_SYNC = "broker_sync"   # 券商成交同步
    OTHER = "other"               # 其他


@dataclass
class TradeRecord:
    """交易记录数据结构"""
    id: int = 0                        # 数据库自增ID
    trade_id: str = ""                 # 唯一交易标识
    broker_order_id: int = -1          # 券商委托单号
    stock_code: str = ""               # 股票代码
    stock_name: str = ""               # 股票名称
    direction: str = ""                # 交易方向: buy/sell
    price: float = 0.0                 # 成交价格
    volume: int = 0                    # 成交数量
    amount: float = 0.0                # 成交金额
    commission: float = 0.0            # 手续费（佣金）
    stamp_tax: float = 0.0             # 印花税
    transfer_fee: float = 0.0          # 过户费
    trade_date: str = ""               # 成交日期
    source: str = "manual"             # 来源/策略
    strategy_id: str = ""              # 策略标识
    virtual_account_id: str = ""       # 虚拟子账户标识
    intent_id: str = ""                # 下单意图标识
    remark: str = ""                   # 备注
    created_at: str = ""               # 记录创建时间
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.trade_date:
            self.trade_date = datetime.now().strftime("%Y-%m-%d")
        if not self.trade_id:
            stamp = datetime.now().strftime("%H%M%S%f")
            order_part = int(self.broker_order_id or 0)
            self.trade_id = f"{self.trade_date}_{self.stock_code}_{self.direction}_{order_part}_{stamp}"
    
    @property
    def total_fee(self) -> float:
        """总费用（佣金+印花税+过户费）"""
        return self.commission + self.stamp_tax + self.transfer_fee
    
    @property
    def direction_display(self) -> str:
        """方向显示"""
        return "买入" if self.direction == TradeDirection.BUY.value else "卖出"
    
    @property
    def source_display(self) -> str:
        """来源显示"""
        source_map = {
            TradeSource.MANUAL.value: "手动",
            TradeSource.CONDITIONAL.value: "条件单",
            TradeSource.ETF_GRID.value: "ETF网格",
            TradeSource.AI_AGENT.value: "AI智能",
            TradeSource.BROKER_SYNC.value: "成交同步",
            TradeSource.OTHER.value: "其他",
        }
        return source_map.get(self.source, self.source)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradeRecord':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_row(cls, row: tuple, columns: List[str]) -> 'TradeRecord':
        """从数据库行创建"""
        data = dict(zip(columns, row))
        return cls.from_dict(data)


@dataclass
class OrderRecord:
    """委托生命周期记录"""
    id: int = 0
    request_id: str = ""
    broker_order_id: int = -1
    fingerprint: str = ""
    stock_code: str = ""
    stock_name: str = ""
    direction: str = ""
    price: float = 0.0
    price_type: int = 0
    order_volume: int = 0
    source: str = "manual"
    trigger: str = "manual"
    strategy_name: str = ""
    strategy_id: str = ""
    virtual_account_id: str = ""
    intent_id: str = ""
    execution_mode: str = "live"
    status: str = "created"
    validation_message: str = ""
    order_status_code: int = 0
    order_status_text: str = ""
    executed_price: float = 0.0
    executed_volume: int = 0
    linked_trade_record_id: int = 0
    decision_record_id: str = ""
    remark: str = ""
    archived: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OrderRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


_BROKER_ORDER_STATUS_MAP = {
    48: "unreported",
    49: "pending",
    50: "submitted",
    51: "submitted",
    52: "partial_fill",
    53: "cancelled",
    54: "cancelled",
    55: "partial_fill",
    56: "filled",
    57: "rejected",
}


@dataclass
class DailyPnlSnapshot:
    """每日盈亏快照数据结构"""
    id: int = 0                        # 数据库自增ID
    snapshot_date: str = ""             # 快照日期 YYYY-MM-DD
    total_asset: float = 0.0           # 总资产
    cash: float = 0.0                  # 可用资金
    market_value: float = 0.0          # 持仓市值
    total_profit: float = 0.0          # 当日总盈亏（相对于前一日总资产）
    total_profit_pct: float = 0.0      # 当日总收益率 %
    cumulative_return: float = 0.0     # 累计收益率 %
    position_count: int = 0            # 持仓数量
    remark: str = ""                   # 备注
    created_at: str = ""               # 创建时间
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.snapshot_date:
            self.snapshot_date = datetime.now().strftime("%Y-%m-%d")
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'snapshot_date': self.snapshot_date,
            'total_asset': self.total_asset,
            'cash': self.cash,
            'market_value': self.market_value,
            'total_profit': self.total_profit,
            'total_profit_pct': self.total_profit_pct,
            'cumulative_return': self.cumulative_return,
            'position_count': self.position_count,
            'remark': self.remark,
            'created_at': self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DailyPnlSnapshot':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DailyPositionSnapshot:
    """日终持仓明细快照"""
    id: int = 0
    snapshot_date: str = ""
    stock_code: str = ""
    stock_name: str = ""
    volume: int = 0
    can_use_volume: int = 0
    open_price: float = 0.0
    market_value: float = 0.0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.snapshot_date:
            self.snapshot_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DailyPositionSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyDailyPnlSnapshot:
    id: int = 0
    snapshot_date: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    virtual_account_id: str = ""
    total_asset: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    position_count: int = 0
    capital_limit: float = 0.0
    invested_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    remark: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.snapshot_date:
            self.snapshot_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyDailyPnlSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyPositionSnapshot:
    id: int = 0
    snapshot_date: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    virtual_account_id: str = ""
    stock_code: str = ""
    stock_name: str = ""
    volume: int = 0
    can_use_volume: int = 0
    open_price: float = 0.0
    cost_price: float = 0.0
    cost_amount: float = 0.0
    market_value: float = 0.0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.snapshot_date:
            self.snapshot_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyPositionSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyDailyTradeSummary:
    id: int = 0
    snapshot_date: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    virtual_account_id: str = ""
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_buy_amount: float = 0.0
    total_sell_amount: float = 0.0
    total_commission: float = 0.0
    remark: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.snapshot_date:
            self.snapshot_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyDailyTradeSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TradeSummary:
    """交易统计摘要"""
    total_trades: int = 0              # 总交易次数
    buy_count: int = 0                 # 买入次数
    sell_count: int = 0                # 卖出次数
    total_buy_amount: float = 0.0      # 总买入金额
    total_sell_amount: float = 0.0     # 总卖出金额
    total_commission: float = 0.0      # 总手续费
    win_count: int = 0                 # 盈利次数（卖出价>买入价）
    loss_count: int = 0                # 亏损次数
    total_profit: float = 0.0          # 总盈亏
    win_rate: float = 0.0              # 胜率
    avg_profit: float = 0.0            # 平均盈亏
    max_profit: float = 0.0            # 最大单笔盈利
    max_loss: float = 0.0              # 最大单笔亏损


@dataclass
class TradeAuditIssue:
    category: str = ""
    severity: str = "warning"
    stock_code: str = ""
    stock_name: str = ""
    trade_date: str = ""
    direction: str = ""
    record_ids: List[int] = field(default_factory=list)
    action_record_ids: List[int] = field(default_factory=list)
    summary: str = ""
    details: str = ""
    suggested_trade_date: str = ""

    @property
    def category_display(self) -> str:
        mapping = {
            "duplicate": "重复记录",
            "invalid_date": "日期异常",
            "position_mismatch": "持仓不符",
        }
        return mapping.get(self.category, self.category or "-")

    @property
    def direction_display(self) -> str:
        if self.direction == TradeDirection.BUY.value:
            return "买入"
        if self.direction == TradeDirection.SELL.value:
            return "卖出"
        return "-"


@dataclass
class TradeAuditReport:
    scanned_records: int = 0
    duplicate_issue_count: int = 0
    date_issue_count: int = 0
    position_issue_count: int = 0
    broker_connected: bool = False
    broker_position_count: int = 0
    issues: List[TradeAuditIssue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class TradeRecordService(QObject):
    """
    交易记录服务
    
    负责管理交易记录的存储、查询和统计
    
    信号：
        record_added: 新增记录信号
        records_changed: 记录变化信号
        log_message: 日志消息信号
    """
    
    record_added = pyqtSignal(object)  # TradeRecord
    order_record_added = pyqtSignal(object)  # OrderRecord
    order_record_updated = pyqtSignal(object)  # OrderRecord
    records_changed = pyqtSignal()
    pnl_snapshot_saved = pyqtSignal(object)  # DailyPnlSnapshot
    log_message = pyqtSignal(str)
    
    DB_FILE = "trade_records.db"
    
    # 手续费率默认值（配置文件读不到时的 fallback）
    # 当前默认按 2023 年后通行规则：佣金万1、印花税万5（卖出）、过户费万0.1 双边、最低 5 元
    COMMISSION_RATE = 0.0001   # 券商佣金 0.01%
    STAMP_TAX_RATE = 0.0005    # 印花税 0.05%（仅卖出）
    TRANSFER_FEE_RATE = 0.00001  # 过户费 0.001%（沪深双边）
    MIN_COMMISSION = 5.0       # 最低佣金

    # 配置文件路径（延迟读取，支持运行期首次调用时加载）
    _FEE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "trade_fee_config.json"
    _fee_config_cache: Optional[Dict[str, object]] = None

    @classmethod
    def _load_fee_config(cls) -> Dict[str, object]:
        """读取 ``trade_fee_config.json``；读不到或字段缺失时退回类常量默认值。

        结果缓存在类属性上，整进程只读一次；手工修改配置需要重启应用生效。
        """
        if cls._fee_config_cache is not None:
            return cls._fee_config_cache
        defaults: Dict[str, object] = {
            "commission_rate": cls.COMMISSION_RATE,
            "min_commission": cls.MIN_COMMISSION,
            "stamp_tax_rate": cls.STAMP_TAX_RATE,
            "transfer_fee_rate": cls.TRANSFER_FEE_RATE,
            "etf_exempt_stamp_tax": True,
            "etf_exempt_transfer_fee": True,
            "etf_code_prefixes": ("51", "56", "58", "15", "16"),
        }
        try:
            if cls._FEE_CONFIG_PATH.exists():
                with open(cls._FEE_CONFIG_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                for key, value in raw.items():
                    if str(key).startswith("_"):
                        continue
                    if key in defaults:
                        defaults[key] = value
        except Exception as exc:
            logger.warning("读取手续费配置失败，使用默认值: %s", exc)
        defaults["etf_code_prefixes"] = tuple(
            str(p).strip() for p in (defaults.get("etf_code_prefixes") or ()) if str(p).strip()
        )
        cls._fee_config_cache = defaults
        return defaults

    @classmethod
    def reload_fee_config(cls) -> Dict[str, object]:
        """强制重新读取手续费配置（运维调参时调用）。"""
        cls._fee_config_cache = None
        return cls._load_fee_config()

    @classmethod
    def _is_etf_code(cls, stock_code: str) -> bool:
        code = (stock_code or "").strip().lower()
        for prefix in ("sh", "sz", "bj"):
            if code.startswith(prefix):
                code = code[len(prefix):]
                break
        code = code.lstrip(".")
        if not code:
            return False
        prefixes = cls._load_fee_config().get("etf_code_prefixes") or ()
        return any(code.startswith(p) for p in prefixes)

    @classmethod
    def normalize_stock_name(cls, stock_code: str, stock_name: str) -> str:
        code = str(stock_code or "").split(".")[0].strip()
        name = str(stock_name or "").strip()
        if name and name != code:
            return name
        alias = ETF_NAME_ALIASES.get(code, "").strip()
        if alias:
            return alias
        return name or code

    @classmethod
    def estimate_trade_fees(
        cls,
        *,
        direction: str,
        amount: float,
        stock_code: str = "",
    ) -> Dict[str, float]:
        """统一的手续费估算公式（所有"估算"口径的唯一出口）。

        费率来自 ``trade_fee_config.json``（缺省 fallback 到类常量）。
        ETF（沪 51/56/58、深 15/16 前缀）默认免印花税、免过户费。
        作为 AI / ETF / 影子模式 / sync_from_orders 等所有没有券商真实手续费时的兜底算法。

        Returns:
            {"commission": ..., "stamp_tax": ..., "transfer_fee": ..., "total_fee": ...}
        """
        amount = max(float(amount or 0.0), 0.0)
        direction = (direction or "").strip().lower()
        code = (stock_code or "").strip()
        cfg = cls._load_fee_config()

        commission_rate = float(cfg.get("commission_rate", cls.COMMISSION_RATE) or 0.0)
        min_commission = float(cfg.get("min_commission", cls.MIN_COMMISSION) or 0.0)
        stamp_tax_rate = float(cfg.get("stamp_tax_rate", cls.STAMP_TAX_RATE) or 0.0)
        transfer_fee_rate = float(cfg.get("transfer_fee_rate", cls.TRANSFER_FEE_RATE) or 0.0)
        is_etf = cls._is_etf_code(code)
        etf_exempt_stamp = bool(cfg.get("etf_exempt_stamp_tax", True))
        etf_exempt_transfer = bool(cfg.get("etf_exempt_transfer_fee", True))

        commission = round(max(amount * commission_rate, min_commission), 2) if amount > 0 else 0.0

        stamp_tax = 0.0
        if direction == "sell" and amount > 0 and not (is_etf and etf_exempt_stamp):
            stamp_tax = round(amount * stamp_tax_rate, 2)

        transfer_fee = 0.0
        if amount > 0 and not (is_etf and etf_exempt_transfer):
            # 当前 A 股过户费沪深双边统一收取（2022.4 起），不再只对 6 开头
            transfer_fee = round(amount * transfer_fee_rate, 2)

        total = round(commission + stamp_tax + transfer_fee, 2)
        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "total_fee": total,
        }

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据库路径
        self.data_dir = Path(__file__).resolve().parents[2] / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / self.DB_FILE
        self._write_lock = threading.RLock()
        
        # 初始化数据库
        self._init_database()
        self._init_pnl_table()
        self._init_position_snapshot_table()
        self._init_strategy_snapshot_tables()
        
        logger.info(f"交易记录服务初始化完成，数据库路径: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table_name})")
        return any(str(row[1]) == column_name for row in cursor.fetchall())

    def _ensure_column(self, cursor: sqlite3.Cursor, table_name: str, column_name: str, definition: str) -> None:
        if not self._column_exists(cursor, table_name, column_name):
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 创建交易记录表（不含具体交易时间，只保留日期）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                broker_order_id INTEGER DEFAULT -1,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                direction TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER NOT NULL,
                amount REAL NOT NULL,
                commission REAL DEFAULT 0,
                stamp_tax REAL DEFAULT 0,
                transfer_fee REAL DEFAULT 0,
                trade_date TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                strategy_id TEXT DEFAULT '',
                virtual_account_id TEXT DEFAULT '',
                intent_id TEXT DEFAULT '',
                remark TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        ''')
        self._ensure_column(cursor, 'trades', 'strategy_id', "TEXT DEFAULT ''")
        self._ensure_column(cursor, 'trades', 'virtual_account_id', "TEXT DEFAULT ''")
        self._ensure_column(cursor, 'trades', 'intent_id', "TEXT DEFAULT ''")
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_date ON trades(trade_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_code ON trades(stock_code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_direction ON trades(direction)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON trades(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades(strategy_id)')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                broker_order_id INTEGER DEFAULT -1,
                fingerprint TEXT DEFAULT '',
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                direction TEXT NOT NULL,
                price REAL DEFAULT 0,
                price_type INTEGER DEFAULT 0,
                order_volume INTEGER DEFAULT 0,
                source TEXT DEFAULT 'manual',
                trigger TEXT DEFAULT 'manual',
                strategy_name TEXT DEFAULT '',
                strategy_id TEXT DEFAULT '',
                virtual_account_id TEXT DEFAULT '',
                intent_id TEXT DEFAULT '',
                execution_mode TEXT DEFAULT 'live',
                status TEXT DEFAULT 'created',
                validation_message TEXT DEFAULT '',
                order_status_code INTEGER DEFAULT 0,
                order_status_text TEXT DEFAULT '',
                executed_price REAL DEFAULT 0,
                executed_volume INTEGER DEFAULT 0,
                linked_trade_record_id INTEGER DEFAULT 0,
                decision_record_id TEXT DEFAULT '',
                remark TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        self._ensure_column(cursor, 'order_records', 'strategy_id', "TEXT DEFAULT ''")
        self._ensure_column(cursor, 'order_records', 'virtual_account_id', "TEXT DEFAULT ''")
        self._ensure_column(cursor, 'order_records', 'intent_id', "TEXT DEFAULT ''")
        self._ensure_column(cursor, 'order_records', 'archived', "INTEGER DEFAULT 0")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_request_id ON order_records(request_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_broker_order_id ON order_records(broker_order_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_fingerprint ON order_records(fingerprint)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_created_at ON order_records(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_strategy_id ON order_records(strategy_id)')
        
        # 创建每日汇总视图（先删除再创建，以便更新结构）
        cursor.execute('DROP VIEW IF EXISTS daily_summary')
        cursor.execute('''
            CREATE VIEW daily_summary AS
            SELECT 
                trade_date,
                COUNT(*) as trade_count,
                SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) as sell_count,
                SUM(CASE WHEN direction = 'buy' THEN amount ELSE 0 END) as total_buy,
                SUM(CASE WHEN direction = 'sell' THEN amount ELSE 0 END) as total_sell,
                SUM(commission + stamp_tax + transfer_fee) as total_fee
            FROM trades
            GROUP BY trade_date
            ORDER BY trade_date DESC
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("数据库表初始化完成")
    
    def _log(self, message: str):
        """发送日志"""
        logger.info(message)
        self.log_message.emit(f"[交易记录] {message}")
    
    def _trigger_auto_stop_loss(self, stock_code: str, stock_name: str, 
                                price: float, volume: int, source: str):
        """
        触发自动止损（买入成交后）
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            price: 买入价格（成本价）
            volume: 买入数量
            source: 交易来源
        """
        global _auto_stop_loss_service_getter
        if _auto_stop_loss_service_getter is None:
            return
        
        try:
            auto_stop_loss_service = _auto_stop_loss_service_getter()
            if auto_stop_loss_service and auto_stop_loss_service.is_enabled:
                auto_stop_loss_service.on_buy_trade_added(
                    stock_code, stock_name, price, volume, source
                )
        except Exception as e:
            logger.error(f"触发自动止损失败: {e}")
    
    def calculate_commission(self, direction: str, price: float, volume: int,
                            stock_code: str = "") -> float:
        """计算综合交易费用（委托给 ``estimate_trade_fees``，保持配置一致）。"""
        amount = max(float(price or 0.0) * int(volume or 0), 0.0)
        fees = self.estimate_trade_fees(
            direction=direction,
            amount=amount,
            stock_code=stock_code,
        )
        return float(fees.get("total_fee", 0.0) or 0.0)
    
    def add_record(self,
                   stock_code: str,
                   stock_name: str,
                   direction: str,
                   price: float,
                   volume: int,
                   broker_order_id: int = -1,
                   trade_date: str = None,
                   source: str = "manual",
                   strategy_id: str = "",
                   virtual_account_id: str = "",
                   intent_id: str = "",
                   remark: str = "",
                   commission: float = None,
                   stamp_tax: float = None,
                   transfer_fee: float = None,
                   emit_signals: bool = True) -> Optional[TradeRecord]:
        """
        添加交易记录
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            direction: 交易方向 (buy/sell)
            price: 成交价格
            volume: 成交数量
            broker_order_id: 券商委托单号
            trade_date: 成交日期 (YYYY-MM-DD)
            source: 来源/策略
            remark: 备注
            commission: 手续费/佣金（可选，不传则自动计算）
            stamp_tax: 印花税（可选）
            transfer_fee: 过户费（可选）
            
        Returns:
            创建的交易记录对象
        """
        # 处理股票代码（去掉后缀）
        code = stock_code.split('.')[0] if '.' in stock_code else stock_code
        normalized_stock_name = self.normalize_stock_name(code, stock_name)
        
        # 处理日期
        if not trade_date:
            trade_date = datetime.now().strftime("%Y-%m-%d")
        
        # 计算金额
        amount = round(price * volume, 2)
        
        # 计算费用（如果未提供）——统一走 estimate_trade_fees（含 ETF 免印花/免过户费规则）
        if commission is None or stamp_tax is None or transfer_fee is None:
            fees = self.estimate_trade_fees(
                direction=direction,
                amount=amount,
                stock_code=code,
            )
            if commission is None:
                commission = fees["commission"]
            if stamp_tax is None:
                stamp_tax = fees["stamp_tax"]
            if transfer_fee is None:
                transfer_fee = fees["transfer_fee"]
        
        # 创建记录
        record = TradeRecord(
            broker_order_id=broker_order_id,
            stock_code=code,
            stock_name=normalized_stock_name,
            direction=direction,
            price=price,
            volume=volume,
            amount=amount,
            commission=round(commission, 2),
            stamp_tax=round(stamp_tax, 2),
            transfer_fee=round(transfer_fee, 2),
            trade_date=trade_date,
            source=source,
            strategy_id=strategy_id,
            virtual_account_id=virtual_account_id,
            intent_id=intent_id,
            remark=remark
        )

        duplicate = self._find_exact_trade_duplicate(record)
        if duplicate is not None:
            return duplicate
        
        # 保存到数据库
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    trade_id, broker_order_id, stock_code, stock_name, direction,
                    price, volume, amount, commission, stamp_tax, transfer_fee,
                    trade_date, source, strategy_id, virtual_account_id, intent_id, remark, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.trade_id, record.broker_order_id, record.stock_code,
                record.stock_name, record.direction, record.price, record.volume,
                record.amount, record.commission, record.stamp_tax, record.transfer_fee,
                record.trade_date, record.source, record.strategy_id, record.virtual_account_id,
                record.intent_id, record.remark, record.created_at
            ))
            
            record.id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self._log(f"新增交易记录: {normalized_stock_name}({code}) {record.direction_display} "
                     f"{volume}股 @ {price:.3f}")
            
            if emit_signals:
                self.record_added.emit(record)
                self.records_changed.emit()
                if direction == TradeDirection.BUY.value:
                    self._trigger_auto_stop_loss(code, normalized_stock_name, price, volume, source)
            
            return record
            
        except sqlite3.IntegrityError as e:
            logger.warning(f"交易记录已存在: {record.trade_id}")
            return None
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
            return None

    def _find_exact_trade_duplicate(self, record: TradeRecord) -> Optional[TradeRecord]:
        broker_order_id = int(getattr(record, "broker_order_id", 0) or 0)
        if broker_order_id <= 0:
            return None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT * FROM trades
                WHERE broker_order_id = ?
                  AND stock_code = ?
                  AND direction = ?
                  AND trade_date = ?
                  AND ABS(price - ?) < 0.0001
                  AND volume = ?
                  AND COALESCE(strategy_id, '') = ?
                  AND COALESCE(virtual_account_id, '') = ?
                ORDER BY id DESC
                LIMIT 1
                ''',
                (
                    broker_order_id,
                    str(record.stock_code or ""),
                    str(record.direction or ""),
                    str(record.trade_date or ""),
                    float(record.price or 0.0),
                    int(record.volume or 0),
                    str(record.strategy_id or ""),
                    str(record.virtual_account_id or ""),
                ),
            )
            row = cursor.fetchone()
            conn.close()
            return TradeRecord.from_dict(dict(row)) if row else None
        except Exception:
            logger.debug("检查重复成交记录失败", exc_info=True)
            return None

    def add_order_record(
        self,
        *,
        request_id: str,
        stock_code: str,
        stock_name: str,
        direction: str,
        order_volume: int,
        price: float,
        price_type: int,
        source: str,
        trigger: str,
        strategy_name: str = "",
        strategy_id: str = "",
        virtual_account_id: str = "",
        intent_id: str = "",
        execution_mode: str = "live",
        status: str = "created",
        broker_order_id: int = -1,
        fingerprint: str = "",
        decision_record_id: str = "",
        remark: str = "",
        validation_message: str = "",
    ) -> Optional[OrderRecord]:
        code = stock_code.split('.')[0] if '.' in stock_code else stock_code
        normalized_stock_name = self.normalize_stock_name(code, stock_name)
        record = OrderRecord(
            request_id=request_id,
            broker_order_id=broker_order_id,
            fingerprint=fingerprint,
            stock_code=code,
            stock_name=normalized_stock_name,
            direction=direction,
            price=price,
            price_type=price_type,
            order_volume=order_volume,
            source=source,
            trigger=trigger,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            virtual_account_id=virtual_account_id,
            intent_id=intent_id,
            execution_mode=execution_mode,
            status=status,
            validation_message=validation_message,
            decision_record_id=decision_record_id,
            remark=remark,
        )
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO order_records (
                    request_id, broker_order_id, fingerprint, stock_code, stock_name,
                    direction, price, price_type, order_volume, source, trigger,
                    strategy_name, strategy_id, virtual_account_id, intent_id,
                    execution_mode, status, validation_message,
                    order_status_code, order_status_text, executed_price, executed_volume,
                    linked_trade_record_id, decision_record_id, remark, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    record.request_id, record.broker_order_id, record.fingerprint, record.stock_code,
                    record.stock_name, record.direction, record.price, record.price_type,
                    record.order_volume, record.source, record.trigger, record.strategy_name,
                    record.strategy_id, record.virtual_account_id, record.intent_id,
                    record.execution_mode, record.status, record.validation_message,
                    record.order_status_code, record.order_status_text, record.executed_price,
                    record.executed_volume, record.linked_trade_record_id, record.decision_record_id,
                    record.remark, record.created_at, record.updated_at,
                ),
            )
            record.id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.order_record_added.emit(record)
            self.records_changed.emit()
            return record
        except sqlite3.IntegrityError:
            logger.warning("委托记录 request_id 已存在: %s", request_id)
            return self.get_order_record_by_request_id(request_id)
        except Exception as e:
            logger.error("保存委托记录失败: %s", e)
            return None

    def update_order_record(self, request_id: str, **fields) -> bool:
        if not request_id:
            return False
        if not fields:
            return False
        allowed = {
            "broker_order_id", "fingerprint", "stock_code", "stock_name", "direction",
            "price", "price_type", "order_volume", "source", "trigger", "strategy_name",
            "strategy_id", "virtual_account_id", "intent_id",
            "execution_mode", "status", "validation_message", "order_status_code",
            "order_status_text", "executed_price", "executed_volume", "linked_trade_record_id",
            "decision_record_id", "remark",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = ", ".join(f"{key} = ?" for key in updates.keys())
        params = list(updates.values()) + [request_id]
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"UPDATE order_records SET {sql} WHERE request_id = ?", params)
            changed = cursor.rowcount > 0
            conn.commit()
            conn.close()
            if changed:
                record = self.get_order_record_by_request_id(request_id)
                if record is not None:
                    self.order_record_updated.emit(record)
                self.records_changed.emit()
            return changed
        except Exception as e:
            logger.error("更新委托记录失败: %s", e)
            return False

    def get_order_record_by_request_id(self, request_id: str) -> Optional[OrderRecord]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM order_records WHERE request_id = ? LIMIT 1", (request_id,))
        row = cursor.fetchone()
        conn.close()
        return OrderRecord.from_dict(dict(row)) if row else None

    def find_recent_order_record(self, fingerprint: str, within_seconds: int = 30) -> Optional[OrderRecord]:
        if not fingerprint:
            return None
        cutoff = (datetime.now() - timedelta(seconds=max(int(within_seconds), 1))).strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM order_records
            WHERE fingerprint = ? AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (fingerprint, cutoff),
        )
        row = cursor.fetchone()
        conn.close()
        return OrderRecord.from_dict(dict(row)) if row else None

    def get_order_records(
        self,
        *,
        start_time: str = "",
        end_time: str = "",
        status: str = "",
        source: str = "",
        strategy_id: str = "",
        virtual_account_id: str = "",
        include_archived: bool = False,
        limit: int = 1000,
    ) -> List[OrderRecord]:
        conn = self._get_connection()
        cursor = conn.cursor()
        conditions = []
        params = []
        if start_time:
            conditions.append("created_at >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("created_at <= ?")
            params.append(end_time)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if virtual_account_id:
            conditions.append("virtual_account_id = ?")
            params.append(virtual_account_id)
        if not include_archived:
            conditions.append("COALESCE(archived, 0) = 0")
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        cursor.execute(
            f"SELECT * FROM order_records WHERE {where_clause} ORDER BY created_at DESC, id DESC LIMIT ?",
            [*params, limit],
        )
        rows = cursor.fetchall()
        conn.close()
        return [OrderRecord.from_dict(dict(row)) for row in rows]

    def archive_order_records(self, request_ids: list[str]) -> int:
        """批量归档委托记录（标记 archived=1，不从表中物理删除）。

        返回成功归档的行数。异常订单 widget 的 \"忽略\" 操作会调用它。
        """
        ids = [str(rid or "").strip() for rid in request_ids if str(rid or "").strip()]
        if not ids:
            return 0
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholders = ",".join(["?"] * len(ids))
        cursor.execute(
            f"UPDATE order_records SET archived = 1, updated_at = ? WHERE request_id IN ({placeholders})",
            [now, *ids],
        )
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        if affected > 0:
            for rid in ids:
                record = self.get_order_record_by_request_id(rid)
                if record is not None:
                    try:
                        self.order_record_updated.emit(record)
                    except Exception:
                        pass
        return int(affected or 0)

    def _infer_strategy_identity(
        self,
        stock_code: str,
        *,
        strategy_id: str = "",
        virtual_account_id: str = "",
        intent_id: str = "",
        broker_order_id: int = 0,
        direction: str = "",
    ) -> tuple[str, str, str]:
        if strategy_id:
            return strategy_id, virtual_account_id, intent_id
        broker_order_id = int(broker_order_id or 0)
        if broker_order_id > 0:
            try:
                code = str(stock_code or "").split(".")[0]
                conditions = [
                    "broker_order_id = ?",
                    "COALESCE(strategy_id, '') != ''",
                ]
                params: list[Any] = [broker_order_id]
                if code:
                    conditions.append("stock_code = ?")
                    params.append(code)
                normalized_direction = str(direction or "").strip().lower()
                if normalized_direction:
                    conditions.append("direction = ?")
                    params.append(normalized_direction)
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT strategy_id, virtual_account_id, intent_id
                    FROM order_records
                    WHERE {" AND ".join(conditions)}
                    ORDER BY updated_at DESC, created_at DESC, id DESC
                    LIMIT 1
                    """,
                    params,
                )
                row = cursor.fetchone()
                conn.close()
                if row is not None:
                    return (
                        str(row["strategy_id"] or ""),
                        str(row["virtual_account_id"] or ""),
                        intent_id or str(row["intent_id"] or ""),
                    )
            except Exception as exc:
                logger.debug("按委托号推断策略归属失败: order_id=%s error=%s", broker_order_id, exc)
        try:
            from trading_app.services.strategy.strategy_registry_service import get_strategy_registry_service

            owner = get_strategy_registry_service().get_owner(stock_code)
        except Exception:
            owner = None
        if owner is None:
            return "", "", intent_id
        return owner.strategy_id, owner.virtual_account_id, intent_id

    def sync_order_records_from_orders(self, orders: list) -> int:
        updated = 0
        if not orders:
            return updated
        for order in orders:
            try:
                order_id = int(getattr(order, "order_id", 0) or 0)
                if order_id <= 0:
                    continue
                status_code = int(getattr(order, "order_status", 0) or 0)
                traded_volume = int(getattr(order, "traded_volume", 0) or 0)
                traded_price = float(getattr(order, "traded_price", 0) or 0)
                status_text = {
                    48: "未报",
                    49: "待报",
                    50: "已报",
                    51: "已报待撤",
                    52: "部成待撤",
                    53: "部撤",
                    54: "已撤",
                    55: "部成",
                    56: "已成",
                    57: "废单",
                }.get(status_code, str(status_code))
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT request_id FROM order_records WHERE broker_order_id = ? ORDER BY id DESC LIMIT 1",
                    (order_id,),
                )
                row = cursor.fetchone()
                conn.close()
                if not row:
                    continue
                request_id = str(row["request_id"])
                ok = self.update_order_record(
                    request_id,
                    status=_BROKER_ORDER_STATUS_MAP.get(status_code, "submitted"),
                    order_status_code=status_code,
                    order_status_text=status_text,
                    executed_price=traded_price,
                    executed_volume=traded_volume,
                    validation_message=getattr(order, "status_msg", "") or status_text,
                )
                if ok:
                    updated += 1
            except Exception as exc:
                logger.debug("同步委托记录状态失败: %s", exc)
        return updated
    
    def is_trade_exists(self, traded_id, trade_date: str = None) -> bool:
        """
        检查成交记录是否已存在（基于券商成交ID）
        
        Args:
            traded_id: 券商成交ID（int或str）
            trade_date: 成交日期（可选，用于更精确匹配）
            
        Returns:
            是否已存在
        """
        # 转换为字符串进行匹配
        traded_id_str = str(traded_id) if traded_id else ""
        if not traded_id_str or traded_id_str == "0":
            return False
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 使用 remark 字段存储的 traded_id 来检查
        # remark 格式: "成交号:123456"
        if trade_date:
            cursor.execute(
                "SELECT 1 FROM trades WHERE remark LIKE ? AND trade_date = ? LIMIT 1",
                (f"成交号:{traded_id_str}%", trade_date)
            )
        else:
            cursor.execute(
                "SELECT 1 FROM trades WHERE remark LIKE ? LIMIT 1",
                (f"成交号:{traded_id_str}%",)
            )
        
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    @staticmethod
    def _broker_field(item: Any, field: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(field, default)
        return getattr(item, field, default)

    def sync_broker_trades(
        self,
        broker_trades: list,
        source: str = "broker_sync",
        *,
        strategy_id: str = "",
        virtual_account_id: str = "",
        intent_id: str = "",
    ) -> int:
        """
        同步券商成交回报到本地数据库
        
        基于券商的 traded_id 进行去重，只新增不存在的记录。
        
        Args:
            broker_trades: 券商成交回报列表，每个元素可以是对象或 dict：
                - traded_id: 成交编号
                - stock_code: 股票代码
                - stock_name: 股票名称（可选）
                - order_type: 委托类型（23=买入, 24=卖出）
                - traded_price: 成交价格
                - traded_volume: 成交数量
                - traded_amount: 成交金额
                - traded_time: 成交时间（可选）
            source: 来源标记
            
        Returns:
            新增的记录数量
        """
        added_count = 0
        today = datetime.now().strftime("%Y-%m-%d")
        
        for trade in broker_trades:
            try:
                # 获取成交ID（可能是字符串或整数）
                traded_id_raw = self._broker_field(trade, 'traded_id', 0)
                try:
                    traded_id = int(traded_id_raw) if traded_id_raw else 0
                except (ValueError, TypeError):
                    traded_id = 0
                    
                stock_code_raw = self._broker_field(trade, 'stock_code', '')
                
                if traded_id <= 0:
                    continue
                
                # 检查是否已存在
                trade_date = self._normalize_broker_time_to_date(
                    self._broker_field(trade, 'traded_time', None),
                    today,
                )
                if self.is_trade_exists(traded_id, trade_date):
                    continue
                
                # 解析交易数据
                stock_code = str(stock_code_raw).split('.')[0]
                stock_name = self._broker_field(trade, 'stock_name', '') or stock_code
                try:
                    order_type = int(self._broker_field(trade, 'order_type', 0) or 0)
                except (TypeError, ValueError):
                    order_type = 0
                direction = TradeDirection.BUY.value if order_type == 23 else TradeDirection.SELL.value
                broker_order_id = int(self._broker_field(trade, 'order_id', 0) or 0)
                inferred_strategy_id, inferred_virtual_account_id, inferred_intent_id = self._infer_strategy_identity(
                    stock_code,
                    strategy_id=strategy_id,
                    virtual_account_id=virtual_account_id,
                    intent_id=intent_id,
                    broker_order_id=broker_order_id,
                    direction=direction,
                )
                price = float(self._broker_field(trade, 'traded_price', 0))
                volume = int(self._broker_field(trade, 'traded_volume', 0))
                amount = float(self._broker_field(trade, 'traded_amount', 0)) or round(price * volume, 2)
                
                if price <= 0 or volume <= 0:
                    continue

                fees = self.estimate_trade_fees(
                    direction=direction, amount=amount, stock_code=stock_code,
                )

                # 添加记录
                record = self.add_record(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    direction=direction,
                    price=price,
                    volume=volume,
                    broker_order_id=broker_order_id,
                    trade_date=trade_date,
                    source=source,
                    strategy_id=inferred_strategy_id,
                    virtual_account_id=inferred_virtual_account_id,
                    intent_id=inferred_intent_id,
                    remark=f"成交号:{traded_id}",
                    commission=fees["commission"],
                    stamp_tax=fees["stamp_tax"],
                    transfer_fee=fees["transfer_fee"],
                )
                
                if record:
                    added_count += 1
                    
            except Exception as e:
                logger.error(f"同步成交记录失败: {e}")
                continue
        
        if added_count > 0:
            self._log(f"同步成交记录完成，新增 {added_count} 条")
        
        return added_count
    
    @staticmethod
    def _order_snapshot_has_fill(status_code: int, traded_volume: int) -> bool:
        # 次日补拉时，MiniQMT 可能拿不到成交列表，但委托里仍会带回成交数量。
        return int(status_code or 0) in (52, 55, 56) or int(traded_volume or 0) > 0

    def sync_from_orders(
        self,
        orders: list,
        source: str = "broker_sync",
        name_map: dict = None,
        *,
        strategy_id: str = "",
        virtual_account_id: str = "",
        intent_id: str = "",
    ) -> int:
        """
        从委托数据中同步已成交的记录
        
        筛选状态为"已成交(56)"或"部成(55)"的委托，同步到交易记录。
        使用 order_id 作为去重标识。
        
        Args:
            orders: 委托列表
            source: 来源标记
            name_map: 股票代码到名称的映射表
            
        Returns:
            新增的记录数量
        """
        added_count = 0
        today = datetime.now().strftime("%Y-%m-%d")
        name_map = name_map or {}
        
        for order in orders:
            try:
                order_status = int(getattr(order, 'order_status', 0) or getattr(order, 'status_code', 0) or 0)
                traded_volume = int(getattr(order, 'traded_volume', 0) or getattr(order, 'traded', 0) or 0)
                if not self._order_snapshot_has_fill(order_status, traded_volume):
                    continue

                # 获取委托ID
                order_id = getattr(order, 'order_id', 0) or getattr(order, 'orderid', 0)
                if not order_id:
                    continue

                # 检查是否已存在（使用委托号去重）
                trade_date = self._normalize_broker_time_to_date(
                    getattr(order, 'traded_time', None) or getattr(order, 'order_time', None),
                    today,
                )
                # 解析交易数据
                stock_code = str(getattr(order, 'stock_code', '') or getattr(order, 'symbol', '')).split('.')[0]
                if self._is_order_synced(order_id, trade_date, stock_code):
                    continue

                try:
                    order_type = int(getattr(order, 'order_type', 0) or 0)
                except (TypeError, ValueError):
                    order_type = 0
                direction_value = str(getattr(getattr(order, "direction", ""), "value", getattr(order, "direction", "")) or "").lower()
                direction = direction_value if direction_value in (TradeDirection.BUY.value, TradeDirection.SELL.value) else (TradeDirection.BUY.value if order_type == 23 else TradeDirection.SELL.value)
                inferred_strategy_id, inferred_virtual_account_id, inferred_intent_id = self._infer_strategy_identity(
                    stock_code,
                    strategy_id=strategy_id,
                    virtual_account_id=virtual_account_id,
                    intent_id=intent_id,
                    broker_order_id=int(order_id or 0),
                    direction=direction,
                )
                
                # 获取股票名称：优先从name_map获取，其次从委托数据，最后用xtdata
                stock_name = name_map.get(stock_code, '')
                if not stock_name:
                    stock_name = getattr(order, 'stock_name', '') or ''
                if not stock_name:
                    # 尝试使用 xtdata.get_instrument_detail 获取
                    try:
                        from xtquant import xtdata
                        xt_code = f"{stock_code}.SH" if stock_code.startswith(('5', '6', '9')) else f"{stock_code}.SZ"
                        detail = xtdata.get_instrument_detail(xt_code)
                        stock_name = detail.get('InstrumentName', stock_code) if detail else stock_code
                    except:
                        stock_name = stock_code
                
                # 成交价格和数量
                price = float(getattr(order, 'traded_price', 0) or getattr(order, 'price', 0) or 0)
                volume = traded_volume
                
                if price <= 0 or volume <= 0:
                    continue
                
                amount = round(price * volume, 2)

                fees = self.estimate_trade_fees(
                    direction=direction, amount=amount, stock_code=stock_code,
                )

                # 添加记录
                record = self.add_record(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    direction=direction,
                    price=price,
                    volume=volume,
                    broker_order_id=int(order_id or 0),
                    trade_date=trade_date,
                    source=source,
                    strategy_id=inferred_strategy_id,
                    virtual_account_id=inferred_virtual_account_id,
                    intent_id=inferred_intent_id,
                    remark=f"委托号:{order_id}",
                    commission=fees["commission"],
                    stamp_tax=fees["stamp_tax"],
                    transfer_fee=fees["transfer_fee"],
                )
                
                if record:
                    added_count += 1
                    
            except Exception as e:
                logger.error(f"从委托同步记录失败: {e}")
                continue
        
        if added_count > 0:
            self._log(f"同步成交记录完成，新增 {added_count} 条")
        
        return added_count

    def sync_from_order_records(
        self,
        order_records: list,
        source: str = "broker_sync",
    ) -> int:
        """
        从本地委托生命周期记录中回填缺失的成交记录。

        适用于次日补跑时券商成交明细为空，但本地 order_records 已保留成交量/成交价的场景。
        """
        with self._write_lock:
            added_count = self._sync_from_order_records_unlocked(order_records, source=source)
        if added_count > 0:
            self._log(f"本地委托回填成交完成，新增 {added_count} 条")
            self.records_changed.emit()
        return added_count

    def _sync_from_order_records_unlocked(
        self,
        order_records: list,
        source: str = "broker_sync",
    ) -> int:
        added_count = 0

        for order in order_records or []:
            try:
                order_id = int(getattr(order, "broker_order_id", 0) or 0)
                if order_id <= 0:
                    continue

                status_code = int(getattr(order, "order_status_code", 0) or 0)
                executed_volume = int(getattr(order, "executed_volume", 0) or 0)
                if not self._order_snapshot_has_fill(status_code, executed_volume):
                    continue

                trade_date = self._normalize_broker_time_to_date(
                    getattr(order, "created_at", None) or getattr(order, "updated_at", None),
                    datetime.now().strftime("%Y-%m-%d"),
                )
                stock_code = str(getattr(order, "stock_code", "") or "").split(".")[0]
                if not stock_code:
                    continue
                # miniQMT 委托号会跨日、甚至跨品种复用，必须带日期+代码判断。
                if self._is_order_synced(order_id, trade_date, stock_code):
                    continue

                price = float(getattr(order, "executed_price", 0) or 0)
                volume = executed_volume
                if price <= 0 or price > 1e6 or volume <= 0:
                    continue

                inferred_strategy_id, inferred_virtual_account_id, inferred_intent_id = self._infer_strategy_identity(
                    stock_code,
                    strategy_id=str(getattr(order, "strategy_id", "") or ""),
                    virtual_account_id=str(getattr(order, "virtual_account_id", "") or ""),
                    intent_id=str(getattr(order, "intent_id", "") or ""),
                )
                stock_name = str(getattr(order, "stock_name", "") or stock_code)
                direction = str(getattr(order, "direction", "") or "").lower()
                if direction not in (TradeDirection.BUY.value, TradeDirection.SELL.value):
                    continue

                amount = round(price * volume, 2)
                fees = self.estimate_trade_fees(
                    direction=direction, amount=amount, stock_code=stock_code,
                )

                record = self.add_record(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    direction=direction,
                    price=price,
                    volume=volume,
                    broker_order_id=order_id,
                    trade_date=trade_date,
                    source=source,
                    strategy_id=inferred_strategy_id,
                    virtual_account_id=inferred_virtual_account_id,
                    intent_id=inferred_intent_id,
                    remark=f"委托号:{order_id} 本地委托回填",
                    commission=fees["commission"],
                    stamp_tax=fees["stamp_tax"],
                    transfer_fee=fees["transfer_fee"],
                    emit_signals=False,
                )
                if record is not None and str(getattr(record, "trade_date", "") or "") == trade_date:
                    added_count += 1
            except Exception as exc:
                logger.error("从本地委托回填成交失败: %s", exc)
                continue

        return added_count

    def save_daily_position_snapshots(self, snapshot_date: str, positions: list) -> int:
        saved_count = 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM daily_position_snapshots WHERE snapshot_date = ?', (snapshot_date,))
            for pos in positions or []:
                volume = int(getattr(pos, "volume", 0) or 0)
                if volume <= 0:
                    continue
                stock_code = str(getattr(pos, "stock_code", "") or "").split(".")[0]
                if not stock_code:
                    continue
                cursor.execute(
                    '''
                    INSERT INTO daily_position_snapshots (
                        snapshot_date, stock_code, stock_name, volume, can_use_volume,
                        open_price, market_value, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        snapshot_date,
                        stock_code,
                        getattr(pos, "stock_name", "") or stock_code,
                        volume,
                        int(getattr(pos, "can_use_volume", 0) or 0),
                        round(float(getattr(pos, "open_price", 0) or 0), 4),
                        round(float(getattr(pos, "market_value", 0) or 0), 2),
                        now,
                    ),
                )
                saved_count += 1
            conn.commit()
        finally:
            conn.close()
        if saved_count > 0:
            self._log(f"保存日终持仓快照完成，共 {saved_count} 条")
        return saved_count

    def get_daily_position_snapshots(self, snapshot_date: str) -> List[DailyPositionSnapshot]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM daily_position_snapshots WHERE snapshot_date = ? ORDER BY stock_code ASC',
            (snapshot_date,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [DailyPositionSnapshot.from_dict(dict(row)) for row in rows]

    def get_daily_position_snapshot_count(self, snapshot_date: str) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) FROM daily_position_snapshots WHERE snapshot_date = ?',
            (snapshot_date,),
        )
        count = int(cursor.fetchone()[0] or 0)
        conn.close()
        return count

    def save_strategy_position_snapshots(
        self,
        snapshot_date: str,
        positions_by_strategy: Dict[str, Dict[str, Any]],
    ) -> int:
        saved_count = 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            strategy_ids = [
                str(strategy_id or "").strip()
                for strategy_id in (positions_by_strategy or {}).keys()
                if str(strategy_id or "").strip()
            ]
            if not strategy_ids:
                return 0
            placeholders = ",".join("?" for _ in strategy_ids)
            cursor.execute(
                f"DELETE FROM strategy_position_snapshots WHERE snapshot_date = ? AND strategy_id IN ({placeholders})",
                [snapshot_date, *strategy_ids],
            )
            for strategy_id, payload in (positions_by_strategy or {}).items():
                strategy_name = str(payload.get("strategy_name", "") or "")
                virtual_account_id = str(payload.get("virtual_account_id", "") or "")
                for pos in payload.get("positions", []) or []:
                    volume = int(pos.get("volume", 0) or 0)
                    if volume <= 0:
                        continue
                    stock_code = str(pos.get("stock_code", "") or "").split(".")[0]
                    if not stock_code:
                        continue
                    cost_price = round(
                        float(pos.get("cost_price", pos.get("avg_cost", 0)) or 0),
                        4,
                    )
                    cost_amount = round(
                        float(pos.get("cost_amount", 0) or (cost_price * volume)),
                        2,
                    )
                    cursor.execute(
                        '''
                        INSERT INTO strategy_position_snapshots (
                            snapshot_date, strategy_id, strategy_name, virtual_account_id,
                            stock_code, stock_name, volume, can_use_volume, open_price,
                            cost_price, cost_amount, market_value, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            snapshot_date,
                            strategy_id,
                            strategy_name,
                            virtual_account_id,
                            stock_code,
                            pos.get("stock_name", "") or stock_code,
                            volume,
                            int(pos.get("can_use_volume", 0) or 0),
                            cost_price,
                            cost_price,
                            cost_amount,
                            round(float(pos.get("market_value", 0) or 0), 2),
                            now,
                        ),
                    )
                    saved_count += 1
            conn.commit()
        finally:
            conn.close()
        return saved_count

    def save_strategy_daily_pnl_snapshot(
        self,
        *,
        snapshot_date: str,
        strategy_id: str,
        strategy_name: str,
        virtual_account_id: str,
        total_asset: float,
        cash: float,
        market_value: float,
        position_count: int = 0,
        capital_limit: float = 0.0,
        invested_cost: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        total_pnl: float = 0.0,
        remark: str = "",
    ) -> Optional[StrategyDailyPnlSnapshot]:
        snapshot = StrategyDailyPnlSnapshot(
            snapshot_date=snapshot_date,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            total_asset=round(total_asset, 2),
            cash=round(cash, 2),
            market_value=round(market_value, 2),
            position_count=int(position_count or 0),
            capital_limit=round(capital_limit, 2),
            invested_cost=round(invested_cost, 2),
            realized_pnl=round(realized_pnl, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            total_pnl=round(total_pnl, 2),
            remark=remark,
        )
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO strategy_daily_pnl (
                    snapshot_date, strategy_id, strategy_name, virtual_account_id,
                    total_asset, cash, market_value, position_count,
                    capital_limit, invested_cost, realized_pnl, unrealized_pnl, total_pnl,
                    remark, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_date, strategy_id) DO UPDATE SET
                    strategy_name = excluded.strategy_name,
                    virtual_account_id = excluded.virtual_account_id,
                    total_asset = excluded.total_asset,
                    cash = excluded.cash,
                    market_value = excluded.market_value,
                    position_count = excluded.position_count,
                    capital_limit = excluded.capital_limit,
                    invested_cost = excluded.invested_cost,
                    realized_pnl = excluded.realized_pnl,
                    unrealized_pnl = excluded.unrealized_pnl,
                    total_pnl = excluded.total_pnl,
                    remark = excluded.remark,
                    created_at = excluded.created_at
                ''',
                (
                    snapshot.snapshot_date,
                    snapshot.strategy_id,
                    snapshot.strategy_name,
                    snapshot.virtual_account_id,
                    snapshot.total_asset,
                    snapshot.cash,
                    snapshot.market_value,
                    snapshot.position_count,
                    snapshot.capital_limit,
                    snapshot.invested_cost,
                    snapshot.realized_pnl,
                    snapshot.unrealized_pnl,
                    snapshot.total_pnl,
                    snapshot.remark,
                    snapshot.created_at,
                ),
            )
            conn.commit()
            conn.close()
            return snapshot
        except Exception as exc:
            logger.error("保存策略日终权益快照失败: %s", exc)
            return None

    def save_strategy_daily_trade_summary(
        self,
        *,
        snapshot_date: str,
        strategy_id: str,
        strategy_name: str,
        virtual_account_id: str,
        trade_count: int,
        buy_count: int,
        sell_count: int,
        total_buy_amount: float,
        total_sell_amount: float,
        total_commission: float,
        remark: str = "",
    ) -> Optional[StrategyDailyTradeSummary]:
        summary = StrategyDailyTradeSummary(
            snapshot_date=snapshot_date,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
            trade_count=int(trade_count or 0),
            buy_count=int(buy_count or 0),
            sell_count=int(sell_count or 0),
            total_buy_amount=round(total_buy_amount, 2),
            total_sell_amount=round(total_sell_amount, 2),
            total_commission=round(total_commission, 2),
            remark=remark,
        )
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO strategy_daily_trade_summary (
                    snapshot_date, strategy_id, strategy_name, virtual_account_id,
                    trade_count, buy_count, sell_count, total_buy_amount,
                    total_sell_amount, total_commission, remark, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_date, strategy_id) DO UPDATE SET
                    strategy_name = excluded.strategy_name,
                    virtual_account_id = excluded.virtual_account_id,
                    trade_count = excluded.trade_count,
                    buy_count = excluded.buy_count,
                    sell_count = excluded.sell_count,
                    total_buy_amount = excluded.total_buy_amount,
                    total_sell_amount = excluded.total_sell_amount,
                    total_commission = excluded.total_commission,
                    remark = excluded.remark,
                    created_at = excluded.created_at
                ''',
                (
                    summary.snapshot_date,
                    summary.strategy_id,
                    summary.strategy_name,
                    summary.virtual_account_id,
                    summary.trade_count,
                    summary.buy_count,
                    summary.sell_count,
                    summary.total_buy_amount,
                    summary.total_sell_amount,
                    summary.total_commission,
                    summary.remark,
                    summary.created_at,
                ),
            )
            conn.commit()
            conn.close()
            return summary
        except Exception as exc:
            logger.error("保存策略日成交汇总失败: %s", exc)
            return None

    def summarize_trades_by_strategy(self, snapshot_date: str) -> List[dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT
                strategy_id,
                MAX(COALESCE(strategy_id, '')) AS strategy_id_max,
                MAX(COALESCE(virtual_account_id, '')) AS virtual_account_id,
                COUNT(*) AS trade_count,
                SUM(CASE WHEN direction = 'buy' THEN 1 ELSE 0 END) AS buy_count,
                SUM(CASE WHEN direction = 'sell' THEN 1 ELSE 0 END) AS sell_count,
                SUM(CASE WHEN direction = 'buy' THEN amount ELSE 0 END) AS total_buy_amount,
                SUM(CASE WHEN direction = 'sell' THEN amount ELSE 0 END) AS total_sell_amount,
                SUM(commission + stamp_tax + transfer_fee) AS total_commission
            FROM trades
            WHERE trade_date = ? AND COALESCE(strategy_id, '') != ''
            GROUP BY strategy_id
            ORDER BY strategy_id ASC
            ''',
            (snapshot_date,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def count_order_records_by_broker_ids(self, broker_order_ids: List[int]) -> int:
        ids = [int(order_id) for order_id in broker_order_ids if int(order_id or 0) > 0]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT COUNT(DISTINCT broker_order_id) FROM order_records WHERE broker_order_id IN ({placeholders})',
            ids,
        )
        count = int(cursor.fetchone()[0] or 0)
        conn.close()
        return count

    def count_trade_records_by_trade_ids(self, traded_ids: List[int], trade_date: str) -> int:
        ids = [int(traded_id) for traded_id in traded_ids if int(traded_id or 0) > 0]
        if not ids:
            return 0
        conn = self._get_connection()
        cursor = conn.cursor()
        total = 0
        for traded_id in ids:
            cursor.execute(
                "SELECT 1 FROM trades WHERE remark LIKE ? AND trade_date = ? LIMIT 1",
                (f"成交号:{traded_id}%", trade_date),
            )
            if cursor.fetchone():
                total += 1
        conn.close()
        return total
    
    def _is_order_synced(self, order_id, trade_date: str = None, stock_code: str = "") -> bool:
        """检查委托是否已同步。

        miniQMT 的委托号可能跨交易日复用，所以有日期/代码时按
        broker_order_id + trade_date + stock_code 判断，避免误跳过新成交。
        """
        order_id_int = int(order_id or 0)
        if order_id_int <= 0:
            return False

        normalized_code = str(stock_code or "").strip().upper().split(".")[0]
        conn = self._get_connection()
        cursor = conn.cursor()

        if trade_date and normalized_code:
            cursor.execute(
                """
                SELECT 1 FROM trades
                WHERE broker_order_id = ? AND trade_date = ? AND stock_code = ?
                LIMIT 1
                """,
                (order_id_int, trade_date, normalized_code),
            )
        elif trade_date:
            cursor.execute(
                "SELECT 1 FROM trades WHERE broker_order_id = ? AND trade_date = ? LIMIT 1",
                (order_id_int, trade_date)
            )
        else:
            cursor.execute(
                "SELECT 1 FROM trades WHERE broker_order_id = ? LIMIT 1",
                (order_id_int,)
            )

        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def _is_order_synced_any_date(self, order_id) -> bool:
        """检查委托号是否已在任意日期同步过成交记录。"""
        order_id_int = int(order_id or 0)
        if order_id_int <= 0:
            return False

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1
            FROM trades
            WHERE broker_order_id = ?
            LIMIT 1
            """,
            (order_id_int,),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def get_record_by_id(self, record_id: int) -> Optional[TradeRecord]:
        """根据ID获取记录"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades WHERE id = ?', (record_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return TradeRecord.from_dict(dict(row))
        return None

    def get_records_by_broker_order_id(
        self,
        broker_order_id: int,
        *,
        source: str = None,
        stock_code: str = None,
    ) -> List[TradeRecord]:
        """根据券商委托单号查询交易记录。"""
        if broker_order_id is None or int(broker_order_id) <= 0:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = ["broker_order_id = ?"]
        params = [int(broker_order_id)]

        if source:
            conditions.append("source = ?")
            params.append(source)
        if stock_code:
            code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            conditions.append("stock_code = ?")
            params.append(code)

        where_clause = " AND ".join(conditions)
        cursor.execute(
            f'''
            SELECT * FROM trades
            WHERE {where_clause}
            ORDER BY trade_date ASC, id ASC
            '''
            ,
            params,
        )
        rows = cursor.fetchall()
        conn.close()
        return [TradeRecord.from_dict(dict(row)) for row in rows]

    def get_latest_record_by_broker_order_id(
        self,
        broker_order_id: int,
        *,
        source: str = None,
        stock_code: str = None,
    ) -> Optional[TradeRecord]:
        """根据券商委托单号获取最新一条交易记录。"""
        records = self.get_records_by_broker_order_id(
            broker_order_id,
            source=source,
            stock_code=stock_code,
        )
        return records[-1] if records else None
    
    def get_records(self,
                    start_date: str = None,
                    end_date: str = None,
                    stock_code: str = None,
                    direction: str = None,
                    source: str = None,
                    strategy_id: str = None,
                    virtual_account_id: str = None,
                    limit: int = 1000,
                    offset: int = 0) -> List[TradeRecord]:
        """
        查询交易记录
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            stock_code: 股票代码
            direction: 交易方向
            source: 来源
            limit: 最大返回数量
            offset: 偏移量
            
        Returns:
            交易记录列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        if stock_code:
            code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            conditions.append("stock_code = ?")
            params.append(code)
        if direction:
            conditions.append("direction = ?")
            params.append(direction)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if virtual_account_id:
            conditions.append("virtual_account_id = ?")
            params.append(virtual_account_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f'''
            SELECT * FROM trades 
            WHERE {where_clause}
            ORDER BY trade_date DESC, id DESC
            LIMIT ? OFFSET ?
        '''
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [TradeRecord.from_dict(dict(row)) for row in rows]
    
    def get_records_count(self,
                         start_date: str = None,
                         end_date: str = None,
                         stock_code: str = None,
                         direction: str = None,
                         source: str = None,
                         strategy_id: str = None,
                         virtual_account_id: str = None) -> int:
        """获取符合条件的记录总数"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        if stock_code:
            code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            conditions.append("stock_code = ?")
            params.append(code)
        if direction:
            conditions.append("direction = ?")
            params.append(direction)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if virtual_account_id:
            conditions.append("virtual_account_id = ?")
            params.append(virtual_account_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor.execute(f'SELECT COUNT(*) FROM trades WHERE {where_clause}', params)
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_today_records(
        self,
        *,
        strategy_id: str = None,
        virtual_account_id: str = None,
        source: str = None,
    ) -> List[TradeRecord]:
        """获取今日交易记录"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_records(
            start_date=today,
            end_date=today,
            strategy_id=strategy_id,
            virtual_account_id=virtual_account_id,
            source=source,
        )

    def get_strategy_daily_pnl_snapshots(
        self,
        *,
        strategy_id: str = "",
        strategy_ids: Optional[List[str]] = None,
        start_date: str = "",
        end_date: str = "",
        limit: int = 365,
    ) -> List[StrategyDailyPnlSnapshot]:
        """查询策略日终快照。

        - 传 `strategy_id` (单个) 或 `strategy_ids` (多个) 均可；都不传表示"所有策略"
        - 按 snapshot_date 升序返回，方便直接画曲线
        """
        ids: List[str] = []
        if strategy_id:
            ids.append(str(strategy_id).strip())
        for sid in strategy_ids or []:
            val = str(sid or "").strip()
            if val and val not in ids:
                ids.append(val)
        conn = self._get_connection()
        cursor = conn.cursor()
        conditions: List[str] = []
        params: List[Any] = []
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conditions.append(f"strategy_id IN ({placeholders})")
            params.extend(ids)
        if start_date:
            conditions.append("snapshot_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("snapshot_date <= ?")
            params.append(end_date)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        cursor.execute(
            f"""
            SELECT * FROM strategy_daily_pnl
            WHERE {where_clause}
            ORDER BY snapshot_date ASC, strategy_id ASC
            LIMIT ?
            """,
            [*params, limit],
        )
        rows = cursor.fetchall()
        conn.close()
        return [StrategyDailyPnlSnapshot.from_dict(dict(row)) for row in rows]

    def get_portfolio_equity_curve(
        self,
        *,
        strategy_ids: Optional[List[str]] = None,
        start_date: str = "",
        end_date: str = "",
    ) -> List[dict]:
        """基于每日策略快照拼出"组合净值曲线"，把同一天所有策略汇总成一行。

        Returns:
            [{
                snapshot_date,
                capital_limit, total_asset, cash, market_value,
                invested_cost, realized_pnl, unrealized_pnl, total_pnl,
                position_count, strategy_count,
            }, ...]  按 snapshot_date 升序
        """
        rows = self.get_strategy_daily_pnl_snapshots(
            strategy_ids=strategy_ids,
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )
        buckets: Dict[str, dict] = {}
        for snap in rows:
            bucket = buckets.setdefault(
                snap.snapshot_date,
                {
                    "snapshot_date": snap.snapshot_date,
                    "capital_limit": 0.0,
                    "total_asset": 0.0,
                    "cash": 0.0,
                    "market_value": 0.0,
                    "invested_cost": 0.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "total_pnl": 0.0,
                    "position_count": 0,
                    "strategy_count": 0,
                },
            )
            bucket["capital_limit"] += float(snap.capital_limit or 0.0)
            bucket["total_asset"] += float(snap.total_asset or 0.0)
            bucket["cash"] += float(snap.cash or 0.0)
            bucket["market_value"] += float(snap.market_value or 0.0)
            bucket["invested_cost"] += float(snap.invested_cost or 0.0)
            bucket["realized_pnl"] += float(snap.realized_pnl or 0.0)
            bucket["unrealized_pnl"] += float(snap.unrealized_pnl or 0.0)
            bucket["total_pnl"] += float(snap.total_pnl or 0.0)
            bucket["position_count"] += int(snap.position_count or 0)
            bucket["strategy_count"] += 1
        result = []
        for date_key in sorted(buckets.keys()):
            item = buckets[date_key]
            for field in (
                "capital_limit", "total_asset", "cash", "market_value",
                "invested_cost", "realized_pnl", "unrealized_pnl", "total_pnl",
            ):
                item[field] = round(float(item[field]), 2)
            result.append(item)
        return result
    
    def get_stock_records(self, stock_code: str, limit: int = 100) -> List[TradeRecord]:
        """获取指定股票的交易记录"""
        return self.get_records(stock_code=stock_code, limit=limit)
    
    def get_daily_summary(self, days: int = 30) -> List[dict]:
        """
        获取每日交易汇总
        
        Args:
            days: 最近天数
            
        Returns:
            每日汇总列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        cursor.execute('''
            SELECT * FROM daily_summary 
            WHERE trade_date >= ?
            ORDER BY trade_date DESC
        ''', (start_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ==================================================================
    # 统一成交统计口径（供实盘收益 / 报表等调用，单一真源）
    # ==================================================================

    def _normalize_strategy_ids(self, strategy_ids) -> Optional[List[str]]:
        if strategy_ids is None:
            return None
        if isinstance(strategy_ids, str):
            ids = [strategy_ids]
        else:
            ids = list(strategy_ids)
        return [sid for sid in (str(x or "").strip() for x in ids) if sid]

    def _collect_records(
        self,
        strategy_ids: Optional[List[str]],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[TradeRecord]:
        """按策略 ID 集合汇总成交记录（内部复用）。"""
        records: List[TradeRecord] = []
        if strategy_ids is None:
            records.extend(
                self.get_records(
                    start_date=start_date,
                    end_date=end_date,
                    limit=1000000,
                )
            )
        else:
            for sid in strategy_ids:
                records.extend(
                    self.get_records(
                        start_date=start_date,
                        end_date=end_date,
                        strategy_id=sid,
                        limit=1000000,
                    )
                )
        return records

    def get_period_stats(
        self,
        strategy_ids=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """统一的成交期间汇总（原子口径，所有报表一律调此方法）。

        Args:
            strategy_ids: 单个 ID / ID 列表；None 表示不限策略
            start_date / end_date: 成交日期范围（YYYY-MM-DD）

        Returns:
            {
                "total_trades":  总成交数,
                "buy_count":     买入笔数,
                "sell_count":    卖出笔数,
                "buy_amount":    买入金额合计,
                "sell_amount":   卖出金额合计,
                "total_fee":     手续费合计（佣金+印花税+过户费）,
                "net_inflow":    净流入 = sell_amount - buy_amount - total_fee,
            }
        """
        sids = self._normalize_strategy_ids(strategy_ids)
        records = self._collect_records(sids, start_date=start_date, end_date=end_date)
        buy_count = 0
        sell_count = 0
        buy_amount = 0.0
        sell_amount = 0.0
        total_fee = 0.0
        for rec in records:
            direction = str(getattr(rec, "direction", "") or "").lower()
            amount = float(getattr(rec, "amount", 0.0) or 0.0)
            fee = float(getattr(rec, "total_fee", 0.0) or 0.0)
            total_fee += fee
            if direction == TradeDirection.BUY.value:
                buy_count += 1
                buy_amount += amount
            elif direction == TradeDirection.SELL.value:
                sell_count += 1
                sell_amount += amount
        return {
            "total_trades": len(records),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_amount": round(buy_amount, 2),
            "sell_amount": round(sell_amount, 2),
            "total_fee": round(total_fee, 2),
            "net_inflow": round(sell_amount - buy_amount - total_fee, 2),
        }

    def get_daily_stats(
        self,
        strategy_ids=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[dict]:
        """按交易日分组的汇总，字段与 get_period_stats 一致，外加 trade_date。

        返回顺序：按日期倒序（最近的在前）。
        """
        sids = self._normalize_strategy_ids(strategy_ids)
        records = self._collect_records(sids, start_date=start_date, end_date=end_date)
        buckets: Dict[str, dict] = {}
        for rec in records:
            trade_date = str(getattr(rec, "trade_date", "") or "")
            if not trade_date:
                continue
            item = buckets.setdefault(
                trade_date,
                {
                    "trade_date": trade_date,
                    "total_trades": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "buy_amount": 0.0,
                    "sell_amount": 0.0,
                    "total_fee": 0.0,
                },
            )
            direction = str(getattr(rec, "direction", "") or "").lower()
            amount = float(getattr(rec, "amount", 0.0) or 0.0)
            fee = float(getattr(rec, "total_fee", 0.0) or 0.0)
            item["total_trades"] += 1
            item["total_fee"] += fee
            if direction == TradeDirection.BUY.value:
                item["buy_count"] += 1
                item["buy_amount"] += amount
            elif direction == TradeDirection.SELL.value:
                item["sell_count"] += 1
                item["sell_amount"] += amount
        result: List[dict] = []
        for trade_date in sorted(buckets.keys(), reverse=True):
            item = buckets[trade_date]
            item["buy_amount"] = round(float(item["buy_amount"]), 2)
            item["sell_amount"] = round(float(item["sell_amount"]), 2)
            item["total_fee"] = round(float(item["total_fee"]), 2)
            item["net_inflow"] = round(
                item["sell_amount"] - item["buy_amount"] - item["total_fee"], 2
            )
            result.append(item)
        return result

    def get_closed_trades(
        self,
        strategy_ids=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[dict]:
        """基于 FIFO 配对得到"已平仓交易"。

        算法：
          1) 把全部历史成交按 (strategy_id, stock_code) 分组（为保证配对完整，FIFO
             输入不受 start/end 过滤，只按 strategy_id 过滤）。
          2) 每组按时间升序遍历：
             - 买入 → 入队（units = volume，unit_cost = price，buy_fee 按比例摊销）
             - 卖出 → 从队首依次配对：
                 cost_value  = Σ matched_units * buy_unit_cost
                 buy_fee     = Σ 已摊销的买入手续费
                 proceeds    = matched_units * sell_price
                 sell_fee    = 当笔卖出手续费按比例摊销到 matched_units
                 pnl         = proceeds - cost_value - buy_fee - sell_fee
                 close_date  = 卖出 trade_date
                 open_date   = 最早那一笔买入 trade_date（取队首）
          3) 最后按 close_date 在 [start_date, end_date] 内筛出结果。

        返回 [{ strategy_id, stock_code, stock_name, qty, cost_value, proceeds,
                buy_fee, sell_fee, total_fee, pnl, pnl_pct, is_win,
                open_date, close_date }]，按 close_date 升序。
        """
        sids = self._normalize_strategy_ids(strategy_ids)
        records = self._collect_records(sids)
        records.sort(key=lambda r: (str(r.trade_date or ""), int(r.id or 0)))

        from collections import deque

        buckets: Dict[Tuple[str, str], deque] = {}
        closed: List[dict] = []

        for rec in records:
            direction = str(getattr(rec, "direction", "") or "").lower()
            code = str(getattr(rec, "stock_code", "") or "")
            sid = str(getattr(rec, "strategy_id", "") or "")
            if not code:
                continue
            key = (sid, code)
            queue = buckets.setdefault(key, deque())
            volume = int(getattr(rec, "volume", 0) or 0)
            price = float(getattr(rec, "price", 0.0) or 0.0)
            fee = float(getattr(rec, "total_fee", 0.0) or 0.0)
            if volume <= 0 or price <= 0:
                continue
            if direction == TradeDirection.BUY.value:
                queue.append(
                    {
                        "trade_date": str(rec.trade_date or ""),
                        "stock_name": str(getattr(rec, "stock_name", "") or ""),
                        "remaining": volume,
                        "unit_cost": price,
                        "fee_per_unit": (fee / volume) if volume > 0 else 0.0,
                    }
                )
                continue
            if direction != TradeDirection.SELL.value or not queue:
                # 空仓卖出（历史数据不全）直接跳过，不参与胜率统计
                continue
            sell_fee_per_unit = (fee / volume) if volume > 0 else 0.0
            remaining_sell = volume
            matched_qty = 0
            cost_value = 0.0
            buy_fee_accum = 0.0
            earliest_open = None
            while remaining_sell > 0 and queue:
                head = queue[0]
                take = min(head["remaining"], remaining_sell)
                cost_value += take * head["unit_cost"]
                buy_fee_accum += take * head["fee_per_unit"]
                if earliest_open is None:
                    earliest_open = head["trade_date"]
                head["remaining"] -= take
                remaining_sell -= take
                matched_qty += take
                if head["remaining"] <= 0:
                    queue.popleft()
            if matched_qty <= 0:
                continue
            proceeds = matched_qty * price
            sell_fee = matched_qty * sell_fee_per_unit
            pnl = proceeds - cost_value - buy_fee_accum - sell_fee
            pnl_pct = (pnl / cost_value * 100.0) if cost_value > 0 else 0.0
            closed.append(
                {
                    "strategy_id": sid,
                    "stock_code": code,
                    "stock_name": str(getattr(rec, "stock_name", "") or ""),
                    "qty": int(matched_qty),
                    "cost_value": round(cost_value, 2),
                    "proceeds": round(proceeds, 2),
                    "buy_fee": round(buy_fee_accum, 2),
                    "sell_fee": round(sell_fee, 2),
                    "total_fee": round(buy_fee_accum + sell_fee, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 4),
                    "is_win": pnl > 0,
                    "open_date": earliest_open or "",
                    "close_date": str(rec.trade_date or ""),
                }
            )

        if start_date:
            closed = [c for c in closed if c["close_date"] and c["close_date"] >= start_date]
        if end_date:
            closed = [c for c in closed if c["close_date"] and c["close_date"] <= end_date]
        closed.sort(key=lambda c: c["close_date"])
        return closed

    def get_win_rate_stats(
        self,
        strategy_ids=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """基于 get_closed_trades 的统一胜率口径。

        Returns:
            {
                "closed_count": 已平仓交易数,
                "win_count":    盈利交易数,
                "loss_count":   亏损交易数,
                "flat_count":   盈亏为 0 的交易数,
                "win_rate":     win_count / closed_count（0.0 ~ 1.0）,
                "realized_pnl": 所有已平仓交易累计已实现盈亏,
                "gross_profit": 盈利交易的盈利合计,
                "gross_loss":   亏损交易的亏损合计（正数）,
                "profit_factor": gross_profit / gross_loss（若分母=0 则为 None）,
                "avg_win":      平均每笔盈利,
                "avg_loss":     平均每笔亏损（正数）,
            }
        """
        closed = self.get_closed_trades(strategy_ids, start_date, end_date)
        win_count = sum(1 for c in closed if c["pnl"] > 0)
        loss_count = sum(1 for c in closed if c["pnl"] < 0)
        flat_count = len(closed) - win_count - loss_count
        realized_pnl = round(sum(c["pnl"] for c in closed), 2)
        gross_profit = round(sum(c["pnl"] for c in closed if c["pnl"] > 0), 2)
        gross_loss = round(-sum(c["pnl"] for c in closed if c["pnl"] < 0), 2)
        closed_count = len(closed)
        win_rate = (win_count / closed_count) if closed_count > 0 else 0.0
        profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else None
        avg_win = round(gross_profit / win_count, 2) if win_count > 0 else 0.0
        avg_loss = round(gross_loss / loss_count, 2) if loss_count > 0 else 0.0
        return {
            "closed_count": closed_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "flat_count": flat_count,
            "win_rate": round(win_rate, 4),
            "realized_pnl": realized_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    def get_statistics(self, 
                      start_date: str = None,
                      end_date: str = None,
                      stock_code: str = None) -> TradeSummary:
        """
        获取交易统计
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_code: 股票代码（可选）
            
        Returns:
            交易统计摘要
        """
        records = self.get_records(
            start_date=start_date,
            end_date=end_date,
            stock_code=stock_code,
            limit=100000
        )
        
        summary = TradeSummary()
        summary.total_trades = len(records)
        
        # 按股票分组计算盈亏
        stock_trades: Dict[str, List[TradeRecord]] = {}
        
        for record in records:
            if record.direction == TradeDirection.BUY.value:
                summary.buy_count += 1
                summary.total_buy_amount += record.amount
            else:
                summary.sell_count += 1
                summary.total_sell_amount += record.amount
            
            summary.total_commission += record.total_fee
            
            # 分组
            if record.stock_code not in stock_trades:
                stock_trades[record.stock_code] = []
            stock_trades[record.stock_code].append(record)
        
        # 计算每只股票的盈亏
        profits = []
        for code, trades in stock_trades.items():
            # 按日期排序
            trades.sort(key=lambda x: (x.trade_date, x.id))
            
            # 简单计算：卖出金额 - 买入金额
            buy_amount = sum(t.amount for t in trades if t.direction == TradeDirection.BUY.value)
            sell_amount = sum(t.amount for t in trades if t.direction == TradeDirection.SELL.value)
            
            if sell_amount > 0:  # 有卖出才计算盈亏
                profit = sell_amount - buy_amount
                profits.append(profit)
                
                if profit > 0:
                    summary.win_count += 1
                elif profit < 0:
                    summary.loss_count += 1
        
        # 计算统计指标
        if profits:
            summary.total_profit = sum(profits)
            summary.avg_profit = summary.total_profit / len(profits)
            summary.max_profit = max(profits) if profits else 0
            summary.max_loss = min(profits) if profits else 0
        
        if summary.win_count + summary.loss_count > 0:
            summary.win_rate = summary.win_count / (summary.win_count + summary.loss_count) * 100
        
        return summary
    
    def get_stock_holding_cost(self, stock_code: str) -> Tuple[float, int]:
        """
        计算指定股票的持仓成本
        
        Args:
            stock_code: 股票代码
            
        Returns:
            (平均成本价, 当前持仓数量)
        """
        records = self.get_stock_records(stock_code, limit=10000)
        records.sort(key=lambda x: (x.trade_date, x.id))
        
        total_cost = 0.0
        total_volume = 0
        
        for record in records:
            if record.direction == TradeDirection.BUY.value:
                total_cost += record.amount + record.total_fee
                total_volume += record.volume
            else:  # SELL
                if total_volume > 0:
                    avg_cost = total_cost / total_volume
                    sell_volume = min(record.volume, total_volume)
                    total_cost -= avg_cost * sell_volume
                    total_volume -= sell_volume
        
        avg_price = total_cost / total_volume if total_volume > 0 else 0
        return round(avg_price, 3), total_volume
    
    def delete_record(self, record_id: int) -> bool:
        """删除交易记录"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM trades WHERE id = ?', (record_id,))
            affected = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if affected > 0:
                self._log(f"删除交易记录: ID={record_id}")
                self.records_changed.emit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"删除交易记录失败: {e}")
            return False

    def realign_broker_sync_records_by_ownership(self) -> int:
        """
        按股票归属修正 broker_sync 成交记录的策略归属，避免账户级同步串到错误策略。
        """
        try:
            from trading_app.services.strategy.strategy_registry_service import get_strategy_registry_service

            registry = get_strategy_registry_service()
        except Exception as exc:
            logger.debug("加载策略归属注册表失败，跳过 broker_sync 纠偏: %s", exc)
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()
        corrected = 0
        deleted = 0
        try:
            cursor.execute(
                """
                SELECT id, stock_code, trade_date, broker_order_id, source, strategy_id, virtual_account_id
                FROM trades
                WHERE source = 'broker_sync'
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                record_id = int(row["id"] or 0)
                stock_code = str(row["stock_code"] or "")
                owner = registry.get_owner(stock_code)
                if owner is None or not owner.enabled:
                    continue

                target_strategy_id = str(owner.strategy_id or "")
                target_virtual_account_id = str(owner.virtual_account_id or "")
                current_strategy_id = str(row["strategy_id"] or "")
                current_virtual_account_id = str(row["virtual_account_id"] or "")
                if (
                    current_strategy_id == target_strategy_id
                    and current_virtual_account_id == target_virtual_account_id
                ):
                    continue

                cursor.execute(
                    """
                    SELECT id FROM trades
                    WHERE id != ?
                      AND source = ?
                      AND trade_date = ?
                      AND broker_order_id = ?
                      AND stock_code = ?
                      AND COALESCE(strategy_id, '') = ?
                      AND COALESCE(virtual_account_id, '') = ?
                    LIMIT 1
                    """,
                    (
                        record_id,
                        str(row["source"] or ""),
                        str(row["trade_date"] or ""),
                        int(row["broker_order_id"] or 0),
                        stock_code,
                        target_strategy_id,
                        target_virtual_account_id,
                    ),
                )
                duplicate = cursor.fetchone()
                if duplicate is not None:
                    cursor.execute("DELETE FROM trades WHERE id = ?", (record_id,))
                    deleted += max(cursor.rowcount, 0)
                    continue

                cursor.execute(
                    """
                    UPDATE trades
                    SET strategy_id = ?, virtual_account_id = ?
                    WHERE id = ?
                    """,
                    (target_strategy_id, target_virtual_account_id, record_id),
                )
                corrected += max(cursor.rowcount, 0)

            if corrected > 0 or deleted > 0:
                conn.commit()
                self.records_changed.emit()
                logger.info(
                    "按股票归属修正 broker_sync 成交记录完成: corrected=%d deleted=%d",
                    corrected,
                    deleted,
                )
            return corrected + deleted
        finally:
            conn.close()

    @staticmethod
    def _is_valid_trade_date_text(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        except Exception:
            return False
        return 2000 <= dt.year <= 2100

    @staticmethod
    def _extract_broker_trade_id_from_remark(remark: str) -> str:
        text = str(remark or "").strip()
        if not text:
            return ""
        match = re.search(r"成交号:(\d+)", text)
        return match.group(1) if match else ""

    def audit_trade_records(
        self,
        *,
        start_date: str = "",
        end_date: str = "",
        stock_code: str = "",
        direction: str = "",
        source: str = "",
        limit: int = 200000,
    ) -> TradeAuditReport:
        report = TradeAuditReport()
        scoped_records = self.get_records(
            start_date=start_date or None,
            end_date=end_date or None,
            stock_code=stock_code or None,
            direction=direction or None,
            source=source or None,
            limit=limit,
            offset=0,
        )
        report.scanned_records = len(scoped_records)
        issue_keys: set[tuple] = set()
        duplicate_group_keys: set[tuple[int, ...]] = set()

        def add_issue(issue: TradeAuditIssue) -> None:
            key = (
                issue.category,
                issue.stock_code,
                issue.trade_date,
                issue.direction,
                tuple(sorted(issue.record_ids)),
                issue.summary,
            )
            if key in issue_keys:
                return
            issue_keys.add(key)
            report.issues.append(issue)
            if issue.category == "duplicate":
                report.duplicate_issue_count += 1
            elif issue.category == "invalid_date":
                report.date_issue_count += 1
            elif issue.category == "position_mismatch":
                report.position_issue_count += 1

        trade_id_groups: Dict[tuple[str, str, str], List[TradeRecord]] = {}
        fingerprint_groups: Dict[tuple[int, str, str, float, int, str, str], List[TradeRecord]] = {}
        name_map: Dict[str, str] = {}

        for record in scoped_records:
            code = str(record.stock_code or "").strip()
            if code and record.stock_name:
                name_map[code] = str(record.stock_name or "").strip()

            broker_trade_id = self._extract_broker_trade_id_from_remark(record.remark)
            if broker_trade_id:
                trade_id_groups.setdefault(
                    (
                        broker_trade_id,
                        str(record.strategy_id or ""),
                        str(record.virtual_account_id or ""),
                    ),
                    [],
                ).append(record)

            broker_order_id = int(record.broker_order_id or 0)
            if broker_order_id > 0:
                fingerprint = (
                    broker_order_id,
                    code,
                    str(record.direction or ""),
                    round(float(record.price or 0.0), 4),
                    int(record.volume or 0),
                    str(record.strategy_id or ""),
                    str(record.virtual_account_id or ""),
                )
                fingerprint_groups.setdefault(fingerprint, []).append(record)

        for group in trade_id_groups.values():
            if len(group) <= 1:
                continue
            sorted_group = sorted(
                group,
                key=lambda item: (
                    0 if (self._is_valid_trade_date_text(str(item.trade_date or "")) and str(item.source or "") != TradeSource.BROKER_SYNC.value) else 1,
                    0 if self._is_valid_trade_date_text(str(item.trade_date or "")) else 1,
                    str(item.created_at or ""),
                    int(item.id or 0),
                ),
            )
            canonical = sorted_group[0]
            ids = tuple(sorted(int(item.id or 0) for item in group))
            action_record_ids = [
                int(item.id or 0)
                for item in group
                if int(item.id or 0) != int(canonical.id or 0)
            ]
            duplicate_group_keys.add(ids)
            dates = sorted({str(item.trade_date or "") for item in group})
            sample = canonical
            add_issue(
                TradeAuditIssue(
                    category="duplicate",
                    severity="error",
                    stock_code=str(sample.stock_code or ""),
                    stock_name=str(sample.stock_name or ""),
                    trade_date=" / ".join(dates),
                    direction=str(sample.direction or ""),
                    record_ids=list(ids),
                    action_record_ids=action_record_ids,
                    summary="同一成交号出现多条记录",
                    details=(
                        f"成交号 {self._extract_broker_trade_id_from_remark(sample.remark)} 对应记录 ID: "
                        f"{', '.join(str(i) for i in ids)}，建议保留 ID {int(canonical.id or 0)}"
                    ),
                )
            )

        for fingerprint, group in fingerprint_groups.items():
            if len(group) <= 1:
                continue
            sorted_group = sorted(
                group,
                key=lambda item: (
                    0 if self._is_valid_trade_date_text(str(item.trade_date or "")) else 1,
                    str(item.created_at or ""),
                    int(item.id or 0),
                ),
            )
            canonical = sorted_group[0]
            ids = tuple(sorted(int(item.id or 0) for item in group))
            if ids in duplicate_group_keys:
                continue
            duplicate_group_keys.add(ids)
            broker_order_id, code, _direction, price, volume, _strategy_id, _virtual_account_id = fingerprint
            sample = canonical
            action_record_ids = [
                int(item.id or 0)
                for item in group
                if int(item.id or 0) != int(canonical.id or 0)
            ]
            dates = sorted({str(item.trade_date or "") for item in group})
            add_issue(
                TradeAuditIssue(
                    category="duplicate",
                    severity="warning",
                    stock_code=str(code or ""),
                    stock_name=str(sample.stock_name or ""),
                    trade_date=" / ".join(dates),
                    direction=str(sample.direction or ""),
                    record_ids=list(ids),
                    action_record_ids=action_record_ids,
                    summary="同一委托成交指纹出现多条记录",
                    details=(
                        f"委托号 {broker_order_id}，价格 {price:.4f}，数量 {volume}，"
                        f"记录 ID: {', '.join(str(i) for i in ids)}，建议保留 ID {int(canonical.id or 0)}"
                    ),
                )
            )

        try:
            from live_rotation.holiday_calendar import is_trading_day, get_non_trading_reason
        except Exception:
            is_trading_day = None
            get_non_trading_reason = None

        for record in scoped_records:
            trade_date = str(record.trade_date or "").strip()
            if not self._is_valid_trade_date_text(trade_date):
                add_issue(
                    TradeAuditIssue(
                        category="invalid_date",
                        severity="error",
                        stock_code=str(record.stock_code or ""),
                        stock_name=str(record.stock_name or ""),
                        trade_date=trade_date or "-",
                        direction=str(record.direction or ""),
                        record_ids=[int(record.id or 0)],
                        summary="交易日期格式非法",
                        details=f"记录 ID {int(record.id or 0)} 的 trade_date={trade_date!r} 无法解析为 YYYY-MM-DD",
                    )
                )
                continue
            if is_trading_day is None or is_trading_day(trade_date):
                continue

            suggested_date = ""
            broker_order_id = int(record.broker_order_id or 0)
            if broker_order_id > 0:
                fingerprint = (
                    broker_order_id,
                    str(record.stock_code or ""),
                    str(record.direction or ""),
                    round(float(record.price or 0.0), 4),
                    int(record.volume or 0),
                    str(record.strategy_id or ""),
                    str(record.virtual_account_id or ""),
                )
                for peer in fingerprint_groups.get(fingerprint, []):
                    peer_date = str(peer.trade_date or "").strip()
                    if int(peer.id or 0) == int(record.id or 0):
                        continue
                    if self._is_valid_trade_date_text(peer_date) and is_trading_day(peer_date):
                        suggested_date = peer_date
                        break

            reason = get_non_trading_reason(trade_date) if get_non_trading_reason else "非交易日"
            details = f"记录 ID {int(record.id or 0)} 的 trade_date={trade_date}，{reason}"
            if suggested_date:
                details += f"，疑似应为 {suggested_date}"
            add_issue(
                TradeAuditIssue(
                    category="invalid_date",
                    severity="error",
                    stock_code=str(record.stock_code or ""),
                    stock_name=str(record.stock_name or ""),
                    trade_date=trade_date,
                    direction=str(record.direction or ""),
                    record_ids=[int(record.id or 0)],
                    action_record_ids=[int(record.id or 0)],
                    summary="交易日期落在非交易日",
                    details=details,
                    suggested_trade_date=suggested_date or self.suggest_valid_trade_date(trade_date),
                )
            )

        history_records = self.get_records(
            stock_code=stock_code or None,
            limit=limit,
            offset=0,
        )
        net_positions: Dict[str, int] = {}
        for record in history_records:
            code = str(record.stock_code or "").strip()
            if not code:
                continue
            if code and record.stock_name:
                name_map[code] = str(record.stock_name or "").strip()
            delta = int(record.volume or 0)
            if str(record.direction or "") == TradeDirection.SELL.value:
                delta = -delta
            net_positions[code] = net_positions.get(code, 0) + delta

        broker_positions: Dict[str, int] = {}
        broker_connected = False
        try:
            from common.broker_session_service import get_broker_session_service

            broker_service = get_broker_session_service()
            broker_connected = bool(getattr(broker_service, "is_connected", False))
            report.broker_connected = broker_connected
            if broker_connected:
                for position in broker_service.query_stock_positions() or []:
                    code = str(getattr(position, "stock_code", "") or "").split(".")[0]
                    if not code:
                        continue
                    broker_positions[code] = int(getattr(position, "volume", 0) or 0)
                report.broker_position_count = len(broker_positions)
        except Exception:
            broker_connected = False
            report.broker_connected = False

        for code, history_volume in sorted(net_positions.items()):
            if history_volume < 0:
                add_issue(
                    TradeAuditIssue(
                        category="position_mismatch",
                        severity="error",
                        stock_code=code,
                        stock_name=name_map.get(code, ""),
                        record_ids=[],
                        summary="历史净持仓为负数",
                        details=f"{code} 历史累计买卖后净持仓为 {history_volume} 股，说明存在缺买单、重复卖单或导入不完整。",
                    )
                )

        if broker_connected:
            all_codes = {code for code, volume in net_positions.items() if volume != 0} | {
                code for code, volume in broker_positions.items() if volume != 0
            }
            for code in sorted(all_codes):
                history_volume = int(net_positions.get(code, 0) or 0)
                broker_volume = int(broker_positions.get(code, 0) or 0)
                if history_volume == broker_volume:
                    continue
                add_issue(
                    TradeAuditIssue(
                        category="position_mismatch",
                        severity="warning",
                        stock_code=code,
                        stock_name=name_map.get(code, ""),
                        record_ids=[],
                        summary="历史净持仓与当前持仓不一致",
                        details=(
                            f"{code} 历史净持仓 {history_volume} 股，"
                            f"券商当前持仓 {broker_volume} 股，差额 {broker_volume - history_volume:+d} 股。"
                        ),
                    )
                )
        else:
            report.notes.append("当前未连接券商，'历史买卖数 vs 当前持仓数' 仅能检查负持仓异常，无法校验实时持仓。")

        report.notes.append("重复记录与日期异常按当前筛选范围检查；持仓对账按该股票全量历史成交检查。")
        severity_rank = {"error": 0, "warning": 1, "info": 2}
        report.issues.sort(
            key=lambda item: (
                severity_rank.get(item.severity, 9),
                item.category,
                item.stock_code,
                item.trade_date,
                tuple(item.record_ids),
            )
        )
        return report

    def suggest_valid_trade_date(self, trade_date: str) -> str:
        text = str(trade_date or "").strip()
        if not self._is_valid_trade_date_text(text):
            return ""
        try:
            from live_rotation.holiday_calendar import is_trading_day
        except Exception:
            return text
        try:
            current = datetime.strptime(text[:10], "%Y-%m-%d").date()
        except Exception:
            return ""
        if is_trading_day(current.isoformat()):
            return current.isoformat()
        for _ in range(10):
            current = current - timedelta(days=1)
            if is_trading_day(current.isoformat()):
                return current.isoformat()
        return ""

    def delete_trade_records(self, record_ids: List[int]) -> int:
        ids = sorted({int(item) for item in (record_ids or []) if int(item or 0) > 0})
        if not ids:
            return 0
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(f"DELETE FROM trades WHERE id IN ({placeholders})", ids)
        deleted = int(cursor.rowcount or 0)
        conn.commit()
        conn.close()
        if deleted > 0:
            self.records_changed.emit()
            self._log(f"已删除 {deleted} 条交易记录")
        return deleted

    def update_trade_record_dates(self, record_ids: List[int], trade_date: str) -> int:
        ids = sorted({int(item) for item in (record_ids or []) if int(item or 0) > 0})
        normalized_date = str(trade_date or "").strip()
        if not ids or not self._is_valid_trade_date_text(normalized_date):
            return 0
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            f"UPDATE trades SET trade_date = ? WHERE id IN ({placeholders})",
            [normalized_date, *ids],
        )
        updated = int(cursor.rowcount or 0)
        conn.commit()
        conn.close()
        if updated > 0:
            self.records_changed.emit()
            self._log(f"已修正 {updated} 条交易记录日期为 {normalized_date}")
        return updated

    def dedupe_trade_records_by_broker_order(self) -> int:
        """清理相同委托号/交易日/策略/代码/方向下的重复成交记录。

        miniQMT 委托号会跨交易日、甚至跨品种复用，因此去重键必须包含
        trade_date 与 stock_code，不能把不同交易日的成交当成重复单删掉。
        """
        with self._write_lock:
            deleted, updated = self._dedupe_trade_records_by_broker_order_unlocked()
        changed = deleted + updated
        if changed > 0:
            self.records_changed.emit()
            logger.info(
                "按委托号去重成交记录完成: deleted=%d updated=%d",
                deleted,
                updated,
            )
        return changed

    def _dedupe_trade_records_by_broker_order_unlocked(self) -> tuple[int, int]:
        conn = self._get_connection()
        cursor = conn.cursor()
        deleted = 0
        updated = 0
        try:
            cursor.execute(
                """
                SELECT id, broker_order_id, stock_code, direction,
                       COALESCE(strategy_id, '') AS strategy_id,
                       COALESCE(virtual_account_id, '') AS virtual_account_id,
                       trade_date, source, created_at
                FROM trades
                WHERE broker_order_id > 0
                ORDER BY broker_order_id ASC, id ASC
                """
            )
            rows = [dict(row) for row in cursor.fetchall()]
            groups: Dict[tuple, List[dict]] = {}
            for row in rows:
                fallback_date = str(row.get("created_at", "") or "").split(" ")[0] or datetime.now().strftime("%Y-%m-%d")
                normalized_trade_date = self._normalize_broker_time_to_date(
                    row.get("trade_date", ""),
                    fallback_date,
                )
                key = (
                    int(row.get("broker_order_id", 0) or 0),
                    str(row.get("stock_code", "") or ""),
                    str(row.get("direction", "") or ""),
                    normalized_trade_date,
                    str(row.get("strategy_id", "") or ""),
                    str(row.get("virtual_account_id", "") or ""),
                )
                groups.setdefault(key, []).append(row)

            for group_rows in groups.values():
                if len(group_rows) <= 1:
                    continue

                def _score(item: dict) -> tuple:
                    trade_date = str(item.get("trade_date", "") or "")
                    created_at = str(item.get("created_at", "") or "")
                    source = str(item.get("source", "") or "")
                    return (
                        1 if self._is_valid_trade_date_text(trade_date) else 0,
                        1 if source != "broker_sync" else 0,
                        created_at,
                        -int(item.get("id", 0) or 0),
                    )

                canonical = max(group_rows, key=_score)
                fallback_date = str(canonical.get("created_at", "") or "").split(" ")[0] or datetime.now().strftime("%Y-%m-%d")
                normalized_trade_date = self._normalize_broker_time_to_date(
                    canonical.get("trade_date", ""),
                    fallback_date,
                )
                if normalized_trade_date != str(canonical.get("trade_date", "") or ""):
                    cursor.execute(
                        "UPDATE trades SET trade_date = ? WHERE id = ?",
                        (normalized_trade_date, int(canonical.get("id", 0) or 0)),
                    )
                    updated += max(cursor.rowcount, 0)

                for row in group_rows:
                    row_id = int(row.get("id", 0) or 0)
                    if row_id == int(canonical.get("id", 0) or 0):
                        continue
                    cursor.execute("DELETE FROM trades WHERE id = ?", (row_id,))
                    deleted += max(cursor.rowcount, 0)

            if deleted > 0 or updated > 0:
                conn.commit()
            return deleted, updated
        finally:
            conn.close()
    
    def export_to_csv(self, file_path: str,
                     start_date: str = None,
                     end_date: str = None) -> bool:
        """
        导出交易记录到 CSV 文件
        
        Args:
            file_path: 导出文件路径
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            是否成功
        """
        try:
            import csv
            
            records = self.get_records(start_date=start_date, end_date=end_date, limit=100000)
            
            if not records:
                logger.warning("没有可导出的记录")
                return False
            
            headers = [
                '交易日期', '股票代码', '股票名称', '方向',
                '价格', '数量', '金额', '手续费', '印花税', '过户费', '来源', '备注'
            ]
            
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for r in records:
                    writer.writerow([
                        r.trade_date, r.stock_code, r.stock_name,
                        r.direction_display, f"{r.price:.4f}", r.volume,
                        f"{r.amount:.2f}", f"{r.commission:.2f}",
                        f"{r.stamp_tax:.2f}", f"{r.transfer_fee:.2f}",
                        r.source_display, r.remark
                    ])
            
            self._log(f"导出 {len(records)} 条记录到 {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return False
    
    # ==================== Daily PnL Snapshot Methods ====================
    
    def _init_pnl_table(self):
        """Initialize daily_pnl snapshot table"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT UNIQUE NOT NULL,
                total_asset REAL DEFAULT 0,
                cash REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                total_profit_pct REAL DEFAULT 0,
                cumulative_return REAL DEFAULT 0,
                position_count INTEGER DEFAULT 0,
                remark TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pnl_date ON daily_pnl(snapshot_date)')
        
        conn.commit()
        conn.close()
        logger.info("daily_pnl table initialized")

    def _init_position_snapshot_table(self):
        """Initialize daily position snapshot table"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_position_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                volume INTEGER DEFAULT 0,
                can_use_volume INTEGER DEFAULT 0,
                open_price REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(snapshot_date, stock_code)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_snapshot_date ON daily_position_snapshots(snapshot_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_snapshot_code ON daily_position_snapshots(stock_code)')

        conn.commit()
        conn.close()
        logger.info("daily_position_snapshots table initialized")

    def _init_strategy_snapshot_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS strategy_daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT DEFAULT '',
                virtual_account_id TEXT DEFAULT '',
                total_asset REAL DEFAULT 0,
                cash REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                position_count INTEGER DEFAULT 0,
                remark TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(snapshot_date, strategy_id)
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS strategy_position_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT DEFAULT '',
                virtual_account_id TEXT DEFAULT '',
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                volume INTEGER DEFAULT 0,
                can_use_volume INTEGER DEFAULT 0,
                open_price REAL DEFAULT 0,
                cost_price REAL DEFAULT 0,
                cost_amount REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(snapshot_date, strategy_id, stock_code)
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS strategy_daily_trade_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_name TEXT DEFAULT '',
                virtual_account_id TEXT DEFAULT '',
                trade_count INTEGER DEFAULT 0,
                buy_count INTEGER DEFAULT 0,
                sell_count INTEGER DEFAULT 0,
                total_buy_amount REAL DEFAULT 0,
                total_sell_amount REAL DEFAULT 0,
                total_commission REAL DEFAULT 0,
                remark TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(snapshot_date, strategy_id)
            )
            '''
        )
        self._ensure_column(cursor, 'strategy_daily_pnl', 'capital_limit', "REAL DEFAULT 0")
        self._ensure_column(cursor, 'strategy_daily_pnl', 'invested_cost', "REAL DEFAULT 0")
        self._ensure_column(cursor, 'strategy_daily_pnl', 'realized_pnl', "REAL DEFAULT 0")
        self._ensure_column(cursor, 'strategy_daily_pnl', 'unrealized_pnl', "REAL DEFAULT 0")
        self._ensure_column(cursor, 'strategy_daily_pnl', 'total_pnl', "REAL DEFAULT 0")
        self._ensure_column(cursor, 'strategy_position_snapshots', 'cost_price', "REAL DEFAULT 0")
        self._ensure_column(cursor, 'strategy_position_snapshots', 'cost_amount', "REAL DEFAULT 0")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_pnl_date ON strategy_daily_pnl(snapshot_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_pnl_id ON strategy_daily_pnl(strategy_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_position_date ON strategy_position_snapshots(snapshot_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_position_id ON strategy_position_snapshots(strategy_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_trade_summary_date ON strategy_daily_trade_summary(snapshot_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_trade_summary_id ON strategy_daily_trade_summary(strategy_id)')
        conn.commit()
        conn.close()
        logger.info("strategy snapshot tables initialized")

    @staticmethod
    def _normalize_broker_time_to_date(raw_value, fallback_date: str) -> str:
        if raw_value in (None, "", 0, "0"):
            return fallback_date
        value = str(raw_value).strip()
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits.isdigit() and len(digits) in (10, 13):
            try:
                ts = int(digits)
                if len(digits) == 13:
                    ts = ts / 1000
                dt = datetime.fromtimestamp(ts)
                if 2000 <= dt.year <= 2100:
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y%m%d",
            "%Y-%m-%d",
        ]
        candidates = [value]
        if digits and digits != value:
            candidates.append(digits)
        for candidate in candidates:
            for pattern in patterns:
                try:
                    dt = datetime.strptime(candidate, pattern)
                    if 2000 <= dt.year <= 2100:
                        return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        if len(digits) >= 8:
            try:
                dt = datetime.strptime(digits[:8], "%Y%m%d")
                if 2000 <= dt.year <= 2100:
                    return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return fallback_date
    
    def save_daily_pnl(self, snapshot_date: str, total_asset: float, cash: float,
                       market_value: float, position_count: int = 0,
                       remark: str = "") -> Optional[DailyPnlSnapshot]:
        """
        Save or update daily PnL snapshot.
        
        Automatically computes daily P&L and cumulative return based on
        the previous snapshot.
        
        Args:
            snapshot_date: Date string YYYY-MM-DD
            total_asset: Total account asset value
            cash: Available cash
            market_value: Market value of positions
            position_count: Number of positions held
            remark: Optional note
            
        Returns:
            DailyPnlSnapshot or None on failure
        """
        # Get previous snapshot to calculate daily PnL
        prev = self.get_previous_pnl_snapshot(snapshot_date)
        
        total_profit = 0.0
        total_profit_pct = 0.0
        cumulative_return = 0.0
        
        if prev and prev.total_asset > 0:
            total_profit = total_asset - prev.total_asset
            total_profit_pct = (total_profit / prev.total_asset) * 100
            # Cumulative return: chain from previous
            cumulative_return = (1 + prev.cumulative_return / 100) * (1 + total_profit_pct / 100) - 1
            cumulative_return *= 100
        elif prev is None:
            # First snapshot ever, no PnL to calculate
            total_profit = 0.0
            total_profit_pct = 0.0
            cumulative_return = 0.0
        
        snapshot = DailyPnlSnapshot(
            snapshot_date=snapshot_date,
            total_asset=round(total_asset, 2),
            cash=round(cash, 2),
            market_value=round(market_value, 2),
            total_profit=round(total_profit, 2),
            total_profit_pct=round(total_profit_pct, 4),
            cumulative_return=round(cumulative_return, 4),
            position_count=position_count,
            remark=remark,
        )
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # UPSERT: insert or replace if same date exists
            cursor.execute('''
                INSERT INTO daily_pnl (
                    snapshot_date, total_asset, cash, market_value,
                    total_profit, total_profit_pct, cumulative_return,
                    position_count, remark, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_date) DO UPDATE SET
                    total_asset=excluded.total_asset,
                    cash=excluded.cash,
                    market_value=excluded.market_value,
                    total_profit=excluded.total_profit,
                    total_profit_pct=excluded.total_profit_pct,
                    cumulative_return=excluded.cumulative_return,
                    position_count=excluded.position_count,
                    remark=excluded.remark,
                    created_at=excluded.created_at
            ''', (
                snapshot.snapshot_date, snapshot.total_asset, snapshot.cash,
                snapshot.market_value, snapshot.total_profit, snapshot.total_profit_pct,
                snapshot.cumulative_return, snapshot.position_count,
                snapshot.remark, snapshot.created_at
            ))
            
            snapshot.id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self._log(f"Daily PnL saved: {snapshot_date} total={total_asset:,.2f} "
                     f"pnl={total_profit:+,.2f} ({total_profit_pct:+.2f}%)")
            
            self.pnl_snapshot_saved.emit(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to save daily PnL: {e}")
            return None
    
    def get_previous_pnl_snapshot(self, before_date: str) -> Optional[DailyPnlSnapshot]:
        """Get the most recent snapshot before the given date"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM daily_pnl WHERE snapshot_date < ? ORDER BY snapshot_date DESC LIMIT 1',
            (before_date,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return DailyPnlSnapshot.from_dict(dict(row))
        return None
    
    def get_pnl_snapshot(self, snapshot_date: str) -> Optional[DailyPnlSnapshot]:
        """Get snapshot for a specific date"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM daily_pnl WHERE snapshot_date = ?', (snapshot_date,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return DailyPnlSnapshot.from_dict(dict(row))
        return None
    
    def get_pnl_snapshots(self, start_date: str = None, end_date: str = None,
                          limit: int = 365) -> List[DailyPnlSnapshot]:
        """
        Get PnL snapshots within a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max number of records
            
        Returns:
            List of DailyPnlSnapshot, ordered by date ascending
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("snapshot_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("snapshot_date <= ?")
            params.append(end_date)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor.execute(
            f'SELECT * FROM daily_pnl WHERE {where_clause} ORDER BY snapshot_date ASC LIMIT ?',
            params + [limit]
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [DailyPnlSnapshot.from_dict(dict(row)) for row in rows]
    
    def get_pnl_snapshots_count(self) -> int:
        """Get total count of PnL snapshots"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM daily_pnl')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def delete_pnl_snapshot(self, snapshot_date: str) -> bool:
        """Delete a PnL snapshot by date"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM daily_pnl WHERE snapshot_date = ?', (snapshot_date,))
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            if affected > 0:
                self._log(f"Deleted PnL snapshot: {snapshot_date}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete PnL snapshot: {e}")
            return False
    
    def calculate_max_drawdown(self, snapshots: List[DailyPnlSnapshot]) -> Tuple[float, str, str]:
        """
        Calculate maximum drawdown from a series of snapshots.
        
        Returns:
            (max_drawdown_pct, peak_date, trough_date)
        """
        if not snapshots:
            return 0.0, "", ""
        
        peak_asset = snapshots[0].total_asset
        peak_date = snapshots[0].snapshot_date
        max_drawdown = 0.0
        max_dd_peak_date = ""
        max_dd_trough_date = ""
        
        for s in snapshots:
            if s.total_asset > peak_asset:
                peak_asset = s.total_asset
                peak_date = s.snapshot_date
            
            if peak_asset > 0:
                drawdown = (peak_asset - s.total_asset) / peak_asset * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_dd_peak_date = peak_date
                    max_dd_trough_date = s.snapshot_date
        
        return round(max_drawdown, 2), max_dd_peak_date, max_dd_trough_date
    
    def calculate_sharpe_ratio(self, snapshots: List[DailyPnlSnapshot],
                               risk_free_rate: float = 0.015) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            snapshots: List of snapshots
            risk_free_rate: Annual risk-free rate (default 1.5%)
            
        Returns:
            Sharpe ratio
        """
        if len(snapshots) < 2:
            return 0.0
        
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_asset = snapshots[i - 1].total_asset
            if prev_asset > 0:
                ret = (snapshots[i].total_asset - prev_asset) / prev_asset
                daily_returns.append(ret)
        
        if not daily_returns:
            return 0.0
        
        import math
        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_return = math.sqrt(variance) if variance > 0 else 0
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        daily_rf = risk_free_rate / 252
        sharpe = (avg_return - daily_rf) / std_return * math.sqrt(252)
        return round(sharpe, 2)
    
    def export_pnl_to_csv(self, file_path: str, start_date: str = None,
                          end_date: str = None) -> bool:
        """Export PnL snapshots to CSV"""
        try:
            import csv
            
            snapshots = self.get_pnl_snapshots(start_date, end_date, limit=100000)
            if not snapshots:
                logger.warning("No PnL snapshots to export")
                return False
            
            headers = [
                'Date', 'Total Asset', 'Cash', 'Market Value',
                'Daily P&L', 'Daily Return %', 'Cumulative Return %',
                'Positions', 'Remark'
            ]
            
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for s in snapshots:
                    writer.writerow([
                        s.snapshot_date, f"{s.total_asset:.2f}", f"{s.cash:.2f}",
                        f"{s.market_value:.2f}", f"{s.total_profit:.2f}",
                        f"{s.total_profit_pct:.4f}", f"{s.cumulative_return:.4f}",
                        s.position_count, s.remark
                    ])
            
            self._log(f"Exported {len(snapshots)} PnL snapshots to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export PnL snapshots: {e}")
            return False

    def get_all_stocks(self) -> List[Tuple[str, str]]:
        """获取所有交易过的股票列表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT stock_code, stock_name 
            FROM trades 
            ORDER BY stock_code
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [(row['stock_code'], row['stock_name']) for row in rows]


# 全局单例
_trade_record_service: Optional[TradeRecordService] = None


def get_trade_record_service() -> TradeRecordService:
    """
    获取全局交易记录服务实例（单例模式）
    
    Returns:
        TradeRecordService 实例
    """
    global _trade_record_service
    if _trade_record_service is None:
        _trade_record_service = TradeRecordService()
    return _trade_record_service

