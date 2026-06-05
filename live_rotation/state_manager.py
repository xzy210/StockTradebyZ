"""
ETF轮动实盘 - 状态持久化

当前已切换为统一策略账本存储。
旧的 live_rotation/config/rotation_state.json 仅作为一次性迁移来源，
迁移完成后会自动删除。
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

from trading_app.services.strategy.strategy_budget_service import (
    StrategyBudgetState,
    get_strategy_budget_service,
)
from trading_app.services.strategy.strategy_constants import normalize_symbol_code
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    time: str
    action: str           # BUY / SELL / SELL_ALL
    code: str
    name: str
    price: float
    quantity: int
    amount: float
    reason: str = ""
    broker_order_id: int = -1
    success: bool = True
    error_msg: str = ""
    pnl: float = 0.0      # 本笔盈亏（仅 SELL 有效）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TradeRecord':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class CapitalLedgerEntry:
    """专用资金账本流水条目"""
    date: str
    time: str
    action: str            # 初始化 / 买入划出 / 卖出回收 / 手动重置
    code: str = ""
    name: str = ""
    amount: float = 0.0    # 变动金额（正=增加，负=减少）
    commission: float = 0.0
    balance: float = 0.0   # 操作后余额
    fee_source: str = ""   # [实际] / [估算]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CapitalLedgerEntry':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class OrderRecord:
    """委托 / 成交明细"""
    order_id: int
    date: str
    time: str
    action: str            # 买入 / 卖出
    code: str
    name: str
    ordered_qty: int
    ordered_price: float
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = -1.0   # -1 = 未知/估算
    status: str = "pending_submit"
    reason: str = ""
    pnl: float = 0.0           # 本笔盈亏（仅卖出有效）

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'OrderRecord':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class RotationState:
    """ETF轮动实盘状态"""
    current_holding: Optional[str] = None       # 当前持仓ETF代码
    current_holding_name: str = ""              # 持仓ETF名称
    current_score: float = 0.0                  # 当前持仓得分
    buy_price: float = 0.0                      # 买入价格
    buy_date: str = ""                          # 买入日期
    buy_quantity: int = 0                       # 买入数量
    last_check_date: str = ""                   # 最近一次信号检查日期
    last_check_time: str = ""                   # 最近一次信号检查时间
    last_signal: str = ""                       # 最近信号: HOLD / SWITCH / SELL_ALL / BUY
    trades_today: int = 0                       # 今日已交易次数
    trades_today_date: str = ""                 # trades_today 对应的日期
    total_invested: float = 0.0                 # 累计投入资金
    total_pnl: float = 0.0                      # 累计盈亏

    holding_high_price: float = 0.0             # 持仓期间最高价（移动止盈用）
    account_peak: float = 0.0                   # 账户资产峰值（回撤保护用）
    cooldown_remaining: int = 0                 # 回撤保护冷却剩余天数
    cooldown_last_decrement_date: str = ""      # 冷却天数最近一次递减日期
    check_count: int = 0                        # 信号检查计数（调仓周期用）

    # --- ETF 自动调度元状态（与业务检查结果分离）---
    auto_data_task_date: str = ""               # 自动数据更新所属业务日期
    auto_data_task_schedule_time: str = ""      # 自动数据更新时间配置
    auto_data_task_trigger: str = ""            # scheduled / manual
    auto_data_task_status: str = ""             # running / completed / failed
    auto_data_task_started_at: str = ""
    auto_data_task_completed_at: str = ""
    auto_data_task_error: str = ""

    auto_signal_task_date: str = ""             # 自动信号检查所属业务日期
    auto_signal_task_schedule_time: str = ""    # 自动检查时间配置
    auto_signal_task_trigger: str = ""          # scheduled / manual
    auto_signal_task_status: str = ""           # running / completed / failed
    auto_signal_task_started_at: str = ""
    auto_signal_task_completed_at: str = ""
    auto_signal_task_error: str = ""

    # --- 专用资金账本（真实账户模式下的资金隔离）---
    dedicated_cash: float = 0.0                 # 策略可用现金（0=尚未初始化）

    last_scores: Dict[str, float] = field(default_factory=dict)
    trade_history: List[dict] = field(default_factory=list)

    # --- 扩展分析数据 ---
    capital_ledger: List[dict] = field(default_factory=list)   # 资金流水（最近500条）
    daily_equity: Dict[str, float] = field(default_factory=dict)  # 净值曲线 {日期: 净值}
    order_records: List[dict] = field(default_factory=list)    # 委托明细（最近200条）

    def get_trades_today(self) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        if self.trades_today_date != today:
            return 0
        return self.trades_today

    def increment_trades_today(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.trades_today_date != today:
            self.trades_today = 0
            self.trades_today_date = today
        self.trades_today += 1

    def add_trade(self, record: TradeRecord):
        self.trade_history.append(record.to_dict())
        self.increment_trades_today()
        # 保留最近200条
        if len(self.trade_history) > 200:
            self.trade_history = self.trade_history[-200:]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RotationState':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


class StateManager:
    """统一策略账本适配器。"""

    LEGACY_STATE_FILE = "rotation_state.json"

    def __init__(
        self,
        config_dir: Optional[str] = None,
        *,
        strategy_id: str = "",
        strategy_name: str = "",
        virtual_account_id: str = "",
    ):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        default_spec = get_strategy_spec_service().etf_rotation()
        self.strategy_id = (strategy_id or default_spec.strategy_id or "etf_rotation").strip()
        self.strategy_name = (strategy_name or default_spec.strategy_name or "ETF轮动实盘").strip()
        self.virtual_account_id = (
            virtual_account_id or default_spec.virtual_account_id or f"va_{self.strategy_id}"
        ).strip()
        self.legacy_state_path = self.config_dir / self.LEGACY_STATE_FILE
        self.budget_service = get_strategy_budget_service()
        self._state: Optional[RotationState] = None

    @property
    def state(self) -> RotationState:
        if self._state is None:
            self._state = self.load()
        return self._state

    def load(self) -> RotationState:
        try:
            legacy_state = self._load_legacy_state()
            if legacy_state is not None:
                self._state = legacy_state
                self.save(legacy_state)
                self._delete_legacy_state()
                logger.info("已将 ETF 旧账本迁移到统一策略账本并删除旧文件")
                return legacy_state

            record = self.budget_service.get_strategy_state_record(
                self.strategy_id,
                strategy_name=self.strategy_name,
                virtual_account_id=self.virtual_account_id,
                real_total_asset=0.0,
            )
            state = self._state_from_budget_record(record)
            repaired = self._repair_suspicious_startup_cash_adjustments(state, record)
            if repaired:
                self._state = state
                self.save(state)
            logger.info("已从统一策略账本加载 ETF 状态: 持仓=%s", state.current_holding or "无")
            return state
        except Exception as e:
            logger.error(f"加载统一策略状态失败: {e}")
            return RotationState()

    def save(self, state: Optional[RotationState] = None):
        if state is not None:
            self._state = state
        if self._state is None:
            return
        try:
            record = self._budget_record_from_state(self._state)
            self.budget_service.save_strategy_state_record(record)
            logger.debug("ETF 状态已保存到统一策略账本")
        except Exception as e:
            logger.error(f"保存统一策略状态失败: {e}")

    def _load_legacy_state(self) -> Optional[RotationState]:
        if not self.legacy_state_path.exists():
            return None
        try:
            with open(self.legacy_state_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            logger.info("检测到 ETF 旧账本，准备迁移: 持仓=%s", data.get("current_holding", "无"))
            return RotationState.from_dict(data)
        except Exception as exc:
            logger.error("读取 ETF 旧账本失败: %s", exc)
            return None

    def _delete_legacy_state(self) -> None:
        try:
            if self.legacy_state_path.exists():
                self.legacy_state_path.unlink()
        except Exception as exc:
            logger.warning("删除 ETF 旧账本失败: %s", exc)

    def _state_from_budget_record(self, record: StrategyBudgetState) -> RotationState:
        runtime = dict(getattr(record, "runtime_state", {}) or {})
        positions = record.get_positions()
        current_holding = normalize_symbol_code(str(runtime.get("current_holding", "") or ""))
        current_position = positions.get(current_holding) if current_holding else None
        if current_position is None:
            current_position = next(
                (pos for pos in positions.values() if int(getattr(pos, "quantity", 0) or 0) > 0),
                None,
            )
            current_holding = current_position.symbol_code if current_position is not None else None
        buy_quantity = int(getattr(current_position, "quantity", 0) or 0) if current_position else 0
        buy_price = float(getattr(current_position, "avg_cost", 0.0) or 0.0) if current_position else 0.0
        total_pnl = float(runtime.get("total_pnl", getattr(record, "realized_pnl", 0.0)) or 0.0)
        return RotationState(
            current_holding=current_holding or None,
            current_holding_name=str(runtime.get("current_holding_name", "") or ""),
            current_score=float(runtime.get("current_score", 0.0) or 0.0),
            buy_price=buy_price,
            buy_date=str(runtime.get("buy_date", "") or ""),
            buy_quantity=buy_quantity,
            last_check_date=str(runtime.get("last_check_date", "") or ""),
            last_check_time=str(runtime.get("last_check_time", "") or ""),
            last_signal=str(runtime.get("last_signal", "") or ""),
            trades_today=int(runtime.get("trades_today", 0) or 0),
            trades_today_date=str(runtime.get("trades_today_date", "") or ""),
            total_invested=float(runtime.get("total_invested", 0.0) or 0.0),
            total_pnl=total_pnl,
            holding_high_price=float(runtime.get("holding_high_price", 0.0) or 0.0),
            account_peak=float(runtime.get("account_peak", 0.0) or 0.0),
            cooldown_remaining=int(runtime.get("cooldown_remaining", 0) or 0),
            cooldown_last_decrement_date=str(runtime.get("cooldown_last_decrement_date", "") or ""),
            check_count=int(runtime.get("check_count", 0) or 0),
            auto_data_task_date=str(runtime.get("auto_data_task_date", "") or ""),
            auto_data_task_schedule_time=str(runtime.get("auto_data_task_schedule_time", "") or ""),
            auto_data_task_trigger=str(runtime.get("auto_data_task_trigger", "") or ""),
            auto_data_task_status=str(runtime.get("auto_data_task_status", "") or ""),
            auto_data_task_started_at=str(runtime.get("auto_data_task_started_at", "") or ""),
            auto_data_task_completed_at=str(runtime.get("auto_data_task_completed_at", "") or ""),
            auto_data_task_error=str(runtime.get("auto_data_task_error", "") or ""),
            auto_signal_task_date=str(runtime.get("auto_signal_task_date", "") or ""),
            auto_signal_task_schedule_time=str(runtime.get("auto_signal_task_schedule_time", "") or ""),
            auto_signal_task_trigger=str(runtime.get("auto_signal_task_trigger", "") or ""),
            auto_signal_task_status=str(runtime.get("auto_signal_task_status", "") or ""),
            auto_signal_task_started_at=str(runtime.get("auto_signal_task_started_at", "") or ""),
            auto_signal_task_completed_at=str(runtime.get("auto_signal_task_completed_at", "") or ""),
            auto_signal_task_error=str(runtime.get("auto_signal_task_error", "") or ""),
            dedicated_cash=round(float(getattr(record, "cash_balance", 0.0) or 0.0), 2),
            last_scores=dict(runtime.get("last_scores", {}) or {}),
            trade_history=list(getattr(record, "trade_history", []) or []),
            capital_ledger=list(getattr(record, "capital_ledger", []) or []),
            daily_equity=dict(getattr(record, "daily_equity", {}) or {}),
            order_records=list(getattr(record, "order_records", []) or []),
        )

    def _repair_suspicious_startup_cash_adjustments(
        self,
        state: RotationState,
        record: StrategyBudgetState,
    ) -> bool:
        entries = [CapitalLedgerEntry.from_dict(item) for item in list(state.capital_ledger or [])]
        if len(entries) < 2:
            return False

        capital_limit = float(getattr(record, "capital_limit", 0.0) or 0.0)
        position_cost = max(float(state.buy_price or 0.0) * int(state.buy_quantity or 0), 0.0)
        suspicious_threshold = max(capital_limit * 0.5, position_cost * 0.5, 1000.0)

        repaired = False
        running_balance = round(float(entries[0].balance or 0.0), 2)
        repaired_entries: List[CapitalLedgerEntry] = [entries[0]]
        suspicious_total = 0.0

        for entry in entries[1:]:
            amount = round(float(entry.amount or 0.0), 2)
            action = str(entry.action or "")
            suspicious = "对账校准(startup)" in action and amount > suspicious_threshold
            effective_amount = 0.0 if suspicious else amount
            expected_balance = round(running_balance + effective_amount, 2)

            if suspicious:
                suspicious_total = round(suspicious_total + amount, 2)
                repaired = True
            if abs(float(entry.balance or 0.0) - expected_balance) > 0.01:
                repaired = True

            repaired_entries.append(
                CapitalLedgerEntry(
                    date=entry.date,
                    time=entry.time,
                    action=entry.action,
                    code=entry.code,
                    name=entry.name,
                    amount=effective_amount,
                    commission=float(entry.commission or 0.0),
                    balance=expected_balance,
                    fee_source=entry.fee_source,
                )
            )
            running_balance = expected_balance

        if not repaired:
            return False

        state.capital_ledger = [item.to_dict() for item in repaired_entries]
        state.dedicated_cash = round(running_balance, 2)
        logger.warning(
            "检测并修复 ETF 专用资金账本异常: rolled_back=%.2f repaired_cash=%.2f",
            suspicious_total,
            state.dedicated_cash,
        )
        return True

    def _budget_record_from_state(self, state: RotationState) -> StrategyBudgetState:
        record = self.budget_service.get_strategy_state_record(
            self.strategy_id,
            strategy_name=self.strategy_name,
            virtual_account_id=self.virtual_account_id,
            real_total_asset=0.0,
        )
        record.strategy_name = self.strategy_name
        record.virtual_account_id = self.virtual_account_id

        # ── 主账本字段由 commit_buy / commit_sell 维护，此处不再覆盖 ──
        # 仅对老数据（legacy rotation_state.json 迁移 / 旧版 ETF 引擎残留）做一次性 seed：
        #   1) cash_balance 还没被主账本记过（==0），但旧 state.dedicated_cash 有值
        #   2) realized_pnl 还没记过（==0），但旧 state.total_pnl 有值
        #   3) positions 为空，但旧 state.current_holding 有持仓
        legacy_cash = round(float(state.dedicated_cash or 0.0), 2)
        if round(float(record.cash_balance or 0.0), 2) == 0.0 and legacy_cash > 0.0:
            record.cash_balance = legacy_cash
        legacy_pnl = round(float(state.total_pnl or 0.0), 2)
        if round(float(record.realized_pnl or 0.0), 2) == 0.0 and legacy_pnl != 0.0:
            record.realized_pnl = legacy_pnl
        if (
            not record.get_positions()
            and state.current_holding
            and int(state.buy_quantity or 0) > 0
        ):
            code = normalize_symbol_code(state.current_holding)
            record.positions[code] = {
                "symbol_code": code,
                "quantity": int(state.buy_quantity or 0),
                "avg_cost": round(float(state.buy_price or 0.0), 4),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        record.runtime_state = {
            "current_holding": normalize_symbol_code(state.current_holding or "") if state.current_holding else "",
            "current_holding_name": state.current_holding_name,
            "current_score": float(state.current_score or 0.0),
            "buy_date": state.buy_date,
            "last_check_date": state.last_check_date,
            "last_check_time": state.last_check_time,
            "last_signal": state.last_signal,
            "trades_today": int(state.trades_today or 0),
            "trades_today_date": state.trades_today_date,
            "total_invested": round(float(state.total_invested or 0.0), 2),
            "total_pnl": round(float(state.total_pnl or 0.0), 2),
            "holding_high_price": round(float(state.holding_high_price or 0.0), 4),
            "account_peak": round(float(state.account_peak or 0.0), 2),
            "cooldown_remaining": int(state.cooldown_remaining or 0),
            "cooldown_last_decrement_date": state.cooldown_last_decrement_date,
            "check_count": int(state.check_count or 0),
            "auto_data_task_date": state.auto_data_task_date,
            "auto_data_task_schedule_time": state.auto_data_task_schedule_time,
            "auto_data_task_trigger": state.auto_data_task_trigger,
            "auto_data_task_status": state.auto_data_task_status,
            "auto_data_task_started_at": state.auto_data_task_started_at,
            "auto_data_task_completed_at": state.auto_data_task_completed_at,
            "auto_data_task_error": state.auto_data_task_error,
            "auto_signal_task_date": state.auto_signal_task_date,
            "auto_signal_task_schedule_time": state.auto_signal_task_schedule_time,
            "auto_signal_task_trigger": state.auto_signal_task_trigger,
            "auto_signal_task_status": state.auto_signal_task_status,
            "auto_signal_task_started_at": state.auto_signal_task_started_at,
            "auto_signal_task_completed_at": state.auto_signal_task_completed_at,
            "auto_signal_task_error": state.auto_signal_task_error,
            "last_scores": dict(state.last_scores or {}),
        }
        record.trade_history = list(state.trade_history or [])[-200:]
        record.capital_ledger = list(state.capital_ledger or [])[-500:]
        record.daily_equity = {
            str(k): round(float(v or 0.0), 2)
            for k, v in (state.daily_equity or {}).items()
            if str(k)
        }
        if len(record.daily_equity) > 730:
            keys = sorted(record.daily_equity.keys())
            for key in keys[:-730]:
                record.daily_equity.pop(key, None)
        record.order_records = list(state.order_records or [])[-200:]
        return record

    def update_holding(self, code: Optional[str], name: str, score: float,
                       price: float, quantity: int):
        """更新持仓信息"""
        s = self.state
        s.current_holding = code
        s.current_holding_name = name
        s.current_score = score
        s.buy_price = price
        s.buy_quantity = quantity
        s.buy_date = datetime.now().strftime("%Y-%m-%d")
        s.holding_high_price = price
        self.save()

    def clear_holding(self):
        """清空持仓"""
        s = self.state
        s.current_holding = None
        s.current_holding_name = ""
        s.current_score = 0.0
        s.buy_price = 0.0
        s.buy_quantity = 0
        s.buy_date = ""
        s.holding_high_price = 0.0
        self.save()

    def update_check_result(self, signal: str, scores: Dict[str, float]):
        """更新检查结果"""
        s = self.state
        now = datetime.now()
        s.last_check_date = now.strftime("%Y-%m-%d")
        s.last_check_time = now.strftime("%H:%M:%S")
        s.last_signal = signal
        s.last_scores = scores
        self.save()

    def mark_auto_data_task(
        self,
        *,
        status: str,
        schedule_time: str = "",
        trigger: str = "",
        task_date: str = "",
        error: str = "",
    ) -> None:
        self._mark_auto_task_state(
            prefix="auto_data_task",
            status=status,
            schedule_time=schedule_time,
            trigger=trigger,
            task_date=task_date,
            error=error,
        )

    def mark_auto_signal_task(
        self,
        *,
        status: str,
        schedule_time: str = "",
        trigger: str = "",
        task_date: str = "",
        error: str = "",
    ) -> None:
        self._mark_auto_task_state(
            prefix="auto_signal_task",
            status=status,
            schedule_time=schedule_time,
            trigger=trigger,
            task_date=task_date,
            error=error,
        )

    def is_auto_data_task_completed(
        self,
        *,
        task_date: str,
        schedule_time: str = "",
        trigger: str = "scheduled",
    ) -> bool:
        return self._is_auto_task_completed(
            prefix="auto_data_task",
            task_date=task_date,
            schedule_time=schedule_time,
            trigger=trigger,
        )

    def is_auto_signal_task_completed(
        self,
        *,
        task_date: str,
        schedule_time: str = "",
        trigger: str = "scheduled",
    ) -> bool:
        return self._is_auto_task_completed(
            prefix="auto_signal_task",
            task_date=task_date,
            schedule_time=schedule_time,
            trigger=trigger,
        )

    def _mark_auto_task_state(
        self,
        *,
        prefix: str,
        status: str,
        schedule_time: str = "",
        trigger: str = "",
        task_date: str = "",
        error: str = "",
    ) -> None:
        s = self.state
        now = datetime.now()
        resolved_date = str(task_date or now.strftime("%Y-%m-%d"))
        setattr(s, f"{prefix}_date", resolved_date)
        if schedule_time:
            setattr(s, f"{prefix}_schedule_time", str(schedule_time or ""))
        if trigger:
            setattr(s, f"{prefix}_trigger", str(trigger or ""))
        setattr(s, f"{prefix}_status", str(status or ""))
        if status == "running":
            setattr(s, f"{prefix}_started_at", now.strftime("%Y-%m-%d %H:%M:%S"))
            setattr(s, f"{prefix}_completed_at", "")
            setattr(s, f"{prefix}_error", "")
        else:
            if not getattr(s, f"{prefix}_started_at", ""):
                setattr(s, f"{prefix}_started_at", now.strftime("%Y-%m-%d %H:%M:%S"))
            setattr(s, f"{prefix}_completed_at", now.strftime("%Y-%m-%d %H:%M:%S"))
            setattr(s, f"{prefix}_error", str(error or ""))
        self.save()

    def _is_auto_task_completed(
        self,
        *,
        prefix: str,
        task_date: str,
        schedule_time: str = "",
        trigger: str = "scheduled",
    ) -> bool:
        s = self.state
        if str(getattr(s, f"{prefix}_date", "") or "") != str(task_date or ""):
            return False
        if str(getattr(s, f"{prefix}_status", "") or "") != "completed":
            return False
        if schedule_time and str(getattr(s, f"{prefix}_schedule_time", "") or "") != str(schedule_time):
            return False
        if trigger and str(getattr(s, f"{prefix}_trigger", "") or "") != str(trigger):
            return False
        return True

    # ------------------------------------------------------------------
    #  扩展分析数据辅助方法
    # ------------------------------------------------------------------

    def add_capital_entry(self, entry: 'CapitalLedgerEntry'):
        """追加资金流水条目，最多保留 500 条"""
        self.state.capital_ledger.append(entry.to_dict())
        if len(self.state.capital_ledger) > 500:
            self.state.capital_ledger = self.state.capital_ledger[-500:]
        self.save()

    def add_order_record(self, record: 'OrderRecord'):
        """新增或更新委托记录（同一 order_id 会覆盖旧记录）"""
        self.state.order_records = [
            r for r in self.state.order_records
            if r.get('order_id') != record.order_id
        ]
        self.state.order_records.append(record.to_dict())
        if len(self.state.order_records) > 200:
            self.state.order_records = self.state.order_records[-200:]
        self.save()

    def update_order_record(self, order_id: int, **kwargs):
        """按字段更新指定 order_id 的委托记录"""
        for r in self.state.order_records:
            if r.get('order_id') == order_id:
                r.update(kwargs)
                break
        self.save()

    def record_daily_equity(self, equity: float):
        """记录当日净值快照，最多保留 730 天"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.state.daily_equity[today] = round(equity, 2)
        if len(self.state.daily_equity) > 730:
            keys = sorted(self.state.daily_equity.keys())
            for k in keys[:-730]:
                del self.state.daily_equity[k]
        self.save()
