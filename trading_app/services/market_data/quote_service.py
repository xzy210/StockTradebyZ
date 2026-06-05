# quote_service.py - 实时行情推送服务
"""
基于 xtquant 的实时行情服务

功能：
- 订阅/取消订阅股票实时行情
- 定时轮询获取最新行情快照
- 支持批量订阅
- PyQt 信号推送

采用轮询模式（推荐）：
- 使用 QTimer 定期调用 get_full_tick() 获取快照
- 更稳定，与 PyQt GUI 更好集成
- 适合 UI 显示场景（1-3秒刷新足够）

参考文档: https://dict.thinktrader.net/nativeApi/xtdata.html
"""
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from trading_app.services.market_data.market_data_gateway import get_market_data_gateway, to_xt_code as gateway_to_xt_code
from trading_app.services.market_data.market_data_policy import evaluate_tick_freshness

# 设置日志
logger = logging.getLogger(__name__)

# 检查 xtquant 是否可用
def _has_xtquant() -> bool:
    return get_market_data_gateway().is_available()


@dataclass
class QuoteData:
    """实时行情数据结构"""
    code: str                    # 股票代码 (如 "000001.SZ")
    name: str = ""               # 股票名称
    last_price: float = 0.0      # 最新价
    open_price: float = 0.0      # 开盘价
    high_price: float = 0.0      # 最高价
    low_price: float = 0.0       # 最低价
    prev_close: float = 0.0      # 昨收价
    volume: int = 0              # 成交量（股）
    amount: float = 0.0          # 成交额
    bid_prices: List[float] = field(default_factory=lambda: [0.0] * 5)   # 买1-买5价格
    bid_volumes: List[int] = field(default_factory=lambda: [0] * 5)     # 买1-买5量
    ask_prices: List[float] = field(default_factory=lambda: [0.0] * 5)   # 卖1-卖5价格
    ask_volumes: List[int] = field(default_factory=lambda: [0] * 5)     # 卖1-卖5量
    timestamp: datetime = field(default_factory=datetime.now)           # 兼容旧接口：本地接收时间戳
    source_time: Optional[datetime] = None                              # tick 原始时间戳
    received_time: datetime = field(default_factory=datetime.now)       # 本地接收时间
    age_seconds: Optional[float] = None                                 # tick 距当前秒数
    is_fresh: bool = False                                              # tick 是否新鲜
    source: str = "xtdata.get_full_tick"                                # 行情来源
    
    @property
    def change(self) -> float:
        """涨跌额"""
        if self.prev_close > 0:
            return self.last_price - self.prev_close
        return 0.0
    
    @property
    def change_pct(self) -> float:
        """涨跌幅（百分比）"""
        if self.prev_close > 0:
            return (self.last_price - self.prev_close) / self.prev_close * 100
        return 0.0
    
    @property
    def is_up(self) -> bool:
        """是否上涨"""
        return self.change > 0
    
    @property
    def simple_code(self) -> str:
        """返回6位简单代码"""
        return self.code.split('.')[0] if '.' in self.code else self.code
    
    def to_dict(self) -> dict:
        """转换为字典格式（兼容旧接口）"""
        return {
            'lastPrice': self.last_price,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'lastClose': self.prev_close,
            'volume': self.volume,
            'amount': self.amount,
            'bidPrice': self.bid_prices,
            'bidVol': self.bid_volumes,
            'askPrice': self.ask_prices,
            'askVol': self.ask_volumes,
            'sourceTime': self.source_time.strftime("%Y-%m-%d %H:%M:%S") if self.source_time else "",
            'receivedTime': self.received_time.strftime("%Y-%m-%d %H:%M:%S"),
            'ageSeconds': self.age_seconds,
            'isFresh': self.is_fresh,
            'source': self.source,
        }


def to_xt_code(code: str, is_index: bool = False) -> str:
    """
    将6位股票/指数代码转换为 xtquant 格式
    
    Args:
        code: 6位股票/指数代码或已带后缀的代码
        is_index: 是否为指数代码
    
    Returns:
        xtquant 格式代码，如 "000001.SZ"（股票）或 "000001.SH"（上证指数）
    """
    return gateway_to_xt_code(code, is_index=is_index)


def from_xt_code(xt_code: str) -> str:
    """
    将 xtquant 格式代码转换为6位代码
    
    Args:
        xt_code: xtquant 格式代码，如 "000001.SZ"
    
    Returns:
        6位股票代码
    """
    return xt_code.split('.')[0] if '.' in xt_code else xt_code


class QuoteService(QObject):
    """
    实时行情服务（轮询模式）
    
    使用 QTimer 定时调用 xtdata.get_full_tick() 获取行情快照，
    通过 PyQt 信号发送给订阅者。
    
    信号：
        quote_updated: 单股行情更新信号，参数为 QuoteData
        quotes_batch_updated: 批量行情更新信号，参数为 Dict[str, QuoteData]
        connection_status_changed: 连接状态变化信号，参数为 (bool, str)
    
    使用示例：
        service = QuoteService()
        service.quote_updated.connect(on_quote_update)
        service.subscribe(['000001', '600000'])
    """
    
    # PyQt 信号
    quote_updated = pyqtSignal(object)  # QuoteData
    quotes_batch_updated = pyqtSignal(dict)  # Dict[str, QuoteData]
    connection_status_changed = pyqtSignal(bool, str)  # connected, message
    
    def __init__(self, parent=None, poll_interval: int = 1000):
        """
        初始化行情服务
        
        Args:
            parent: 父对象
            poll_interval: 轮询间隔（毫秒），默认1000ms
        """
        super().__init__(parent)
        
        self._subscribed_codes: Set[str] = set()  # 合并后的有效代码（xtquant格式）
        self._subscriptions_by_owner: Dict[str, Set[str]] = {}
        self._quote_cache: Dict[str, QuoteData] = {}  # 行情缓存
        self._is_running = False
        self._poll_interval = max(1000, poll_interval)
        
        # 轮询定时器
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_quotes)
        
        # 数据更新锁
        self._lock = threading.Lock()
        
    @property
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return _has_xtquant()
    
    @property
    def is_running(self) -> bool:
        """检查服务是否正在运行"""
        return self._is_running
    
    @property
    def subscribed_count(self) -> int:
        """返回已订阅的股票数量"""
        return len(self._subscribed_codes)

    def get_effective_codes(self) -> Set[str]:
        """返回所有 owner 合并后的有效订阅代码。"""
        with self._lock:
            return set(self._subscribed_codes)

    def get_subscription(self, owner_id: str) -> Set[str]:
        """返回指定 owner 的订阅集合。"""
        with self._lock:
            return set(self._subscriptions_by_owner.get(owner_id, set()))

    def list_owner_ids(self) -> List[str]:
        """返回当前所有 owner 标识。"""
        with self._lock:
            return list(self._subscriptions_by_owner.keys())
    
    @property
    def poll_interval(self) -> int:
        """获取轮询间隔"""
        return self._poll_interval
    
    def set_poll_interval(self, interval_ms: int):
        """
        设置轮询间隔
        
        Args:
            interval_ms: 轮询间隔（毫秒），最小1000ms
        """
        self._poll_interval = max(1000, interval_ms)
        if self._poll_timer.isActive():
            self._poll_timer.setInterval(self._poll_interval)
        logger.info(f"轮询间隔已设置为 {self._poll_interval}ms")
    
    def start(self) -> bool:
        """
        启动行情服务
        
        Returns:
            是否成功启动
        """
        if not self.is_available:
            logger.error("xtquant 未安装，无法启动行情服务")
            self.connection_status_changed.emit(False, "xtquant 未安装")
            return False
        
        if self._is_running:
            logger.debug("行情服务已在运行中")
            return True
        
        try:
            ok, message = get_market_data_gateway().check_connection()
            if not ok:
                logger.warning(message)
                self.connection_status_changed.emit(False, message)
                return False
            
            self._is_running = True
            self._poll_timer.start(self._poll_interval)
            
            logger.info(f"实时行情服务已启动 (轮询间隔: {self._poll_interval}ms)")
            self.connection_status_changed.emit(True, "行情服务已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动行情服务失败: {e}")
            self.connection_status_changed.emit(False, f"启动失败: {e}")
            return False
    
    def stop(self):
        """停止行情服务"""
        if not self._is_running:
            return
        
        try:
            self._poll_timer.stop()
            
            with self._lock:
                self._subscribed_codes.clear()
                self._subscriptions_by_owner.clear()
                self._quote_cache.clear()
            
            self._is_running = False
            logger.info("实时行情服务已停止")
            self.connection_status_changed.emit(False, "行情服务已停止")
            
        except Exception as e:
            logger.error(f"停止行情服务失败: {e}")
    
    def subscribe(self, codes: List[str], start_service: bool = True, is_index: bool = False) -> bool:
        """
        订阅股票/指数行情
        
        Args:
            codes: 股票/指数代码列表（支持6位代码或xtquant格式）
            start_service: 是否自动启动服务
            is_index: 是否为指数代码
        
        Returns:
            是否成功
        """
        return self.subscribe_owner(
            "_legacy",
            codes,
            start_service=start_service,
            is_index=is_index,
        )
    
    def unsubscribe(self, codes: List[str]) -> bool:
        """
        取消订阅
        
        Args:
            codes: 股票代码列表
        """
        return self.unsubscribe_owner("_legacy", codes)
    
    def unsubscribe_all(self):
        """取消所有订阅"""
        self.clear_owner_subscription("_legacy")

    def subscribe_owner(
        self,
        owner_id: str,
        codes: List[str],
        *,
        start_service: bool = True,
        is_index: bool = False,
    ) -> bool:
        """为指定 owner 增量添加订阅。"""
        current_codes = self.get_subscription(owner_id)
        merged = list(current_codes | {to_xt_code(code, is_index=is_index) for code in codes})
        return self.replace_subscription(
            owner_id,
            merged,
            start_service=start_service,
            is_index=is_index,
        )

    def replace_subscription(
        self,
        owner_id: str,
        codes: List[str],
        *,
        start_service: bool = True,
        is_index: bool = False,
    ) -> bool:
        """原子替换指定 owner 的订阅集合。"""
        if not self.is_available:
            logger.error("xtquant 未安装，无法订阅")
            return False

        if start_service and not self._is_running:
            if not self.start():
                return False

        xt_codes = {
            to_xt_code(code, is_index=is_index)
            for code in (codes or [])
            if code
        }

        with self._lock:
            previous_codes = set(self._subscriptions_by_owner.get(owner_id, set()))
            if xt_codes:
                self._subscriptions_by_owner[owner_id] = set(xt_codes)
            else:
                self._subscriptions_by_owner.pop(owner_id, None)
            self._rebuild_effective_codes_locked()
            new_codes = list(xt_codes - previous_codes)
            removed_codes = previous_codes - xt_codes
            for code in removed_codes:
                if code not in self._subscribed_codes:
                    self._quote_cache.pop(code, None)

        if new_codes:
            logger.info(
                "owner=%s 订阅 %d 只代码: %s%s",
                owner_id,
                len(new_codes),
                new_codes[:3],
                "..." if len(new_codes) > 3 else "",
            )
            self.refresh_quotes(new_codes)

        if removed_codes:
            logger.info("owner=%s 取消订阅 %d 只代码", owner_id, len(removed_codes))
        return True

    def unsubscribe_owner(self, owner_id: str, codes: List[str], *, is_index: bool = False) -> bool:
        """为指定 owner 取消部分订阅。"""
        if not codes:
            return True
        current_codes = self.get_subscription(owner_id)
        xt_codes = {to_xt_code(code, is_index=is_index) for code in codes if code}
        remaining = list(current_codes - xt_codes)
        return self.replace_subscription(owner_id, remaining, start_service=False)

    def clear_owner_subscription(self, owner_id: str):
        """清空指定 owner 的订阅集合。"""
        with self._lock:
            previous_codes = set(self._subscriptions_by_owner.pop(owner_id, set()))
            self._rebuild_effective_codes_locked()
            for code in previous_codes:
                if code not in self._subscribed_codes:
                    self._quote_cache.pop(code, None)
        if previous_codes:
            logger.info("owner=%s 已取消全部订阅 (%d 只)", owner_id, len(previous_codes))

    def refresh_owner(self, owner_id: str):
        """刷新指定 owner 当前关心的代码。"""
        self.refresh_quotes(list(self.get_subscription(owner_id)))

    def _rebuild_effective_codes_locked(self):
        effective_codes: Set[str] = set()
        for owner_codes in self._subscriptions_by_owner.values():
            effective_codes.update(owner_codes)
        self._subscribed_codes = effective_codes
    
    def get_quote(self, code: str) -> Optional[QuoteData]:
        """
        获取指定股票的最新行情
        
        Args:
            code: 股票代码
        
        Returns:
            QuoteData 或 None
        """
        xt_code = to_xt_code(code) if "." not in code else code
        return self._quote_cache.get(xt_code)
    
    def get_all_quotes(self) -> Dict[str, QuoteData]:
        """获取所有已缓存的行情数据"""
        with self._lock:
            return dict(self._quote_cache)
    
    def refresh_quotes(self, codes: List[str] = None):
        """
        主动刷新行情数据
        
        Args:
            codes: 要刷新的代码列表，None 表示刷新所有已订阅的
        """
        if not self.is_available:
            return
        
        try:
            if codes is None:
                with self._lock:
                    xt_codes = list(self._subscribed_codes)
            else:
                xt_codes = [to_xt_code(c) if '.' not in c else c for c in codes]
            
            if not xt_codes:
                return
            
            # 获取快照数据
            tick_data = get_market_data_gateway().get_full_tick(xt_codes)
            
            if tick_data:
                for xt_code, tick in tick_data.items():
                    self._process_tick(xt_code, tick)
                
        except Exception as e:
            logger.error(f"刷新行情失败: {e}")
    
    def _poll_quotes(self):
        """定时轮询获取行情"""
        if not self._subscribed_codes:
            return
        
        try:
            with self._lock:
                xt_codes = list(self._subscribed_codes)
            
            if not xt_codes:
                return
            
            # 分批获取（避免一次请求过多）
            batch_size = 100
            for i in range(0, len(xt_codes), batch_size):
                batch = xt_codes[i:i + batch_size]
                tick_data = get_market_data_gateway().get_full_tick(batch)
                
                if tick_data:
                    for xt_code, tick in tick_data.items():
                        self._process_tick(xt_code, tick)
            
            # 发送批量更新信号
            self._emit_batch_update()
                
        except Exception as e:
            logger.error(f"轮询行情失败: {e}")
    
    def _process_tick(self, xt_code: str, tick: dict):
        """处理单条 tick 数据"""
        if not tick:
            return
        
        try:
            # 安全获取列表数据
            def safe_list(data, default_len=5):
                if isinstance(data, (list, tuple)):
                    return list(data)[:default_len] + [0] * (default_len - len(data))
                return [0] * default_len

            received_time = datetime.now()
            freshness = evaluate_tick_freshness(tick, received_time)
            quote = QuoteData(
                code=xt_code,
                last_price=float(tick.get('lastPrice') or 0),
                open_price=float(tick.get('open') or 0),
                high_price=float(tick.get('high') or 0),
                low_price=float(tick.get('low') or 0),
                prev_close=float(tick.get('lastClose') or 0),
                volume=int(tick.get('volume') or 0),
                amount=float(tick.get('amount') or 0),
                bid_prices=safe_list(tick.get('bidPrice')),
                bid_volumes=[int(v or 0) for v in safe_list(tick.get('bidVol'))],
                ask_prices=safe_list(tick.get('askPrice')),
                ask_volumes=[int(v or 0) for v in safe_list(tick.get('askVol'))],
                timestamp=received_time,
                source_time=freshness.source_time,
                received_time=received_time,
                age_seconds=freshness.age_seconds,
                is_fresh=freshness.is_fresh,
            )
            
            with self._lock:
                self._quote_cache[xt_code] = quote
            
            # 发送单条更新信号
            self.quote_updated.emit(quote)
            
        except Exception as e:
            logger.error(f"处理 tick 数据失败 {xt_code}: {e}")
    
    def _emit_batch_update(self):
        """发送批量行情更新信号"""
        with self._lock:
            if self._quote_cache:
                self.quotes_batch_updated.emit(dict(self._quote_cache))


# 全局单例
_quote_service_instance: Optional[QuoteService] = None


def get_quote_service() -> QuoteService:
    """
    获取全局行情服务实例（单例模式）
    
    Returns:
        QuoteService 实例
    """
    global _quote_service_instance
    if _quote_service_instance is None:
        _quote_service_instance = QuoteService()
    return _quote_service_instance
