"""
ETF rotation execution context.

The live ETF rotation path submits orders through TradeExecutionService only.
This module only provides read-only broker/market context and an explicit local
simulation stub for tests or manual debugging.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Callable

from common.broker_session_service import BrokerSessionService, get_broker_session_service
from trading_app.services.market_data.market_data_gateway import get_market_data_gateway, to_xt_code as gateway_to_xt_code

logger = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    price: float
    source: str
    tick_time: Optional[datetime] = None
    age_seconds: Optional[float] = None
    is_fresh: bool = False
    message: str = ""


def to_xt_code(code: str) -> str:
    """6位代码 → xtquant 格式"""
    return gateway_to_xt_code(code)


class TradeExecutor(ABC):
    """Read-only execution context abstraction for ETF rotation."""

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def get_current_price(self, code: str) -> float:
        """获取当前价格"""
        ...

    def get_current_price_snapshot(
        self,
        code: str,
        *,
        allow_daily_fallback: bool = True,
        require_fresh: bool = True,
    ) -> PriceSnapshot:
        price = float(self.get_current_price(code) or 0.0)
        return PriceSnapshot(
            price=price,
            source="executor",
            is_fresh=price > 0,
            message="执行器价格" if price > 0 else "执行器未返回有效价格",
        )

    @abstractmethod
    def query_position(self, code: str) -> Tuple[int, float]:
        """
        查询总持仓

        Returns:
            (持仓数量, 成本价)
        """
        ...

    @abstractmethod
    def query_sellable_position(self, code: str) -> Tuple[int, float]:
        """
        查询可卖持仓

        Returns:
            (可卖数量, 成本价)
        """
        ...

    def query_order_fill(self, order_id: int,
                         timeout_secs: float = 5.0) -> dict:
        """
        轮询查询订单实际成交情况（可选实现，子类覆盖以获得准确数据）。

        Returns:
            {
              'filled'      : bool   # 是否全量成交
              'filled_qty'  : int    # 实际成交数量（0 表示未知）
              'filled_price': float  # 实际成交均价（0 表示未知）
              'commission'  : float  # 实际佣金（-1 表示不可用，调用方应估算）
              'timed_out'   : bool   # 是否查询超时
            }
        """
        return {
            'filled': True, 'filled_qty': 0, 'filled_price': 0.0,
            'commission': -1.0, 'timed_out': False,
        }


class BrokerReadOnlyExecutor(TradeExecutor):
    """
    Read-only broker context for ETF rotation.

    It provides market price, position, asset, order and deal queries. All live
    order submissions must still go through TradeExecutionService.
    """

    def __init__(self):
        self._broker_session_service: Optional[BrokerSessionService] = None
        self._xt_trader = None
        self._acc = None
        self._get_price_func: Optional[Callable] = None

    def set_broker_session_service(self, broker_session_service: Optional[BrokerSessionService] = None):
        self._broker_session_service = broker_session_service or get_broker_session_service()
        logger.info("BrokerReadOnlyExecutor: 已绑定共享 BrokerSessionService")

    def set_broker(self, xt_trader, acc):
        """注入券商连接对象"""
        self._xt_trader = xt_trader
        self._acc = acc
        logger.info("BrokerReadOnlyExecutor: 券商连接已注入")

    def set_price_func(self, func: Callable):
        """注入实时价格获取函数: func(code) -> float"""
        self._get_price_func = func

    def is_connected(self) -> bool:
        if self._broker_session_service is not None:
            return self._broker_session_service.is_connected
        return self._xt_trader is not None and self._acc is not None

    def _get_trader_and_account(self):
        if self._broker_session_service is not None:
            return (
                self._broker_session_service.xt_trader,
                self._broker_session_service.account_obj,
            )
        return self._xt_trader, self._acc

    def get_current_price(self, code: str) -> float:
        snapshot = self.get_current_price_snapshot(
            code,
            allow_daily_fallback=True,
            require_fresh=True,
        )
        return snapshot.price if snapshot.price > 0 else 0.0

    def get_current_price_snapshot(
        self,
        code: str,
        *,
        allow_daily_fallback: bool = True,
        require_fresh: bool = True,
    ) -> PriceSnapshot:
        """
        获取价格快照：
          1. 外部注入价格函数
          2. get_full_tick 实时 tick
          3. 非交易时段才允许 get_market_data 日线收盘价兜底
        """
        if self._get_price_func:
            try:
                price = float(self._get_price_func(code) or 0.0)
            except Exception as exc:
                logger.warning("外部价格函数获取 %s 失败: %s", code, exc)
                price = 0.0
            if price > 0:
                return PriceSnapshot(price=price, source="external", is_fresh=True, message="外部实时价格")
            return PriceSnapshot(price=0.0, source="external", is_fresh=False, message="外部价格函数返回无效价格")

        try:
            snapshot = get_market_data_gateway().get_price_snapshot(
                code,
                allow_daily_fallback=allow_daily_fallback,
                require_fresh=require_fresh,
            )
            return PriceSnapshot(
                price=snapshot.price,
                source=snapshot.source,
                tick_time=snapshot.source_time,
                age_seconds=snapshot.age_seconds,
                is_fresh=snapshot.is_fresh,
                message=snapshot.message,
            )
        except Exception as exc:
            logger.error("获取价格失败 %s: %s", code, exc)
        return PriceSnapshot(price=0.0, source="none", is_fresh=False, message="无法获取有效价格")

    def query_position(self, code: str) -> Tuple[int, float]:
        if not self.is_connected():
            return 0, 0.0

        try:
            if self._broker_session_service is not None:
                positions = self._broker_session_service.query_stock_positions()
            else:
                positions = self._xt_trader.query_stock_positions(self._acc)
            xt_code = to_xt_code(code)
            for pos in (positions or []):
                if pos.stock_code == xt_code:
                    return int(getattr(pos, "volume", 0) or 0), float(getattr(pos, "open_price", 0) or 0.0)
        except Exception as e:
            logger.error(f"查询持仓异常: {e}")
        return 0, 0.0

    def query_sellable_position(self, code: str) -> Tuple[int, float]:
        if not self.is_connected():
            return 0, 0.0

        try:
            if self._broker_session_service is not None:
                positions = self._broker_session_service.query_stock_positions()
            else:
                positions = self._xt_trader.query_stock_positions(self._acc)
            xt_code = to_xt_code(code)
            for pos in (positions or []):
                if pos.stock_code == xt_code:
                    return int(getattr(pos, "can_use_volume", 0) or 0), float(getattr(pos, "open_price", 0) or 0.0)
        except Exception as e:
            logger.error(f"查询可卖持仓异常: {e}")
        return 0, 0.0

    def query_available_cash(self) -> float:
        if not self.is_connected():
            return 0.0
        try:
            assets = (
                self._broker_session_service.query_stock_asset()
                if self._broker_session_service is not None
                else self._xt_trader.query_stock_asset(self._acc)
            )
            return float(getattr(assets, "cash", 0.0) or 0.0)
        except Exception as exc:
            logger.error(f"查询可用资金失败: {exc}")
            return 0.0

    def query_total_asset(self) -> float:
        if not self.is_connected():
            return 0.0
        try:
            assets = (
                self._broker_session_service.query_stock_asset()
                if self._broker_session_service is not None
                else self._xt_trader.query_stock_asset(self._acc)
            )
            return float(getattr(assets, "total_asset", 0.0) or 0.0)
        except Exception as exc:
            logger.error(f"查询总资产失败: {exc}")
            return 0.0

    def query_order_fill(self, order_id: int,
                         timeout_secs: float = 5.0) -> dict:
        """
        轮询 miniQMT 查询委托的实际成交情况。
        在工作线程中调用（rotation_engine 的 _confirm_fill 会将此方法
        放到 daemon 线程执行，并通过 QEventLoop 保持 UI 响应）。
        """
        import time

        # 全成(56) / 部撤(53) / 已撤(54) / 废单(57) 均为终态
        TERMINAL = {53, 54, 56, 57}

        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            try:
                order = self._get_order_by_id(order_id)
                if order is not None:
                    traded_qty = int(getattr(order, 'traded_volume', 0) or 0)
                    status = int(getattr(order, 'order_status', -1) or -1)

                    # 终态，或部成(55)且已有成交量 → 可以读结果了
                    if status in TERMINAL or (status == 55 and traded_qty > 0):
                        traded_price = float(
                            getattr(order, 'traded_price', 0) or 0
                        )
                        if traded_price <= 0 and traded_qty > 0:
                            traded_price = float(
                                getattr(order, 'price', 0) or 0
                            )
                        commission = self._query_commission(order_id)
                        return {
                            'filled': status == 56,
                            'filled_qty': traded_qty,
                            'filled_price': traded_price,
                            'commission': commission,
                            'timed_out': False,
                        }
            except Exception as e:
                logger.error(f"query_order_fill 轮询异常: {e}")

            time.sleep(0.5)

        # 超时：尝试最后一次读取，返回已知成交量
        try:
            order = self._get_order_by_id(order_id)
            if order:
                traded_qty = int(getattr(order, 'traded_volume', 0) or 0)
                traded_price = float(getattr(order, 'traded_price', 0) or 0)
                commission = self._query_commission(order_id) if traded_qty > 0 else -1.0
                return {
                    'filled': False,
                    'filled_qty': traded_qty,
                    'filled_price': traded_price,
                    'commission': commission,
                    'timed_out': True,
                }
        except Exception:
            pass

        return {
            'filled': False, 'filled_qty': 0, 'filled_price': 0.0,
            'commission': -1.0, 'timed_out': True,
        }

    def _get_order_by_id(self, order_id: int):
        """查询指定委托，兼容不同版本的 xtquant API"""
        if not self.is_connected():
            return None
        try:
            # 优先使用单条查询（部分版本支持）
            if self._broker_session_service is not None:
                return self._broker_session_service.query_stock_order(order_id)
            if hasattr(self._xt_trader, 'query_stock_order'):
                return self._xt_trader.query_stock_order(self._acc, order_id)
            # 回退：查全部委托后过滤
            orders = self._xt_trader.query_stock_orders(self._acc) or []
            for o in orders:
                if getattr(o, 'order_id', None) == order_id:
                    return o
        except Exception as e:
            logger.error(f"_get_order_by_id({order_id}) 异常: {e}")
        return None

    def _query_commission(self, order_id: int) -> float:
        """
        从成交明细获取指定委托的实际佣金合计。
        无法获取时返回 -1.0（调用方回退到估算）。
        """
        try:
            deals = None
            if self._broker_session_service is not None:
                deals = self._broker_session_service.query_stock_deals()
            for method in ('query_stock_deal', 'query_stock_deals'):
                if deals is not None:
                    break
                if hasattr(self._xt_trader, method):
                    deals = getattr(self._xt_trader, method)(self._acc)
                    break
            if deals:
                total = sum(
                    float(getattr(d, 'commission', 0) or 0)
                    for d in deals
                    if getattr(d, 'order_id', None) == order_id
                )
                return total if total > 0 else -1.0
        except Exception as e:
            logger.error(f"_query_commission({order_id}) 异常: {e}")
        return -1.0


class SimulatedExecutor(TradeExecutor):
    """
    Local simulation context for tests and manual debugging.

    It never submits live orders and must be injected explicitly.
    """

    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: dict = {}  # code -> {quantity, avg_price}
        self._prices: dict = {}   # code -> price
        self._order_seq = 0
        self._connected = True

    def set_prices(self, prices: dict):
        """设置模拟价格 {code: price}"""
        self._prices.update(prices)

    def is_connected(self) -> bool:
        return self._connected

    def get_current_price(self, code: str) -> float:
        return self._prices.get(code, 0.0)

    def query_position(self, code: str) -> Tuple[int, float]:
        pos = self.positions.get(code)
        if pos:
            return pos['quantity'], pos['avg_price']
        return 0, 0.0

    def query_sellable_position(self, code: str) -> Tuple[int, float]:
        return self.query_position(code)


XtQuantExecutor = BrokerReadOnlyExecutor
