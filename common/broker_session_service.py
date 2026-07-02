"""
Shared broker session service for miniQMT / xtquant.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from common.io_utils import atomic_write_json
from common.qmt_client_service import QmtClientService

logger = logging.getLogger(__name__)


try:
    from xtquant.xttrader import XtQuantTraderCallback as _XtCallbackBase
except Exception:
    _XtCallbackBase = object


class _BrokerTradeCallback(_XtCallbackBase):
    """XtQuantTraderCallback adapter that forwards events to a Qt signal emitter.

    xtquant invokes callbacks on its internal thread.  We serialise the data
    into plain dicts and relay them through ``_SignalRelay`` so that Qt
    widgets always receive them on the main thread.
    """

    def __init__(self, relay: "_SignalRelay"):
        if _XtCallbackBase is not object:
            super().__init__()
        self._relay = relay

    @staticmethod
    def _order_to_dict(order: Any) -> dict:
        fields = (
            "account_id", "stock_code", "order_id", "order_sysid",
            "order_time", "order_type", "order_volume", "price_type",
            "price", "traded_volume", "traded_price", "order_status",
            "status_msg", "strategy_name", "order_remark",
        )
        result: dict[str, Any] = {}
        for f in fields:
            val = getattr(order, f, None)
            if val is not None:
                result[f] = val
        return result

    @staticmethod
    def _trade_to_dict(trade: Any) -> dict:
        fields = (
            "account_id", "stock_code", "order_id", "order_sysid",
            "traded_id", "traded_time", "traded_price", "traded_volume",
            "traded_amount", "order_type", "strategy_name", "order_remark",
        )
        result: dict[str, Any] = {}
        for f in fields:
            val = getattr(trade, f, None)
            if val is not None:
                result[f] = val
        return result

    def on_disconnected(self):
        logger.warning("xtquant 回调: 连接断开")
        self._relay.broker_disconnected.emit()

    def on_stock_order(self, order):
        data = self._order_to_dict(order)
        logger.info("xtquant 回调: 委托变更 %s status=%s", data.get("stock_code"), data.get("order_status"))
        self._relay.order_changed.emit(data)

    def on_stock_trade(self, trade):
        data = self._trade_to_dict(trade)
        logger.info(
            "xtquant 回调: 成交 %s price=%s vol=%s",
            data.get("stock_code"), data.get("traded_price"), data.get("traded_volume"),
        )
        self._relay.trade_occurred.emit(data)

    def on_order_error(self, order_error):
        data = {}
        for f in ("account_id", "order_id", "error_id", "error_msg"):
            val = getattr(order_error, f, None)
            if val is not None:
                data[f] = val
        logger.warning("xtquant 回调: 委托错误 %s", data)
        self._relay.order_error.emit(data)

    def on_cancel_error(self, cancel_error):
        logger.warning("xtquant 回调: 撤单错误 %s", getattr(cancel_error, "error_msg", ""))

    def on_order_stock_async_response(self, response):
        pass

    def on_account_status(self, status):
        pass


class _SignalRelay(QObject):
    """Lives on the main thread; exposes pyqtSignals that UI widgets connect to.

    xtquant callbacks emit these signals from their internal thread.
    Because the relay is affined to the main thread, auto-connections
    to main-thread slots are automatically queued by Qt.
    """

    order_changed = pyqtSignal(dict)
    trade_occurred = pyqtSignal(dict)
    order_error = pyqtSignal(dict)
    broker_disconnected = pyqtSignal()


class _BrokerConnectWorker(QThread):
    """Background connector for miniQMT."""

    connected = pyqtSignal(object, object, str, str)  # xt_trader, acc, qmt_path, account
    failed = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, config: dict[str, Any], signal_relay: Optional[_SignalRelay] = None, parent=None):
        super().__init__(parent)
        self.config = dict(config or {})
        self.qmt_path = str(self.config.get("qmt_path", "") or "").strip()
        self.account = str(self.config.get("account", "") or "").strip()
        self.connect_timeout_seconds = max(float(self.config.get("broker_connect_timeout_seconds", 25.0) or 25.0), 5.0)
        self.connect_retry_count = max(int(self.config.get("broker_connect_retry_count", 2) or 2), 1)
        self._cancel_event = threading.Event()
        self._signal_relay = signal_relay

    def cancel(self):
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _log(self, message: str):
        logger.info(message)
        self.log_message.emit(message)

    def run(self):
        try:
            if self._is_cancelled():
                return

            last_error = "连接 QMT 交易端失败"
            for attempt in range(1, self.connect_retry_count + 1):
                if self._is_cancelled():
                    return
                self._log(f"连接券商 ({attempt}/{self.connect_retry_count})")
                ok, result = self._connect_once_with_timeout()
                if self._is_cancelled():
                    if ok:
                        xt_trader, _acc = result
                        try:
                            xt_trader.stop()
                        except Exception:
                            pass
                    return
                if ok:
                    xt_trader, acc = result
                    self._log("券商连接成功")
                    if not self._is_cancelled():
                        self.connected.emit(xt_trader, acc, self.qmt_path, self.account)
                    else:
                        try:
                            xt_trader.stop()
                        except Exception:
                            pass
                    return

                last_error = str(result)
                self._log(last_error)
                if attempt < self.connect_retry_count:
                    self._log("3 秒后重试")
                    waited = 0.0
                    while waited < 3.0:
                        if self._is_cancelled():
                            return
                        time.sleep(0.2)
                        waited += 0.2

            if not self._is_cancelled():
                self.failed.emit(last_error)
        except ImportError:
            if not self._is_cancelled():
                self.failed.emit("未找到 xtquant 库，请确认已安装 miniQMT 并激活对应 Python 环境")
        except Exception as exc:
            if not self._is_cancelled():
                self.failed.emit(f"连接异常: {exc}")

    def _connect_once_with_timeout(self) -> tuple[bool, tuple[object, object] | str]:
        result_box: dict[str, object] = {}
        done = threading.Event()
        abandon = threading.Event()

        def runner():
            xt_trader = None
            try:
                from xtquant import xttrader
                from xtquant.xttype import StockAccount

                if self._is_cancelled():
                    return
                session_id = int(random.randint(100000, 999999))
                self._log(f"创建会话 {session_id}")
                xt_trader = xttrader.XtQuantTrader(self.qmt_path, session_id)
                xt_trader.start()
                self._log("执行 connect()")

                result = xt_trader.connect()
                if abandon.is_set() or self._is_cancelled():
                    try:
                        xt_trader.stop()
                    except Exception:
                        pass
                    return
                if result != 0:
                    result_box["error"] = "连接 QMT 交易端失败，请确认 miniQMT 已启动并登录"
                    return

                if self._signal_relay is not None:
                    self._log("注册交易回调")
                    try:
                        callback = _BrokerTradeCallback(self._signal_relay)
                        xt_trader.register_callback(callback)
                        result_box["callback"] = callback
                    except Exception as exc:
                        self._log(f"注册回调失败（不影响连接）: {exc}")

                self._log("订阅账户")
                acc = StockAccount(self.account)
                res = xt_trader.subscribe(acc)
                if abandon.is_set() or self._is_cancelled():
                    try:
                        xt_trader.stop()
                    except Exception:
                        pass
                    return
                if res != 0:
                    try:
                        xt_trader.stop()
                    except Exception:
                        pass
                    result_box["error"] = f"订阅账户失败（返回码 {res}），请检查资金账号"
                    return

                result_box["trader"] = xt_trader
                result_box["acc"] = acc
            except ImportError:
                result_box["error"] = "未找到 xtquant 库，请确认已安装 miniQMT 并激活对应 Python 环境"
            except Exception as exc:
                if xt_trader is not None:
                    try:
                        xt_trader.stop()
                    except Exception:
                        pass
                result_box["error"] = f"连接异常: {exc}"
            finally:
                done.set()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        if not done.wait(self.connect_timeout_seconds):
            abandon.set()
            return False, f"连接 miniQMT 超时（>{self.connect_timeout_seconds:.0f} 秒），通常是登录后客户端尚未完全就绪"

        trader = result_box.get("trader")
        acc = result_box.get("acc")
        if trader is not None and acc is not None:
            return True, (trader, acc)
        return False, str(result_box.get("error") or "连接 QMT 交易端失败")


class BrokerSessionService(QObject):
    """
    Shared broker session singleton.

    It owns the xtquant trader/account lifecycle and exposes thin query/order APIs.
    """

    connection_changed = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    config_changed = pyqtSignal(dict)
    client_state_changed = pyqtSignal(dict)

    order_changed = pyqtSignal(dict)
    trade_occurred = pyqtSignal(dict)
    order_error = pyqtSignal(dict)
    broker_disconnected = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._xt_trader = None
        self._acc = None
        self._connected = False
        self._connect_worker: Optional[_BrokerConnectWorker] = None
        self._lock = threading.RLock()
        self._connect_token = 0

        self._signal_relay = _SignalRelay(self)
        self._signal_relay.order_changed.connect(self.order_changed)
        self._signal_relay.trade_occurred.connect(self.trade_occurred)
        self._signal_relay.order_error.connect(self.order_error)
        self._signal_relay.broker_disconnected.connect(self.broker_disconnected)

        project_root = Path(__file__).resolve().parent.parent
        self._primary_config_path = project_root / "trading_app" / "config" / "broker_config.json"
        self._fallback_config_paths = [
            project_root / "live_rotation" / "config" / "broker_settings.json",
            project_root / "live_rotation" / "config" / "broker_config.json",
        ]
        self._last_config = self.load_config()
        self._query_timeout_at: dict[str, float] = {}
        self._order_authorization = threading.local()
        self._client_status_cache: Optional[dict[str, Any]] = None
        self._client_status_cache_until = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected and self._xt_trader is not None and self._acc is not None

    @property
    def xt_trader(self):
        return self._xt_trader

    @property
    def account_obj(self):
        return self._acc

    def _emit_log(self, message: str):
        logger.info(message)
        self.log_message.emit(message)

    def _get_cached_client_status(self) -> Optional[dict[str, Any]]:
        if self._client_status_cache is None:
            return None
        if time.time() > float(self._client_status_cache_until or 0.0):
            self._client_status_cache = None
            self._client_status_cache_until = 0.0
            return None
        return dict(self._client_status_cache)

    def _set_cached_client_status(self, status: Optional[dict[str, Any]], *, ttl_seconds: float = 0.0) -> None:
        if not status or ttl_seconds <= 0:
            self._client_status_cache = None
            self._client_status_cache_until = 0.0
            return
        self._client_status_cache = dict(status)
        self._client_status_cache_until = time.time() + max(float(ttl_seconds), 0.1)

    def _build_process_only_client_status(self, *, message: str = "券商已断开") -> dict[str, Any]:
        qmt_path = str(self._last_config.get("qmt_path", "") or "").strip()
        process_ids: list[int] = []
        if qmt_path:
            try:
                process_ids = QmtClientService(self._last_config)._find_process_ids()
            except Exception:
                process_ids = []
        running = bool(process_ids)
        return {
            "running": running,
            "login_window_visible": False,
            "main_window_visible": running,
            "ready": False,
            "process_ids": process_ids,
            "matched_titles": [],
            "message": "miniQMT 进程运行中" if running else message,
        }

    def _normalize_config(self, data: Optional[dict[str, Any]]) -> dict[str, Any]:
        source = dict(data or {})
        return {
            "qmt_path": str(source.get("qmt_path", "") or "").strip(),
            "account": str(source.get("account", "") or "").strip(),
            "process_name": str(source.get("process_name", "") or "").strip(),
            "broker_connect_timeout_seconds": float(source.get("broker_connect_timeout_seconds", 25.0) or 25.0),
            "broker_connect_retry_count": int(source.get("broker_connect_retry_count", 2) or 2),
        }

    def load_config(self) -> dict:
        for path in [self._primary_config_path, *self._fallback_config_paths]:
            if path.exists():
                try:
                    data = json.loads(path.read_text("utf-8"))
                    return self._normalize_config(data)
                except Exception as exc:
                    logger.warning("读取券商配置失败 %s: %s", path, exc)
        return self._normalize_config({})

    def save_config(self, qmt_path: str | dict[str, Any], account: str = "", **extra: Any) -> None:
        if isinstance(qmt_path, dict):
            merged = dict(self._last_config)
            merged.update(qmt_path)
            data = self._normalize_config(merged)
        else:
            merged = dict(self._last_config)
            merged.update(extra)
            merged.update({
                "qmt_path": qmt_path.strip(),
                "account": account.strip(),
            })
            data = self._normalize_config(merged)

        atomic_write_json(self._primary_config_path, data)
        self._last_config = data
        self.config_changed.emit(dict(data))
        self.client_state_changed.emit(self.get_client_status())

    def get_config(self) -> dict:
        self.reload_config()
        return dict(self._last_config)

    def reload_config(self) -> dict:
        latest = self.load_config()
        if latest != self._last_config:
            self._last_config = latest
            self.config_changed.emit(dict(latest))
        return dict(self._last_config)

    def connect_async(self, qmt_path: str, account: str) -> bool:
        self.reload_config()
        qmt_path = qmt_path.strip()
        account = account.strip()
        if not qmt_path or not account:
            self.connection_changed.emit(False, "请先填写 miniQMT 路径和资金账号")
            return False

        if self.is_connected:
            current = self._last_config
            if current.get("qmt_path") == qmt_path and current.get("account") == account:
                self.connection_changed.emit(True, "券商会话已连接")
                return True
            self.disconnect()

        if self._connect_worker and self._connect_worker.isRunning():
            self.connection_changed.emit(False, "券商连接正在进行中")
            return False

        self._emit_log("开始建立共享券商会话...")
        connect_config = dict(self._last_config)
        connect_config.update({
            "qmt_path": qmt_path,
            "account": account,
        })
        self._connect_token += 1
        token = self._connect_token
        self._connect_worker = _BrokerConnectWorker(connect_config, signal_relay=self._signal_relay, parent=self)
        self._connect_worker.connected.connect(
            lambda xt_trader, acc, worker_qmt_path, worker_account, t=token: self._handle_connected(
                t, xt_trader, acc, worker_qmt_path, worker_account
            )
        )
        self._connect_worker.failed.connect(
            lambda message, t=token: self._handle_connect_failed(t, message)
        )
        self._connect_worker.log_message.connect(self.log_message.emit)
        self._connect_worker.start()
        self.connection_changed.emit(False, "正在连接券商...")
        self.client_state_changed.emit(self.get_client_status())
        return True

    def _handle_connected(self, token: int, xt_trader, acc, qmt_path: str, account: str):
        if token != self._connect_token:
            try:
                xt_trader.stop()
            except Exception:
                pass
            return
        self._on_connected(xt_trader, acc, qmt_path, account)

    def _handle_connect_failed(self, token: int, message: str):
        if token != self._connect_token:
            return
        self._on_connect_failed(message)

    def _on_connected(self, xt_trader, acc, qmt_path: str, account: str):
        with self._lock:
            self._xt_trader = xt_trader
            self._acc = acc
            self._connected = True
            self._last_config.update({"qmt_path": qmt_path, "account": account})
            self._connect_worker = None
        self.save_config(dict(self._last_config))
        self.connection_changed.emit(True, "券商连接成功")
        self.client_state_changed.emit(self.get_client_status())

    def _on_connect_failed(self, message: str):
        with self._lock:
            self._xt_trader = None
            self._acc = None
            self._connected = False
            self._connect_worker = None
        self.connection_changed.emit(False, message)
        self.client_state_changed.emit(self.get_client_status())

    def disconnect(self):
        self._connect_token += 1
        worker = self._connect_worker
        self._connect_worker = None
        if worker and worker.isRunning():
            try:
                worker.cancel()
            except Exception:
                pass
        with self._lock:
            trader = self._xt_trader
            self._xt_trader = None
            self._acc = None
            self._connected = False
        if trader is not None:
            try:
                trader.stop()
            except Exception as exc:
                logger.warning("停止券商连接失败: %s", exc)
        light_status = self._build_process_only_client_status(message="券商已断开")
        self._set_cached_client_status(light_status, ttl_seconds=3.0)
        self.connection_changed.emit(False, "券商已断开")
        self.client_state_changed.emit(dict(light_status))

    def get_client_status(self) -> dict:
        self.reload_config()
        cached_status = self._get_cached_client_status()
        if cached_status is not None:
            return cached_status
        worker_running = bool(self._connect_worker is not None and self._connect_worker.isRunning())
        if self.is_connected or worker_running:
            qmt_path = str(self._last_config.get("qmt_path", "") or "").strip()
            process_ids: list[int] = []
            if qmt_path:
                try:
                    process_ids = QmtClientService(self._last_config)._find_process_ids()
                except Exception:
                    process_ids = []
            status = {
                "running": bool(process_ids) or self.is_connected,
                "login_window_visible": False,
                "main_window_visible": bool(process_ids) or self.is_connected,
                "ready": self.is_connected,
                "process_ids": process_ids,
                "matched_titles": [],
                "message": "xtquant 已连接" if self.is_connected else "券商连接进行中",
            }
            self._set_cached_client_status(None)
            return status
        status = QmtClientService(self._last_config).get_status().to_dict()
        self._set_cached_client_status(None)
        return status

    def _require_connected(self):
        if not self.is_connected:
            raise RuntimeError("券商未连接")
        return self._xt_trader, self._acc

    def _query_with_timeout(
        self,
        label: str,
        func: Callable[[], Any],
        *,
        timeout_seconds: float = 8.0,
        default: Any = None,
    ) -> Any:
        done = threading.Event()
        result_box: dict[str, Any] = {}

        def runner():
            try:
                result_box["value"] = func()
            except Exception as exc:
                result_box["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        if not done.wait(max(float(timeout_seconds or 0.0), 0.1)):
            with self._lock:
                self._query_timeout_at[str(label)] = time.monotonic()
            logger.warning("%s 查询超时（>%ss），返回默认值", label, timeout_seconds)
            return default
        with self._lock:
            self._query_timeout_at.pop(str(label), None)
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value", default)

    def was_query_timeout_recently(self, label: str, *, within_seconds: float = 15.0) -> bool:
        with self._lock:
            last_timeout = float(self._query_timeout_at.get(str(label), 0.0) or 0.0)
        if last_timeout <= 0:
            return False
        return (time.monotonic() - last_timeout) < max(float(within_seconds or 0.0), 0.1)

    @contextmanager
    def authorize_order_stock(self, source: str = "", request_id: str = "", token: str = ""):
        """Temporarily allow the unified execution gateway to submit one scoped real order."""
        scoped_request_id = str(request_id or "").strip()
        scoped_token = str(token or (uuid4().hex if scoped_request_id else "")).strip()
        frame = {
            "source": source or "TradeExecutionService",
            "request_id": scoped_request_id,
            "token": scoped_token,
        }
        stack = list(getattr(self._order_authorization, "stack", []) or [])
        stack.append(frame)
        self._order_authorization.stack = stack
        self._order_authorization.depth = len(stack)
        self._order_authorization.source = frame["source"]
        self._order_authorization.request_id = frame["request_id"]
        self._order_authorization.token = frame["token"]
        try:
            yield frame["token"]
        finally:
            stack = list(getattr(self._order_authorization, "stack", []) or [])
            if stack and stack[-1] == frame:
                stack.pop()
            else:
                try:
                    stack.remove(frame)
                except ValueError:
                    pass
            self._order_authorization.stack = stack
            self._order_authorization.depth = len(stack)
            current = stack[-1] if stack else {}
            self._order_authorization.source = str(current.get("source", "") or "")
            self._order_authorization.request_id = str(current.get("request_id", "") or "")
            self._order_authorization.token = str(current.get("token", "") or "")

    def _current_order_authorization(self) -> dict[str, str]:
        stack = list(getattr(self._order_authorization, "stack", []) or [])
        if not stack:
            return {}
        frame = dict(stack[-1] or {})
        return {
            "source": str(frame.get("source", "") or ""),
            "request_id": str(frame.get("request_id", "") or ""),
            "token": str(frame.get("token", "") or ""),
        }

    def _is_order_stock_authorized(self, request_id: str = "", token: str = "") -> bool:
        frame = self._current_order_authorization()
        if not frame:
            return False
        expected_request_id = str(frame.get("request_id", "") or "")
        expected_token = str(frame.get("token", "") or "")
        request_id = str(request_id or "").strip()
        token = str(token or "").strip()
        if expected_request_id or expected_token:
            return bool(request_id and token and request_id == expected_request_id and token == expected_token)
        return True

    def query_stock_asset(self):
        trader, acc = self._require_connected()
        return trader.query_stock_asset(acc)

    def query_stock_positions(self):
        trader, acc = self._require_connected()
        return trader.query_stock_positions(acc)

    def query_stock_orders(self):
        trader, acc = self._require_connected()
        return trader.query_stock_orders(acc)

    def query_stock_trades(self):
        trader, acc = self._require_connected()
        return trader.query_stock_trades(acc)

    def query_stock_orders_safe(self, *, timeout_seconds: float = 8.0):
        trader, acc = self._require_connected()
        return self._query_with_timeout(
            "券商委托",
            lambda: trader.query_stock_orders(acc),
            timeout_seconds=timeout_seconds,
            default=[],
        )

    def query_stock_trades_safe(self, *, timeout_seconds: float = 8.0):
        trader, acc = self._require_connected()
        return self._query_with_timeout(
            "券商成交",
            lambda: trader.query_stock_trades(acc),
            timeout_seconds=timeout_seconds,
            default=[],
        )

    def query_stock_order(self, order_id: int):
        trader, acc = self._require_connected()
        if hasattr(trader, "query_stock_order"):
            return trader.query_stock_order(acc, order_id)
        orders = trader.query_stock_orders(acc) or []
        for order in orders:
            if getattr(order, "order_id", None) == order_id:
                return order
        return None

    def query_stock_deals(self):
        trader, acc = self._require_connected()
        for method_name in ("query_stock_deal", "query_stock_deals"):
            if hasattr(trader, method_name):
                return getattr(trader, method_name)(acc)
        return []

    def query_stock_deals_safe(self, *, timeout_seconds: float = 8.0):
        trader, acc = self._require_connected()

        def _run():
            for method_name in ("query_stock_deal", "query_stock_deals"):
                if hasattr(trader, method_name):
                    return getattr(trader, method_name)(acc)
            return []

        return self._query_with_timeout(
            "券商成交回报",
            _run,
            timeout_seconds=timeout_seconds,
            default=[],
        )

    def order_stock(
        self,
        stock_code: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str = "",
        remark: str = "",
        _authorization_request_id: str = "",
        _authorization_token: str = "",
    ):
        auth = self._current_order_authorization()
        if not self._is_order_stock_authorized(_authorization_request_id, _authorization_token):
            logger.error(
                "Blocked direct BrokerSessionService.order_stock call: stock_code=%s order_type=%s volume=%s source=%s request_id=%s",
                stock_code,
                order_type,
                order_volume,
                auth.get("source", ""),
                _authorization_request_id,
            )
            raise RuntimeError("真实下单必须通过 TradeExecutionService 统一执行网关")
        trader, acc = self._require_connected()
        return trader.order_stock(
            acc,
            stock_code,
            order_type,
            order_volume,
            price_type,
            price,
            strategy_name,
            remark,
        )

    def cancel_order_stock(self, order_id: int):
        trader, acc = self._require_connected()
        return trader.cancel_order_stock(acc, order_id)


_broker_session_service: Optional[BrokerSessionService] = None


def get_broker_session_service() -> BrokerSessionService:
    global _broker_session_service
    if _broker_session_service is None:
        _broker_session_service = BrokerSessionService()
    return _broker_session_service
