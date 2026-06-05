from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional, Sequence

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

try:
    from trading_app.services.ai.auto_trade_config_service import get_auto_trade_config_service
except ImportError:
    from trading_app.services.ai.auto_trade_config_service import get_auto_trade_config_service

try:
    from trading_app.services.execution.trade_record_service import get_trade_record_service
except ImportError:
    from trading_app.services.execution.trade_record_service import get_trade_record_service

try:
    from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service
except ImportError:
    from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

logger = logging.getLogger(__name__)

_EXCEPTION_ORDER_STATUSES = {"blocked", "failed", "cancelled", "rejected"}
_UNMANAGED_RECONCILE_MIN_INTERVAL_SEC = 30.0


class HubStateService(QObject):
    state_changed = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.broker_service = None
        self.startup_orchestrator = None
        self.eod_service = None
        self.strategy_adapters: list = []
        self.alert_service = None
        self.task_service = None
        self._state: dict = {}
        self._timer = QTimer(self)
        self._timer.setInterval(4000)
        self._timer.timeout.connect(self.refresh_state)
        self._auto_trade_config = get_auto_trade_config_service()
        self._trade_service = get_trade_record_service()
        self._budget_service = get_strategy_budget_service()
        self._last_unmanaged_reconcile_ts: float = 0.0
        self._last_broker_connected: bool = False
        try:
            self._budget_service.ensure_unmanaged_strategy()
        except Exception as exc:
            logger.warning("初始化 unmanaged 策略失败: %s", exc)

    def bind(
        self,
        *,
        broker_service,
        startup_orchestrator,
        eod_service,
        alert_service,
        task_service,
        strategy_adapters: Optional[Sequence] = None,
        ai_panel=None,
        etf_panel=None,
    ) -> None:
        self.broker_service = broker_service
        self.startup_orchestrator = startup_orchestrator
        self.eod_service = eod_service
        self.strategy_adapters = list(strategy_adapters or [])
        if not self.strategy_adapters:
            legacy_adapters = []
            if ai_panel is not None:
                legacy_adapters.append(ai_panel)
            if etf_panel is not None:
                legacy_adapters.append(etf_panel)
            self.strategy_adapters = legacy_adapters
        self.alert_service = alert_service
        self.task_service = task_service
        broker_service.connection_changed.connect(self._on_broker_connection_changed)
        broker_service.client_state_changed.connect(lambda *_: self.refresh_state())
        startup_orchestrator.finished.connect(lambda *_: self.refresh_state())
        eod_service.cycle_finished.connect(self._on_eod_cycle_finished)
        alert_service.events_changed.connect(self.refresh_state)
        task_service.tasks_changed.connect(lambda *_: self.refresh_state())
        try:
            self._trade_service.order_record_added.connect(lambda *_: self.refresh_state())
            self._trade_service.order_record_updated.connect(lambda *_: self.refresh_state())
        except Exception:
            pass
        self._timer.start()
        self.refresh_state()

    def refresh_state(self) -> None:
        if self.broker_service is None:
            return
        broker_status = dict(self.broker_service.get_client_status() or {})
        strategy_statuses = self._collect_strategy_statuses()
        ai_status = self._find_strategy_status(strategy_statuses, "ai_trade_decision_center")
        etf_status = self._find_strategy_status(strategy_statuses, "etf_rotation")
        eod_state = {}
        if self.eod_service is not None:
            eod_state = self.eod_service._get_cycle_state()  # noqa: SLF001
        alert_counts = self.alert_service.get_counts() if self.alert_service is not None else {}
        tasks = self.task_service.list_tasks() if self.task_service is not None else []
        cfg = self._auto_trade_config.get_config()
        exception_order_count = self._count_today_exception_orders()
        center_automation_paused = self._is_center_automation_paused()
        risk_summary = self._build_risk_summary(
            cfg=cfg,
            broker_connected=bool(getattr(self.broker_service, "is_connected", False)),
            qmt_running=bool(broker_status.get("running", False)),
            startup_running=bool(getattr(self.startup_orchestrator, "is_running", False)),
            alert_counts=alert_counts,
            exception_order_count=exception_order_count,
            eod_state=eod_state,
            center_automation_paused=center_automation_paused,
        )
        self._state = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "broker_status": broker_status,
            "qmt_running": bool(broker_status.get("running", False)),
            "broker_connected": bool(getattr(self.broker_service, "is_connected", False)),
            "startup_running": bool(getattr(self.startup_orchestrator, "is_running", False)),
            "eod_state": eod_state,
            "strategy_statuses": strategy_statuses,
            "ai_status": ai_status,
            "etf_status": etf_status,
            "alert_counts": alert_counts,
            "exception_order_count": exception_order_count,
            "tasks": tasks,
            "auto_trade_mode": cfg.auto_trade_mode,
            "manual_orders_enabled": bool(cfg.manual_orders_enabled),
            "require_trading_time": bool(cfg.require_trading_time),
            "center_automation_paused": center_automation_paused,
            "risk_summary": risk_summary,
        }
        self.state_changed.emit(dict(self._state))

    def _collect_strategy_statuses(self) -> list[dict]:
        rows: list[dict] = []
        for adapter in list(self.strategy_adapters or []):
            try:
                if hasattr(adapter, "get_status_summary"):
                    status = dict(adapter.get_status_summary() or {})
                elif hasattr(adapter, "get_center_status_summary"):
                    status = dict(adapter.get_center_status_summary() or {})
                else:
                    status = {}
            except Exception as exc:
                logger.warning("读取策略状态失败: %s", exc)
                status = {}
            strategy_id = str(getattr(adapter, "strategy_id", "") or status.get("strategy_id", "") or "")
            strategy_name = str(getattr(adapter, "strategy_name", "") or status.get("strategy_name", "") or strategy_id)
            if strategy_id:
                status.setdefault("strategy_id", strategy_id)
            if strategy_name:
                status.setdefault("strategy_name", strategy_name)
            if status:
                rows.append(status)
        return rows

    @staticmethod
    def _find_strategy_status(strategy_statuses: list[dict], strategy_id: str) -> dict:
        target = str(strategy_id or "").strip()
        for item in strategy_statuses:
            if str(item.get("strategy_id", "") or "").strip() == target:
                return dict(item)
        return {}

    def _is_center_automation_paused(self) -> bool:
        for adapter in list(self.strategy_adapters or []):
            try:
                if hasattr(adapter, "is_automation_paused") and bool(adapter.is_automation_paused()):
                    return True
            except Exception as exc:
                logger.debug("读取策略暂停状态失败: %s", exc)
        return False

    def _build_risk_summary(
        self,
        *,
        cfg,
        broker_connected: bool,
        qmt_running: bool,
        startup_running: bool,
        alert_counts: dict,
        exception_order_count: int,
        eod_state: dict,
        center_automation_paused: bool,
    ) -> dict:
        mode = str(getattr(cfg, "auto_trade_mode", "off") or "off").strip().lower()
        open_alerts = int((alert_counts or {}).get("open", 0) or 0)
        eod_status = str((eod_state or {}).get("status", "") or "").strip().lower()
        items: list[str] = []
        level = "ok"

        def raise_level(candidate: str) -> None:
            nonlocal level
            order = {"ok": 0, "warning": 1, "danger": 2}
            if order.get(candidate, 0) > order.get(level, 0):
                level = candidate

        if mode == "live":
            items.append("执行模式：实盘")
            if not broker_connected or not qmt_running:
                raise_level("danger")
                items.append("实盘连接未完全就绪")
        elif mode == "shadow":
            items.append("执行模式：影子")
        elif mode == "paper":
            items.append("执行模式：模拟")
        else:
            items.append("执行模式：关闭")

        if not bool(getattr(cfg, "manual_orders_enabled", True)):
            raise_level("warning")
            items.append("手动委托已关闭")
        else:
            items.append("手动委托：开启")

        if bool(getattr(cfg, "require_trading_time", True)):
            items.append("交易时段闸：开启")
        else:
            raise_level("warning")
            items.append("交易时段闸：关闭")

        if center_automation_paused:
            raise_level("warning")
            items.append("自动化：已暂停")
        else:
            items.append("自动化：正常")

        if exception_order_count > 0:
            raise_level("danger")
            items.append(f"异常订单：{exception_order_count}")
        if open_alerts > 0:
            raise_level("warning")
            items.append(f"未处理告警：{open_alerts}")
        if eod_status == "failed":
            raise_level("danger")
            items.append("今日日终：失败")
        if startup_running:
            raise_level("warning")
            items.append("启动自检中")

        label_map = {
            "ok": "风控: 正常",
            "warning": "风控: 注意",
            "danger": "风控: 高风险",
        }
        return {
            "level": level,
            "label": label_map.get(level, "风控: -"),
            "items": items,
            "tooltip": "\n".join(items),
        }

    def _count_today_exception_orders(self) -> int:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            records = self._trade_service.get_order_records(
                start_time=f"{today} 00:00:00",
                end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                include_archived=False,
                limit=500,
            )
        except Exception:
            return 0
        return sum(
            1 for item in records
            if str(getattr(item, "status", "") or "").strip().lower() in _EXCEPTION_ORDER_STATUSES
        )

    def get_state(self) -> dict:
        return dict(self._state)

    # ------------------------------------------------------------------
    #  未管理账户对账：broker 连接成功/EOD 时把券商里"无主"现金和持仓
    #  同步进 unmanaged 虚拟账户，保证主账本 ≡ 券商实况
    # ------------------------------------------------------------------

    def _on_broker_connection_changed(self, *args) -> None:
        self.refresh_state()
        connected = bool(getattr(self.broker_service, "is_connected", False))
        if connected and not self._last_broker_connected:
            QTimer.singleShot(3000, lambda: self._reconcile_unmanaged_account(source="broker_connected"))
        self._last_broker_connected = connected

    def _on_eod_cycle_finished(self, *args) -> None:
        self.refresh_state()
        self._reconcile_unmanaged_account(source="eod", force=True)

    def _reconcile_unmanaged_account(self, *, source: str = "", force: bool = False) -> None:
        if self.broker_service is None:
            return
        if not getattr(self.broker_service, "is_connected", False):
            return
        now = time.monotonic()
        if not force and (now - self._last_unmanaged_reconcile_ts) < _UNMANAGED_RECONCILE_MIN_INTERVAL_SEC:
            return
        self._last_unmanaged_reconcile_ts = now
        try:
            asset = self.broker_service.query_stock_asset()
            broker_cash = float(getattr(asset, "cash", 0.0) or 0.0)
        except Exception as exc:
            logger.warning("[%s] 查询券商资产失败，unmanaged 对账跳过: %s", source or "reconcile", exc)
            return
        try:
            raw_positions = self.broker_service.query_stock_positions() or []
        except Exception as exc:
            logger.warning("[%s] 查询券商持仓失败，unmanaged 对账跳过: %s", source or "reconcile", exc)
            return
        broker_positions = []
        for pos in raw_positions:
            try:
                volume = int(getattr(pos, "volume", 0) or 0)
                if volume <= 0:
                    continue
                broker_positions.append({
                    "stock_code": str(getattr(pos, "stock_code", "") or ""),
                    "volume": volume,
                    "open_price": float(getattr(pos, "open_price", 0.0) or 0.0),
                })
            except Exception:
                continue
        try:
            summary = self._budget_service.reconcile_unmanaged_with_broker(
                broker_cash=broker_cash,
                broker_positions=broker_positions,
            )
            logger.info(
                "[%s] unmanaged 账户对账: broker_cash=%.2f claimed=%.2f unmanaged_cash=%.2f positions=%d",
                source or "reconcile",
                summary.get("broker_cash", 0.0),
                summary.get("claimed_cash", 0.0),
                summary.get("unmanaged_cash", 0.0),
                summary.get("unmanaged_position_count", 0),
            )
        except Exception as exc:
            logger.warning("[%s] unmanaged 账户对账执行失败: %s", source or "reconcile", exc)
