from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal

from .models import LiveCenterEvent
from .storage import LiveStrategyCenterStorage, get_live_strategy_center_storage

logger = logging.getLogger(__name__)


class AlertEventService(QObject):
    """实盘策略中枢事件审阅台的数据服务。

    系统异常告警和订单生命周期事件共用 ``live_center_events``。
    状态栏的告警计数默认排除普通订单流水，订单事件可在实盘事件页按分类查询。
    """

    event_recorded = pyqtSignal(dict)
    events_changed = pyqtSignal()

    def __init__(self, storage: Optional[LiveStrategyCenterStorage] = None, parent=None) -> None:
        super().__init__(parent)
        self.storage = storage or get_live_strategy_center_storage()

    @staticmethod
    def _build_event_id(key: str) -> str:
        return hashlib.md5(str(key or "").encode("utf-8")).hexdigest()

    def record_event(
        self,
        *,
        key: str,
        level: str,
        category: str,
        source: str,
        title: str,
        message: str,
        strategy_id: str = "",
        symbol: str = "",
        request_id: str = "",
        broker_order_id: int = 0,
        status: str = "open",
        payload: Optional[dict] = None,
        occurred_at: str = "",
    ) -> str:
        event_id = self._build_event_id(key)
        existing = self.storage.get_event(event_id)
        preserved_status = str(getattr(existing, "status", "") or "").strip().lower()
        effective_status = str(status or "open").strip().lower()
        if preserved_status in {"read", "ignored", "resolved"} and effective_status == "open":
            effective_status = preserved_status
        effective_occurred_at = str(occurred_at or getattr(existing, "occurred_at", "") or "").strip()
        if not effective_occurred_at:
            effective_occurred_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = LiveCenterEvent(
            event_id=event_id,
            occurred_at=effective_occurred_at,
            level=level,
            category=category,
            source=source,
            strategy_id=strategy_id,
            symbol=symbol,
            request_id=request_id,
            broker_order_id=broker_order_id,
            title=title,
            message=message,
            status=effective_status,
            payload_json=self.storage.dumps_payload(payload or {}),
        )
        self.storage.add_event(event)
        payload_dict = event.to_dict()
        payload_dict["payload"] = payload or {}
        self.event_recorded.emit(payload_dict)
        self.events_changed.emit()
        return event.event_id

    def list_events(
        self,
        *,
        status: str = "",
        category: str = "",
        start_time: str = "",
        end_time: str = "",
        include_ignored: bool = True,
        limit: int = 300,
    ) -> list[dict]:
        results = []
        for event in self.storage.list_events(
            status=status,
            category=category,
            start_time=start_time,
            end_time=end_time,
            include_ignored=include_ignored,
            limit=limit,
        ):
            item = event.to_dict()
            payload = self.storage.loads_payload(event.payload_json)
            item["payload"] = payload
            if str(item.get("category", "") or "") == "order_execution":
                self._attach_order_event_context(item, payload)
            results.append(item)
        return results

    @staticmethod
    def _attach_order_event_context(item: dict, payload: dict) -> None:
        event_type = str(payload.get("event_type", "") or "").strip()
        event_scope = str(payload.get("event_scope", "") or "").strip()
        direction = str(payload.get("direction", "") or "").strip()
        order_record_id = int(payload.get("order_record_id", 0) or 0)
        executed_volume = int(payload.get("executed_volume", 0) or 0)
        order_volume = int(payload.get("order_volume", 0) or 0)
        price = float(payload.get("executed_price", 0) or payload.get("price", 0) or 0)
        status_text = str(payload.get("order_status_text", "") or payload.get("latest_status", "") or "").strip()
        broker_order_id = int(item.get("broker_order_id", 0) or 0)
        symbol = str(item.get("symbol", "") or "").strip()

        refs = []
        if symbol:
            refs.append(symbol)
        if broker_order_id:
            refs.append(f"委托 {broker_order_id}")
        if item.get("request_id"):
            refs.append(f"请求 {str(item.get('request_id', ''))[:8]}")
        item["order_reference"] = " / ".join(refs) if refs else "-"
        item["order_event_type"] = event_type
        item["order_event_scope"] = event_scope
        item["order_direction"] = direction
        item["order_record_id"] = order_record_id
        item["order_status_text"] = status_text
        item["order_observable_detail"] = "，".join(
            part
            for part in [
                event_type,
                event_scope if event_scope and event_scope != event_type else "",
                direction,
                f"{executed_volume or order_volume}股" if (executed_volume or order_volume) else "",
                f"{price:.3f}" if price else "",
                status_text,
            ]
            if part
        )

    def get_counts(self, *, include_order_events: bool = False) -> Dict[str, int]:
        exclude = [] if include_order_events else ["order_execution"]
        return self.storage.get_event_counts(exclude_categories=exclude)

    def mark_event_status(self, event_id: str, status: str) -> None:
        self.storage.update_event_status(event_id, status)
        self.events_changed.emit()

    def connect_broker_service(self, broker_service) -> None:
        broker_service.order_error.connect(self._on_order_error)
        broker_service.broker_disconnected.connect(self._on_broker_disconnected)
        broker_service.order_changed.connect(self._on_order_changed)

    def connect_qmt_startup(self, orchestrator) -> None:
        orchestrator.finished.connect(self._on_qmt_finished)

    def connect_end_of_day(self, eod_service) -> None:
        eod_service.cycle_finished.connect(
            self._on_end_of_day_finished,
            Qt.ConnectionType.QueuedConnection,
        )

    def connect_ai_panel(self, ai_panel) -> None:
        try:
            ai_panel.daily_auto_trade.cycle_finished.connect(self._on_ai_cycle_finished)
        except Exception:
            pass

    def connect_etf_panel(self, etf_panel) -> None:
        try:
            etf_panel.engine.trade_executed.connect(self._on_etf_trade_executed)
        except Exception:
            pass

    def _on_order_error(self, payload: dict) -> None:
        order_id = int(payload.get("order_id", 0) or 0)
        self.record_event(
            key=f"broker-order-error:{order_id}:{payload.get('error_id', '')}",
            level="danger",
            category="broker_error",
            source="broker_session",
            title="券商委托错误",
            message=str(payload.get("error_msg", "") or "券商返回委托错误"),
            broker_order_id=order_id,
            payload=payload,
        )

    def _on_broker_disconnected(self) -> None:
        self.record_event(
            key=f"broker-disconnected:{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            level="warning",
            category="broker_disconnected",
            source="broker_session",
            title="券商连接断开",
            message="xtquant 回调显示交易连接已断开，请尽快确认连接状态。",
        )

    def _on_order_changed(self, payload: dict) -> None:
        status = int(payload.get("order_status", 0) or 0)
        if status not in (53, 54, 57):
            return
        order_id = int(payload.get("order_id", 0) or 0)
        status_text = str(payload.get("status_msg", "") or payload.get("order_status", ""))
        self.record_event(
            key=f"broker-order-status:{order_id}:{status}",
            level="warning" if status in (53, 54) else "danger",
            category="order_exception",
            source="broker_session",
            title="异常委托状态",
            message=f"{payload.get('stock_code', '')} 状态异常: {status_text}",
            symbol=str(payload.get("stock_code", "") or ""),
            broker_order_id=order_id,
            payload=payload,
        )

    def _on_qmt_finished(self, success: bool, message: str) -> None:
        # 成功的连接自检由实盘任务/状态栏展示；实盘事件页只关注失败与异常。
        if success:
            return
        self.record_event(
            key=f"qmt-startup:{datetime.now().strftime('%Y-%m-%d')}:{success}:{message}",
            level="danger",
            category="startup",
            source="qmt_startup",
            title="连接自检失败",
            message=message,
        )

    def _on_end_of_day_finished(self, success: bool, message: str, payload: dict) -> None:
        # 成功的日终只在实盘任务体现，避免实盘事件页被成功流水淹没。
        if success:
            return
        self.record_event(
            key=f"eod:{datetime.now().strftime('%Y-%m-%d')}:{success}:{message}",
            level="danger",
            category="end_of_day",
            source="live_strategy_eod",
            title="统一日终失败",
            message=message,
            payload=payload,
        )

    def _on_ai_cycle_finished(self, task_id: str, success: bool, message: str, summary: dict) -> None:
        # AI 实盘决策调度任务的常规成功/心跳不进实盘事件页，由实盘任务展示最近一次状态；只有失败才告警。
        if success:
            return
        self.record_event(
            key=f"ai-cycle:{task_id}:{datetime.now().strftime('%Y-%m-%d %H:%M')}:{success}",
            level="warning",
            category="ai_task",
            source="ai_scheduler",
            title="AI 自动任务失败",
            message=message,
            strategy_id="ai_trade_decision_center",
            payload=summary,
        )

    def _on_etf_trade_executed(self, success: bool, detail: dict) -> None:
        if success:
            return
        self.record_event(
            key=f"etf-trade:{datetime.now().strftime('%Y-%m-%d %H:%M')}:{detail.get('code', '')}:{success}",
            level="warning",
            category="etf_trade",
            source="etf_rotation",
            title="ETF 交易执行失败",
            message=str(detail.get("reason", "") or detail.get("message", "") or "ETF 交易执行失败"),
            strategy_id="etf_rotation",
            symbol=str(detail.get("code", "") or ""),
            payload=detail,
        )
