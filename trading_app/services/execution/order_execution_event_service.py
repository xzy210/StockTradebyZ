from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from trading_app.services.execution.order_state_machine import (
    OrderLifecycle,
    OrderLifecycleEvent,
    OrderStateSnapshot,
    OrderStateTransitionError,
)

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "live_strategy_center.db"
_singleton: Optional["OrderExecutionEventService"] = None


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_time_text(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "").strip()


def _to_timestamp(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class OrderExecutionEvent:
    event_id: str
    level: str = "info"
    category: str = "order_execution"
    source: str = "trade_execution"
    strategy_id: str = ""
    symbol: str = ""
    request_id: str = ""
    broker_order_id: int = 0
    title: str = ""
    message: str = ""
    status: str = "resolved"
    payload: Optional[dict[str, Any]] = None
    occurred_at: str = ""

    @property
    def event_type(self) -> str:
        payload = self.payload or {}
        return str(payload.get("event_type") or "").strip()


class OrderExecutionEventService:
    """Order execution event ledger used by the live strategy center."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS live_center_events (
                event_id TEXT PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                strategy_id TEXT DEFAULT '',
                symbol TEXT DEFAULT '',
                request_id TEXT DEFAULT '',
                broker_order_id INTEGER DEFAULT 0,
                title TEXT DEFAULT '',
                message TEXT DEFAULT '',
                status TEXT DEFAULT 'open',
                payload_json TEXT DEFAULT ''
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_center_events_occurred_at ON live_center_events(occurred_at DESC)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_center_events_status ON live_center_events(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_center_events_category ON live_center_events(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_center_events_request_id ON live_center_events(request_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_center_events_broker_order_id ON live_center_events(broker_order_id)"
        )
        conn.commit()
        conn.close()

    def add_event(self, event: OrderExecutionEvent) -> None:
        payload_json = self.dumps_payload(event.payload)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO live_center_events (
                event_id, occurred_at, level, category, source, strategy_id, symbol,
                request_id, broker_order_id, title, message, status, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.event_id or "").strip(),
                event.occurred_at or _now_text(),
                str(event.level or "info").strip().lower(),
                str(event.category or "order_execution").strip(),
                str(event.source or "trade_execution").strip(),
                str(event.strategy_id or "").strip(),
                str(event.symbol or "").strip(),
                str(event.request_id or "").strip(),
                int(event.broker_order_id or 0),
                str(event.title or "").strip(),
                str(event.message or "").strip(),
                str(event.status or "resolved").strip().lower(),
                payload_json,
            ),
        )
        conn.commit()
        conn.close()

    def get_event(self, event_id: str) -> OrderExecutionEvent | None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM live_center_events WHERE event_id = ?",
            (str(event_id or "").strip(),),
        )
        row = cursor.fetchone()
        conn.close()
        return self._row_to_event(row) if row else None

    def query_by_request_id(self, request_id: str, *, limit: int = 500) -> list[OrderExecutionEvent]:
        request_id = str(request_id or "").strip()
        if not request_id:
            return []
        return self._query_events(
            "category = ? AND request_id = ?",
            ["order_execution", request_id],
            order="ASC",
            limit=limit,
        )

    def query_by_broker_order_id(self, broker_order_id: int, *, limit: int = 500) -> list[OrderExecutionEvent]:
        broker_order_id = int(broker_order_id or 0)
        if broker_order_id <= 0:
            return []
        return self._query_events(
            "category = ? AND broker_order_id = ?",
            ["order_execution", broker_order_id],
            order="ASC",
            limit=limit,
        )

    def query_since(self, since: str | datetime, *, limit: int = 1000) -> list[OrderExecutionEvent]:
        return self._query_events(
            "category = ? AND occurred_at >= ?",
            ["order_execution", _to_time_text(since)],
            order="ASC",
            limit=limit,
        )

    def replay_since(self, since: str | datetime, *, limit: int = 1000) -> list[OrderExecutionEvent]:
        return self.query_since(since, limit=limit)

    def rebuild_state(self, request_id: str, *, strict: bool = False) -> OrderLifecycle | None:
        events = self.query_by_request_id(request_id)
        lifecycle_events = [item for item in (self.to_lifecycle_event(event) for event in events) if item is not None]
        if not lifecycle_events:
            return None
        lifecycle = OrderLifecycle(request_id)
        for event in lifecycle_events:
            try:
                lifecycle.apply_event(event)
            except OrderStateTransitionError:
                if strict:
                    raise
                logger.warning(
                    "Skip illegal order lifecycle event request_id=%s event_type=%s",
                    request_id,
                    event.event_type,
                    exc_info=True,
                )
        return lifecycle

    def query_open_orders(self, *, limit: int = 5000) -> dict[str, OrderLifecycle]:
        events = self._query_events(
            "category = ? AND request_id != ''",
            ["order_execution"],
            order="ASC",
            limit=limit,
        )
        request_ids = list(dict.fromkeys(event.request_id for event in events if event.request_id))
        open_orders: dict[str, OrderLifecycle] = {}
        for request_id in request_ids:
            lifecycle = self.rebuild_state(request_id)
            if lifecycle is not None and not lifecycle.is_terminal:
                open_orders[request_id] = lifecycle
        return open_orders

    def query_open(self, *, limit: int = 5000) -> dict[str, OrderLifecycle]:
        return self.query_open_orders(limit=limit)

    def list_pending(self, *, limit: int = 5000) -> dict[str, OrderLifecycle]:
        return self.query_open_orders(limit=limit)

    def to_lifecycle_event(self, event: OrderExecutionEvent) -> OrderLifecycleEvent | None:
        payload = event.payload or {}
        event_type = str(payload.get("event_type") or "").strip()
        normalized_type = self._normalize_lifecycle_event_type(event_type)
        if not normalized_type:
            return None
        snapshot = self._snapshot_from_payload(payload)
        return OrderLifecycleEvent(
            event_type=normalized_type,
            snapshot=snapshot,
            occurred_at=_to_timestamp(event.occurred_at),
            message=event.message,
            payload={
                **payload,
                "event_id": event.event_id,
                "broker_order_id": event.broker_order_id,
                "source_event_type": event_type,
            },
        )

    def _query_events(
        self,
        where_clause: str,
        params: list[Any],
        *,
        order: str,
        limit: int,
    ) -> list[OrderExecutionEvent]:
        direction = "DESC" if str(order or "").upper() == "DESC" else "ASC"
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT * FROM live_center_events
            WHERE {where_clause}
            ORDER BY occurred_at {direction}, rowid {direction}
            LIMIT ?
            """,
            [*params, max(int(limit or 0), 1)],
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_event(row) for row in rows]

    @classmethod
    def _row_to_event(cls, row: sqlite3.Row) -> OrderExecutionEvent:
        payload_json = str(row["payload_json"] or "")
        return OrderExecutionEvent(
            event_id=str(row["event_id"] or ""),
            occurred_at=str(row["occurred_at"] or ""),
            level=str(row["level"] or "info"),
            category=str(row["category"] or "order_execution"),
            source=str(row["source"] or "trade_execution"),
            strategy_id=str(row["strategy_id"] or ""),
            symbol=str(row["symbol"] or ""),
            request_id=str(row["request_id"] or ""),
            broker_order_id=int(row["broker_order_id"] or 0),
            title=str(row["title"] or ""),
            message=str(row["message"] or ""),
            status=str(row["status"] or "resolved"),
            payload=cls.loads_payload(payload_json),
        )

    @staticmethod
    def _normalize_lifecycle_event_type(event_type: str) -> str:
        aliases = {
            "OrderRequested": "OrderRequested",
            "OrderSubmitted": "OrderSubmitted",
            "OrderAccepted": "OrderAccepted",
            "OrderPartiallyFilled": "OrderPartiallyFilled",
            "OrderFilled": "OrderFilled",
            "OrderRejected": "OrderRejected",
            "OrderCancelled": "OrderCancelled",
            "OrderBlocked": "OrderRejected",
            "OrderSubmitFailed": "OrderRejected",
            "OrderPendingConfirmation": "OrderTimeoutPending",
            "OrderTimeoutPending": "OrderTimeoutPending",
            "OrderShadowRecorded": "OrderFilled",
        }
        return aliases.get(str(event_type or "").strip(), "")

    @staticmethod
    def _snapshot_from_payload(payload: dict[str, Any]) -> OrderStateSnapshot | None:
        if "order_status_code" not in payload:
            return None
        status_code = _to_int(payload.get("order_status_code"))
        return OrderStateSnapshot(
            status_code=status_code,
            status_text=str(payload.get("order_status_text") or status_code),
            status_message=str(payload.get("status_message") or payload.get("message") or ""),
            traded_volume=_to_int(payload.get("executed_volume") or payload.get("traded_volume")),
            traded_price=_to_float(payload.get("executed_price") or payload.get("traded_price")),
        )

    @staticmethod
    def dumps_payload(payload: Optional[dict[str, Any]]) -> str:
        if not payload:
            return ""
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception as exc:
            logger.debug("Serialize order execution event payload failed: %s", exc)
            return ""

    @staticmethod
    def loads_payload(payload_json: str) -> dict[str, Any]:
        if not payload_json:
            return {}
        try:
            payload = json.loads(payload_json)
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.debug("Deserialize order execution event payload failed: %s", exc)
            return {}


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def get_order_execution_event_service() -> OrderExecutionEventService:
    global _singleton
    if _singleton is None:
        _singleton = OrderExecutionEventService()
    return _singleton
