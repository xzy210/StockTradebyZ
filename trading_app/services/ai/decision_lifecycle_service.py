# -*- coding: utf-8 -*-
"""AI 决策生命周期统一持久化门面。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from trading_app.services.ai.decision_session_store import (
    DecisionSession,
    DecisionSessionItem,
    DecisionSessionStore,
)
from trading_app.services.ai.evidence_trace_store import EvidenceTrace, EvidenceTraceStore
from trading_app.services.ai.decision_tracker_service import DecisionTrackerService
from trading_app.services.ai.trade_decision_models import (
    DecisionRecord,
    RiskCheckResult,
    TradeDecision,
)


class DecisionLifecycleService:
    """Keep decision records, evidence traces and sessions on one persistence path."""

    def __init__(
        self,
        *,
        decision_tracker: Optional[DecisionTrackerService] = None,
        evidence_trace_store: Optional[EvidenceTraceStore] = None,
        decision_session_store: Optional[DecisionSessionStore] = None,
    ) -> None:
        self.decision_tracker = decision_tracker or DecisionTrackerService()
        self.evidence_trace_store = evidence_trace_store or EvidenceTraceStore()
        self.decision_session_store = decision_session_store or DecisionSessionStore()

    def save_decision_record(
        self,
        decision: TradeDecision,
        risk_result: RiskCheckResult,
        outcome: str,
        *,
        user_remark: str = "",
        broker_order_id: int = -1,
        entry_price: float = 0.0,
    ) -> DecisionRecord:
        return self.decision_tracker.save_decision(
            decision,
            risk_result,
            outcome,
            user_remark=user_remark,
            broker_order_id=broker_order_id,
            entry_price=entry_price,
        )

    def update_decision_outcome(self, record_id: str, **kwargs) -> bool:
        return self.decision_tracker.update_outcome(record_id, **kwargs)

    def auto_close_by_symbol(self, symbol_code: str, exit_price: float, *, broker_order_id: int = -1) -> list[str]:
        return self.decision_tracker.auto_close_by_symbol(
            symbol_code,
            exit_price,
            broker_order_id=broker_order_id,
        )

    def close_position(self, record_id: str, exit_price: float) -> bool:
        return self.decision_tracker.close_position(record_id, exit_price)

    def query_recent_decisions(self, limit: int = 50) -> list[DecisionRecord]:
        return self.decision_tracker.query_recent(limit=limit)

    def expire_stale_decisions(self) -> int:
        return self.decision_tracker.expire_stale_decisions()

    def get_decision_stats(self) -> dict:
        return self.decision_tracker.get_stats()

    def export_decisions_csv(self, path: Path) -> int:
        return self.decision_tracker.export_csv(path)

    def export_decisions_html_report(self, path: Path) -> int:
        return self.decision_tracker.export_html_report(path)

    def save_evidence_trace(self, trace: EvidenceTrace) -> Path:
        return self.evidence_trace_store.save_trace(trace)

    def load_evidence_trace(self, path: str | Path) -> EvidenceTrace | None:
        return self.evidence_trace_store.load_trace(path)

    def list_recent_evidence_traces(self, limit: int = 100) -> list[tuple[Path, EvidenceTrace]]:
        return self.evidence_trace_store.list_recent(limit=limit)

    def upsert_session(self, session: DecisionSession) -> Path:
        return self.decision_session_store.upsert_session(session)

    def append_session_item(self, session_id: str, item: DecisionSessionItem) -> Path | None:
        return self.decision_session_store.append_item(session_id, item)

    def complete_session(self, session_id: str, *, status: str = "done", summary: str = "") -> Path | None:
        return self.decision_session_store.complete_session(session_id, status=status, summary=summary)

    def list_recent_sessions(self, limit: int = 80) -> list[tuple[Path, DecisionSession]]:
        return self.decision_session_store.list_recent(limit=limit)


__all__ = ["DecisionLifecycleService"]
