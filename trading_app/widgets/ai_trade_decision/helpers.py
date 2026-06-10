# -*- coding: utf-8 -*-
"""AI ???????????"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from common.market_data_policy import is_etf_like_code
from trading_app.services.market_data.market_data_status_service import get_market_data_status_service
from trading_app.services.ai.trade_decision_models import RiskCheckResult, TradeAction, TradeDecision

logger = logging.getLogger(__name__)


class _StatusMessageProxy:
    def __init__(self, owner: QWidget):
        self.owner = owner

    def showMessage(self, message: str):
        logger.info("AI trade panel status: %s", message)


def _check_ai_live_market_data_ready(codes: list, *, require_minute_freshness: bool = False) -> tuple[bool, str]:
    unique_codes: list[str] = []
    seen_codes = set()
    for code in codes or []:
        plain_code = str(code or "").strip().split(".", 1)[0]
        if not plain_code:
            continue
        normalized = plain_code.zfill(6)
        if normalized in seen_codes:
            continue
        seen_codes.add(normalized)
        unique_codes.append(normalized)

    stock_codes = [code for code in unique_codes if not is_etf_like_code(code)]
    etf_codes = [code for code in unique_codes if is_etf_like_code(code)]
    status = get_market_data_status_service().check_status(
        stock_codes=stock_codes,
        etf_codes=etf_codes,
        index_codes=[],
        realtime_probe_codes=unique_codes[:3] if unique_codes else None,
        require_minute_freshness=require_minute_freshness,
    )
    if status.can_run_live_strategy:
        return True, status.summary
    return False, status.summary


# ---------------------------------------------------------------------------
#  Helper: reuse ChatThread from ai_agent_widget to avoid duplication
# ---------------------------------------------------------------------------
def _get_chat_thread_class():
    try:
        from trading_app.widgets.ai_agent_widget import ChatThread
    except ImportError:
        from trading_app.widgets.ai_agent_widget import ChatThread
    return ChatThread


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _make_json_safe(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _build_scan_status_text(decision: Optional[TradeDecision], risk_result: Optional[RiskCheckResult]) -> str:
    if decision is None:
        return "解析失败"
    if risk_result and not getattr(risk_result, "passed", True) and decision.is_actionable:
        reasons = [str(item or "").strip() for item in getattr(risk_result, "blocked_reasons", []) or []]
        if reasons:
            return f"风控拦截: {reasons[0]}"
        return "风控拦截"
    if decision.is_actionable:
        return "可执行"
    if decision.action == TradeAction.WATCH.value:
        return "候选观察"
    if decision.action == TradeAction.REJECT.value:
        return "剔除候选"
    return "继续持有"


def _serialize_scan_result_for_record(result: Dict[str, Any]) -> Dict[str, Any]:
    decision = result.get("decision")
    risk_result = result.get("risk_result")
    scan_item = dict(result.get("scan_item", {}) or {})
    return {
        "symbol_code": str(result.get("symbol_code", "") or ""),
        "symbol_name": str(result.get("symbol_name", "") or ""),
        "decision": _make_json_safe(decision.to_dict() if decision is not None else {}),
        "risk_result": _make_json_safe(risk_result.to_dict() if risk_result is not None else {}),
        "scan_item": _make_json_safe(scan_item),
        "response_text": str(result.get("response_text", "") or ""),
        "decision_record_id": str(result.get("decision_record_id", "") or ""),
        "evidence_trace_path": str(result.get("evidence_trace_path", "") or ""),
        "status_text": _build_scan_status_text(decision, risk_result),
    }


def _build_scheduled_scan_batch_record(task_id: str, task_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    serialized_results = [
        _serialize_scan_result_for_record(item)
        for item in list(payload.get("results", []) or [])
        if isinstance(item, dict)
    ]
    actionable = 0
    risk_blocked = 0
    for item in serialized_results:
        decision_payload = dict(item.get("decision", {}) or {})
        risk_payload = dict(item.get("risk_result", {}) or {})
        action = str(decision_payload.get("action", "") or "").lower()
        if action in {
            TradeAction.BUY.value,
            TradeAction.SELL.value,
            TradeAction.REDUCE.value,
            TradeAction.ADD.value,
        }:
            actionable += 1
        if risk_payload and not bool(risk_payload.get("passed", True)):
            risk_blocked += 1
    scan_label = str(payload.get("scan_label", "") or "定时巡检")
    scan_total = len(serialized_results)
    return {
        "task_id": str(task_id or ""),
        "task_name": str(task_name or task_id or ""),
        "completed_at": str(payload.get("completed_at", "") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "scan_run_id": str(payload.get("scan_run_id", "") or ""),
        "scan_source": str(payload.get("scan_source", "") or ""),
        "scan_scope": str(payload.get("scan_scope", "") or ""),
        "scan_label": scan_label,
        "allow_auto_execute": bool(payload.get("allow_auto_execute", False)),
        "scan_total": scan_total,
        "actionable": actionable,
        "risk_blocked": risk_blocked,
        "summary_text": f"{scan_label}共 {scan_total} 只，可操作 {actionable} 只，风控拦截 {risk_blocked} 只",
        "results": serialized_results,
    }

