from __future__ import annotations

import json
import logging
import re
from typing import Optional

from trading_app.services.ai.trade_decision_models import TradeAction, TradeDecision

logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(
    r"<trade_decision>\s*(.*?)\s*</trade_decision>",
    re.DOTALL,
)

_REQUIRED_FIELDS = {"action", "symbol_code", "symbol_name"}

_VALID_ACTIONS = {a.value for a in TradeAction}


class TradeDecisionExtractor:
    """Extract a structured ``TradeDecision`` from an LLM Markdown response."""

    @staticmethod
    def extract(response_text: str) -> Optional[TradeDecision]:
        if not response_text:
            return None

        match = _TAG_PATTERN.search(response_text)
        if not match:
            logger.debug("No <trade_decision> tag found in response")
            return None

        raw_json = match.group(1).strip()
        # Handle markdown code fences inside the tag
        if raw_json.startswith("```"):
            raw_json = re.sub(r"^```\w*\n?", "", raw_json)
            raw_json = re.sub(r"\n?```$", "", raw_json)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse trade_decision JSON: %s", exc)
            return None

        if not isinstance(data, dict):
            logger.warning("trade_decision JSON is not a dict")
            return None

        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            logger.warning("trade_decision missing required fields: %s", missing)
            return None

        action = str(data.get("action", "")).lower().strip()
        if action not in _VALID_ACTIONS:
            logger.warning("Invalid trade action: %s", action)
            return None

        decision = TradeDecision(
            action=action,
            symbol_code=str(data.get("symbol_code", "")),
            symbol_name=str(data.get("symbol_name", "")),
            confidence=_clamp(data.get("confidence", 0.5), 0.0, 1.0),
            target_price=max(float(data.get("target_price", 0)), 0),
            stop_loss_price=max(float(data.get("stop_loss_price", 0)), 0),
            current_price=max(float(data.get("current_price", 0)), 0),
            position_pct=_clamp(data.get("position_pct", 0.1), 0.0, 1.0),
            risk_score=_clamp(data.get("risk_score", 0.5), 0.0, 1.0),
            time_horizon=str(data.get("time_horizon", "short")),
            invalidation=str(data.get("invalidation", "")),
            reasoning=str(data.get("reasoning", "")),
            bull_case=str(data.get("bull_case", "")),
            bear_case=str(data.get("bear_case", "")),
        )
        return decision


def _clamp(value, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo
