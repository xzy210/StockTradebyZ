from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


ALLOWED_AGENT_ACTIONS = {"connect_broker"}


@dataclass
class AgentActionIntent:
    action: str
    reason: str = ""
    requires_confirmation: bool = True

    @property
    def label(self) -> str:
        return {
            "connect_broker": "连接券商会话",
        }.get(self.action, self.action)


class AgentActionService:
    """Parse and validate side-effect actions proposed by the AI agent."""

    ACTION_BLOCK_RE = re.compile(r"<agent_action>\s*(\{.*?\})\s*</agent_action>", re.DOTALL)

    @classmethod
    def extract_intent(cls, text: str) -> Optional[AgentActionIntent]:
        if not text:
            return None
        match = cls.ACTION_BLOCK_RE.search(text)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except Exception:
            return None

        action = str(payload.get("action", "") or "").strip()
        if action not in ALLOWED_AGENT_ACTIONS:
            return None
        reason = str(payload.get("reason", "") or "").strip()
        return AgentActionIntent(
            action=action,
            reason=reason,
            requires_confirmation=bool(payload.get("requires_confirmation", True)),
        )
