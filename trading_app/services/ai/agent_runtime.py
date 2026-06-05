from __future__ import annotations

from common.agent.agent_runtime import PreparedAgentRequest, StockAgentRuntime as _CommonStockAgentRuntime

from trading_app.services.ai.stock_agent_tools import (
    AgentToolExecutionContext,
    build_default_stock_tool_registry,
    extract_symbol_codes,
)


def _build_execution_context(runtime_context, user_text: str) -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        runtime_context=runtime_context,
        raw_context=runtime_context.raw,
        user_text=user_text,
    )


class StockAgentRuntime(_CommonStockAgentRuntime):
    """Trading app runtime with stock-domain tools registered by default."""

    def __init__(self, tool_registry=None, evidence_service=None):
        super().__init__(
            tool_registry=tool_registry or build_default_stock_tool_registry(),
            evidence_service=evidence_service,
            execution_context_factory=_build_execution_context,
            symbol_code_extractor=extract_symbol_codes,
        )


__all__ = ["PreparedAgentRequest", "StockAgentRuntime"]
