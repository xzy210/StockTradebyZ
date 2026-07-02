from __future__ import annotations

import logging
from typing import Callable, Optional

from common.agent.agent_context_service import (
    AgentRuntimeContext,
    TASK_MODE_GENERAL,
    TASK_MODE_LABELS,
    TASK_MODE_POSITION_DIAGNOSIS,
    TASK_MODE_SYMBOL_ANALYSIS,
    TASK_MODE_TRADE_DECISION,
    TASK_MODE_WATCHLIST_SCAN,
)
from common.agent.agent_response_contract import build_contract_with_citations

logger = logging.getLogger(__name__)

MarketContextProvider = Callable[[AgentRuntimeContext], list[str]]
_market_context_provider: Optional[MarketContextProvider] = None


def set_market_context_provider(provider: Optional[MarketContextProvider]) -> None:
    global _market_context_provider
    _market_context_provider = provider


class AgentPromptBuilder:
    """Build system prompts and quick task prompts for AI agents."""

    @staticmethod
    def build_system_prompt(
        base_prompt: str,
        context: AgentRuntimeContext,
        task_mode: str = TASK_MODE_GENERAL,
    ) -> str:
        sections = [
            (base_prompt or "你是一个专业的股票投资顾问。").strip(),
            "你运行在一个股票/ETF分析桌面应用中，请优先结合当前运行上下文回答。",
            f"当前任务模式: {TASK_MODE_LABELS.get(task_mode, task_mode)}",
            "当前运行上下文:",
            *[f"- {line}" for line in context.to_summary_lines()],
            "",
            "回答要求:",
            "- 优先给出可执行、可验证的结论，而不是空泛表述。",
            "- 涉及交易建议时，请同时说明依据、风险点、失效条件。",
            "- 如果系统补充了证据摘要，请优先基于证据回答，并明确哪些结论来自现有证据。",
            "- 如果上下文不足，请明确指出缺少什么数据。",
            "- 如果用户明确要求你连接券商，请不要声称已经执行；你只能提出动作意图，等待界面确认执行。",
        ]

        if task_mode in (
            TASK_MODE_TRADE_DECISION,
            TASK_MODE_SYMBOL_ANALYSIS,
            TASK_MODE_WATCHLIST_SCAN,
            TASK_MODE_POSITION_DIAGNOSIS,
        ):
            try:
                market_lines = _market_context_provider(context) if _market_context_provider else []
                if market_lines:
                    sections.extend(["", *market_lines])
            except Exception as exc:
                logger.debug("Failed to inject market context: %s", exc)

        if task_mode == TASK_MODE_TRADE_DECISION:
            sections.extend([
                "",
                "交易决策专项要求:",
                "- 你必须结合当前大盘环境做出交易决策。若大盘环境恶劣，应提高风控要求、降低仓位、收紧止损。",
                "- 你必须同时站在看多和看空两个角度分析，不能只给出单方面观点。",
                "- 你必须在回答中输出一个 <trade_decision> JSON 块，包含明确的操作方向、目标价、止损价、仓位、置信度。",
                "- confidence 要反映你对此决策的真实把握程度，不要一律给高值。",
                "- 如果证据不足以做出明确买卖判断，请将 action 设为 hold 并说明原因。",
                "- stop_loss_price 必须设置，买入时止损价应低于现价，卖出时止盈价应高于现价。",
                "- position_pct 表示此笔交易占总资产的比例，请根据置信度和风险合理设置（通常 5%~20%）。",
                "- 买入时须关注组合行业集中度，若同行业已持仓较多，应降低仓位或回避。",
                "- 若持仓行业暴露数据可用，请在分析中体现行业分散化考量。",
            ])
            if context.broker.connected:
                sections.extend([
                    "",
                    "账户信息（请据此调整仓位建议）:",
                    f"- 可用资金: {context.broker.available_cash:,.0f}",
                    f"- 总资产: {context.broker.total_asset:,.0f}",
                    f"- 当前持仓数: {context.broker.position_count}",
                ])

        sections.extend([
            "",
            "券商连接动作协议:",
            '- 当且仅当用户明确要求连接券商时，在正文最后额外输出一个 `<agent_action>` JSON 块。',
            '- 允许的 action 只有: `connect_broker`。',
            '- 示例: `<agent_action>{\"action\":\"connect_broker\",\"reason\":\"用户要求连接券商会话\",\"requires_confirmation\":true}</agent_action>`',
            "- 除这个 JSON 块外，仍然要先用自然语言说明你为什么建议执行该动作。",
            "",
            "输出协议:",
            build_contract_with_citations(task_mode),
        ])
        return "\n".join(section for section in sections if section is not None)

    @staticmethod
    def build_quick_task_prompt(
        task_mode: str,
        context: AgentRuntimeContext,
        extra_instructions: str = "",
    ) -> str:
        if task_mode == TASK_MODE_SYMBOL_ANALYSIS:
            symbol = context.symbol
            return (
                f"请基于当前上下文，对 {symbol.name or '-'}({symbol.code or '-'}) 输出一份结构化分析，"
                "请尽量综合技术面、消息面、基本面三类证据，至少包含：趋势判断、关键支撑/压力位、量价特征、"
                "消息面催化/风险、基本面结论、风险点、操作建议、失效条件。"
                "请严格遵循系统提示中的输出协议。"
            )

        if task_mode == TASK_MODE_WATCHLIST_SCAN:
            group_name = context.watchlist.group_name or "当前列表"
            preview = "、".join(context.watchlist.visible_codes[:12]) or "无可用标的"
            return (
                f"请对 {group_name} 做一轮 AI 巡检。标的样本包括：{preview}。"
                "请输出：最值得关注的3只、风险最高的3只、值得继续跟踪的理由，以及下一步观察清单。"
                "请严格遵循系统提示中的输出协议。"
            )

        if task_mode == TASK_MODE_TRADE_DECISION:
            symbol = context.symbol
            if not symbol.is_available:
                return "请先选择一个标的后再使用交易决策功能。"
            prompt_parts = [
                f"请对 {symbol.name or '-'}({symbol.code or '-'}) 进行交易决策分析。",
                "要求：",
                "1. 先评估当前大盘/市场环境对个股的影响；",
                "2. 同时分析看多和看空理由，引用证据；",
                "3. 给出综合研判和操作建议；",
                "4. 在 <trade_decision> 标签中输出结构化 JSON 决策（含 action/target_price/stop_loss_price/confidence/position_pct/risk_score 等）；",
                "5. 说明风险点和失效条件。",
                "请严格遵循系统提示中的输出协议。",
            ]
            return "\n".join(prompt_parts)

        if task_mode == TASK_MODE_POSITION_DIAGNOSIS:
            return (
                "请基于当前账户持仓上下文做持仓诊断。请输出：仓位概览、主要风险暴露、"
                "应优先复盘的持仓、需要防守的标的，以及后续跟踪建议。"
                "请严格遵循系统提示中的输出协议。"
            )

        if extra_instructions:
            return extra_instructions
        return "请结合当前股票软件上下文回答用户问题。"

    @staticmethod
    def build_context_brief(context: AgentRuntimeContext) -> str:
        return "\n".join(f"- {line}" for line in context.to_summary_lines())


__all__ = [
    "AgentPromptBuilder",
    "MarketContextProvider",
    "set_market_context_provider",
]
