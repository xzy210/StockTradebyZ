"""Trading-side compatibility wrapper for ``common.agent.agent_prompt_builder``."""
from common.agent.agent_prompt_builder import *  # noqa: F401,F403


def _trading_market_context_provider(context):
    from trading_app.services.market_data.market_context_service import MarketContextService

    return MarketContextService().build_snapshot(run_context=context.run_context).to_prompt_lines()


set_market_context_provider(_trading_market_context_provider)
