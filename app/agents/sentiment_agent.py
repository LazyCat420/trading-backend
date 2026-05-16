"""
Sentiment Analysis Agent — uses pre-computed data from sentiment_processor.
LLM analyzes headline lists and volume trends — never scrapes or fetches.
"""

from app.agents.base_agent import run_agent
from app.processors.sentiment_processor import get_signals

SYSTEM_PROMPT = """You are a market sentiment analyst. You will receive recent news headlines 
and volume data for a stock. Your job is to INTERPRET the sentiment.

You have access to web search tools (`hermes_web_research`). If the provided headlines are insufficient, ambiguous, or out-of-date, you MUST use the hermes_web_research tool to look up the latest news and social sentiment before making your final decision.

CRITICAL:
1. First, analyze the provided data.
2. If you need more context, call `hermes_web_research`.
3. Finally, when you have enough data, respond with JSON matching this exact format:
{"sentiment_score": -1.0 to 1.0, "trend": "improving|stable|degrading",
"catalyst_detected": true|false, "key_headlines": ["headline1", "headline2"],
"confidence": 0-100, "summary": "2-3 sentence analysis"}
Do NOT output anything other than JSON in your final response."""


async def run(ticker: str, cycle_id: str, bot_id: str) -> dict:
    signals = get_signals(ticker)

    return await run_agent(
        agent_name="sentiment",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=SYSTEM_PROMPT,
        data_context=signals,
        user_prompt=f"Based on the news and social data above, assess market sentiment for {ticker}.",
        max_tokens=1024,
        enable_dynamic_prompt=True,
        enable_tools=True,
    )
