from __future__ import annotations

from datetime import datetime

from app.agents.base import AgentRun, BaseAgent
from app.agents.context import market_conditions, signal_payload
from app.models.agent_decision import SentimentAgentResult
from app.models.market_data import MarketSnapshot
from app.models.sentiment import SentimentBundle
from app.models.signal import TradeSignal


class SentimentAgent(BaseAgent):
    """One of three specialist agents (news, sentiment, technical) that run
    concurrently -- signal + market conditions + its OWN sentiment evidence
    base, independent of the other two. Only the final decision agent sees
    all three findings together.

    Parallelized 2026-09-18 (was sequential, seeing the news agent's
    findings first): the news->sentiment->technical chain added ~40s of
    pure latency for a benefit (each specialist critiquing the previous
    one) that final_agent's own chain_conflicts resolution already does at
    the synthesis step. The AI review's own latency was found to be the
    direct cause of most MODIFY verdicts (the original entry going stale
    during the wait), so cutting it took priority over that layered
    critique. The separate sentiment bundle is still what makes this
    agent's read independent -- without its own source material it could
    only guess at market mood.
    """

    prompt_file = "sentiment_agent.md"
    response_model = SentimentAgentResult
    agent_name = "sentiment_agent"
    task_instruction = (
        "Independently assess market sentiment for this asset from the sentiment "
        "sources provided, then judge whether it supports or conflicts with the "
        "proposed trade."
    )

    def build_payload(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        sentiment: SentimentBundle,
        now: datetime | None = None,
    ) -> dict:
        return {
            "chain_position": "1 of 3 parallel specialists (sentiment) -> final decision",
            "signal": signal_payload(signal, now),
            "market_conditions": market_conditions(market),
            "sentiment_sources": {
                "lookback_minutes": sentiment.lookback_minutes,
                "sources_queried": sentiment.sources_queried,
                "data_is_degraded": sentiment.is_degraded,
                "degraded_reason": sentiment.degraded_reason,
                "low_quality_source_ratio": round(sentiment.low_quality_ratio, 3),
                "item_count": len(sentiment.items),
                "items": [
                    {
                        "source": item.source,
                        "kind": item.kind.value,
                        "source_quality": item.quality.value,
                        "text": item.text,
                        "timestamp": item.timestamp,
                        "age_minutes": round(item.age_minutes(now), 1),
                        "vendor_score": item.score,
                    }
                    for item in sentiment.items
                ],
            },
        }

    async def run(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        sentiment: SentimentBundle,
        now: datetime | None = None,
    ) -> AgentRun:
        return await self._call(self.build_payload(signal, market, sentiment, now))
