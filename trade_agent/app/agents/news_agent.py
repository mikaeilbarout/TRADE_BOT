from __future__ import annotations

from datetime import datetime

from app.agents.base import AgentRun, BaseAgent
from app.agents.context import market_conditions, signal_payload
from app.config.assets import AssetMeta
from app.models.agent_decision import NewsAgentResult
from app.models.market_data import MarketSnapshot
from app.models.news import NewsBundle
from app.models.signal import TradeSignal


class NewsAgent(BaseAgent):
    """One of three parallel specialists. Has no upstream analysis to
    weigh, so its independence is structural: it sees only the signal,
    live market conditions, and the news bundle."""

    prompt_file = "news_agent.md"
    response_model = NewsAgentResult
    agent_name = "news_agent"
    task_instruction = (
        "Judge whether the proposed trade is compatible with the current news "
        "and macro environment for this asset."
    )

    def build_payload(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        news: NewsBundle,
        asset_meta: AssetMeta,
        now: datetime | None = None,
    ) -> dict:
        return {
            "chain_position": "1 of 3 parallel specialists (news & macro) -> final decision",
            "signal": signal_payload(signal, now),
            "market_conditions": market_conditions(market),
            "asset_profile": {
                "asset_type": asset_meta.asset_type.value,
                "base_asset": asset_meta.base_asset,
                "quote_asset": asset_meta.quote_asset,
                "relevant_factors": asset_meta.relevant_factors,
                "news_categories": asset_meta.news_categories,
            },
            "news": {
                "lookback_minutes": news.lookback_minutes,
                "sources_queried": news.sources_queried,
                "data_is_degraded": news.is_degraded,
                "degraded_reason": news.degraded_reason,
                "item_count": len(news.items),
                "items": [
                    {
                        "title": item.title,
                        "source": item.source,
                        "timestamp": item.timestamp,
                        "age_minutes": round(item.age_minutes(now), 1),
                        "is_breaking": item.is_breaking,
                        "summary": item.summary,
                        "category": item.category,
                    }
                    for item in news.items
                ],
            },
        }

    async def run(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        news: NewsBundle,
        asset_meta: AssetMeta,
        now: datetime | None = None,
    ) -> AgentRun:
        return await self._call(self.build_payload(signal, market, news, asset_meta, now))
