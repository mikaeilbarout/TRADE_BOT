from __future__ import annotations

from datetime import datetime

from app.agents.base import AgentRun, BaseAgent
from app.agents.context import full_technical_conditions, signal_payload
from app.config.settings import Settings
from app.models.agent_decision import UnifiedDecisionResult
from app.models.market_data import MarketSnapshot
from app.models.news import NewsBundle
from app.models.sentiment import SentimentBundle
from app.models.signal import TradeSignal
from app.services.risk_service import AccountState


class UnifiedAgent(BaseAgent):
    """Single-agent replacement for the news/sentiment/technical/final
    four-agent chain: one call sees the full multi-timeframe technical
    picture, news, sentiment and risk context together and makes one
    decision. There is no upstream chain to weigh."""

    prompt_file = "unified_agent.md"
    response_model = UnifiedDecisionResult
    agent_name = "unified_agent"
    task_instruction = (
        "Review the complete picture below -- market structure, news, sentiment and "
        "risk context -- and decide once: APPROVE, REJECT or WAIT."
    )

    def build_payload(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        news: NewsBundle,
        sentiment: SentimentBundle,
        account: AccountState,
        settings: Settings,
        now: datetime | None = None,
    ) -> dict:
        return {
            "signal": signal_payload(signal, now),
            "market_conditions": full_technical_conditions(market),
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
            "risk_context": {
                "account_balance_reported": account.balance is not None,
                "daily_loss_pct": account.daily_loss_pct,
                "trades_today": account.trades_today,
                "open_positions": account.open_positions,
                "exposure_by_asset_pct": account.exposure_by_asset_pct,
                "current_leverage": account.current_leverage,
                "market_open": account.market_open,
                "recent_loss_streak": account.recent_loss_streak,
            },
            "policy": {
                "min_confidence_required": settings.min_confidence,
                "high_impact_news_blackout_minutes": settings.high_impact_news_blackout_minutes,
                # None when this signal's strategy has no validated ADX
                # backstop (e.g. donchian_h1) -- the deterministic override
                # will not fire either way, so weigh D1 ADX only as far as
                # your own judgment finds it useful.
                "strong_trend_adx_threshold_for_this_strategy": (
                    settings.single_agent_strong_trend_adx_thresholds.get(signal.strategy)
                ),
            },
        }

    async def run(
        self,
        signal: TradeSignal,
        market: MarketSnapshot,
        news: NewsBundle,
        sentiment: SentimentBundle,
        account: AccountState,
        settings: Settings,
        now: datetime | None = None,
    ) -> AgentRun:
        return await self._call(
            self.build_payload(signal, market, news, sentiment, account, settings, now)
        )
