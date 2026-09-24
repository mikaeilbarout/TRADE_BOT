from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.base import AgentError
from app.agents.news_agent import NewsAgent
from app.config.assets import get_asset_meta
from app.models.enums import FinalDecision, GateDecision, TechnicalDecision
from app.models.news import NewsBundle
from app.models.trade import ModifiedTrade
from app.providers.llm.mock_provider import MockLLMProvider
from app.providers.market_data.base import MarketDataProvider, MarketDataProviderError
from app.providers.news.base import NewsProvider, NewsProviderError
from app.providers.news.mock_provider import MockNewsProvider
from app.services.market_data import MarketDataService, MarketDataUnavailableError
from app.services.news_service import NewsService, NewsUnavailableError
from tests.conftest import (
    build_test_pipeline,
    make_account,
    make_final_result,
    make_llm,
    make_market_snapshot,
    make_news_result,
    make_sentiment_result,
    make_settings,
    make_signal,
    make_technical_result,
)


class _AlwaysFailsMarketData(MarketDataProvider):
    name = "always_fails"

    async def get_quote(self, symbol: str):
        raise MarketDataProviderError("feed disconnected")

    async def get_candles(self, symbol: str, timeframe: str, count: int = 200):
        raise MarketDataProviderError("feed disconnected")


class _AlwaysFailsNews(NewsProvider):
    name = "always_fails"

    async def fetch(self, query_terms: list[str], lookback_minutes: int):
        raise NewsProviderError("news vendor 500")


async def test_market_data_service_fails_closed_on_provider_error():
    service = MarketDataService(_AlwaysFailsMarketData(), make_settings())
    with pytest.raises(MarketDataUnavailableError):
        await service.get_snapshot("XAUUSD", ["H1"])


async def test_news_service_fails_closed_when_no_cache_available():
    service = NewsService(_AlwaysFailsNews(), make_settings())
    with pytest.raises(NewsUnavailableError):
        await service.get_news("XAUUSD")


async def test_news_service_serves_degraded_cache_on_transient_failure():
    class FlakyProvider(NewsProvider):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        async def fetch(self, query_terms, lookback_minutes):
            self.calls += 1
            if self.calls == 1:
                return await MockNewsProvider().fetch(query_terms, lookback_minutes)
            raise NewsProviderError("temporary outage")

    service = NewsService(FlakyProvider(), make_settings(news_cache_ttl_seconds=0))
    first = await service.get_news("XAUUSD")
    assert not first.is_degraded

    second = await service.get_news("XAUUSD")
    assert second.is_degraded is True
    assert second.degraded_reason is not None


async def test_news_service_refuses_cache_older_than_max_fallback_age():
    """A very old cached bundle must not be used to decide a trade -- better
    to fail closed than to reason about ancient headlines."""

    class FlakyProvider(NewsProvider):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        async def fetch(self, query_terms, lookback_minutes):
            self.calls += 1
            if self.calls == 1:
                return await MockNewsProvider().fetch(query_terms, lookback_minutes)
            raise NewsProviderError("outage")

    settings = make_settings(news_cache_ttl_seconds=0, news_max_fallback_age_seconds=0)
    service = NewsService(FlakyProvider(), settings)
    await service.get_news("XAUUSD")
    with pytest.raises(NewsUnavailableError):
        await service.get_news("XAUUSD")


async def test_agent_raises_on_malformed_llm_output():
    def bad_factory(system_prompt, user_prompt, response_model):
        raise ValidationError.from_exception_data("NewsAgentResult", [])

    agent = NewsAgent(MockLLMProvider(factory=bad_factory), timeout_seconds=5.0, max_retries=0)
    with pytest.raises(AgentError):
        await agent.run(
            make_signal(), make_market_snapshot(), NewsBundle(symbol="XAUUSD", items=[]),
            get_asset_meta("XAUUSD"),
        )


async def test_modified_trade_with_inverted_levels_is_rejected_at_schema_level():
    """An LLM proposing a BUY with its stop above entry never reaches the
    execution layer -- the schema refuses it."""
    with pytest.raises(ValidationError):
        ModifiedTrade(
            symbol="XAUUSD", side="BUY", entry=3650, stop_loss=3660, take_profit=3670
        )


async def test_pipeline_resolves_conflicting_chain_without_crashing():
    """News bullish/PASS, sentiment supportive, technical strongly conflicting.
    The chain must resolve to a decision, never raise. With
    veto_on_technical_block explicitly enabled, a technical BLOCK still
    force-rejects even an APPROVE from final_decision_agent (the mechanism
    itself still exists -- see app/config/settings.py, it is just off by
    default for the live service as of 2026-09-18)."""
    pipeline = build_test_pipeline(
        make_llm(
            technical=make_technical_result(
                decision=TechnicalDecision.BLOCK,
                aligned_with_higher_tf=False,
                higher_tf_trend="DOWNTREND",
                summary="Counter-trend on H4 with no confluence.",
            ),
            final=make_final_result(decision=FinalDecision.APPROVE),
        ),
        veto_on_technical_block=True,
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.REJECT
    assert result.veto_triggered is True
    assert result.technical_result.decision == TechnicalDecision.BLOCK


async def test_pipeline_fails_closed_when_each_agent_is_unreachable():
    """Every link in the chain, one at a time: a missing canned response
    simulates that agent's LLM call failing. All must fail closed."""
    full = {
        "NewsAgentResult": make_news_result(),
        "SentimentAgentResult": make_sentiment_result(),
        "TechnicalAgentResult": make_technical_result(),
        "FinalDecisionResult": make_final_result(),
    }
    for missing in list(full):
        responses = {k: v for k, v in full.items() if k != missing}
        pipeline = build_test_pipeline(MockLLMProvider(responses=responses))
        result = await pipeline.run(make_signal(), make_account())
        assert result.decision == FinalDecision.REJECT, missing
        assert result.degraded is True, missing
        assert result.errors, missing


async def test_pipeline_fails_closed_when_market_data_is_down():
    from app.agents.final_decision_agent import FinalDecisionAgent
    from app.agents.sentiment_agent import SentimentAgent
    from app.agents.technical_agent import TechnicalAgent
    from app.services.decision_policy import DecisionPolicy
    from app.services.pipeline import DecisionPipeline
    from app.services.risk_service import RiskService
    from app.services.sentiment_service import SentimentService
    from app.providers.sentiment.mock_provider import MockSentimentProvider

    settings = make_settings()
    llm = make_llm()
    pipeline = DecisionPipeline(
        settings=settings,
        market_data_service=MarketDataService(_AlwaysFailsMarketData(), settings),
        news_service=NewsService(MockNewsProvider(), settings),
        sentiment_service=SentimentService(MockSentimentProvider(), settings),
        risk_service=RiskService(settings),
        decision_policy=DecisionPolicy(settings),
        news_agent=NewsAgent(llm, settings.agent_timeout_seconds),
        sentiment_agent=SentimentAgent(llm, settings.agent_timeout_seconds),
        technical_agent=TechnicalAgent(llm, settings.agent_timeout_seconds),
        final_agent=FinalDecisionAgent(llm, settings.agent_timeout_seconds),
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.REJECT
    assert result.degraded is True
    assert result.agent_traces == []  # never paid for an LLM call


async def test_news_block_without_imminent_event_still_rejects():
    """BLOCK that isn't time-sensitive is a REJECT, not a WAIT, when
    veto_on_news_block is explicitly enabled (off by default, see
    app/config/settings.py)."""
    pipeline = build_test_pipeline(
        make_llm(
            news=make_news_result(
                decision=GateDecision.BLOCK, high_impact_event_within_minutes=False
            )
        ),
        short_circuit_on_critical_news=True,
        veto_on_news_block=True,
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.REJECT
    assert result.short_circuited is False  # full chain still ran


async def test_invalid_settings_weights_are_rejected_at_startup():
    with pytest.raises(ValidationError):
        make_settings(weight_news=0.9, weight_sentiment=0.9)
