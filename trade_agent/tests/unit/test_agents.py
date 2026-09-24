from __future__ import annotations

import asyncio

import pytest

from app.agents.base import AgentError
from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.news_agent import NewsAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.config.assets import get_asset_meta
from app.models.news import NewsBundle
from app.models.sentiment import SentimentBundle, SentimentItem, SentimentKind, SourceQuality
from app.providers.llm.mock_provider import MockLLMProvider
from tests.conftest import (
    make_account,
    make_final_result,
    make_market_snapshot,
    make_news_result,
    make_sentiment_result,
    make_settings,
    make_signal,
    make_technical_result,
)


def _sentiment_bundle() -> SentimentBundle:
    from datetime import datetime, timezone

    return SentimentBundle(
        symbol="XAUUSD",
        items=[
            SentimentItem(
                source="desk",
                kind=SentimentKind.MARKET_COMMENTARY,
                quality=SourceQuality.HIGH,
                text="Constructive tone into the London fix.",
                timestamp=datetime.now(timezone.utc),
                score=0.3,
            )
        ],
    )


async def test_news_agent_payload_includes_signal_and_market_conditions():
    llm = MockLLMProvider(responses={"NewsAgentResult": make_news_result()})
    agent = NewsAgent(llm, timeout_seconds=5.0, model_name="test-model")
    run = await agent.run(
        make_signal(), make_market_snapshot(), NewsBundle(symbol="XAUUSD", items=[]),
        get_asset_meta("XAUUSD"),
    )
    assert run.result.decision.value == "PASS"
    assert run.result.model == "test-model"  # stamped for the audit trail
    assert "signal" in run.input_snapshot
    assert "market_conditions" in run.input_snapshot
    assert run.input_snapshot["market_conditions"]["last_price"] == 3650.0


async def test_sentiment_agent_is_independent_with_own_sources():
    # Parallelized 2026-09-18: sentiment no longer receives the news
    # agent's chain -- it runs concurrently with news/technical, seeing
    # only the signal, market, and its own sentiment evidence base.
    llm = MockLLMProvider(responses={"SentimentAgentResult": make_sentiment_result()})
    agent = SentimentAgent(llm, timeout_seconds=5.0)
    run = await agent.run(make_signal(), make_market_snapshot(), _sentiment_bundle())
    snapshot = run.input_snapshot
    assert "upstream_chain" not in snapshot
    assert snapshot["sentiment_sources"]["item_count"] == 1
    assert snapshot["sentiment_sources"]["items"][0]["source_quality"] == "HIGH"


async def test_technical_agent_is_independent_with_full_ohlcv():
    # Parallelized 2026-09-18: technical no longer receives the news/
    # sentiment chain -- it runs concurrently with them, seeing only the
    # signal and full market data.
    from app.models.market_data import Candle, TimeframeSeries
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    candles = [
        Candle(
            timestamp=now - timedelta(minutes=5 * i),
            open=3650, high=3652, low=3648, close=3651, volume=10,
        )
        for i in range(80, 0, -1)
    ]
    market = make_market_snapshot(
        timeframes={"M5": TimeframeSeries(timeframe="M5", candles=candles)}
    )

    llm = MockLLMProvider(responses={"TechnicalAgentResult": make_technical_result()})
    agent = TechnicalAgent(llm, timeout_seconds=5.0)
    run = await agent.run(make_signal(), market)
    snapshot = run.input_snapshot
    assert "upstream_chain" not in snapshot
    ohlcv = snapshot["market_conditions"]["recent_ohlcv_by_timeframe"]["M5"]
    assert len(ohlcv) == 60  # tail only (candles_per_timeframe=60), so the prompt stays bounded


async def test_final_agent_receives_complete_chain_and_policy():
    llm = MockLLMProvider(responses={"FinalDecisionResult": make_final_result()})
    agent = FinalDecisionAgent(llm, timeout_seconds=5.0)
    run = await agent.run(
        make_signal(),
        make_market_snapshot(),
        make_news_result(),
        make_sentiment_result(),
        make_technical_result(),
        make_account(),
        make_settings(),
    )
    snapshot = run.input_snapshot
    assert [link["agent"] for link in snapshot["agent_chain"]] == [
        "news_agent",
        "sentiment_agent",
        "technical_agent",
    ]
    assert snapshot["policy"]["dimension_weights"]["technical"] == 0.35
    assert snapshot["risk_context"]["account_balance_reported"] is True


async def test_agent_raises_after_exhausting_retries():
    llm = MockLLMProvider(fail=True)
    agent = NewsAgent(llm, timeout_seconds=5.0, max_retries=2)
    with pytest.raises(AgentError) as exc_info:
        await agent.run(
            make_signal(), make_market_snapshot(), NewsBundle(symbol="XAUUSD", items=[]),
            get_asset_meta("XAUUSD"),
        )
    assert "3 attempt(s)" in str(exc_info.value)


async def test_agent_retry_recovers_from_transient_failure():
    calls = {"n": 0}

    def flaky(system_prompt, user_prompt, response_model):
        calls["n"] += 1
        if calls["n"] == 1:
            from app.providers.llm.base import LLMProviderError

            raise LLMProviderError("transient 529 overloaded")
        return make_news_result()

    agent = NewsAgent(MockLLMProvider(factory=flaky), timeout_seconds=5.0, max_retries=1)
    run = await agent.run(
        make_signal(), make_market_snapshot(), NewsBundle(symbol="XAUUSD", items=[]),
        get_asset_meta("XAUUSD"),
    )
    assert run.attempts == 2
    assert run.result.decision.value == "PASS"


async def test_agent_times_out_when_llm_hangs():
    async def slow_factory(system_prompt, user_prompt, response_model):
        await asyncio.sleep(1.0)
        return make_news_result()

    agent = NewsAgent(
        MockLLMProvider(factory=slow_factory), timeout_seconds=0.05, max_retries=0
    )
    with pytest.raises(AgentError) as exc_info:
        await agent.run(
            make_signal(), make_market_snapshot(), NewsBundle(symbol="XAUUSD", items=[]),
            get_asset_meta("XAUUSD"),
        )
    assert "timed out" in str(exc_info.value)
