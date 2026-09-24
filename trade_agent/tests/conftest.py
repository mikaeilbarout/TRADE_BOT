from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config.settings import Settings
from app.models.agent_decision import (
    DimensionScores,
    FinalDecisionResult,
    NewsAgentResult,
    SentimentAgentResult,
    TechnicalAgentResult,
)
from app.models.enums import (
    AgentAgreement,
    Bias,
    ContradictionLevel,
    FinalDecision,
    GateDecision,
    MarketEnvironment,
    SentimentStrength,
    TechnicalDecision,
    TradeCompatibility,
)
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal
from app.providers.llm.mock_provider import MockLLMProvider
from app.services.risk_service import AccountState

# Balance large enough that the default 0.5%-per-trade limit passes for the
# standard 0.1-lot XAUUSD test signal (10 points x 0.1 x 100 = $100 risk).
TEST_BALANCE = 100_000.0


def make_settings(**overrides) -> Settings:
    defaults = dict(
        llm_provider="mock",
        market_data_provider="mock",
        news_provider="mock",
        sentiment_provider="mock",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_account(**overrides) -> AccountState:
    defaults = dict(balance=TEST_BALANCE)
    defaults.update(overrides)
    return AccountState(**defaults)


def make_signal(**overrides) -> TradeSignal:
    defaults = dict(
        symbol="XAUUSD",
        side="BUY",
        entry=3650.0,
        stop_loss=3640.0,
        take_profit=3670.0,
        volume=0.1,
        timeframe="M5",
        strategy="test_strategy",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return TradeSignal(**defaults)


def make_market_snapshot(**overrides) -> MarketSnapshot:
    defaults = dict(
        symbol="XAUUSD",
        bid=3649.9,
        ask=3650.1,
        last_price=3650.0,
        spread=0.2,
        session="LONDON",
        atr=2.0,
        volatility_pct=0.05,
        quote_timestamp=datetime.now(timezone.utc),
        is_stale=False,
        freshness_seconds=0.5,
        entry_timeframe="M5",
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


def make_news_result(**overrides) -> NewsAgentResult:
    defaults = dict(
        symbol="XAUUSD",
        signal_side="BUY",
        confidence=0.85,
        reasoning="USD weakness and falling yields support gold.",
        decision=GateDecision.PASS,
        news_bias=Bias.BULLISH,
        market_environment=MarketEnvironment.SUPPORTIVE,
        major_events_detected=False,
        high_impact_event_within_minutes=False,
        risk_of_news_reversal="LOW",
        summary="Recent USD weakness and falling yields support gold.",
        sources=[],
        agreement_with_previous=AgentAgreement.NOT_APPLICABLE,
        independent_finding="Yields fell after the auction, not on data.",
    )
    defaults.update(overrides)
    return NewsAgentResult(**defaults)


def make_sentiment_result(**overrides) -> SentimentAgentResult:
    defaults = dict(
        symbol="XAUUSD",
        signal_side="BUY",
        confidence=0.78,
        reasoning="Desk commentary is constructive; retail chatter is noise.",
        decision=GateDecision.PASS,
        overall_sentiment=Bias.BULLISH,
        sentiment_score=0.6,
        sentiment_strength=SentimentStrength.MODERATE,
        sentiment_momentum="RISING",
        contradiction_level=ContradictionLevel.LOW,
        trade_compatibility=TradeCompatibility.SUPPORT,
        low_quality_source_ratio=0.33,
        manipulation_suspected=False,
        summary="Sentiment moderately bullish, supports the trade.",
        agreement_with_previous=AgentAgreement.PARTIAL,
        agreement_explanation="Agree on direction but positioning is more crowded than news implies.",
        independent_finding="Positioning is stretched, which caps upside follow-through.",
    )
    defaults.update(overrides)
    return SentimentAgentResult(**defaults)


def make_technical_result(**overrides) -> TechnicalAgentResult:
    defaults = dict(
        symbol="XAUUSD",
        confidence=0.8,
        reasoning="H4/H1 uptrend, breakout confirmed on M15.",
        decision=TechnicalDecision.PASS,
        higher_tf_trend="UPTREND",
        aligned_with_higher_tf=True,
        entry_valid=True,
        is_overextended=False,
        breakout_confirmed=True,
        false_breakout_risk="LOW",
        risk_reward_ratio=2.0,
        stop_loss_logical=True,
        take_profit_realistic=True,
        volatility_acceptable=True,
        confluence_factors=["uptrend on H4", "confirmed breakout on M15"],
        conflicting_factors=[],
        summary="Technical setup is valid as proposed.",
        agreement_with_previous=AgentAgreement.AGREE,
        agreement_explanation="Chart independently confirms the bullish read.",
        independent_finding="M15 breakout retested and held, which neither upstream agent saw.",
    )
    defaults.update(overrides)
    return TechnicalAgentResult(**defaults)


def make_final_result(**overrides) -> FinalDecisionResult:
    defaults = dict(
        symbol="XAUUSD",
        confidence=0.82,
        reasoning="All dimensions supportive, risk acceptable.",
        decision=FinalDecision.APPROVE,
        scores=DimensionScores(
            news_score=85, sentiment_score=78, technical_score=80, risk_score=90,
            weighted_total=83,
        ),
        veto_triggered=False,
        veto_reason=None,
        chain_conflicts=[],
        summary="News, sentiment, and technical all support this trade.",
        agreement_with_previous=AgentAgreement.AGREE,
        agreement_explanation="Chain is coherent; technical evidence weighted highest.",
        independent_finding="All three dimensions align without relying on the weakest source.",
    )
    defaults.update(overrides)
    return FinalDecisionResult(**defaults)


def make_llm(**result_overrides) -> MockLLMProvider:
    """Mock LLM wired with a full, healthy chain of canned agent results.
    Pass e.g. final=make_final_result(decision=...) to vary one link."""
    return MockLLMProvider(
        responses={
            "NewsAgentResult": result_overrides.get("news", make_news_result()),
            "SentimentAgentResult": result_overrides.get(
                "sentiment", make_sentiment_result()
            ),
            "TechnicalAgentResult": result_overrides.get(
                "technical", make_technical_result()
            ),
            "FinalDecisionResult": result_overrides.get("final", make_final_result()),
        }
    )


def build_test_pipeline(llm: MockLLMProvider, **settings_overrides):
    """The same wiring the API uses, with mock providers -- so integration
    tests exercise the real chain rather than a test-only arrangement.

    Pinned to the four-agent chain regardless of Settings' own default
    (single_agent since 2026-09-19): every test using this helper asserts
    on that chain's own behavior by name (agent order, independence,
    etc.), so it needs the four-agent pipeline specifically, not whichever
    mode happens to be the production default. Pass
    pipeline_mode="single_agent" explicitly to exercise the other one.
    """
    from app.bootstrap import build_pipeline

    settings = make_settings(pipeline_mode="four_agent", **settings_overrides)
    pipeline, _risk = build_pipeline(settings, llm)
    return pipeline


@pytest.fixture
def signal_factory():
    return make_signal


@pytest.fixture
def account_factory():
    return make_account


@pytest.fixture
def news_result_factory():
    return make_news_result


@pytest.fixture
def sentiment_result_factory():
    return make_sentiment_result


@pytest.fixture
def technical_result_factory():
    return make_technical_result


@pytest.fixture
def final_result_factory():
    return make_final_result


@pytest.fixture(autouse=True)
def isolate_ingestion_credentials(monkeypatch):
    """Keep every test isolated from the machine's real credentials.

    Two hazards this closes. A developer with a live `FRED_API_KEY` in `.env`
    would otherwise see different test outcomes from CI -- and worse, tests
    asserting a "no key" refusal would instead attempt real network calls and
    hang on timeouts. So the `.env` loader is disabled and the credential
    variables are cleared for the duration of every test.

    Patching `research_data.cli.load_env_file` only blocks that package's OWN
    lightweight loader. `app.config.settings.Settings` and
    `research.ai.settings.AISettings` are pydantic-settings `BaseSettings`
    subclasses with their own `env_file=".env"` -- pydantic-settings reads
    that file directly during `__init__`, entirely bypassing
    `research_data.cli.load_env_file` and unaffected by `monkeypatch.delenv`
    below. Without also neutralizing THIS loader, any test that builds a
    fresh `Settings()`/`AISettings()` (e.g. `ExperimentConfig.ai_settings()`
    when `config.ai is None`) still reads the real `.env` on disk -- which,
    on a machine with a live ANTHROPIC_API_KEY configured for actual trading,
    means a "no key configured" test silently makes a real, billed API call
    instead of asserting the refusal it claims to test.
    """
    import research_data.cli as ingest_cli
    from app.config.settings import Settings
    from research.ai.settings import AISettings

    monkeypatch.setattr(ingest_cli, "load_env_file", lambda *a, **k: [])
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    monkeypatch.setattr(AISettings, "model_config", {**AISettings.model_config, "env_file": None})
    for name in ("FRED_API_KEY", "ANTHROPIC_API_KEY", "NEWSAPI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
