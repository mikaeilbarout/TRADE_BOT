from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.backtest.harness import _build_pipeline_for_record, backtest_settings, load_fixtures, run_backtest
from app.backtest.historical_providers import (
    HistoricalMarketDataProvider,
    HistoricalNewsProvider,
    HistoricalSentimentProvider,
    LookaheadViolationError,
)
from app.backtest.simulate import simulate_outcome
from app.models.enums import Bias, FinalDecision, ImpactLevel
from app.models.market_data import Candle
from app.models.news import NewsItem
from app.models.sentiment import SentimentItem, SentimentKind, SourceQuality
from app.models.trade import ModifiedTrade
from tests.conftest import make_account, make_final_result, make_llm, make_settings

FIXTURE = "fixtures/sample_backtest.json"


def _candle(ts, o, h, l, c):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=10)


def _backtest_settings(**overrides):
    return make_settings(
        default_higher_timeframes=["H4"],
        default_medium_timeframes=[],
        default_entry_timeframes=["M15"],
        **overrides,
    )


def test_simulate_outcome_win_on_buy():
    now = datetime.now(timezone.utc)
    candles = [_candle(now + timedelta(minutes=5), 100, 112, 99, 111)]
    outcome = simulate_outcome("BUY", 100, 95, 110, candles)
    assert outcome.status == "WIN"
    assert outcome.r_multiple == pytest.approx(2.0)


def test_simulate_outcome_loss_on_buy():
    now = datetime.now(timezone.utc)
    candles = [_candle(now + timedelta(minutes=5), 100, 101, 93, 95)]
    outcome = simulate_outcome("BUY", 100, 95, 110, candles)
    assert outcome.status == "LOSS"
    assert outcome.r_multiple == -1.0


def test_simulate_outcome_assumes_stop_first_when_both_hit_same_candle():
    now = datetime.now(timezone.utc)
    candles = [_candle(now + timedelta(minutes=5), 100, 115, 90, 112)]
    assert simulate_outcome("BUY", 100, 95, 110, candles).status == "LOSS"


def test_simulate_outcome_not_filled_when_limit_never_touched():
    now = datetime.now(timezone.utc)
    candles = [_candle(now + timedelta(minutes=5), 120, 122, 118, 121)]
    outcome = simulate_outcome("BUY", 100, 95, 110, candles, immediate_fill=False)
    assert outcome.status == "NOT_FILLED"
    assert outcome.r_multiple == 0.0


def test_historical_market_provider_rejects_lookahead_candles():
    as_of = datetime.now(timezone.utc)
    future = _candle(as_of + timedelta(hours=1), 100, 101, 99, 100)
    with pytest.raises(LookaheadViolationError):
        HistoricalMarketDataProvider(
            as_of=as_of, candles_by_timeframe={"H1": [future]}, bid=99.9, ask=100.1
        )


def test_historical_news_provider_rejects_lookahead_items():
    as_of = datetime.now(timezone.utc)
    with pytest.raises(LookaheadViolationError):
        HistoricalNewsProvider(
            as_of,
            [
                NewsItem(
                    title="Tomorrow's CPI print",
                    source="wire",
                    timestamp=as_of + timedelta(minutes=30),
                    impact=ImpactLevel.HIGH,
                    bias=Bias.BEARISH,
                )
            ],
        )


def test_historical_sentiment_provider_rejects_lookahead_items():
    as_of = datetime.now(timezone.utc)
    with pytest.raises(LookaheadViolationError):
        HistoricalSentimentProvider(
            as_of,
            [
                SentimentItem(
                    source="social",
                    kind=SentimentKind.SOCIAL,
                    quality=SourceQuality.LOW,
                    text="posted after the decision",
                    timestamp=as_of + timedelta(minutes=1),
                )
            ],
        )


def test_historical_providers_accept_naive_timestamps_as_utc():
    as_of = datetime(2025, 1, 6, 13, 0, 0)  # tz-naive, treated as UTC
    provider = HistoricalNewsProvider(
        as_of,
        [
            NewsItem(
                title="Earlier headline",
                source="wire",
                timestamp=datetime(2025, 1, 6, 12, 0, 0),
            )
        ],
    )
    assert provider is not None


def test_load_fixtures_parses_sample_file():
    fixtures = load_fixtures(FIXTURE)
    assert len(fixtures) == 1
    assert fixtures[0].signal.symbol == "XAUUSD"
    assert len(fixtures[0].future_candles) == 2
    assert fixtures[0].sentiment_items  # sentiment evidence is part of a fixture


def test_fixture_defaults_when_optional_lists_omitted():
    """Regression: these defaults were previously built with
    dataclasses.field() inside a Pydantic model, which produced a Field
    object instead of an empty list."""
    from app.backtest.harness import HistoricalSignalFixture
    from tests.conftest import make_signal

    fixture = HistoricalSignalFixture(
        signal=make_signal(), market_history={}, future_candles=[]
    )
    assert fixture.news_items == []
    assert fixture.sentiment_items == []


async def test_run_backtest_end_to_end_with_approve_decision():
    fixtures = load_fixtures(FIXTURE)
    report = await run_backtest(
        fixtures,
        _backtest_settings(),
        make_llm(final=make_final_result(decision=FinalDecision.APPROVE)),
        make_account(),
    )
    assert report.raw.trades_taken == 1
    assert report.ai_filtered.trades_taken == 1
    assert report.ai_filtered.total_r == pytest.approx(2.0)  # TP hit at 2R
    assert report.to_dict()["decision_breakdown"]["APPROVE"] == 1


async def test_run_backtest_counts_false_rejection_when_ai_rejects_a_winner():
    fixtures = load_fixtures(FIXTURE)
    report = await run_backtest(
        fixtures,
        _backtest_settings(),
        make_llm(final=make_final_result(decision=FinalDecision.REJECT)),
        make_account(),
    )
    assert report.ai_filtered.trades_taken == 0
    assert report.false_rejections == 1
    assert report.avoided_losses == 0


async def test_backtest_keeps_risk_rules_active_while_disabling_staleness():
    """Replay disables signal-age/staleness only; real risk limits still bite."""
    report = await run_backtest(
        load_fixtures(FIXTURE),
        _backtest_settings(min_risk_reward_ratio=99.0),
        make_llm(),
        make_account(),
    )
    assert report.records[0].ai_decision == FinalDecision.REJECT.value
    assert report.ai_filtered.trades_taken == 0


async def test_replay_is_anchored_to_the_signal_time_not_the_wall_clock():
    """Regression for a bug found by diffing a replayed payload against a
    live one: every "how old is this" computation used datetime.now(), so
    a year-old fixture reached the agents as a year-old signal with
    data_age_seconds in the tens of millions, the session computed from
    today's clock, and EVERY news/sentiment item filtered out as older
    than the lookback -- the agents decided on empty evidence while the
    fixture was full. The pipeline now takes its "now" from an injected
    clock that the harness pins to the fixture's decision time.
    """
    fixture = load_fixtures(FIXTURE)[0]
    assert fixture.news_items and fixture.sentiment_items  # the fixture is not empty
    settings = backtest_settings(_backtest_settings())
    pipeline = _build_pipeline_for_record(settings, make_llm(), fixture)
    result = await pipeline.run(fixture.signal, make_account())
    assert result.stage_reached.value == "complete", result.reason

    inputs = {t.agent_name: t.input_snapshot for t in result.agent_traces}
    assert inputs["news_agent"]["signal"]["age_seconds"] == 0.0
    assert inputs["technical_agent"]["market_conditions"]["data_age_seconds"] == 0.0
    assert inputs["news_agent"]["news"]["item_count"] == len(fixture.news_items)
    assert inputs["sentiment_agent"]["sentiment_sources"]["item_count"] == len(
        fixture.sentiment_items
    )
    ages = [i["age_minutes"] for i in inputs["news_agent"]["news"]["items"]]
    assert ages and max(ages) <= settings.news_lookback_minutes
