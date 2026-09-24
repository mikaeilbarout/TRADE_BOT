from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.services.risk_service import AccountState, RiskService
from research.ai.agents import technical_block
from research.ai.checkpoint import CheckpointStore, SignalProgress
from research.ai.client import (
    AgentCallError,
    AgentRequest,
    AnthropicAgentClient,
    MockAgentClient,
    build_strict_tool,
    extract_usage,
    parse_tool_result,
)
from research.ai.cost import TokenUsage
from research.ai.gate import DeterministicGate, replay_risk_settings
from research.ai.pit_store import CalendarPitStore, NewsPitStore, SentimentPitStore
from research.ai.prompts import PROMPTS, all_prompt_versions, prompt_version
from research.ai.runner import AIBacktestRunner, select_pilot_signals
from research.ai.schemas import (
    Bias,
    FinalAction,
    FinalVerdict,
    Gate,
    NewsVerdict,
    RiskLevel,
    SentimentVerdict,
    TechnicalVerdict,
)
from research.ai.settings import AISettings, FailClosedPolicy
from research.backtest.engine import BacktestEngine
from research.backtest.indicators import compute_indicator_frame
from research.config import BacktestConfig, RiskModel
from research.strategy.base import StrategySignal
from tests.conftest import make_settings

UTC = timezone.utc
BASE = datetime(2025, 6, 3, 8, 0, tzinfo=UTC)


# --- fixtures --------------------------------------------------------------


def make_bars(n: int = 300) -> pd.DataFrame:
    """Indicator-complete bar FIXTURES for exercising the AI plumbing."""
    closes = [2400 + i * 0.4 + (i % 5) * 0.6 for i in range(n)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([BASE + timedelta(minutes=15 * i) for i in range(n)], utc=True),
            "open": closes,
            "high": [c + 1.5 for c in closes],
            "low": [c - 1.5 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "tick_count": [80] * n,
            "bid_close": [c - 0.15 for c in closes],
            "ask_close": [c + 0.15 for c in closes],
            "spread_mean": [0.3] * n,
            "spread_max": [0.5] * n,
            "is_partial": [False] * n,
        }
    )
    return compute_indicator_frame(frame, ema_fast=20, ema_slow=100, volatility_window=50)


def make_ai_signal(bars: pd.DataFrame, bar_index: int = 250, **overrides) -> StrategySignal:
    bar = bars.iloc[bar_index]
    entry = float(bar["close"])
    defaults = dict(
        signal_id=f"sig-{bar_index}",
        bar_index=bar_index,
        signal_time=bar["timestamp"].to_pydatetime(),
        symbol="XAUUSD",
        side="BUY",
        entry=entry,
        stop_loss=entry - 8.0,
        take_profit=entry + 16.0,
        entry_reason="fixture breakout",
        market_conditions={"session": "LONDON"},
    )
    defaults.update(overrides)
    return StrategySignal(**defaults)


def ai_settings(**overrides) -> AISettings:
    defaults = dict(anthropic_api_key="test-key", ai_cost_limit_usd=100.0)
    defaults.update(overrides)
    return AISettings(**defaults)


def technical_verdict(**overrides) -> TechnicalVerdict:
    defaults = dict(
        decision=Gate.PASS, confidence=0.8, bias=Bias.BULLISH, risk_level=RiskLevel.LOW,
        reason_codes=["TREND_ALIGNED", "GOOD_RR"], htf_aligned=True,
        entry_quality=RiskLevel.HIGH,
    )
    defaults.update(overrides)
    return TechnicalVerdict(**defaults)


def news_verdict(**overrides) -> NewsVerdict:
    defaults = dict(
        decision=Gate.PASS, confidence=0.75, bias=Bias.BULLISH,
        reason_codes=["NO_HIGH_IMPACT_NEWS"], high_impact_within_window=False,
    )
    defaults.update(overrides)
    return NewsVerdict(**defaults)


def sentiment_verdict(**overrides) -> SentimentVerdict:
    defaults = dict(
        decision=Gate.PASS, confidence=0.7, bias=Bias.BULLISH,
        reason_codes=["SENTIMENT_SUPPORTS"],
    )
    defaults.update(overrides)
    return SentimentVerdict(**defaults)


def final_verdict(**overrides) -> FinalVerdict:
    defaults = dict(
        action=FinalAction.APPROVE, confidence=0.82, risk_level=RiskLevel.LOW,
        reason_codes=["ALL_ALIGNED"],
    )
    defaults.update(overrides)
    return FinalVerdict(**defaults)


def mock_client(**overrides) -> MockAgentClient:
    verdicts = {
        "technical": overrides.pop("technical", technical_verdict()),
        "news": overrides.pop("news", news_verdict()),
        "sentiment": overrides.pop("sentiment", sentiment_verdict()),
        "final": overrides.pop("final", final_verdict()),
    }
    return MockAgentClient(verdicts=verdicts, **overrides)


def news_frame(times: list[datetime]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times, utc=True),
            "source": ["reuters"] * len(times),
            "headline": [f"headline {i}" for i in range(len(times))],
            "category": ["macro"] * len(times),
        }
    )


def build_runner(bars: pd.DataFrame, client=None, settings=None, tmp_path=None, **store_kw):
    config = BacktestConfig(risk=RiskModel(initial_balance=100_000.0, risk_per_trade_pct=0.5))
    app_settings = replay_risk_settings(make_settings())
    risk = RiskService(app_settings)
    engine = BacktestEngine(config)
    gate = DeterministicGate(config, risk, engine)
    checkpoint = CheckpointStore(tmp_path / "cp.sqlite")

    # Records are placed just before the default signal bar (index 250) so
    # the lookback window actually contains them.
    near_signal = bars["timestamp"].iloc[250].to_pydatetime() - timedelta(minutes=20)
    news_store = store_kw.get("news_store", NewsPitStore(news_frame([near_signal])))
    sentiment_store = store_kw.get("sentiment_store", SentimentPitStore(None))

    return AIBacktestRunner(
        config=config,
        ai_settings=settings or ai_settings(),
        client=client or mock_client(),
        gate=gate,
        risk_service=risk,
        engine=engine,
        checkpoint=checkpoint,
        news_store=news_store,
        sentiment_store=sentiment_store,
        calendar_store=store_kw.get("calendar_store"),
    ), checkpoint


# --- prompt caching design ------------------------------------------------


def test_static_prompts_are_byte_stable_across_calls():
    """Any per-call variation in the static prefix destroys the cache."""
    assert prompt_version("technical") == prompt_version("technical")
    assert all_prompt_versions() == all_prompt_versions()


def test_static_prompts_contain_no_timestamps_or_ids():
    import re

    for agent, prompt in PROMPTS.items():
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:", prompt), agent
        assert "uuid" not in prompt.lower(), agent


def test_static_prompts_are_long_enough_to_be_cacheable():
    """Caching silently does nothing below the model's minimum cacheable
    prefix (512-4096 tokens, model-dependent). At roughly 4 chars/token,
    >=8k chars clears a 2048-token floor. This is not padding: a cached
    3k-token prefix bills ~300 effective tokens per call (0.1x read) versus
    1250 uncached, so clearing the floor is also the cheaper option."""
    for agent, prompt in PROMPTS.items():
        assert len(prompt) > 8_000, f"{agent} prompt is only {len(prompt)} chars"


def test_cache_control_is_placed_on_the_last_static_block():
    client = AnthropicAgentClient(ai_settings(), client=object())
    request = AgentRequest(
        agent="technical", signal_id="s1", static_system=PROMPTS["technical"],
        user_payload={"a": 1}, response_model=TechnicalVerdict,
        prompt_version="v1", model="claude-haiku-4-5",
    )
    body = client._request_body(request)

    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # Volatile content must live in messages, AFTER the breakpoint.
    assert "a" in body["messages"][0]["content"]
    assert "1" in body["messages"][0]["content"]


def test_cache_control_absent_when_caching_disabled():
    client = AnthropicAgentClient(ai_settings(ai_use_prompt_cache=False), client=object())
    request = AgentRequest(
        agent="news", signal_id="s1", static_system=PROMPTS["news"], user_payload={},
        response_model=NewsVerdict, prompt_version="v1", model="claude-haiku-4-5",
    )
    assert "cache_control" not in client._request_body(request)["system"][-1]


def test_effort_is_only_sent_to_models_that_accept_it():
    """Haiku 4.5 400s on output_config.effort -- sending it would break the
    entire cheap path."""
    client = AnthropicAgentClient(ai_settings(), client=object())
    haiku = AgentRequest(
        agent="technical", signal_id="s", static_system="x", user_payload={},
        response_model=TechnicalVerdict, prompt_version="v", model="claude-haiku-4-5",
    )
    sonnet = haiku.__class__(**{**haiku.__dict__, "model": "claude-sonnet-5"})

    assert "output_config" not in client._request_body(haiku)
    assert client._request_body(sonnet)["output_config"] == {"effort": "low"}


def test_user_payload_serialization_is_deterministic():
    request = AgentRequest(
        agent="technical", signal_id="s", static_system="x",
        user_payload={"b": 2, "a": 1}, response_model=TechnicalVerdict,
        prompt_version="v", model="claude-haiku-4-5",
    )
    assert request.rendered_user_text() == '{"a":1,"b":2}'


# --- structured output ----------------------------------------------------


def test_strict_tool_schema_forbids_extra_properties():
    tool = build_strict_tool(FinalVerdict)
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False


def test_forced_tool_choice_is_requested():
    client = AnthropicAgentClient(ai_settings(), client=object())
    body = client._request_body(
        AgentRequest(
            agent="final", signal_id="s", static_system="x", user_payload={},
            response_model=FinalVerdict, prompt_version="v", model="claude-sonnet-5",
        )
    )
    assert body["tool_choice"] == {"type": "tool", "name": "emit"}


class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Response:
    def __init__(self, content, stop_reason="tool_use", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class _Usage:
    def __init__(self, **kw):
        for field in (
            "input_tokens", "cache_creation_input_tokens",
            "cache_read_input_tokens", "output_tokens",
        ):
            setattr(self, field, kw.get(field, 0))


def test_parse_tool_result_validates_against_schema():
    response = _Response([_Block("tool_use", input={"action": "APPROVE", "confidence": 0.9})])
    verdict = parse_tool_result(response, FinalVerdict)
    assert verdict.action == FinalAction.APPROVE


def test_parse_tool_result_rejects_refusal():
    with pytest.raises(AgentCallError, match="refused"):
        parse_tool_result(_Response([], stop_reason="refusal"), FinalVerdict)


def test_parse_tool_result_rejects_missing_tool_block():
    with pytest.raises(AgentCallError, match="no tool_use block"):
        parse_tool_result(_Response([_Block("text", text="hello")]), FinalVerdict)


def test_parse_tool_result_rejects_schema_violation():
    bad = _Response([_Block("tool_use", input={"action": "MAYBE", "confidence": 0.5})])
    with pytest.raises(AgentCallError, match="schema validation"):
        parse_tool_result(bad, FinalVerdict)


def test_parse_tool_result_parses_json_string_input():
    response = _Response([_Block("tool_use", input='{"action":"REJECT","confidence":0.4}')])
    assert parse_tool_result(response, FinalVerdict).action == FinalAction.REJECT


def test_extract_usage_reads_all_four_token_fields():
    usage = extract_usage(
        _Response([], usage=_Usage(
            input_tokens=100, cache_creation_input_tokens=2000,
            cache_read_input_tokens=3000, output_tokens=50,
        ))
    )
    assert usage.input_tokens == 100
    assert usage.cache_creation_tokens == 2000
    assert usage.cache_read_tokens == 3000
    assert usage.output_tokens == 50
    assert usage.total_input_equivalent == 5100


def test_extract_usage_tolerates_missing_cache_fields():
    assert extract_usage(_Response([], usage=_Usage(input_tokens=10))).cache_read_tokens == 0


# --- payload economy -----------------------------------------------------


def test_technical_payload_downsamples_the_bar_tail():
    bars = make_bars()
    settings = ai_settings(ai_technical_lookback_bars=60)
    block = technical_block(bars, 250, settings)
    # 60 bars of lookback must not become 60 rows in the prompt.
    assert block["bar_count"] < 60
    assert block["bar_count"] >= 12


def test_technical_payload_rounds_prices():
    bars = make_bars()
    block = technical_block(bars, 250, ai_settings())
    for row in block["bars_ohlc"]:
        for price in row[1:]:
            assert round(price, 2) == price


def test_static_rules_live_in_the_prompt_not_the_payload():
    """Rules belong in the cached system prefix. Repeating them per signal
    would pay full price for them on every single call."""
    bars = make_bars()
    block = technical_block(bars, 250, ai_settings())
    serialized = str(block)

    assert "Reason code catalogue" in PROMPTS["technical"]
    for rule_text in ("Reason code", "reason_codes", "confidence is", "UNAVAILABLE"):
        assert rule_text not in serialized


# --- point-in-time stores ------------------------------------------------


def test_news_store_returns_only_past_records():
    as_of = BASE + timedelta(hours=5)
    frame = news_frame([as_of - timedelta(minutes=30), as_of + timedelta(minutes=30)])
    result = NewsPitStore(frame).query(as_of, lookback_minutes=600, max_items=10)
    assert result.available
    assert len(result.records) == 1
    assert result.records[0].timestamp <= as_of


def test_store_reports_unavailable_when_dataset_missing():
    result = NewsPitStore(None).query(BASE, 60, 5)
    assert result.available is False
    assert "not loaded" in result.reason


def test_store_reports_unavailable_outside_coverage():
    frame = news_frame([BASE])
    result = NewsPitStore(frame).query(BASE + timedelta(days=400), 60, 5)
    assert result.available is False
    assert "does not cover" in result.reason


def test_available_but_empty_window_is_distinguishable_from_missing_data():
    """'No news happened' and 'we have no news data' are different claims."""
    frame = news_frame([BASE, BASE + timedelta(days=2)])
    quiet = NewsPitStore(frame).query(BASE + timedelta(days=1), lookback_minutes=10, max_items=5)
    assert quiet.available is True
    assert quiet.records == []
    assert quiet.is_empty


def test_calendar_withholds_released_values_until_release_time():
    """A scheduled release is known in advance; its OUTCOME is not."""
    as_of = BASE + timedelta(hours=2)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [as_of - timedelta(minutes=30), as_of + timedelta(minutes=30)], utc=True
            ),
            "name": ["CPI (already out)", "NFP (pending)"],
            "importance": ["HIGH", "HIGH"],
            "currency": ["USD", "USD"],
            "forecast_value": ["3.1%", "180k"],
            "released_value": ["3.4%", "195k"],
        }
    )
    result = CalendarPitStore(frame).query_events(as_of, window_minutes=120)

    by_name = {event.name: event for event in result.events}
    assert by_name["CPI (already out)"].released_value == "3.4%"
    # The pending release must NOT leak its actual value.
    assert by_name["NFP (pending)"].released_value is None
    assert by_name["NFP (pending)"].forecast_value == "180k"


def test_store_filter_boundary_is_inclusive_of_the_signal_instant():
    """A record published exactly AT the signal timestamp was knowable; one a
    second later was not."""
    as_of = BASE + timedelta(hours=3)
    frame = news_frame([as_of, as_of + timedelta(seconds=1)])
    result = NewsPitStore(frame).query(as_of, lookback_minutes=600, max_items=10)
    assert len(result.records) == 1
    assert result.records[0].timestamp == as_of


# --- deterministic gate ---------------------------------------------------


def test_gate_passes_a_healthy_signal(tmp_path):
    bars = make_bars()
    runner, _ = build_runner(bars, tmp_path=tmp_path)
    result = runner._gate.check(
        make_ai_signal(bars), bars, AccountState(balance=100_000.0)
    )
    assert result.passed, result.violations
    assert result.sized_volume > 0
    assert result.trade_signal is not None


def test_gate_enforces_risk_per_trade_with_real_position_size(tmp_path):
    """The gate sizes with the same engine execution uses, so the
    risk-per-trade limit is checked against a real volume."""
    bars = make_bars()
    config = BacktestConfig(risk=RiskModel(initial_balance=100_000.0, risk_per_trade_pct=5.0))
    app_settings = replay_risk_settings(make_settings(max_risk_per_trade_pct=0.01))
    gate = DeterministicGate(config, RiskService(app_settings), BacktestEngine(config))

    result = gate.check(make_ai_signal(bars), bars, AccountState(balance=100_000.0))
    assert not result.passed
    assert any("risk per trade" in v for v in result.violations)


def test_gate_rejects_poor_risk_reward(tmp_path):
    bars = make_bars()
    config = BacktestConfig()
    app_settings = replay_risk_settings(make_settings(min_risk_reward_ratio=10.0))
    gate = DeterministicGate(config, RiskService(app_settings), BacktestEngine(config))
    result = gate.check(make_ai_signal(bars), bars, AccountState(balance=100_000.0))
    assert not result.passed
    assert any("risk/reward" in v for v in result.violations)


async def test_gate_rejection_costs_zero_tokens(tmp_path):
    """The core cost lever: deterministic rejections never reach the LLM."""
    bars = make_bars()
    client = mock_client()
    settings = ai_settings()
    config = BacktestConfig()
    app_settings = replay_risk_settings(make_settings(min_risk_reward_ratio=10.0))
    risk = RiskService(app_settings)
    engine = BacktestEngine(config)
    runner = AIBacktestRunner(
        config=config, ai_settings=settings, client=client,
        gate=DeterministicGate(config, risk, engine), risk_service=risk, engine=engine,
        checkpoint=CheckpointStore(tmp_path / "cp.sqlite"),
        news_store=NewsPitStore(None), sentiment_store=SentimentPitStore(None),
    )
    outcome = await runner.run(
        [make_ai_signal(bars)], bars, AccountState(balance=100_000.0)
    )

    assert outcome.decisions[0].action == FinalAction.REJECT
    assert outcome.decisions[0].gate_passed is False
    assert client.calls == []  # not one token spent
    assert runner.ledger.spent_usd == 0.0


# --- runner behavior -----------------------------------------------------


async def test_full_chain_approves_and_records_costs(tmp_path):
    bars = make_bars()
    near_signal = bars["timestamp"].iloc[250].to_pydatetime() - timedelta(minutes=15)
    sentiment = SentimentPitStore(news_frame([near_signal]))
    runner, _ = build_runner(bars, tmp_path=tmp_path, sentiment_store=sentiment)
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))

    decision = outcome.decisions[0]
    assert decision.action == FinalAction.APPROVE
    assert decision.agents_called == ["technical", "news", "sentiment", "final"]
    assert decision.cost_usd > 0
    assert runner.ledger.spent_usd > 0


async def test_agent_is_skipped_when_its_data_is_unavailable(tmp_path):
    """Paying a model to say UNAVAILABLE about data we know is absent is
    waste; the skip is recorded so the final agent still sees the gap."""
    bars = make_bars()
    client = mock_client()
    runner, _ = build_runner(
        bars, client=client, tmp_path=tmp_path, sentiment_store=SentimentPitStore(None)
    )
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))

    decision = outcome.decisions[0]
    assert "sentiment" not in decision.agents_called
    assert "sentiment" in decision.skip_reasons
    assert decision.sentiment is None
    assert [c.agent for c in client.calls] == ["technical", "news", "final"]


async def test_high_impact_news_forces_wait_regardless_of_final_approval(tmp_path):
    bars = make_bars()
    client = mock_client(
        news=news_verdict(high_impact_within_window=True),
        final=final_verdict(action=FinalAction.APPROVE, confidence=0.95),
    )
    runner, _ = build_runner(bars, client=client, tmp_path=tmp_path)
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))

    decision = outcome.decisions[0]
    assert decision.action == FinalAction.WAIT
    assert "blackout" in decision.reason


async def test_agent_block_vetoes_an_approval(tmp_path):
    bars = make_bars()
    client = mock_client(
        technical=technical_verdict(decision=Gate.BLOCK),
        final=final_verdict(action=FinalAction.APPROVE, confidence=0.9),
    )
    runner, _ = build_runner(bars, client=client, tmp_path=tmp_path)
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))
    assert outcome.decisions[0].action == FinalAction.REJECT
    # Assert the rule identifier, not the sentence: the wording belongs to
    # the shared policy core and is not part of the behaviour under test.
    assert outcome.decisions[0].deciding_rule == "BLOCK_VETO"
    assert outcome.decisions[0].blocking_agent == "technical"
    assert "technical agent returned BLOCK" in outcome.decisions[0].reason


async def test_low_confidence_approval_becomes_wait(tmp_path):
    bars = make_bars()
    client = mock_client(final=final_verdict(action=FinalAction.APPROVE, confidence=0.2))
    runner, _ = build_runner(
        bars, client=client, tmp_path=tmp_path, settings=ai_settings(ai_min_final_confidence=0.6)
    )
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))
    assert outcome.decisions[0].action == FinalAction.WAIT


async def test_modify_without_levels_fails_closed(tmp_path):
    bars = make_bars()
    client = mock_client(final=final_verdict(action=FinalAction.MODIFY, entry=None))
    runner, _ = build_runner(bars, client=client, tmp_path=tmp_path)
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))
    assert outcome.decisions[0].action == FinalAction.REJECT
    assert outcome.decisions[0].deciding_rule == "MODIFY_MISSING_LEVELS"
    assert "without modified trade parameters" in outcome.decisions[0].reason


async def test_modify_with_incoherent_levels_is_rejected(tmp_path):
    bars = make_bars()
    signal = make_ai_signal(bars)
    # BUY with the stop ABOVE entry: must never reach execution.
    client = mock_client(
        final=final_verdict(
            action=FinalAction.MODIFY,
            entry=signal.entry,
            stop_loss=signal.entry + 5,
            take_profit=signal.entry + 15,
        )
    )
    runner, _ = build_runner(bars, client=client, tmp_path=tmp_path)
    outcome = await runner.run([signal], bars, AccountState(balance=100_000.0))
    assert outcome.decisions[0].action == FinalAction.REJECT


async def test_valid_modify_is_carried_through(tmp_path):
    bars = make_bars()
    signal = make_ai_signal(bars)
    client = mock_client(
        final=final_verdict(
            action=FinalAction.MODIFY,
            entry=signal.entry - 1.0,
            stop_loss=signal.entry - 9.0,
            take_profit=signal.entry + 15.0,
            confidence=0.85,
        )
    )
    runner, _ = build_runner(bars, client=client, tmp_path=tmp_path)
    outcome = await runner.run([signal], bars, AccountState(balance=100_000.0))

    decision = outcome.decisions[0]
    assert decision.action == FinalAction.MODIFY
    assert decision.was_modified is True
    assert decision.modified_entry == pytest.approx(signal.entry - 1.0)


async def test_agent_failure_fails_closed_to_no_trade(tmp_path):
    bars = make_bars()
    client = mock_client()
    client.fail_agents = {"final"}
    runner, _ = build_runner(bars, client=client, tmp_path=tmp_path)
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))

    decision = outcome.decisions[0]
    assert decision.action == FinalAction.REJECT
    assert decision.failed_closed is True
    assert "Failed closed" in decision.reason


async def test_fail_closed_policy_can_be_wait(tmp_path):
    bars = make_bars()
    client = mock_client()
    client.fail_agents = {"technical"}
    runner, _ = build_runner(
        bars, client=client, tmp_path=tmp_path,
        settings=ai_settings(ai_fail_closed_policy=FailClosedPolicy.WAIT),
    )
    outcome = await runner.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))
    assert outcome.decisions[0].action == FinalAction.WAIT


async def test_budget_stop_halts_the_run_safely(tmp_path):
    bars = make_bars()
    client = mock_client()
    # A tiny budget: the first signal's calls exhaust it, the rest must not run.
    runner, checkpoint = build_runner(
        bars, client=client, tmp_path=tmp_path,
        settings=ai_settings(ai_cost_limit_usd=0.0005),
    )
    signals = [make_ai_signal(bars, bar_index=i) for i in (200, 220, 240, 260)]
    outcome = await runner.run(signals, bars, AccountState(balance=100_000.0))

    assert outcome.budget_stopped is True
    assert "cost limit reached" in outcome.stopped_reason
    assert len(outcome.decisions) < len(signals)


# --- checkpoint / resume -------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path / "cp.sqlite")
    store.save_progress(
        SignalProgress(
            signal_id="s1", status="done", decision="APPROVE", confidence=0.8,
            payload={"action": "APPROVE"}, cost_usd=0.002,
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )
    )
    loaded = store.get_progress("s1")
    assert loaded.status == "done"
    assert loaded.usage.input_tokens == 100
    assert store.completed_signal_ids() == {"s1"}


def test_failed_signals_are_retried_on_resume_but_done_ones_are_not(tmp_path):
    store = CheckpointStore(tmp_path / "cp.sqlite")
    store.save_progress(SignalProgress(signal_id="done1", status="done"))
    store.save_progress(SignalProgress(signal_id="skip1", status="skipped_deterministic"))
    store.save_progress(SignalProgress(signal_id="fail1", status="failed"))

    completed = store.completed_signal_ids()
    assert completed == {"done1", "skip1"}
    assert "fail1" not in completed  # transient failure gets another chance


async def test_resume_does_not_re_call_completed_signals(tmp_path):
    """The headline requirement: restarting must not re-pay for prior work."""
    bars = make_bars()
    signals = [make_ai_signal(bars, bar_index=i) for i in (200, 230, 260)]

    first_client = mock_client()
    runner_one, _ = build_runner(bars, client=first_client, tmp_path=tmp_path)
    await runner_one.run(signals, bars, AccountState(balance=100_000.0))
    first_call_count = len(first_client.calls)
    assert first_call_count > 0

    # Same checkpoint file, fresh client: nothing should be re-requested.
    second_client = mock_client()
    runner_two, _ = build_runner(bars, client=second_client, tmp_path=tmp_path)
    outcome = await runner_two.run(signals, bars, AccountState(balance=100_000.0))

    assert second_client.calls == []
    assert outcome.resumed_count == len(signals)
    assert len(outcome.decisions) == len(signals)  # decisions restored from disk


async def test_resumed_spend_is_recovered_into_the_budget(tmp_path):
    bars = make_bars()
    runner_one, _ = build_runner(bars, tmp_path=tmp_path)
    await runner_one.run([make_ai_signal(bars)], bars, AccountState(balance=100_000.0))
    spent = runner_one.ledger.spent_usd
    assert spent > 0

    runner_two, _ = build_runner(bars, tmp_path=tmp_path)
    assert runner_two.ledger.spent_usd == pytest.approx(spent)


def test_call_log_persists_token_detail(tmp_path):
    from research.ai.cost import CallRecord

    store = CheckpointStore(tmp_path / "cp.sqlite")
    store.log_call(
        CallRecord(
            signal_id="s1", agent="technical", model="claude-haiku-4-5",
            usage=TokenUsage(input_tokens=400, cache_read_tokens=3000, output_tokens=55),
            cost_usd=0.0012, latency_seconds=0.8, prompt_version="abc123",
        )
    )
    calls = store.all_calls()
    assert len(calls) == 1
    assert calls[0].usage.cache_read_tokens == 3000
    assert calls[0].prompt_version == "abc123"
    assert store.total_spend() == pytest.approx(0.0012)


# --- pilot sampling ------------------------------------------------------


def test_pilot_sampling_is_chronological_and_reproducible():
    bars = make_bars()
    signals = [make_ai_signal(bars, bar_index=i) for i in range(150, 290)]
    first = select_pilot_signals(signals, 100)
    second = select_pilot_signals(list(reversed(signals)), 100)

    assert len(first) == 100
    assert [s.signal_id for s in first] == [s.signal_id for s in second]
    times = [s.signal_time for s in first]
    assert times == sorted(times)


def test_evenly_spaced_sampling_covers_the_whole_period():
    bars = make_bars()
    signals = [make_ai_signal(bars, bar_index=i) for i in range(150, 290)]
    sample = select_pilot_signals(signals, 10, method="evenly_spaced")
    assert len(sample) == 10
    assert sample[-1].signal_time > sample[0].signal_time
    # Spans more of the period than the first 10 chronologically would.
    assert sample[-1].bar_index > signals[9].bar_index


def test_pilot_sampling_returns_all_when_fewer_signals_than_requested():
    bars = make_bars()
    signals = [make_ai_signal(bars, bar_index=i) for i in (200, 210)]
    assert len(select_pilot_signals(signals, 100)) == 2


def test_unknown_sampling_method_is_rejected():
    with pytest.raises(ValueError, match="unknown sampling method"):
        select_pilot_signals([], 10, method="cherry_pick_winners")
