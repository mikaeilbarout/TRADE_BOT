from __future__ import annotations

from app.models.enums import (
    ApprovalStatus,
    FinalDecision,
    GateDecision,
    PipelineStage,
    TechnicalDecision,
)
from app.providers.llm.mock_provider import MockLLMProvider
from tests.conftest import (
    build_test_pipeline,
    make_account,
    make_final_result,
    make_llm,
    make_news_result,
    make_sentiment_result,
    make_signal,
    make_technical_result,
)


async def test_full_chain_approves_when_everything_supportive():
    pipeline = build_test_pipeline(make_llm())
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.APPROVE
    assert result.stage_reached == PipelineStage.COMPLETE
    assert not result.errors


async def test_chain_runs_all_four_agents_in_order():
    pipeline = build_test_pipeline(make_llm())
    result = await pipeline.run(make_signal(), make_account())
    assert [t.agent_name for t in result.agent_traces] == [
        "news_agent",
        "sentiment_agent",
        "technical_agent",
        "final_decision_agent",
    ]


async def test_specialists_are_independent_and_final_sees_all_three():
    """Parallelized 2026-09-18: news/sentiment/technical each see only the
    signal and market conditions (no upstream_chain -- they run
    concurrently, not as a chain). Only the final decision agent sees all
    three findings together."""
    pipeline = build_test_pipeline(make_llm())
    result = await pipeline.run(make_signal(), make_account())
    traces = {t.agent_name: t.input_snapshot for t in result.agent_traces}

    for snapshot in traces.values():
        assert "signal" in snapshot
        assert "market_conditions" in snapshot

    assert "upstream_chain" not in traces["news_agent"]
    assert "upstream_chain" not in traces["sentiment_agent"]
    assert "upstream_chain" not in traces["technical_agent"]
    assert [link["agent"] for link in traces["final_decision_agent"]["agent_chain"]] == [
        "news_agent",
        "sentiment_agent",
        "technical_agent",
    ]


async def test_sentiment_agent_gets_its_own_evidence_base():
    pipeline = build_test_pipeline(make_llm())
    result = await pipeline.run(make_signal(), make_account())
    snapshot = next(
        t.input_snapshot for t in result.agent_traces if t.agent_name == "sentiment_agent"
    )
    assert snapshot["sentiment_sources"]["item_count"] > 0
    assert result.data_sources.sentiment_item_count > 0


async def test_pipeline_rejects_on_final_agent_reject():
    pipeline = build_test_pipeline(
        make_llm(final=make_final_result(decision=FinalDecision.REJECT))
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.REJECT


async def test_pipeline_short_circuits_on_critical_news():
    # Parallelized 2026-09-18: news/sentiment/technical now run concurrently,
    # so a critical-news short-circuit can no longer skip PAYING for
    # sentiment/technical (they're already in flight by the time news
    # resolves) -- it only skips calling the final decision agent. Both
    # canned responses are needed here since the mock pipeline genuinely
    # calls all three specialists now.
    llm = make_llm(
        news=make_news_result(
            decision=GateDecision.BLOCK,
            high_impact_event_within_minutes=True,
            summary="FOMC decision imminent, extreme volatility expected.",
        )
    )
    pipeline = build_test_pipeline(llm, short_circuit_on_critical_news=True)
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.WAIT
    assert result.short_circuited is True
    assert result.sentiment_result is not None
    assert result.technical_result is not None
    assert len(result.agent_traces) == 3  # news + sentiment + technical, not final


async def test_pipeline_rejects_bad_signal_before_calling_any_agent():
    llm = MockLLMProvider()  # no canned responses; any agent call raises
    pipeline = build_test_pipeline(llm, min_risk_reward_ratio=5.0)
    result = await pipeline.run(make_signal(), make_account())  # RR 2.0
    assert result.decision == FinalDecision.REJECT
    assert result.stage_reached == PipelineStage.HARD_RISK_PRECHECK
    assert result.agent_traces == []


async def test_policy_veto_overrides_ai_approval_in_pipeline():
    """The veto mechanism itself still exists end-to-end when explicitly
    enabled -- off by default for the live service as of 2026-09-18, see
    app/config/settings.py."""
    pipeline = build_test_pipeline(
        make_llm(technical=make_technical_result(decision=TechnicalDecision.BLOCK)),
        veto_on_technical_block=True,
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.REJECT
    assert result.ai_decision == FinalDecision.APPROVE  # what the AI wanted
    assert result.veto_triggered is True


async def test_technical_block_no_longer_overrides_ai_approval_by_default():
    """As of 2026-09-18, a technical BLOCK is strong input to
    final_decision_agent, not an automatic override of its own decision --
    if final_decision_agent still says APPROVE, that stands."""
    pipeline = build_test_pipeline(
        make_llm(technical=make_technical_result(decision=TechnicalDecision.BLOCK))
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.APPROVE
    assert result.veto_triggered is False


async def test_low_confidence_approval_is_downgraded_to_wait():
    pipeline = build_test_pipeline(
        make_llm(final=make_final_result(decision=FinalDecision.APPROVE, confidence=0.4)),
        min_confidence=0.7,
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.WAIT


async def test_missing_account_balance_rejects_before_agents():
    from app.services.risk_service import AccountState

    pipeline = build_test_pipeline(make_llm())
    result = await pipeline.run(make_signal(), AccountState(balance=None))
    assert result.decision == FinalDecision.REJECT
    assert any("account balance" in v for v in result.guard_violations)


async def test_pipeline_records_data_sources_and_latencies():
    pipeline = build_test_pipeline(make_llm())
    result = await pipeline.run(make_signal(), make_account())
    assert result.data_sources.llm_provider == "mock"
    assert result.data_sources.news_provider == "mock"
    assert result.data_sources.sentiment_provider == "mock"
    assert result.data_sources.news_item_count >= 1
    stages = {latency.stage for latency in result.latencies}
    assert {"news_agent", "sentiment_agent", "technical_agent", "final_decision_agent"} <= stages
    assert result.total_latency_seconds > 0
    assert result.approval_status == ApprovalStatus.NOT_REQUIRED


async def test_pipeline_never_returns_modified_levels():
    """The executed trade is always the bot's own signal."""
    pipeline = build_test_pipeline(
        make_llm(technical=make_technical_result(decision=TechnicalDecision.WARNING))
    )
    result = await pipeline.run(make_signal(), make_account())
    assert result.decision == FinalDecision.APPROVE
    assert result.modified_trade is None
    assert result.to_api_response()["modified_trade"] is None

