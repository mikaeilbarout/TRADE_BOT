from __future__ import annotations

import pytest

from app.models.agent_decision import DimensionScores
from app.models.enums import FinalDecision, GateDecision, TechnicalDecision
from app.models.market_data import TechnicalIndicators
from app.services.decision_policy import DecisionPolicy
from tests.conftest import (
    make_final_result,
    make_market_snapshot,
    make_news_result,
    make_sentiment_result,
    make_settings,
    make_signal,
    make_technical_result,
)


def _apply(**overrides):
    settings = make_settings(**overrides.pop("settings", {}))
    policy = DecisionPolicy(settings)
    return policy.apply(
        overrides.pop("signal", make_signal()),
        overrides.pop("final", make_final_result()),
        overrides.pop("news", make_news_result()),
        overrides.pop("sentiment", make_sentiment_result()),
        overrides.pop("technical", make_technical_result()),
        overrides.pop("market", make_market_snapshot()),
        data_degraded=overrides.pop("data_degraded", False),
    )


def test_healthy_chain_passes_through_untouched():
    outcome = _apply()
    assert outcome.decision == FinalDecision.APPROVE
    assert not outcome.veto_triggered


def test_weighted_score_uses_configured_weights():
    policy = DecisionPolicy(make_settings())
    final = make_final_result(
        scores=DimensionScores(
            news_score=100, sentiment_score=0, technical_score=100, risk_score=0,
            weighted_total=99,  # agent's own arithmetic is not trusted
        )
    )
    # 100*0.25 + 0*0.20 + 100*0.35 + 0*0.20 = 60
    assert policy.weighted_score(final) == pytest.approx(60.0)


def test_news_block_vetoes_when_explicitly_enabled():
    # All three hard vetoes default to False (2026-09-18): a specialist's
    # BLOCK is strong input to final_decision_agent, not an automatic
    # override of its own decision. The mechanism itself still exists and
    # can be re-enabled per agent, which this test covers.
    outcome = _apply(
        news=make_news_result(decision=GateDecision.BLOCK),
        settings={"veto_on_news_block": True},
    )
    assert outcome.decision == FinalDecision.REJECT
    assert outcome.veto_triggered
    assert any("news agent returned BLOCK" in r for r in outcome.veto_reasons)


def test_sentiment_block_vetoes_when_explicitly_enabled():
    outcome = _apply(
        sentiment=make_sentiment_result(decision=GateDecision.BLOCK),
        settings={"veto_on_sentiment_block": True},
    )
    assert outcome.decision == FinalDecision.REJECT
    assert any("sentiment agent returned BLOCK" in r for r in outcome.veto_reasons)


def test_technical_block_vetoes_when_explicitly_enabled():
    outcome = _apply(
        technical=make_technical_result(decision=TechnicalDecision.BLOCK),
        settings={"veto_on_technical_block": True},
    )
    assert outcome.decision == FinalDecision.REJECT
    assert any("technical agent returned BLOCK" in r for r in outcome.veto_reasons)


def test_vetoes_are_off_by_default_final_agent_decision_stands():
    """A specialist BLOCK no longer auto-rejects; if final_decision_agent
    still says APPROVE (and confidence/score thresholds clear), that stands."""
    outcome = _apply(
        news=make_news_result(decision=GateDecision.BLOCK),
        sentiment=make_sentiment_result(decision=GateDecision.BLOCK),
        technical=make_technical_result(decision=TechnicalDecision.BLOCK),
    )
    assert outcome.decision == FinalDecision.APPROVE
    assert not outcome.veto_triggered


def test_veto_can_be_disabled_per_agent():
    outcome = _apply(
        sentiment=make_sentiment_result(decision=GateDecision.BLOCK),
        settings={"veto_on_sentiment_block": False},
    )
    assert outcome.decision == FinalDecision.APPROVE


def test_high_impact_event_forces_wait_even_with_excellent_scores():
    """The spec's own example: strong technical/sentiment/news scores but a
    high-impact event inside the danger window must still WAIT."""
    outcome = _apply(
        news=make_news_result(high_impact_event_within_minutes=True),
        final=make_final_result(
            confidence=0.99,
            scores=DimensionScores(
                news_score=90, sentiment_score=85, technical_score=90, risk_score=95,
                weighted_total=90,
            ),
        ),
    )
    assert outcome.decision == FinalDecision.WAIT
    assert any("blackout window" in r for r in outcome.veto_reasons)


def test_degraded_upstream_analysis_forces_wait():
    outcome = _apply(sentiment=make_sentiment_result(is_degraded=True))
    assert outcome.decision == FinalDecision.WAIT
    assert any("degraded" in r for r in outcome.veto_reasons)


def test_stale_market_data_forces_wait():
    outcome = _apply(market=make_market_snapshot(is_stale=True))
    assert outcome.decision == FinalDecision.WAIT
    assert any("stale" in r for r in outcome.veto_reasons)


def test_low_confidence_downgraded_to_wait():
    outcome = _apply(final=make_final_result(confidence=0.4))
    assert outcome.decision == FinalDecision.WAIT
    assert "confidence" in outcome.reason.lower()


def test_low_weighted_score_rejects_despite_approve():
    outcome = _apply(
        final=make_final_result(
            scores=DimensionScores(
                news_score=40, sentiment_score=40, technical_score=40, risk_score=40,
                weighted_total=95,  # inflated by the agent; recomputed here as 40
            )
        )
    )
    assert outcome.decision == FinalDecision.REJECT
    assert "Weighted evidence score" in outcome.reason


def test_reject_is_never_upgraded():
    outcome = _apply(final=make_final_result(decision=FinalDecision.REJECT))
    assert outcome.decision == FinalDecision.REJECT


def test_contradiction_with_computed_trend_is_flagged():
    """The technical agent claims higher-timeframe alignment for a BUY while
    the deterministically computed D1 trend is a downtrend."""
    market = make_market_snapshot(
        indicators={"D1": TechnicalIndicators(timeframe="D1", trend="DOWNTREND")}
    )
    outcome = _apply(market=market, technical=make_technical_result(aligned_with_higher_tf=True))
    assert any("claims higher-timeframe alignment" in w for w in outcome.warnings)


# --- execution_blocked: which WAITs actually stop a trade -----------------
# WAIT means two different things and the label cannot separate them. These
# pin down which one each condition produces, because a caller that trades
# through "not confident" must still be stopped by "an event is imminent".


def test_blackout_wait_blocks_execution():
    outcome = _apply(news=make_news_result(high_impact_event_within_minutes=True))
    assert outcome.decision == FinalDecision.WAIT
    assert outcome.execution_blocked is True


def test_stale_market_wait_blocks_execution():
    outcome = _apply(market=make_market_snapshot(is_stale=True))
    assert outcome.decision == FinalDecision.WAIT
    assert outcome.execution_blocked is True


def test_degraded_data_wait_does_not_block_execution():
    """Thin/unreliable evidence is an uncertainty signal; whether to trade
    on uncertainty is the caller's policy, not the policy core's."""
    outcome = _apply(sentiment=make_sentiment_result(is_degraded=True))
    assert outcome.decision == FinalDecision.WAIT
    assert outcome.execution_blocked is False


def test_low_confidence_wait_does_not_block_execution():
    outcome = _apply(final=make_final_result(confidence=0.4))
    assert outcome.decision == FinalDecision.WAIT
    assert outcome.execution_blocked is False


def test_agent_authored_wait_still_blocks_on_a_safety_condition():
    """The agent's own WAIT bypasses the vetoes (rule 1 passes it through),
    so the safety facts have to be evaluated on that path too."""
    outcome = _apply(
        final=make_final_result(decision=FinalDecision.WAIT),
        news=make_news_result(high_impact_event_within_minutes=True),
    )
    assert outcome.execution_blocked is True


def test_reject_always_blocks_execution():
    outcome = _apply(final=make_final_result(decision=FinalDecision.REJECT))
    assert outcome.execution_blocked is True


def test_clean_approval_does_not_block_execution():
    outcome = _apply()
    assert outcome.decision == FinalDecision.APPROVE
    assert outcome.execution_blocked is False


def test_deterministic_degraded_data_fires_even_when_agents_stay_silent():
    """The rule used to read only the LLM-authored is_degraded fields, so a
    model that omitted the flag silently disabled it."""
    outcome = _apply(data_degraded=True)
    assert outcome.decision == FinalDecision.WAIT
    assert any("degraded" in r for r in outcome.veto_reasons)


def test_live_vocabulary_has_no_modify():
    """2026-09-19: the AI may approve or refuse a trade, never redraw it. The
    guarantee is schema-level -- the enum the model's tool schema is built
    from has no MODIFY, so no prompt drift can bring it back."""
    from app.models.enums import FinalDecision, TechnicalDecision

    assert "MODIFY" not in FinalDecision.__members__
    assert "MODIFY" not in TechnicalDecision.__members__
    assert set(TechnicalDecision.__members__) == {"PASS", "WARNING", "BLOCK"}


def test_technical_warning_is_not_a_veto():
    outcome = _apply(technical=make_technical_result(decision=TechnicalDecision.WARNING))
    assert outcome.decision == FinalDecision.APPROVE
    assert not outcome.veto_triggered

