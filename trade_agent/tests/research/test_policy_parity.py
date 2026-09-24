from __future__ import annotations

import pytest

from app.models.agent_decision import DimensionScores
from app.models.enums import FinalDecision, GateDecision, TechnicalDecision
from app.services.decision_policy import DecisionPolicy, thresholds_from_settings
from app.services.policy_core import (
    ComponentScores,
    GateSignal,
    PolicyAction,
    PolicyInputs,
    PolicyThresholds,
    evaluate_policy,
)
from research.ai.runner import thresholds_from_ai_settings
from research.experiment import ExperimentConfig
from tests.conftest import (
    make_final_result,
    make_market_snapshot,
    make_news_result,
    make_sentiment_result,
    make_settings,
    make_signal,
    make_technical_result,
)

"""Proof that the live service and the research backtest run ONE policy.

The risk this guards against is specific: the two paths used to implement the
same rules separately, so a backtest could measure a policy the live system
does not run. These tests assert (a) both paths derive the same thresholds
from one experiment config, and (b) the same inputs produce the same outcome
whichever path builds them.
"""


# --- thresholds come from one source --------------------------------------
def test_live_and_research_thresholds_agree_when_built_from_one_config():
    config = ExperimentConfig()
    live = thresholds_from_settings(config.risk_settings(make_settings()))
    research = thresholds_from_ai_settings(config.ai_settings())
    assert live == research


def test_a_changed_experiment_policy_moves_both_paths_together():
    config = ExperimentConfig()
    config.policy.min_confidence = 0.85
    config.policy.min_weighted_score = 72.0
    config.policy.veto_on_sentiment_block = False

    live = thresholds_from_settings(config.risk_settings(make_settings()))
    research = thresholds_from_ai_settings(config.ai_settings())

    assert live.min_confidence == research.min_confidence == 0.85
    assert live.min_weighted_score == research.min_weighted_score == 72.0
    assert live.veto_on_sentiment_block is research.veto_on_sentiment_block is False


def test_the_old_position_and_daily_cap_conflict_is_resolved():
    """The engine and the risk service previously disagreed on both caps."""
    config = ExperimentConfig()
    settings = config.risk_settings(make_settings())
    assert settings.max_simultaneous_positions == config.backtest.risk.max_concurrent_positions
    assert settings.max_trades_per_day == config.backtest.risk.max_trades_per_day


# --- the same inputs produce the same outcome ------------------------------
def _live_outcome(
    *,
    decision: FinalDecision = FinalDecision.APPROVE,
    confidence: float = 0.9,
    scores: tuple[float, float, float, float] = (80, 80, 80, 80),
    news_gate: GateDecision = GateDecision.PASS,
    sentiment_gate: GateDecision = GateDecision.PASS,
    technical_gate: TechnicalDecision = TechnicalDecision.PASS,
    high_impact: bool = False,
    degraded: bool = False,
    stale: bool = False,
    modified_trade=None,
):
    """Run the live policy over the shared test factories."""
    policy = DecisionPolicy(make_settings())
    return policy.apply(
        make_signal(),
        make_final_result(
            decision=decision,
            confidence=confidence,
            scores=DimensionScores(
                news_score=scores[0],
                sentiment_score=scores[1],
                technical_score=scores[2],
                risk_score=scores[3],
                weighted_total=sum(scores) / 4,
            ),
            modified_trade=modified_trade,
        ),
        make_news_result(
            decision=news_gate,
            high_impact_event_within_minutes=high_impact,
            is_degraded=degraded,
        ),
        make_sentiment_result(decision=sentiment_gate),
        make_technical_result(decision=technical_gate),
        make_market_snapshot(is_stale=stale),
    )


def _core_outcome(
    thresholds: PolicyThresholds,
    *,
    action: PolicyAction = PolicyAction.APPROVE,
    confidence: float = 0.9,
    scores: tuple[float, float, float, float] = (80, 80, 80, 80),
    news_gate: GateSignal = GateSignal.PASS,
    sentiment_gate: GateSignal = GateSignal.PASS,
    technical_gate: GateSignal = GateSignal.PASS,
    high_impact: bool = False,
    degraded: bool = False,
    stale: bool = False,
    has_levels: bool = True,
):
    return evaluate_policy(
        PolicyInputs(
            action=action,
            confidence=confidence,
            news_gate=news_gate,
            sentiment_gate=sentiment_gate,
            technical_gate=technical_gate,
            high_impact_within_window=high_impact,
            degraded=degraded,
            market_stale=stale,
            has_modified_levels=has_levels,
            component_scores=ComponentScores(*scores),
        ),
        thresholds,
    )


PARITY_CASES = [
    ("clean approval", {}, {}),
    ("news block", {"news_gate": GateDecision.BLOCK}, {"news_gate": GateSignal.BLOCK}),
    (
        "technical block",
        {"technical_gate": TechnicalDecision.BLOCK},
        {"technical_gate": GateSignal.BLOCK},
    ),
    (
        "sentiment block",
        {"sentiment_gate": GateDecision.BLOCK},
        {"sentiment_gate": GateSignal.BLOCK},
    ),
    ("blackout window", {"high_impact": True}, {"high_impact": True}),
    ("degraded inputs", {"degraded": True}, {"degraded": True}),
    ("low confidence", {"confidence": 0.3}, {"confidence": 0.3}),
    (
        "weak weighted score",
        {"scores": (30, 30, 30, 30)},
        {"scores": (30, 30, 30, 30)},
    ),
    (
        "block outranks blackout",
        {"news_gate": GateDecision.BLOCK, "high_impact": True},
        {"news_gate": GateSignal.BLOCK, "high_impact": True},
    ),
    (
        "a rejection passes through untouched",
        {"decision": FinalDecision.REJECT, "confidence": 0.2},
        {"action": PolicyAction.REJECT, "confidence": 0.2},
    ),
]


@pytest.mark.parametrize("name,live_kwargs,core_kwargs", PARITY_CASES)
def test_live_and_core_agree(name, live_kwargs, core_kwargs):
    """The live path and the shared core reach the same action."""
    settings = make_settings()
    live = _live_outcome(**live_kwargs)
    core = _core_outcome(thresholds_from_settings(settings), **core_kwargs)
    assert live.decision.value == core.action.value, name
    assert live.deciding_rule == core.deciding_rule, name


def test_research_resolution_matches_the_core_for_the_same_inputs():
    """The research path's own thresholds, fed identical inputs, agree.

    This is the parity that matters: both paths call one function, so the
    only way they can diverge is by being configured differently -- which the
    experiment config prevents and the first tests here assert.
    """
    config = ExperimentConfig()
    live_thresholds = thresholds_from_settings(config.risk_settings(make_settings()))
    research_thresholds = thresholds_from_ai_settings(config.ai_settings())

    for case in (
        {"technical_gate": GateSignal.BLOCK},
        {"high_impact": True},
        {"confidence": 0.1},
        {"scores": (20, 20, 20, 20)},
        {"action": PolicyAction.MODIFY, "has_levels": False},
        {},
    ):
        assert _core_outcome(live_thresholds, **case) == _core_outcome(
            research_thresholds, **case
        )


# --- the offline exception is deliberate and documented --------------------
def test_stale_market_waits_live_and_is_not_applied_offline():
    """Live, a stale quote forces WAIT. Offline it is meaningless.

    The research path passes market_stale=False on purpose, because every
    historical quote is old against the wall clock. The rule itself is NOT
    removed -- the core still applies it whenever the input is True.
    """
    live = _live_outcome(stale=True)
    assert live.decision == FinalDecision.WAIT

    thresholds = thresholds_from_settings(make_settings())
    assert _core_outcome(thresholds, stale=True).action == PolicyAction.WAIT
    assert _core_outcome(thresholds, stale=False).action == PolicyAction.APPROVE


# --- no safety rule was lost in the extraction -----------------------------
def test_policy_can_only_make_an_outcome_more_conservative():
    thresholds = thresholds_from_settings(make_settings())
    for action in (PolicyAction.REJECT, PolicyAction.WAIT):
        result = _core_outcome(thresholds, action=action, confidence=1.0, scores=(99, 99, 99, 99))
        assert result.action == action  # never upgraded to APPROVE


def test_missing_component_scores_skip_the_rule_and_are_recorded():
    """An absent score is not a passing score."""
    thresholds = thresholds_from_settings(make_settings())
    result = evaluate_policy(
        PolicyInputs(action=PolicyAction.APPROVE, confidence=0.9, component_scores=None),
        thresholds,
    )
    assert result.action == PolicyAction.APPROVE
    assert result.weighted_score is None
    assert "WEIGHTED_SCORE_UNAVAILABLE" in result.notes


def test_modify_without_levels_rejects_in_the_core():
    thresholds = thresholds_from_settings(make_settings())
    result = _core_outcome(thresholds, action=PolicyAction.MODIFY, has_levels=False)
    assert result.action == PolicyAction.REJECT
    assert result.deciding_rule == "MODIFY_MISSING_LEVELS"


def test_modify_disabled_rejects_in_the_core():
    thresholds = thresholds_from_settings(make_settings())
    disabled = PolicyThresholds(
        min_confidence=thresholds.min_confidence,
        min_weighted_score=thresholds.min_weighted_score,
        allow_modify=False,
    )
    result = _core_outcome(disabled, action=PolicyAction.MODIFY)
    assert result.action == PolicyAction.REJECT
    assert result.deciding_rule == "MODIFY_DISABLED"
