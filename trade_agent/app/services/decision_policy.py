from __future__ import annotations

from dataclasses import dataclass, field

from app.config.settings import Settings
from app.models.agent_decision import (
    FinalDecisionResult,
    NewsAgentResult,
    SentimentAgentResult,
    TechnicalAgentResult,
)
from app.models.enums import FinalDecision, GateDecision, TechnicalDecision
from app.models.market_data import MarketSnapshot
from app.models.signal import TradeSignal
from app.services.policy_core import (
    ComponentScores,
    GateSignal,
    PolicyAction,
    PolicyInputs,
    PolicyThresholds,
    evaluate_policy,
)
from app.services.technical_service import higher_tf_alignment


@dataclass
class PolicyOutcome:
    decision: FinalDecision
    reason: str
    weighted_score: float
    veto_triggered: bool = False
    veto_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Which shared-core rule decided this, so the audit trail names a rule
    # rather than only carrying an English sentence.
    deciding_rule: str = "AGENT_DECISION"
    # See PolicyResult.execution_blocked -- the decision label alone does
    # not tell a caller whether it may trade.
    execution_blocked: bool = False


def thresholds_from_settings(settings: Settings) -> PolicyThresholds:
    """The live service's view of the shared policy numbers.

    `research.experiment.ExperimentConfig` builds the research side's
    thresholds from the same fields, and a parity test asserts the two agree
    -- which is what stops the backtest from measuring a policy the live
    system does not run.
    """
    return PolicyThresholds(
        min_confidence=settings.min_confidence,
        min_weighted_score=settings.min_weighted_score,
        veto_on_news_block=settings.veto_on_news_block,
        veto_on_sentiment_block=settings.veto_on_sentiment_block,
        veto_on_technical_block=settings.veto_on_technical_block,
        allow_modify=True,
        weight_news=settings.weight_news,
        weight_sentiment=settings.weight_sentiment,
        weight_technical=settings.weight_technical,
        weight_risk=settings.weight_risk,
    )


_GATE_MAP = {
    GateDecision.PASS: GateSignal.PASS,
    GateDecision.WARNING: GateSignal.WARN,
    GateDecision.BLOCK: GateSignal.BLOCK,
}

_TECHNICAL_GATE_MAP = {
    TechnicalDecision.PASS: GateSignal.PASS,
    TechnicalDecision.WARNING: GateSignal.WARN,
    TechnicalDecision.BLOCK: GateSignal.BLOCK,
}


class DecisionPolicy:
    """Deterministic backstop applied to the final agent's decision.

    The final agent is instructed to respect vetoes and thresholds, but an
    instruction is not a guarantee -- so the same rules are enforced here in
    code, where they cannot be argued with:

      * A BLOCK from any gating agent vetoes approval (configurable per
        agent), because a specialist saying "this conflicts with my
        dimension" outranks a favorable aggregate (section 10).
      * A high-impact event inside the blackout window forces WAIT even
        when every score is strong -- the spec's explicit example.
      * The weighted score (using the configured dimension weights) must
        clear min_weighted_score; a naive-average-looking APPROVE with weak
        components does not pass.
      * Confidence must clear min_confidence.
      * Degraded upstream data forces WAIT rather than APPROVE.

    It also cross-checks the technical agent's structural claims against
    the indicators computed deterministically from the same candles. A
    contradiction does not by itself veto (the computed trend heuristic is
    deliberately crude), but it is recorded as a warning so blind trust in
    the LLM's read is visible in the audit trail.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def weighted_score(self, final_result: FinalDecisionResult) -> float:
        s = self._settings
        scores = final_result.scores
        return (
            scores.news_score * s.weight_news
            + scores.sentiment_score * s.weight_sentiment
            + scores.technical_score * s.weight_technical
            + scores.risk_score * s.weight_risk
        )

    def apply(
        self,
        signal: TradeSignal,
        final_result: FinalDecisionResult,
        news_result: NewsAgentResult,
        sentiment_result: SentimentAgentResult,
        technical_result: TechnicalAgentResult,
        market: MarketSnapshot,
        data_degraded: bool = False,
    ) -> PolicyOutcome:
        """`data_degraded` is the DETERMINISTIC fact -- did the news or
        sentiment service actually serve a degraded/fallback bundle. The
        per-agent `is_degraded` fields below are LLM-authored, so relying
        on them alone made a deterministic safety rule depend on the model
        volunteering a flag; the two are OR-ed so the rule fires on the
        known fact even when the model omits it."""
        s = self._settings
        decision = final_result.decision
        score = self.weighted_score(final_result)
        vetoes: list[str] = []
        warnings: list[str] = []

        # --- Cross-check LLM claims against computed indicators ----------
        higher_tf = next(
            (
                tf
                for tf in s.default_higher_timeframes
                if tf in market.indicators
            ),
            None,
        )
        if higher_tf is not None:
            computed_trend = market.indicators[higher_tf].trend or "RANGE"
            computed_alignment = higher_tf_alignment(signal.side.value, computed_trend)
            if technical_result.aligned_with_higher_tf and not computed_alignment:
                warnings.append(
                    f"technical agent claims higher-timeframe alignment, but computed "
                    f"{higher_tf} trend is {computed_trend} for a {signal.side.value}"
                )
            elif computed_alignment and not technical_result.aligned_with_higher_tf:
                warnings.append(
                    f"technical agent reports no higher-timeframe alignment, but computed "
                    f"{higher_tf} trend is {computed_trend} for a {signal.side.value}"
                )

        # --- Delegate the decision itself to the shared policy core ------
        # Every rule below this point is implemented once, in
        # app/services/policy_core.py, and the research backtest calls the
        # same function with the same thresholds.
        inputs = PolicyInputs(
            action=PolicyAction(decision.value),
            confidence=final_result.confidence,
            news_gate=_GATE_MAP.get(news_result.decision, GateSignal.UNAVAILABLE),
            sentiment_gate=_GATE_MAP.get(sentiment_result.decision, GateSignal.UNAVAILABLE),
            technical_gate=_TECHNICAL_GATE_MAP.get(
                technical_result.decision, GateSignal.UNAVAILABLE
            ),
            high_impact_within_window=news_result.high_impact_event_within_minutes,
            degraded=(
                data_degraded
                or news_result.is_degraded
                or sentiment_result.is_degraded
                or technical_result.is_degraded
            ),
            # Live: a stale quote is a real hazard, so it is reported as-is.
            market_stale=market.is_stale,
            # The live vocabulary has no MODIFY any more; the flag only
            # exists for the research path, which still replays one.
            has_modified_levels=True,
            component_scores=ComponentScores(
                news_score=final_result.scores.news_score,
                sentiment_score=final_result.scores.sentiment_score,
                technical_score=final_result.scores.technical_score,
                risk_score=final_result.scores.risk_score,
            ),
        )
        result = evaluate_policy(inputs, thresholds_from_settings(s))

        # The core reports veto reasons generically; the live audit trail
        # names the configured blackout length, so it is restated here.
        vetoes = [
            reason.replace(
                "the news blackout window",
                f"the {s.high_impact_news_blackout_minutes}-minute blackout window",
            )
            for reason in result.veto_reasons
        ]
        reason = result.reason
        if result.deciding_rule != "AGENT_DECISION":
            reason = f"{reason}. Agent reasoning: {final_result.summary}"
            for original, restated in zip(result.veto_reasons, vetoes):
                reason = reason.replace(original, restated)
        else:
            reason = final_result.summary

        return PolicyOutcome(
            decision=FinalDecision(result.action.value),
            reason=reason,
            weighted_score=result.weighted_score if result.weighted_score is not None else score,
            veto_triggered=result.veto_triggered,
            veto_reasons=vetoes,
            warnings=warnings,
            deciding_rule=result.deciding_rule,
            execution_blocked=result.execution_blocked,
        )
