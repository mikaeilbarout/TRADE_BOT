from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

"""The single deterministic decision policy, shared by the live service and
the research backtest.

Why this module exists: the same rules were previously written twice -- once
in `app/services/decision_policy.py` for live trading and once in
`research/ai/runner._resolve` for the historical replay. Two copies of a
safety policy drift, and a backtest that measures a policy the live system
does not run is worse than no backtest: it produces a number that looks like
evidence for a system nobody is going to trade.

Everything here is a pure function of explicit inputs. No settings objects,
no market snapshots, no LLM types -- both callers translate their own
domain objects into `PolicyInputs`, which is what makes a parity test
possible at all.

The one deliberate difference between the two callers is documented on
`PolicyInputs.market_stale`.
"""


class PolicyAction(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"
    WAIT = "WAIT"


class GateSignal(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ComponentScores:
    """The four dimension scores (0-100) the final agent reports.

    Optional: when a caller cannot supply them the weighted-score rule is
    skipped and `WEIGHTED_SCORE_UNAVAILABLE` is recorded, rather than a
    missing score being silently treated as a pass.
    """

    news_score: float
    sentiment_score: float
    technical_score: float
    risk_score: float


@dataclass(frozen=True)
class PolicyThresholds:
    """Numeric policy. Built from one source of truth per experiment so the
    live path and the research path cannot be configured differently."""

    min_confidence: float = 0.70
    min_weighted_score: float = 60.0
    veto_on_news_block: bool = True
    veto_on_sentiment_block: bool = True
    veto_on_technical_block: bool = True
    allow_modify: bool = True
    weight_news: float = 0.25
    weight_sentiment: float = 0.20
    weight_technical: float = 0.35
    weight_risk: float = 0.20

    def weights_sum(self) -> float:
        return (
            self.weight_news
            + self.weight_sentiment
            + self.weight_technical
            + self.weight_risk
        )

    def weighted_score(self, scores: ComponentScores) -> float:
        return (
            scores.news_score * self.weight_news
            + scores.sentiment_score * self.weight_sentiment
            + scores.technical_score * self.weight_technical
            + scores.risk_score * self.weight_risk
        )


@dataclass(frozen=True)
class PolicyInputs:
    """Everything the policy is allowed to look at.

    `market_stale` is the documented offline exception. Live, a stale quote
    forces WAIT because the price the decision was made on may no longer
    exist. In historical replay every quote is by definition "old" against
    the wall clock, so the research path passes `market_stale=False` and
    relies on the bar timestamps themselves; see
    `research.ai.gate.replay_risk_settings`. This is the ONLY input the two
    callers are expected to populate differently, and it is a property of
    the clock, not of the policy.

    An agent that was skipped because its point-in-time dataset had no
    coverage passes `UNAVAILABLE`, not `None`: absence is an input, and the
    policy treats it as neither a pass nor a block.
    """

    action: PolicyAction
    confidence: float
    news_gate: GateSignal = GateSignal.UNAVAILABLE
    sentiment_gate: GateSignal = GateSignal.UNAVAILABLE
    technical_gate: GateSignal = GateSignal.UNAVAILABLE
    high_impact_within_window: bool = False
    degraded: bool = False
    market_stale: bool = False
    has_modified_levels: bool = True
    component_scores: ComponentScores | None = None


@dataclass
class PolicyResult:
    action: PolicyAction
    reason: str
    weighted_score: float | None = None
    veto_triggered: bool = False
    veto_reasons: list[str] = field(default_factory=list)
    # Which rule decided the outcome, for the audit trail and for the
    # parity test to compare on more than the final action alone.
    deciding_rule: str = "AGENT_DECISION"
    notes: list[str] = field(default_factory=list)
    # Whether execution must be prevented, independent of the action label.
    #
    # WAIT carries two very different meanings and the label alone cannot
    # separate them: "not enough evidence to be confident" (the caller may
    # legitimately choose to trade anyway) versus "a high-impact event is
    # imminent" or "the price this was decided on is already dead" (the
    # caller must NOT trade). Callers that treat only REJECT as blocking
    # silently traded through the second kind. This flag is the machine-
    # readable answer, derived from the deterministic facts rather than
    # from which label an LLM happened to choose.
    execution_blocked: bool = False


# Rule identifiers, so "which rule rejected this" is a value rather than a
# substring match on an English sentence.
RULE_PASS_THROUGH = "AGENT_DECISION"
RULE_BLOCK_VETO = "BLOCK_VETO"
RULE_TIME_SENSITIVE_VETO = "TIME_SENSITIVE_VETO"
RULE_CONFIDENCE_FLOOR = "CONFIDENCE_FLOOR"
RULE_WEIGHTED_SCORE_FLOOR = "WEIGHTED_SCORE_FLOOR"
RULE_MODIFY_MISSING_LEVELS = "MODIFY_MISSING_LEVELS"
RULE_MODIFY_DISABLED = "MODIFY_DISABLED"

NOTE_SCORE_UNAVAILABLE = "WEIGHTED_SCORE_UNAVAILABLE"

# Time-sensitive veto reasons, as values rather than English substrings, so
# "is this one of the two that must actually stop a trade" is a comparison
# and not a fragile string match.
REASON_BLACKOUT = "high-impact event inside the news blackout window"
REASON_DEGRADED = "one or more agents reported degraded analysis inputs"
REASON_STALE = "market data went stale before the decision was finalized"

# Of those three, the two that mean "do not trade", as opposed to merely
# "we are not confident". Degraded inputs are deliberately NOT here: an
# incomplete evidence picture is an uncertainty signal, and whether to
# trade on uncertainty is the caller's policy, not this module's.
BLOCKING_WAIT_REASONS = frozenset({REASON_BLACKOUT, REASON_STALE})


def evaluate_policy(inputs: PolicyInputs, thresholds: PolicyThresholds) -> PolicyResult:
    """Apply the deterministic policy to an agent decision.

    Rule order is significant and matches the live service's established
    precedence:

      1. A non-approving decision passes through untouched. The policy only
         ever makes an outcome MORE conservative; it can never turn a
         REJECT into an APPROVE.
      2. A BLOCK from a gating agent rejects outright -- a specialist
         objecting on its own dimension outranks a favorable aggregate.
      3. A MODIFY without levels, or with modification disabled, rejects --
         an ambiguous instruction is never forwarded to execution. This
         runs BEFORE the time-sensitive vetoes below: it used to run last,
         so a MODIFY that carried no levels got downgraded to WAIT by a
         veto and never reached this rule at all, and a caller that trades
         through WAIT then executed the ORIGINAL levels the agent had just
         disavowed (seen twice in live data).
      4. Time-sensitive conditions (event inside the blackout window,
         degraded inputs, stale quote) become WAIT: the setup may be valid
         once the condition clears, so rejecting it permanently would be
         wrong.
      5. Confidence below the floor becomes WAIT.
      6. Weighted evidence score below the floor rejects.

    Separately from the action, `execution_blocked` says whether execution
    must actually be prevented -- see PolicyResult for why the action label
    alone is not enough.
    """
    notes: list[str] = []
    veto_reasons: list[str] = []

    scores = inputs.component_scores
    weighted = thresholds.weighted_score(scores) if scores is not None else None
    if scores is None:
        notes.append(NOTE_SCORE_UNAVAILABLE)

    # Deterministic "must not trade" facts, evaluated for every path
    # including the pass-throughs below: an agent that returns WAIT itself
    # (its prompt reserves WAIT for exactly these conditions) has to block
    # execution just as firmly as the same condition caught by a veto here.
    safety_block = inputs.high_impact_within_window or inputs.market_stale

    if inputs.action not in (PolicyAction.APPROVE, PolicyAction.MODIFY):
        return PolicyResult(
            action=inputs.action,
            reason="agent decision is not an approval; policy applies no change",
            weighted_score=weighted,
            deciding_rule=RULE_PASS_THROUGH,
            notes=notes,
            execution_blocked=safety_block or inputs.action == PolicyAction.REJECT,
        )

    # --- 2. BLOCK vetoes ------------------------------------------------
    block_reasons: list[str] = []
    if thresholds.veto_on_news_block and inputs.news_gate == GateSignal.BLOCK:
        block_reasons.append("news agent returned BLOCK")
    if thresholds.veto_on_sentiment_block and inputs.sentiment_gate == GateSignal.BLOCK:
        block_reasons.append("sentiment agent returned BLOCK")
    if thresholds.veto_on_technical_block and inputs.technical_gate == GateSignal.BLOCK:
        block_reasons.append("technical agent returned BLOCK")

    # --- 4. time-sensitive vetoes (computed here, applied after rule 3) --
    wait_reasons: list[str] = []
    if inputs.high_impact_within_window:
        wait_reasons.append(REASON_BLACKOUT)
    if inputs.degraded:
        wait_reasons.append(REASON_DEGRADED)
    if inputs.market_stale:
        wait_reasons.append(REASON_STALE)

    veto_reasons = block_reasons + wait_reasons

    if block_reasons:
        return PolicyResult(
            action=PolicyAction.REJECT,
            reason="Policy veto overrode AI decision: " + "; ".join(veto_reasons),
            weighted_score=weighted,
            veto_triggered=True,
            veto_reasons=veto_reasons,
            deciding_rule=RULE_BLOCK_VETO,
            notes=notes,
            execution_blocked=True,
        )

    # --- 3. modification integrity ---------------------------------------
    if inputs.action == PolicyAction.MODIFY:
        if not thresholds.allow_modify:
            return PolicyResult(
                action=PolicyAction.REJECT,
                reason="MODIFY returned but modification is disabled by configuration",
                weighted_score=weighted,
                deciding_rule=RULE_MODIFY_DISABLED,
                notes=notes,
                execution_blocked=True,
            )
        if not inputs.has_modified_levels:
            return PolicyResult(
                action=PolicyAction.REJECT,
                reason=(
                    "MODIFY returned without modified trade parameters; failing "
                    "closed rather than forwarding an ambiguous instruction"
                ),
                weighted_score=weighted,
                deciding_rule=RULE_MODIFY_MISSING_LEVELS,
                notes=notes,
                execution_blocked=True,
            )

    if wait_reasons:
        return PolicyResult(
            action=PolicyAction.WAIT,
            reason="Policy veto (time-sensitive) overrode AI decision: "
            + "; ".join(veto_reasons),
            weighted_score=weighted,
            veto_triggered=True,
            veto_reasons=veto_reasons,
            deciding_rule=RULE_TIME_SENSITIVE_VETO,
            notes=notes,
            execution_blocked=any(r in BLOCKING_WAIT_REASONS for r in wait_reasons),
        )

    # --- 5. confidence floor ---------------------------------------------
    if inputs.confidence < thresholds.min_confidence:
        return PolicyResult(
            action=PolicyAction.WAIT,
            reason=(
                f"confidence {inputs.confidence:.2f} is below the required minimum "
                f"{thresholds.min_confidence}; downgraded to WAIT"
            ),
            weighted_score=weighted,
            deciding_rule=RULE_CONFIDENCE_FLOOR,
            notes=notes,
            execution_blocked=safety_block,
        )

    # --- 6. weighted evidence floor --------------------------------------
    if weighted is not None and weighted < thresholds.min_weighted_score:
        return PolicyResult(
            action=PolicyAction.REJECT,
            reason=(
                f"Weighted evidence score {weighted:.1f} is below the required "
                f"minimum {thresholds.min_weighted_score}"
            ),
            weighted_score=weighted,
            deciding_rule=RULE_WEIGHTED_SCORE_FLOOR,
            notes=notes,
            execution_blocked=True,
        )

    return PolicyResult(
        action=inputs.action,
        reason="policy clear: no veto and all thresholds met",
        weighted_score=weighted,
        deciding_rule=RULE_PASS_THROUGH,
        notes=notes,
        execution_blocked=safety_block,
    )
