from __future__ import annotations

from pydantic import BaseModel, Field

from research.backtest.engine import BacktestResult
from research.backtest.executor import BY_AI
from research.backtest.trade import SkippedSignal, Trade

"""Diagnostic analysis of what the AI layer let through and what it stopped.

This answers the questions the experiment is actually for, none of which the
equity curves answer on their own:

  * How many profitable trades did the AI reject?
  * How many losing trades did it correctly reject?
  * How many winners did it approve, and how many losers?
  * Which agent caused the most false rejections?
  * What percentage of rejected signals would have been profitable?

Every number here is diagnostic. Counterfactual P&L is computed on a fixed
notional balance and is never added to either arm's equity curve, so nothing
in this module can move a reported return.

One caveat that belongs next to the results rather than in a footnote: a
"false rejection" here means the rejected trade would have won. That is not
the same as the rejection being wrong. Declining a trade that happened to win
but ran 2R against you first can be correct risk management, which is why
maximum adverse excursion is reported alongside the outcome.
"""


class DecisionQuadrant(BaseModel):
    """The confusion matrix of the AI layer as a trade filter."""

    approved_winners: int = 0
    approved_losers: int = 0
    approved_breakeven: int = 0
    rejected_would_have_won: int = 0
    rejected_would_have_lost: int = 0
    rejected_would_have_broken_even: int = 0
    rejected_unscoreable: int = 0

    @property
    def approved_total(self) -> int:
        return self.approved_winners + self.approved_losers + self.approved_breakeven

    @property
    def rejected_scored(self) -> int:
        return (
            self.rejected_would_have_won
            + self.rejected_would_have_lost
            + self.rejected_would_have_broken_even
        )

    @property
    def precision(self) -> float | None:
        """Share of approved trades that won."""
        return (
            self.approved_winners / self.approved_total
            if self.approved_total
            else None
        )

    @property
    def rejection_accuracy(self) -> float | None:
        """Share of rejections that avoided a loss."""
        return (
            self.rejected_would_have_lost / self.rejected_scored
            if self.rejected_scored
            else None
        )


class AgentAttribution(BaseModel):
    agent: str
    rejections: int = 0
    would_have_won: int = 0
    would_have_lost: int = 0
    unscoreable: int = 0
    profit_forgone: float = 0.0   # sum of positive counterfactual P&L
    loss_avoided: float = 0.0     # sum of |negative counterfactual P&L|
    net_contribution: float = 0.0  # loss_avoided - profit_forgone
    mean_adverse_excursion_r_of_winners: float | None = None

    @property
    def false_rejection_rate(self) -> float | None:
        scored = self.would_have_won + self.would_have_lost
        return self.would_have_won / scored if scored else None


class CounterfactualAnalysis(BaseModel):
    run_id: str
    signals_considered: int
    executed: int
    rejected_by_ai: int
    quadrant: DecisionQuadrant = Field(default_factory=DecisionQuadrant)

    pct_rejected_would_have_been_profitable: float | None = None
    profit_forgone: float = 0.0
    loss_avoided: float = 0.0
    net_contribution_of_rejections: float = 0.0
    mean_r_of_rejected: float | None = None
    mean_r_of_approved: float | None = None

    by_agent: list[AgentAttribution] = Field(default_factory=list)
    by_reason_code: dict[str, int] = Field(default_factory=dict)
    by_deciding_rule: dict[str, int] = Field(default_factory=dict)
    by_skipped_by: dict[str, int] = Field(default_factory=dict)

    # Counterfactual trades that could not be scored (e.g. a signal on the
    # final bar). Reported rather than dropped so percentages have a stated
    # denominator.
    unscoreable: int = 0
    unscoreable_reasons: dict[str, int] = Field(default_factory=dict)

    def headline(self) -> str:
        parts = [
            f"{self.quadrant.approved_winners} winners and "
            f"{self.quadrant.approved_losers} losers approved",
            f"{self.quadrant.rejected_would_have_lost} losers and "
            f"{self.quadrant.rejected_would_have_won} winners rejected",
        ]
        if self.net_contribution_of_rejections:
            direction = (
                "saved" if self.net_contribution_of_rejections > 0 else "cost"
            )
            parts.append(
                f"rejections {direction} "
                f"{abs(self.net_contribution_of_rejections):,.2f} in counterfactual P&L"
            )
        return "; ".join(parts)


def _ai_rejections(result: BacktestResult) -> list[SkippedSignal]:
    """Only signals the AI layer itself declined.

    A signal blocked by a deterministic rule or an engine constraint is not an
    AI decision, and counting it as one would credit or blame the AI for the
    risk engine's work.
    """
    return [s for s in result.skipped if s.skipped_by == BY_AI]


def analyze_counterfactuals(
    result: BacktestResult, baseline: BacktestResult | None = None
) -> CounterfactualAnalysis:
    """Build the diagnostic analysis from an AI run's result."""
    rejections = _ai_rejections(result)
    analysis = CounterfactualAnalysis(
        run_id=result.run_id,
        signals_considered=result.signals_generated,
        executed=len(result.trades),
        rejected_by_ai=len(rejections),
    )

    for trade in result.trades:
        if trade.profit > 0:
            analysis.quadrant.approved_winners += 1
        elif trade.profit < 0:
            analysis.quadrant.approved_losers += 1
        else:
            analysis.quadrant.approved_breakeven += 1

    approved_r = [t.r_multiple for t in result.trades]
    analysis.mean_r_of_approved = (
        sum(approved_r) / len(approved_r) if approved_r else None
    )

    attribution: dict[str, AgentAttribution] = {}
    winner_excursions: dict[str, list[float]] = {}
    rejected_r: list[float] = []

    for skipped in rejections:
        agent = skipped.blocking_agent or "unattributed"
        record = attribution.setdefault(agent, AgentAttribution(agent=agent))
        record.rejections += 1

        analysis.by_skipped_by[skipped.skipped_by] = (
            analysis.by_skipped_by.get(skipped.skipped_by, 0) + 1
        )
        if skipped.ai_deciding_rule:
            analysis.by_deciding_rule[skipped.ai_deciding_rule] = (
                analysis.by_deciding_rule.get(skipped.ai_deciding_rule, 0) + 1
            )
        for code in skipped.ai_reason_codes:
            analysis.by_reason_code[code] = analysis.by_reason_code.get(code, 0) + 1

        if not skipped.counterfactual_available:
            analysis.quadrant.rejected_unscoreable += 1
            analysis.unscoreable += 1
            reason = skipped.counterfactual_unavailable_reason or "unknown"
            analysis.unscoreable_reasons[reason] = (
                analysis.unscoreable_reasons.get(reason, 0) + 1
            )
            record.unscoreable += 1
            continue

        profit = skipped.counterfactual_profit or 0.0
        rejected_r.append(skipped.counterfactual_r or 0.0)
        if profit > 0:
            analysis.quadrant.rejected_would_have_won += 1
            analysis.profit_forgone += profit
            record.would_have_won += 1
            record.profit_forgone += profit
            if skipped.counterfactual_mae_r is not None:
                winner_excursions.setdefault(agent, []).append(
                    skipped.counterfactual_mae_r
                )
        elif profit < 0:
            analysis.quadrant.rejected_would_have_lost += 1
            analysis.loss_avoided += abs(profit)
            record.would_have_lost += 1
            record.loss_avoided += abs(profit)
        else:
            analysis.quadrant.rejected_would_have_broken_even += 1

    for agent, record in attribution.items():
        record.net_contribution = record.loss_avoided - record.profit_forgone
        excursions = winner_excursions.get(agent)
        if excursions:
            record.mean_adverse_excursion_r_of_winners = round(
                sum(excursions) / len(excursions), 4
            )

    analysis.by_agent = sorted(
        attribution.values(), key=lambda a: (a.would_have_won, a.rejections), reverse=True
    )
    analysis.net_contribution_of_rejections = analysis.loss_avoided - analysis.profit_forgone
    analysis.mean_r_of_rejected = (
        sum(rejected_r) / len(rejected_r) if rejected_r else None
    )
    scored = analysis.quadrant.rejected_scored
    analysis.pct_rejected_would_have_been_profitable = (
        round(analysis.quadrant.rejected_would_have_won / scored * 100, 2)
        if scored
        else None
    )
    return analysis


def render_markdown(analysis: CounterfactualAnalysis) -> str:
    q = analysis.quadrant
    lines = [
        "## AI decision quality (diagnostic)",
        "",
        f"Run `{analysis.run_id}`: {analysis.headline()}.",
        "",
        "Counterfactual P&L below is computed on a fixed notional balance and is NOT",
        "part of either experiment's equity curve.",
        "",
        "| | Would have won | Would have lost | Breakeven | Unscoreable |",
        "|---|---|---|---|---|",
        f"| **Approved** (executed) | {q.approved_winners} | {q.approved_losers} | "
        f"{q.approved_breakeven} | - |",
        f"| **Rejected / waited** | {q.rejected_would_have_won} | "
        f"{q.rejected_would_have_lost} | {q.rejected_would_have_broken_even} | "
        f"{q.rejected_unscoreable} |",
        "",
    ]

    pct = analysis.pct_rejected_would_have_been_profitable
    lines += [
        "| Question | Answer |",
        "|---|---|",
        f"| How many profitable trades did the AI reject? | {q.rejected_would_have_won} |",
        f"| How many losing trades did it correctly reject? | {q.rejected_would_have_lost} |",
        f"| How many winners did it approve? | {q.approved_winners} |",
        f"| How many losers did it approve? | {q.approved_losers} |",
        f"| % of rejected signals that would have been profitable | "
        f"{'n/a' if pct is None else f'{pct:.1f}%'} |",
        f"| Approved-trade win rate | "
        f"{'n/a' if q.precision is None else f'{q.precision * 100:.1f}%'} |",
        f"| Rejections that avoided a loss | "
        f"{'n/a' if q.rejection_accuracy is None else f'{q.rejection_accuracy * 100:.1f}%'} |",
        f"| Profit forgone on rejected winners | {analysis.profit_forgone:,.2f} |",
        f"| Loss avoided on rejected losers | {analysis.loss_avoided:,.2f} |",
        f"| Net contribution of rejections | "
        f"{analysis.net_contribution_of_rejections:,.2f} |",
        "",
    ]

    if analysis.by_agent:
        lines += [
            "### Attribution by blocking agent",
            "",
            "| Agent | Rejections | Would have won | Would have lost | False-rejection rate "
            "| Profit forgone | Loss avoided | Net | Mean MAE(R) of rejected winners |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for record in analysis.by_agent:
            rate = record.false_rejection_rate
            mae = record.mean_adverse_excursion_r_of_winners
            lines.append(
                f"| {record.agent} | {record.rejections} | {record.would_have_won} | "
                f"{record.would_have_lost} | "
                f"{'n/a' if rate is None else f'{rate * 100:.1f}%'} | "
                f"{record.profit_forgone:,.2f} | {record.loss_avoided:,.2f} | "
                f"{record.net_contribution:,.2f} | "
                f"{'n/a' if mae is None else f'{mae:.2f}'} |"
            )
        lines += [
            "",
            "A high false-rejection rate is not automatically a fault. The last column is "
            "there for that reason: a rejected winner that first ran 2R against the entry "
            "was a trade worth declining.",
            "",
        ]

    if analysis.by_reason_code:
        lines += ["### Rejections by reason code", ""]
        for code, count in sorted(
            analysis.by_reason_code.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(f"- `{code}`: {count}")
        lines.append("")

    if analysis.unscoreable:
        lines += [
            f"### Unscoreable counterfactuals ({analysis.unscoreable})",
            "",
        ]
        for reason, count in analysis.unscoreable_reasons.items():
            lines.append(f"- {reason}: {count}")
        lines.append("")

    return "\n".join(lines)
