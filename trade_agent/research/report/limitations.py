from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

"""Known limitations, carried with the results rather than kept apart from them.

These are written down, attached to every manifest and printed in every
report, because a result presented without them invites a reader to believe
something the experiment cannot support. Each entry says what the limitation
is, why the obvious mitigation is insufficient, and what would actually
resolve it.
"""


class Severity(StrEnum):
    # Could change the direction of the conclusion.
    CRITICAL = "CRITICAL"
    # Could change the magnitude, not the direction.
    MATERIAL = "MATERIAL"
    # Worth disclosing; unlikely to change a decision.
    MINOR = "MINOR"


class Limitation(BaseModel):
    code: str
    severity: Severity
    title: str
    description: str
    why_not_mitigated: str = ""
    what_would_resolve_it: str = ""
    affects: list[str] = Field(default_factory=list)


MODEL_KNOWLEDGE_LEAKAGE = Limitation(
    code="MODEL_KNOWLEDGE_LEAKAGE",
    severity=Severity.CRITICAL,
    title="The language model may know what happened after the signal timestamp",
    description=(
        "Every agent in this experiment is a language model trained on text published "
        "after the period being replayed. When it is shown a XAUUSD setup dated "
        "2024-03-08, it may recognise the period and carry knowledge of what gold did "
        "next: rate decisions, geopolitical events, the shape of the trend. That "
        "knowledge is not in the prompt and cannot be removed from the weights. Any "
        "measured improvement from the AI layer is therefore an UPPER BOUND on what "
        "the same layer would achieve on genuinely unseen future data."
    ),
    why_not_mitigated=(
        "The prompts instruct the agents to reason only from the supplied data, and the "
        "point-in-time stores make sure no future data is supplied. Neither addresses "
        "this. An instruction cannot remove information from a model's weights, and a "
        "model does not need a future headline in its context to recall the period. "
        "Claiming that prompt discipline eliminates this risk would be false; it "
        "reduces explicit look-ahead, which is a different problem. Nor is the risk "
        "removed by the fact that the model is not asked to predict prices: recognising "
        "'this is the week before the March 2024 breakout' is enough to bias a verdict."
    ),
    what_would_resolve_it=(
        "Three controls, in increasing strength: (1) a DATE-STRIPPED PLACEBO ARM -- "
        "re-run the pilot with all absolute dates and period-identifying details removed "
        "from agent payloads (`AI_STRIP_DATES=true`) and compare decision distributions; "
        "a large difference indicates the dates were carrying information. (2) A "
        "SHUFFLED-PERIOD CONTROL -- present real setups with dates from unrelated "
        "periods and check whether verdict quality tracks the true period. (3) The only "
        "conclusive test: FORWARD PAPER TRADING on data generated after the model's "
        "training cutoff. Until (3) runs, the out-of-sample result is evidence about "
        "the layer's ceiling, not its expected live performance."
    ),
    affects=["experiment_b", "ai_vs_baseline_comparison", "all_ai_decisions"],
)

NO_INTRABAR_REPLAY = Limitation(
    code="NO_INTRABAR_REPLAY",
    severity=Severity.MATERIAL,
    title="Fills are resolved on 15-minute bars, not ticks",
    description=(
        "When one bar's range contains both the stop and the target, the true order of "
        "events is unknown without tick replay. The engine always assumes the stop came "
        "first, and assumes the pessimistic side of every other ambiguity."
    ),
    why_not_mitigated=(
        "Tick-level replay is implementable -- the tick data is the source of the bars -- "
        "but it was deliberately deferred until the pilot infrastructure is proven. The "
        "current assumption is conservative, so it understates rather than flatters "
        "results."
    ),
    what_would_resolve_it=(
        "Replay the stored ticks inside any bar where both levels are touched, and "
        "compare the two sets of results to size the effect."
    ),
    affects=["experiment_a", "experiment_b", "counterfactuals"],
)

NO_SWAP_COSTS = Limitation(
    code="NO_SWAP_COSTS",
    severity=Severity.MINOR,
    title="Overnight financing is not charged",
    description=(
        "Commission, spread and slippage are modelled; swap/financing on positions held "
        "overnight is not. The strategy's holding periods are intraday-to-short, so the "
        "omission is small, but it is an omission and it flatters both arms equally."
    ),
    why_not_mitigated="Deliberately deferred until after the pilot.",
    what_would_resolve_it=(
        "Add a per-night financing charge from the broker's published rates, applied to "
        "both arms."
    ),
    affects=["experiment_a", "experiment_b"],
)

SINGLE_INSTRUMENT_SINGLE_STRATEGY = Limitation(
    code="SINGLE_INSTRUMENT_SINGLE_STRATEGY",
    severity=Severity.MATERIAL,
    title="One instrument, one strategy, one out-of-sample period",
    description=(
        "The result describes this strategy on XAUUSD over one contiguous test period. "
        "It is a single observation, not a distribution, and the test period has its own "
        "regime."
    ),
    why_not_mitigated=(
        "It is inherent to the 70/30 design, which is the correct design for the "
        "question asked: it buys one honest out-of-sample answer at the cost of "
        "statistical breadth."
    ),
    what_would_resolve_it=(
        "Repeat on other instruments and other strategies, and report the distribution "
        "of outcomes rather than one number."
    ),
    affects=["all_conclusions"],
)

AGENT_DATA_UNAVAILABLE = Limitation(
    code="AGENT_DATA_UNAVAILABLE",
    severity=Severity.CRITICAL,
    title="Agents whose point-in-time dataset is missing cannot contribute",
    description=(
        "An agent with no data for a timestamp answers UNAVAILABLE and the fail-closed "
        "policy applies. If news and sentiment datasets are absent, the four-agent chain "
        "is effectively a technical agent plus an adjudicator, and the result measures "
        "THAT system -- not the one the design describes."
    ),
    why_not_mitigated=(
        "Deliberately. The alternative -- generating plausible historical headlines or "
        "scoring archived text with a present-day model -- would produce numbers that "
        "look complete and mean nothing."
    ),
    what_would_resolve_it=(
        "Supply genuine point-in-time datasets: timestamped news with publication times, "
        "sentiment captured live (not retrospectively scored), and an economic calendar "
        "with original-release values."
    ),
    affects=["experiment_b", "news_agent", "sentiment_agent"],
)

PILOT_SAMPLE_SIZE = Limitation(
    code="PILOT_SAMPLE_SIZE",
    severity=Severity.MATERIAL,
    title="A 100-signal pilot measures cost reliably and performance barely",
    description=(
        "The pilot exists to measure cost per signal and to prove the pipeline. With "
        "roughly 100 signals, a difference in win rate or expectancy between the arms is "
        "well inside noise."
    ),
    why_not_mitigated=(
        "It is the point of a pilot: measure the bill and validate the machinery before "
        "spending on the full run."
    ),
    what_would_resolve_it=(
        "The full out-of-sample run, once the pilot's measured cost justifies it."
    ),
    affects=["pilot_results"],
)

ALL_LIMITATIONS: list[Limitation] = [
    MODEL_KNOWLEDGE_LEAKAGE,
    AGENT_DATA_UNAVAILABLE,
    NO_INTRABAR_REPLAY,
    SINGLE_INSTRUMENT_SINGLE_STRATEGY,
    PILOT_SAMPLE_SIZE,
    NO_SWAP_COSTS,
]


def limitations_for(run_kind: str, dataset_gaps: list[str] | None = None) -> list[Limitation]:
    """The limitations that apply to a given run.

    The baseline arm involves no model, so the knowledge-leakage and
    data-availability entries do not apply to it; everything mechanical does.
    """
    selected = [
        limitation
        for limitation in ALL_LIMITATIONS
        if run_kind == "ai" or "experiment_a" in limitation.affects
        or "all_conclusions" in limitation.affects
    ]
    if dataset_gaps:
        extra = Limitation(
            code="DATASETS_MISSING",
            severity=Severity.CRITICAL,
            title=f"Dataset(s) unavailable for this run: {', '.join(dataset_gaps)}",
            description=(
                "The run executed with these datasets reported UNAVAILABLE. Every "
                "dependent agent answered UNAVAILABLE and the fail-closed policy "
                "applied. No substitute data was generated."
            ),
            why_not_mitigated="Fabricating the missing data is explicitly out of scope.",
            what_would_resolve_it="Supply the datasets and re-run.",
            affects=["experiment_b"],
        )
        selected = [extra, *selected]
    return selected


def as_dicts(limitations: list[Limitation]) -> list[dict]:
    return [limitation.model_dump(mode="json") for limitation in limitations]


def render_markdown(limitations: list[Limitation]) -> str:
    lines = ["## Known limitations", ""]
    for limitation in limitations:
        lines.append(f"### {limitation.severity.value} - {limitation.title}")
        lines.append("")
        lines.append(limitation.description)
        if limitation.why_not_mitigated:
            lines.append("")
            lines.append(f"**Why this is not mitigated:** {limitation.why_not_mitigated}")
        if limitation.what_would_resolve_it:
            lines.append("")
            lines.append(f"**What would resolve it:** {limitation.what_would_resolve_it}")
        lines.append("")
        lines.append(f"_Affects: {', '.join(limitation.affects)}_")
        lines.append("")
    return "\n".join(lines)
