from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from research.ai.cost import CallRecord, CostSummary, summarize_costs
from research.ai.models import PRICING_SNAPSHOT_DATE, get_pricing
from research.backtest.engine import BacktestResult
from research.report.limitations import Limitation, as_dicts

"""The pilot's cost report, in both machine-readable and human-readable form.

Every figure is MEASURED from recorded token counts. The only computed
quantities are the projections, which are explicit multiples of the measured
average and labelled as such -- the difference between "this run cost $0.83"
and "a full run would cost about $41" matters, and the report keeps them
apart.

The post-join figures (cost per approved trade, AI cost as a share of profit)
require the execution bridge, so they appear only when a `BacktestResult` is
supplied. They are never estimated from assumptions when the measurement is
available, and are reported as unavailable when it is not.
"""


class AgentCostBreakdown(BaseModel):
    agent: str
    model: str | None = None
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_share_pct: float = 0.0
    avg_cost_per_call: float = 0.0
    cache_read_ratio: float = 0.0


class ProfitImpact(BaseModel):
    """AI spend measured against what the AI arm actually produced."""

    available: bool = False
    unavailable_reason: str | None = None
    approved_trades: int = 0
    cost_per_approved_trade: float | None = None
    gross_profit: float | None = None
    net_profit: float | None = None
    ai_cost_pct_of_gross_profit: float | None = None
    ai_cost_pct_of_net_profit: float | None = None
    net_profit_after_ai_cost: float | None = None
    note: str | None = None


class CostReport(BaseModel):
    """The machine-readable pilot cost report."""

    run_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pricing_snapshot_date: str = PRICING_SNAPSHOT_DATE
    models_used: dict[str, str] = Field(default_factory=dict)
    batch_used: bool = False
    prompt_cache_enabled: bool = True
    budget_usd: float | None = None
    budget_stopped: bool = False
    stopped_reason: str | None = None

    # --- signal counts ---------------------------------------------------
    total_signals: int = 0
    processed_signals: int = 0
    skipped_deterministic_signals: int = 0
    resumed_signals: int = 0
    failed_closed_signals: int = 0
    agent_calls_skipped_for_missing_data: int = 0

    # --- measured tokens --------------------------------------------------
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    total_input_equivalent_tokens: int = 0
    cache_read_ratio: float = 0.0
    cache_hit_rate: float = 0.0

    # --- measured cost ----------------------------------------------------
    total_cost_usd: float = 0.0
    api_calls: int = 0
    failed_calls: int = 0
    avg_cost_per_signal: float = 0.0
    median_cost_per_signal: float = 0.0
    min_cost_per_signal: float = 0.0
    max_cost_per_signal: float = 0.0
    avg_cost_per_call: float = 0.0
    avg_latency_seconds: float = 0.0
    cost_by_agent: dict[str, float] = Field(default_factory=dict)
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    by_agent: list[AgentCostBreakdown] = Field(default_factory=list)

    # --- projections (computed, not measured) -----------------------------
    projected_1k_usd: float = 0.0
    projected_5k_usd: float = 0.0
    projected_10k_usd: float = 0.0
    projected_full_oos_usd: float | None = None
    full_oos_signal_count: int | None = None
    projection_basis: str = ""

    # --- post-join -------------------------------------------------------
    profit_impact: ProfitImpact = Field(default_factory=ProfitImpact)

    decision_counts: dict[str, int] = Field(default_factory=dict)
    limitations: list[dict] = Field(default_factory=list)

    def save_json(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        return Path(path)


def _breakdowns(records: list[CallRecord], total_cost: float) -> list[AgentCostBreakdown]:
    grouped: dict[str, AgentCostBreakdown] = {}
    for record in records:
        entry = grouped.setdefault(
            record.agent, AgentCostBreakdown(agent=record.agent, model=record.model)
        )
        entry.calls += 1
        if record.error:
            entry.failed_calls += 1
        entry.input_tokens += record.usage.input_tokens
        entry.cache_creation_tokens += record.usage.cache_creation_tokens
        entry.cache_read_tokens += record.usage.cache_read_tokens
        entry.output_tokens += record.usage.output_tokens
        entry.cost_usd += record.cost_usd

    for entry in grouped.values():
        entry.cost_share_pct = (
            round(entry.cost_usd / total_cost * 100, 2) if total_cost else 0.0
        )
        entry.avg_cost_per_call = entry.cost_usd / entry.calls if entry.calls else 0.0
        equivalent = (
            entry.input_tokens + entry.cache_creation_tokens + entry.cache_read_tokens
        )
        entry.cache_read_ratio = (
            round(entry.cache_read_tokens / equivalent, 4) if equivalent else 0.0
        )
    return sorted(grouped.values(), key=lambda e: e.cost_usd, reverse=True)


def profit_impact(
    total_cost_usd: float, result: BacktestResult | None
) -> ProfitImpact:
    """AI cost against realised profit, from measurements only."""
    if result is None:
        return ProfitImpact(
            available=False,
            unavailable_reason=(
                "no AI-arm backtest result supplied: cost per approved trade and cost as "
                "a share of profit require the decision-to-trade join, so they are "
                "reported as unavailable rather than estimated"
            ),
        )

    trades = result.trades
    gross = sum(t.gross_profit for t in trades)
    net = sum(t.profit for t in trades)
    impact = ProfitImpact(
        available=True,
        approved_trades=len(trades),
        cost_per_approved_trade=(total_cost_usd / len(trades)) if trades else None,
        gross_profit=gross,
        net_profit=net,
        net_profit_after_ai_cost=net - total_cost_usd,
    )
    if gross > 0:
        impact.ai_cost_pct_of_gross_profit = round(total_cost_usd / gross * 100, 4)
    if net > 0:
        impact.ai_cost_pct_of_net_profit = round(total_cost_usd / net * 100, 4)
    else:
        impact.note = (
            "the AI arm was not profitable over this period, so AI cost as a percentage "
            "of profit is undefined; the absolute cost and the net-after-cost figure are "
            "the meaningful ones"
        )
    return impact


def build_cost_report(
    run_id: str,
    records: list[CallRecord],
    total_signals: int,
    processed_signals: int,
    skipped_deterministic: int,
    models_used: dict[str, str],
    *,
    resumed_signals: int = 0,
    failed_closed_signals: int = 0,
    agent_calls_skipped: int = 0,
    decision_counts: dict[str, int] | None = None,
    ai_result: BacktestResult | None = None,
    full_oos_signal_count: int | None = None,
    budget_usd: float | None = None,
    budget_stopped: bool = False,
    stopped_reason: str | None = None,
    prompt_cache_enabled: bool = True,
    limitations: list[Limitation] | None = None,
) -> CostReport:
    """Assemble the measured cost report."""
    summary: CostSummary = summarize_costs(
        records, processed_signals, full_oos_signal_count
    )
    total_cost = summary.total_cost_usd

    input_tokens = sum(r.usage.input_tokens for r in records)
    cache_creation = sum(r.usage.cache_creation_tokens for r in records)
    cache_read = sum(r.usage.cache_read_tokens for r in records)
    output_tokens = sum(r.usage.output_tokens for r in records)
    equivalent = input_tokens + cache_creation + cache_read

    per_call = [r.cost_usd for r in records] or [0.0]

    report = CostReport(
        run_id=run_id,
        models_used=models_used,
        batch_used=summary.batch_used,
        prompt_cache_enabled=prompt_cache_enabled,
        budget_usd=budget_usd,
        budget_stopped=budget_stopped,
        stopped_reason=stopped_reason,
        total_signals=total_signals,
        processed_signals=processed_signals,
        skipped_deterministic_signals=skipped_deterministic,
        resumed_signals=resumed_signals,
        failed_closed_signals=failed_closed_signals,
        agent_calls_skipped_for_missing_data=agent_calls_skipped,
        input_tokens=input_tokens,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
        output_tokens=output_tokens,
        total_input_equivalent_tokens=equivalent,
        cache_read_ratio=round(cache_read / equivalent, 4) if equivalent else 0.0,
        cache_hit_rate=round(summary.cache_hit_rate, 4),
        total_cost_usd=total_cost,
        api_calls=summary.api_calls,
        failed_calls=summary.failed_calls,
        avg_cost_per_signal=summary.avg_cost_per_signal,
        median_cost_per_signal=summary.median_cost_per_signal,
        min_cost_per_signal=summary.min_cost_per_signal,
        max_cost_per_signal=summary.max_cost_per_signal,
        avg_cost_per_call=statistics.fmean(per_call),
        avg_latency_seconds=summary.avg_latency_seconds,
        cost_by_agent=summary.cost_by_agent,
        cost_by_model=summary.cost_by_model,
        by_agent=_breakdowns(records, total_cost),
        projected_1k_usd=summary.projected_1k,
        projected_5k_usd=summary.projected_5k,
        projected_10k_usd=summary.projected_10k,
        projected_full_oos_usd=summary.projected_full_run,
        full_oos_signal_count=full_oos_signal_count,
        projection_basis=(
            f"measured average of ${summary.avg_cost_per_signal:.6f} per signal over "
            f"{processed_signals} processed signal(s), multiplied by the signal count. "
            "Assumes the same cache-hit behaviour and the same deterministic skip rate; "
            "a longer run should improve both, so these are upper estimates."
        ),
        decision_counts=decision_counts or {},
        limitations=as_dicts(limitations or []),
    )
    report.profit_impact = profit_impact(total_cost, ai_result)
    return report


def render_markdown(report: CostReport) -> str:
    """The human-readable report."""
    r = report
    lines = [
        f"# AI cost report - run `{r.run_id}`",
        "",
        f"Generated {r.generated_at:%Y-%m-%d %H:%M} UTC. "
        f"Pricing snapshot: {r.pricing_snapshot_date}.",
        "",
        "All token and cost figures are MEASURED from the API's own usage reporting.",
        "Projections are explicit multiples of the measured average and are labelled.",
        "",
        "## Headline",
        "",
        f"- **Total measured cost: ${r.total_cost_usd:.4f}**",
        f"- Average per signal: ${r.avg_cost_per_signal:.6f}",
        f"- Signals processed: {r.processed_signals} of {r.total_signals}",
        f"- API calls: {r.api_calls} ({r.failed_calls} failed)",
        f"- Cache read ratio: {r.cache_read_ratio * 100:.1f}% of input-equivalent tokens",
        "",
    ]

    if r.budget_stopped:
        lines += [
            f"> **The run stopped on the budget limit** (${r.budget_usd}): "
            f"{r.stopped_reason}",
            "",
        ]

    lines += [
        "## Signal accounting",
        "",
        "| | Count |",
        "|---|---|",
        f"| Total signals in scope | {r.total_signals} |",
        f"| Processed through the AI layer | {r.processed_signals} |",
        f"| Rejected deterministically before any LLM call (zero cost) | "
        f"{r.skipped_deterministic_signals} |",
        f"| Resumed from checkpoint (not re-charged) | {r.resumed_signals} |",
        f"| Failed closed | {r.failed_closed_signals} |",
        f"| Agent calls skipped for missing point-in-time data | "
        f"{r.agent_calls_skipped_for_missing_data} |",
        "",
        "## Measured tokens",
        "",
        "| Token type | Count |",
        "|---|---|",
        f"| Input (uncached) | {r.input_tokens:,} |",
        f"| Cache creation | {r.cache_creation_tokens:,} |",
        f"| Cache read | {r.cache_read_tokens:,} |",
        f"| Output | {r.output_tokens:,} |",
        f"| Total input-equivalent | {r.total_input_equivalent_tokens:,} |",
        "",
        f"Cache hit rate: {r.cache_hit_rate * 100:.1f}% of calls served cached tokens.",
        "",
        "## Cost distribution",
        "",
        "| Measure | USD |",
        "|---|---|",
        f"| Total | {r.total_cost_usd:.4f} |",
        f"| Mean per signal | {r.avg_cost_per_signal:.6f} |",
        f"| Median per signal | {r.median_cost_per_signal:.6f} |",
        f"| Minimum per signal | {r.min_cost_per_signal:.6f} |",
        f"| Maximum per signal | {r.max_cost_per_signal:.6f} |",
        f"| Mean per API call | {r.avg_cost_per_call:.6f} |",
        "",
    ]

    if r.by_agent:
        lines += [
            "## By agent",
            "",
            "| Agent | Model | Calls | Cost USD | Share | Input | Cache write | "
            "Cache read | Output | Cache read ratio |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in r.by_agent:
            lines.append(
                f"| {entry.agent} | {entry.model or '-'} | {entry.calls} | "
                f"{entry.cost_usd:.4f} | {entry.cost_share_pct:.1f}% | "
                f"{entry.input_tokens:,} | {entry.cache_creation_tokens:,} | "
                f"{entry.cache_read_tokens:,} | {entry.output_tokens:,} | "
                f"{entry.cache_read_ratio * 100:.1f}% |"
            )
        lines.append("")

    if r.cost_by_model:
        lines += ["## By model", "", "| Model | Cost USD | Rates (in/out per MTok) |", "|---|---|---|"]
        for model, cost in sorted(r.cost_by_model.items(), key=lambda kv: kv[1], reverse=True):
            try:
                pricing = get_pricing(model)
                rates = f"${pricing.input_per_mtok}/${pricing.output_per_mtok}"
            except Exception:
                rates = "unknown"
            lines.append(f"| {model} | {cost:.4f} | {rates} |")
        lines.append("")

    lines += [
        "## Projections (computed, not measured)",
        "",
        "| Scale | Projected USD |",
        "|---|---|",
        f"| 1,000 signals | {r.projected_1k_usd:.2f} |",
        f"| 5,000 signals | {r.projected_5k_usd:.2f} |",
        f"| 10,000 signals | {r.projected_10k_usd:.2f} |",
    ]
    if r.projected_full_oos_usd is not None:
        lines.append(
            f"| Full out-of-sample run ({r.full_oos_signal_count} signals) | "
            f"{r.projected_full_oos_usd:.2f} |"
        )
    else:
        lines.append(
            "| Full out-of-sample run | not available: the out-of-sample signal count "
            "is unknown until the candle dataset exists |"
        )
    lines += ["", f"Basis: {r.projection_basis}", ""]

    impact = r.profit_impact
    lines += ["## Cost against profit", ""]
    if not impact.available:
        lines += [f"Not available. {impact.unavailable_reason}", ""]
    else:
        lines += [
            "| Measure | Value |",
            "|---|---|",
            f"| Approved trades executed | {impact.approved_trades} |",
            f"| Cost per approved trade | "
            f"{'n/a' if impact.cost_per_approved_trade is None else f'${impact.cost_per_approved_trade:.4f}'} |",
            f"| Gross profit | {impact.gross_profit:,.2f} |",
            f"| Net profit (after commission) | {impact.net_profit:,.2f} |",
            f"| AI cost as % of gross profit | "
            f"{'n/a' if impact.ai_cost_pct_of_gross_profit is None else f'{impact.ai_cost_pct_of_gross_profit:.4f}%'} |",
            f"| AI cost as % of net profit | "
            f"{'n/a' if impact.ai_cost_pct_of_net_profit is None else f'{impact.ai_cost_pct_of_net_profit:.4f}%'} |",
            f"| Net profit after AI cost | {impact.net_profit_after_ai_cost:,.2f} |",
            "",
        ]
        if impact.note:
            lines += [f"> {impact.note}", ""]

    if r.decision_counts:
        lines += ["## Decisions", "", "| Action | Count |", "|---|---|"]
        for action, count in sorted(r.decision_counts.items()):
            lines.append(f"| {action} | {count} |")
        lines.append("")

    if r.limitations:
        lines += [
            "## Limitations carried by this run",
            "",
        ]
        for limitation in r.limitations:
            lines.append(f"- **{limitation['severity']}** {limitation['title']}")
        lines += ["", "See the methodology document for the full statements.", ""]

    return "\n".join(lines)


def write_reports(report: CostReport, directory: Path, stem: str = "cost_report") -> dict:
    """Write both forms side by side and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = report.save_json(directory / f"{stem}.json")
    md_path = directory / f"{stem}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
