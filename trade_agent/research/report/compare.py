from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from research.backtest.engine import BacktestResult
from research.backtest.metrics import PerformanceMetrics, compute_metrics
from research.experiment import (
    ExperimentConfig,
    assert_manifests_match,
)
from research.manifest import RunManifest
from research.report.cost_report import CostReport
from research.report.counterfactual import CounterfactualAnalysis, analyze_counterfactuals
from research.report.limitations import Limitation, as_dicts, render_markdown as render_limits

"""Experiment A vs Experiment B.

Two rules this module enforces before it will report anything:

1. **The manifests must match** outside the AI-only fields. A comparison
   between differently configured runs is not a measurement of the AI layer,
   so `assert_manifests_match` runs first and raises rather than producing a
   number with a caveat.

2. **The signal universe must be stated.** If the AI arm only decided on a
   pilot subset, comparing its equity curve against a baseline that traded
   every signal compares two different experiments. The comparison therefore
   reports the shared signal set explicitly and, when asked, restricts both
   arms to it.
"""


class MetricDelta(BaseModel):
    metric: str
    baseline: float | None
    ai: float | None
    delta: float | None = None
    better: str | None = None  # "ai" | "baseline" | "tie" | None

    @classmethod
    def build(
        cls, metric: str, baseline: float | None, ai: float | None, higher_is_better: bool = True
    ) -> "MetricDelta":
        delta = None if baseline is None or ai is None else ai - baseline
        better: str | None = None
        if delta is not None:
            if abs(delta) < 1e-12:
                better = "tie"
            elif (delta > 0) == higher_is_better:
                better = "ai"
            else:
                better = "baseline"
        return cls(metric=metric, baseline=baseline, ai=ai, delta=delta, better=better)


class ComparisonReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    experiment_name: str
    period_start: datetime | None = None
    period_end: datetime | None = None

    shared_signal_count: int = 0
    baseline_only_signals: int = 0
    ai_only_signals: int = 0
    restricted_to_shared_set: bool = False

    baseline_metrics: PerformanceMetrics
    ai_metrics: PerformanceMetrics
    deltas: list[MetricDelta] = Field(default_factory=list)

    counterfactual: CounterfactualAnalysis | None = None
    cost: CostReport | None = None

    ai_cost_usd: float = 0.0
    ai_net_profit_after_cost: float | None = None
    baseline_net_profit: float | None = None
    ai_beats_baseline_after_cost: bool | None = None

    config_fingerprint_hash: str = ""
    manifests_verified: bool = False
    limitations: list[dict] = Field(default_factory=list)

    def save_json(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        return Path(path)


_COMPARED = (
    ("net_profit", True),
    ("return_pct", True),
    ("total_trades", True),
    ("win_rate", True),
    ("profit_factor", True),
    ("expectancy_r", True),
    ("average_realized_r", True),
    ("max_drawdown_pct", False),
    ("sharpe_ratio", True),
    ("sortino_ratio", True),
    ("max_consecutive_losses", False),
    ("average_trade", True),
    ("total_commission", False),
)


def shared_signal_ids(baseline: BacktestResult, ai: BacktestResult) -> set[str]:
    """Signals both arms actually considered.

    Union of traded and skipped on each side, intersected. A signal the AI arm
    never saw is not evidence about the AI arm.
    """
    def ids(result: BacktestResult) -> set[str]:
        return {t.signal_id for t in result.trades} | {
            s.signal_id for s in result.skipped if s.skipped_by != "no_decision"
        }

    return ids(baseline) & ids(ai)


def restrict(result: BacktestResult, keep: set[str]) -> BacktestResult:
    """A copy of a result containing only the named signals.

    The equity curve is rebuilt sequentially from the retained trades, so the
    restricted result's balances are internally consistent rather than being a
    filtered view of a curve that included other trades.
    """
    trades = [t for t in result.trades if t.signal_id in keep]
    balance = result.initial_balance
    equity: list[dict] = []
    rebuilt = []
    for trade in sorted(trades, key=lambda t: t.entry_time):
        copy = trade.model_copy(
            update={
                "balance_before": balance,
                "balance_after": balance + trade.profit,
            }
        )
        balance = copy.balance_after
        rebuilt.append(copy)
        equity.append(
            {"timestamp": copy.exit_time, "balance": balance, "trade_id": copy.trade_id}
        )

    if result.equity_curve:
        equity.insert(0, result.equity_curve[0])

    return result.model_copy(
        update={
            "trades": rebuilt,
            "skipped": [s for s in result.skipped if s.signal_id in keep],
            "equity_curve": equity,
            "final_balance": balance,
            "signals_generated": len(keep),
        }
    )


def compare(
    config: ExperimentConfig,
    baseline: BacktestResult,
    ai: BacktestResult,
    baseline_manifest: RunManifest | None = None,
    ai_manifest: RunManifest | None = None,
    cost: CostReport | None = None,
    limitations: list[Limitation] | None = None,
    restrict_to_shared: bool = True,
) -> ComparisonReport:
    """Compare the two arms, refusing to do so if they are not comparable."""
    manifests_verified = False
    if baseline_manifest is not None and ai_manifest is not None:
        # Raises on any difference outside the AI-only fields.
        assert_manifests_match(baseline_manifest, ai_manifest)
        manifests_verified = True

    shared = shared_signal_ids(baseline, ai)
    baseline_only = len(
        {t.signal_id for t in baseline.trades}
        | {s.signal_id for s in baseline.skipped}
    ) - len(shared)
    ai_only = len(
        {t.signal_id for t in ai.trades} | {s.signal_id for s in ai.skipped}
    ) - len(shared)

    baseline_used, ai_used = baseline, ai
    if restrict_to_shared:
        baseline_used = restrict(baseline, shared)
        ai_used = restrict(ai, shared)

    baseline_metrics = compute_metrics(baseline_used)
    ai_metrics = compute_metrics(ai_used)

    deltas = [
        MetricDelta.build(
            name,
            getattr(baseline_metrics, name, None),
            getattr(ai_metrics, name, None),
            higher_is_better,
        )
        for name, higher_is_better in _COMPARED
    ]

    ai_cost = cost.total_cost_usd if cost else sum(t.ai_cost_usd for t in ai_used.trades)
    ai_net_after_cost = ai_metrics.net_profit - ai_cost

    report = ComparisonReport(
        experiment_name=config.name,
        period_start=ai.period_start,
        period_end=ai.period_end,
        shared_signal_count=len(shared),
        baseline_only_signals=max(0, baseline_only),
        ai_only_signals=max(0, ai_only),
        restricted_to_shared_set=restrict_to_shared,
        baseline_metrics=baseline_metrics,
        ai_metrics=ai_metrics,
        deltas=deltas,
        counterfactual=analyze_counterfactuals(ai_used, baseline_used),
        cost=cost,
        ai_cost_usd=ai_cost,
        ai_net_profit_after_cost=ai_net_after_cost,
        baseline_net_profit=baseline_metrics.net_profit,
        ai_beats_baseline_after_cost=ai_net_after_cost > baseline_metrics.net_profit,
        config_fingerprint_hash=config.fingerprint_hash(),
        manifests_verified=manifests_verified,
        limitations=as_dicts(limitations or []),
    )
    return report


def render_markdown(report: ComparisonReport) -> str:
    from research.report.counterfactual import render_markdown as render_cf

    lines = [
        f"# Experiment A vs B - {report.experiment_name}",
        "",
        f"Generated {report.generated_at:%Y-%m-%d %H:%M} UTC. "
        f"Config fingerprint `{report.config_fingerprint_hash[:16]}`.",
        "",
        "- **A (baseline):** the frozen strategy, through the deterministic gate and "
        "execution guard, with no AI layer.",
        "- **B (AI):** the same signals, same gate, same guard, same fill model, with the "
        "four-agent decision layer inserted.",
        "",
    ]

    lines.append(
        "Manifest equality verified: "
        + ("yes" if report.manifests_verified else "NOT verified (no manifests supplied)")
    )
    lines.append("")
    lines += [
        "## Signal universe",
        "",
        f"- Signals decided by both arms: **{report.shared_signal_count}**",
        f"- Present only in the baseline arm: {report.baseline_only_signals}",
        f"- Present only in the AI arm: {report.ai_only_signals}",
        "",
    ]
    if report.restricted_to_shared_set:
        lines += [
            "Both arms are restricted to the shared signal set for this comparison, so a "
            "pilot that decided on a subset is not compared against a baseline that "
            "traded everything.",
            "",
        ]

    lines += [
        "## Performance",
        "",
        "| Metric | A (baseline) | B (AI) | Delta | Better |",
        "|---|---|---|---|---|",
    ]
    for delta in report.deltas:
        def fmt(value):
            return "n/a" if value is None else f"{value:,.4f}".rstrip("0").rstrip(".")

        lines.append(
            f"| {delta.metric} | {fmt(delta.baseline)} | {fmt(delta.ai)} | "
            f"{fmt(delta.delta)} | {delta.better or '-'} |"
        )

    lines += [
        "",
        "## Net of AI cost",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| A net profit | {report.baseline_net_profit:,.2f} |",
        f"| B net profit (before AI cost) | {report.ai_metrics.net_profit:,.2f} |",
        f"| AI cost | {report.ai_cost_usd:.4f} |",
        f"| B net profit after AI cost | {report.ai_net_profit_after_cost:,.2f} |",
        f"| B beats A after cost | "
        f"{'yes' if report.ai_beats_baseline_after_cost else 'no'} |",
        "",
    ]

    if report.counterfactual:
        lines.append(render_cf(report.counterfactual))

    if report.limitations:
        limitations = [Limitation.model_validate(entry) for entry in report.limitations]
        lines.append(render_limits(limitations))

    return "\n".join(lines)


def write_reports(
    report: ComparisonReport, directory: Path, stem: str = "comparison"
) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = report.save_json(directory / f"{stem}.json")
    md_path = directory / f"{stem}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
