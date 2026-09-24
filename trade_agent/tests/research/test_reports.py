from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.services.risk_service import RiskService
from research.ai.cost import CallRecord, TokenUsage
from research.ai.decision import SignalDecision
from research.ai.gate import replay_risk_settings
from research.ai.schemas import FinalAction
from research.backtest.engine import BacktestEngine
from research.backtest.executor import DecisionExecutor
from research.config import BacktestConfig
from research.experiment import ExperimentConfig
from research.report.compare import compare, restrict, shared_signal_ids
from research.report.cost_report import build_cost_report, render_markdown, write_reports
from research.report.counterfactual import analyze_counterfactuals
from research.report.counterfactual import render_markdown as render_counterfactual
from research.report.limitations import ALL_LIMITATIONS, limitations_for
from tests.conftest import make_settings
from tests.research.test_executor import decision, rising_bars, signal

UTC = timezone.utc


# --- counterfactual analysis ------------------------------------------------
def _ai_run(bars, spec: list[tuple[int, str, FinalAction, str | None]]):
    """Build an AI run from (bar_index, signal_id, action, blocking_agent)."""
    config = BacktestConfig()
    executor = DecisionExecutor(
        config=config,
        engine=BacktestEngine(config),
        risk_service=RiskService(replay_risk_settings(make_settings())),
        counterfactual_balance=config.risk.initial_balance,
    )
    signals = [signal(bars, index, sid) for index, sid, _, _ in spec]
    decisions = [
        decision(sid, action, blocking_agent=agent, reason_codes=["TECHNICAL_VETO"])
        for _, sid, action, agent in spec
    ]
    return executor.run_with_decisions(bars, signals, decisions), executor


def test_quadrant_counts_approvals_and_rejections():
    """WAIT (s3) now executes like an implicit APPROVE (only REJECT actually
    blocks a trade -- see executor.DecisionExecutor), so it counts toward
    approved_total, not rejected_by_ai. s2 (REJECT) is the only one the AI
    layer itself skips; s4 separately gets skipped by an unrelated engine
    constraint (a position from s1 is still open), not by the AI decision."""
    bars = rising_bars()
    result, _ = _ai_run(
        bars,
        [
            (100, "s1", FinalAction.APPROVE, None),
            (160, "s2", FinalAction.REJECT, "technical"),
            (220, "s3", FinalAction.WAIT, "news"),
            (260, "s4", FinalAction.APPROVE, None),
        ],
    )
    analysis = analyze_counterfactuals(result)

    assert [t.signal_id for t in result.trades] == ["s1", "s3"]
    assert analysis.quadrant.approved_total == len(result.trades) == 2
    assert analysis.rejected_by_ai == 1
    assert analysis.quadrant.rejected_scored == 1
    # The fixture rises steadily, so every BUY would have won.
    assert analysis.quadrant.rejected_would_have_won == 1
    assert analysis.pct_rejected_would_have_been_profitable == 100.0


def test_analysis_attributes_rejections_to_the_blocking_agent():
    bars = rising_bars()
    result, _ = _ai_run(
        bars,
        [
            (100, "s1", FinalAction.REJECT, "technical"),
            (160, "s2", FinalAction.REJECT, "technical"),
            (220, "s3", FinalAction.REJECT, "news"),
        ],
    )
    analysis = analyze_counterfactuals(result)
    by_agent = {record.agent: record for record in analysis.by_agent}

    assert by_agent["technical"].rejections == 2
    assert by_agent["news"].rejections == 1
    assert by_agent["technical"].false_rejection_rate == 1.0  # all would have won
    assert by_agent["technical"].profit_forgone > 0
    assert analysis.by_agent[0].agent == "technical"  # ranked by false rejections


def test_analysis_only_counts_ai_decisions_not_risk_engine_blocks():
    """A signal the risk engine refused is not the AI layer's rejection."""
    bars = rising_bars()
    config = BacktestConfig()
    executor = DecisionExecutor(
        config=config,
        engine=BacktestEngine(config),
        risk_service=RiskService(replay_risk_settings(make_settings())),
    )
    good = signal(bars, 100, "s1")
    bad = signal(bars, 200, "s2")
    bad = bad.model_copy(update={"take_profit": bad.entry + 2.0})  # RR below floor
    result = executor.run_with_decisions(
        bars,
        [good, bad],
        [decision("s1", FinalAction.REJECT, blocking_agent="news"),
         decision("s2", FinalAction.APPROVE)],
    )
    analysis = analyze_counterfactuals(result)
    assert analysis.rejected_by_ai == 1
    assert "risk_engine" not in {record.agent for record in analysis.by_agent}


def test_counterfactuals_do_not_affect_reported_profit():
    bars = rising_bars()
    result, _ = _ai_run(bars, [(100, "s1", FinalAction.REJECT, "news")])
    analysis = analyze_counterfactuals(result)

    assert result.final_balance == result.initial_balance
    assert analysis.profit_forgone > 0  # a real forgone profit...
    # ...that appears nowhere in the run's own P&L.
    assert sum(t.profit for t in result.trades) == 0


def test_unscoreable_counterfactual_is_reported_with_its_reason():
    bars = rising_bars(n=120)
    # A signal on the very last bar has no next bar to fill at.
    result, _ = _ai_run(bars, [(119, "s1", FinalAction.REJECT, "news")])
    analysis = analyze_counterfactuals(result)

    assert analysis.unscoreable == 1
    assert analysis.quadrant.rejected_unscoreable == 1
    assert any("final bar" in reason for reason in analysis.unscoreable_reasons)
    # Excluded from the percentage rather than counted as a loss.
    assert analysis.pct_rejected_would_have_been_profitable is None


def test_counterfactual_records_excursions_and_levels():
    bars = rising_bars()
    result, _ = _ai_run(bars, [(100, "s1", FinalAction.REJECT, "technical")])
    skipped = result.skipped[0]

    assert skipped.counterfactual_entry_price is not None
    assert skipped.counterfactual_exit_price is not None
    assert skipped.counterfactual_entry_time < skipped.counterfactual_exit_time
    assert skipped.counterfactual_exit_reason in {"STOP_LOSS", "TAKE_PROFIT", "END_OF_DATA"}
    assert skipped.counterfactual_mfe_r >= 0
    assert skipped.counterfactual_mae_r >= 0
    assert skipped.counterfactual_volume > 0
    assert skipped.counterfactual_is_win is True
    assert skipped.direction.value == "BUY"


def test_counterfactual_markdown_answers_the_six_questions():
    bars = rising_bars()
    result, _ = _ai_run(
        bars,
        [
            (100, "s1", FinalAction.APPROVE, None),
            (160, "s2", FinalAction.REJECT, "technical"),
        ],
    )
    text = render_counterfactual(analyze_counterfactuals(result))
    for question in (
        "How many profitable trades did the AI reject?",
        "How many losing trades did it correctly reject?",
        "How many winners did it approve?",
        "How many losers did it approve?",
        "% of rejected signals that would have been profitable",
    ):
        assert question in text
    assert "Attribution by blocking agent" in text


# --- cost report ------------------------------------------------------------
def records(n: int = 4, cost: float = 0.001) -> list[CallRecord]:
    agents = ["technical", "news", "sentiment", "final"]
    out = []
    for i in range(n):
        agent = agents[i % 4]
        out.append(
            CallRecord(
                signal_id=f"s{i // 4}",
                agent=agent,
                model="claude-haiku-4-5" if agent != "final" else "claude-sonnet-5",
                usage=TokenUsage(
                    input_tokens=200,
                    cache_creation_tokens=2000 if i < 4 else 0,
                    cache_read_tokens=0 if i < 4 else 2000,
                    output_tokens=120,
                ),
                cost_usd=cost,
                latency_seconds=1.5,
            )
        )
    return out


def test_cost_report_contains_every_required_field():
    report = build_cost_report(
        run_id="r1",
        records=records(8),
        total_signals=10,
        processed_signals=2,
        skipped_deterministic=8,
        models_used={"technical": "claude-haiku-4-5", "final": "claude-sonnet-5"},
        full_oos_signal_count=4000,
    )
    for field in (
        "total_signals",
        "processed_signals",
        "skipped_deterministic_signals",
        "input_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
        "output_tokens",
        "total_cost_usd",
        "avg_cost_per_signal",
        "median_cost_per_signal",
        "min_cost_per_signal",
        "max_cost_per_signal",
        "cost_by_agent",
        "cost_by_model",
        "cache_read_ratio",
        "projected_1k_usd",
        "projected_5k_usd",
        "projected_10k_usd",
        "projected_full_oos_usd",
    ):
        assert hasattr(report, field), field

    assert report.total_signals == 10
    assert report.skipped_deterministic_signals == 8
    assert report.input_tokens == 8 * 200
    assert report.cache_read_tokens == 4 * 2000
    assert report.total_cost_usd == pytest.approx(0.008)
    assert set(report.cost_by_agent) == {"technical", "news", "sentiment", "final"}
    assert set(report.cost_by_model) == {"claude-haiku-4-5", "claude-sonnet-5"}
    assert 0 < report.cache_read_ratio < 1
    assert report.projected_full_oos_usd == pytest.approx(
        report.avg_cost_per_signal * 4000
    )


def test_projections_are_labelled_as_computed():
    report = build_cost_report(
        run_id="r1",
        records=records(4),
        total_signals=1,
        processed_signals=1,
        skipped_deterministic=0,
        models_used={},
    )
    assert "measured average" in report.projection_basis
    assert "upper estimates" in report.projection_basis
    assert "Projections (computed, not measured)" in render_markdown(report)


def test_post_join_figures_require_the_execution_bridge():
    """Without a result they are reported unavailable, never estimated."""
    without = build_cost_report(
        run_id="r1",
        records=records(4),
        total_signals=1,
        processed_signals=1,
        skipped_deterministic=0,
        models_used={},
    )
    assert without.profit_impact.available is False
    assert "decision-to-trade join" in without.profit_impact.unavailable_reason
    assert without.profit_impact.cost_per_approved_trade is None


def test_post_join_figures_are_measured_when_the_result_exists():
    bars = rising_bars()
    result, _ = _ai_run(
        bars,
        [(100, "s1", FinalAction.APPROVE, None), (200, "s2", FinalAction.APPROVE, None)],
    )
    report = build_cost_report(
        run_id="r1",
        records=records(8, cost=0.01),
        total_signals=2,
        processed_signals=2,
        skipped_deterministic=0,
        models_used={},
        ai_result=result,
    )
    impact = report.profit_impact
    assert impact.available
    assert impact.approved_trades == len(result.trades)
    assert impact.cost_per_approved_trade == pytest.approx(
        report.total_cost_usd / len(result.trades)
    )
    assert impact.gross_profit == pytest.approx(sum(t.gross_profit for t in result.trades))
    assert impact.net_profit == pytest.approx(sum(t.profit for t in result.trades))
    assert impact.ai_cost_pct_of_gross_profit > 0
    assert impact.net_profit_after_ai_cost == pytest.approx(
        impact.net_profit - report.total_cost_usd
    )


def test_unprofitable_ai_arm_reports_absolute_cost_not_a_ratio():
    """Cost as a percentage of profit is undefined when there is no profit."""
    from research.backtest.engine import BacktestResult

    empty = BacktestResult(run_id="r", run_kind="ai", initial_balance=100_000.0,
                           final_balance=100_000.0)
    report = build_cost_report(
        run_id="r1", records=records(4), total_signals=1, processed_signals=1,
        skipped_deterministic=0, models_used={}, ai_result=empty,
    )
    assert report.profit_impact.available
    assert report.profit_impact.ai_cost_pct_of_net_profit is None
    assert "not profitable" in report.profit_impact.note


def test_budget_stop_is_surfaced_in_both_forms():
    report = build_cost_report(
        run_id="r1", records=records(4), total_signals=100, processed_signals=1,
        skipped_deterministic=0, models_used={}, budget_usd=10.0,
        budget_stopped=True, stopped_reason="cost limit reached",
    )
    assert report.budget_stopped
    assert "stopped on the budget limit" in render_markdown(report)


def test_cost_report_writes_json_and_markdown(tmp_path):
    report = build_cost_report(
        run_id="r1", records=records(8), total_signals=4, processed_signals=2,
        skipped_deterministic=2, models_used={"final": "claude-sonnet-5"},
        limitations=limitations_for("ai"),
    )
    written = write_reports(report, tmp_path)

    assert written["json"].exists() and written["markdown"].exists()
    payload = json.loads(written["json"].read_text())
    assert payload["run_id"] == "r1"
    assert payload["total_cost_usd"] == pytest.approx(report.total_cost_usd)
    text = written["markdown"].read_text()
    assert "# AI cost report" in text
    assert "MEASURED" in text
    assert "By agent" in text


def test_per_agent_breakdown_shares_sum_to_one_hundred():
    report = build_cost_report(
        run_id="r1", records=records(8), total_signals=2, processed_signals=2,
        skipped_deterministic=0, models_used={},
    )
    assert sum(entry.cost_share_pct for entry in report.by_agent) == pytest.approx(
        100.0, abs=0.2
    )


# --- limitations ------------------------------------------------------------
def test_knowledge_leakage_limitation_is_carried_on_every_ai_run():
    limitations = limitations_for("ai")
    codes = [limitation.code for limitation in limitations]
    assert "MODEL_KNOWLEDGE_LEAKAGE" in codes

    leakage = next(l for l in limitations if l.code == "MODEL_KNOWLEDGE_LEAKAGE")
    assert leakage.severity.value == "CRITICAL"
    # It must NOT claim prompt instructions solve it.
    assert "cannot remove information from a model's weights" in leakage.why_not_mitigated
    assert "placebo" in leakage.what_would_resolve_it.lower()
    assert "forward paper trading" in leakage.what_would_resolve_it.lower()


def test_every_limitation_states_a_remedy():
    for limitation in ALL_LIMITATIONS:
        assert limitation.description
        assert limitation.what_would_resolve_it, limitation.code
        assert limitation.affects, limitation.code


def test_baseline_run_does_not_claim_llm_limitations():
    codes = [limitation.code for limitation in limitations_for("baseline")]
    assert "MODEL_KNOWLEDGE_LEAKAGE" not in codes
    assert "NO_INTRABAR_REPLAY" in codes  # mechanical ones still apply


# --- comparison -------------------------------------------------------------
def test_comparison_restricts_both_arms_to_the_shared_signal_set():
    bars = rising_bars()
    config = ExperimentConfig()
    executor = config.executor(make_settings())

    all_signals = [signal(bars, i, f"s{i}") for i in (100, 160, 220, 260)]
    baseline = executor.run_baseline(bars, all_signals)

    # The AI arm only decided on the first two -- a pilot subset.
    pilot = all_signals[:2]
    ai = executor.run_with_decisions(
        bars, pilot, [decision("s100", FinalAction.APPROVE), decision("s160", FinalAction.REJECT)]
    )

    shared = shared_signal_ids(baseline, ai)
    assert shared == {"s100", "s160"}

    report = compare(config, baseline, ai, restrict_to_shared=True)
    assert report.restricted_to_shared_set
    assert report.shared_signal_count == 2
    assert report.baseline_only_signals == 2
    # The baseline arm is cut down to the pilot's signals, so it cannot be
    # credited with trades the AI arm never had the chance to decline.
    assert report.baseline_metrics.total_trades <= 2
    unrestricted = compare(config, baseline, ai, restrict_to_shared=False)
    assert unrestricted.baseline_metrics.total_trades >= (
        report.baseline_metrics.total_trades
    )
    assert not unrestricted.restricted_to_shared_set


def test_restrict_rebuilds_a_consistent_equity_curve():
    bars = rising_bars()
    config = ExperimentConfig()
    executor = config.executor(make_settings())
    signals = [signal(bars, i, f"s{i}") for i in (100, 160, 220)]
    result = executor.run_baseline(bars, signals)
    if len(result.trades) < 2:
        pytest.skip("fixture produced too few trades to restrict")

    keep = {result.trades[0].signal_id}
    restricted = restrict(result, keep)

    assert [t.signal_id for t in restricted.trades] == list(keep)
    assert restricted.trades[0].balance_before == restricted.initial_balance
    assert restricted.final_balance == pytest.approx(restricted.trades[-1].balance_after)


def test_comparison_refuses_mismatched_manifests():
    from research.experiment import ManifestMismatch

    bars = rising_bars()
    config = ExperimentConfig()
    executor = config.executor(make_settings())
    signals = [signal(bars, 100, "s1")]
    baseline = executor.run_baseline(bars, signals)
    ai = executor.run_with_decisions(bars, signals, [decision("s1", FinalAction.APPROVE)])

    drifted = config.model_copy(deep=True)
    drifted.backtest.risk.initial_balance = 1_000.0

    with pytest.raises(ManifestMismatch):
        compare(
            config,
            baseline,
            ai,
            baseline_manifest=config.manifest("a", "baseline"),
            ai_manifest=drifted.manifest("b", "ai"),
        )


def test_comparison_reports_ai_cost_against_the_baseline():
    bars = rising_bars()
    config = ExperimentConfig()
    executor = config.executor(make_settings())
    signals = [signal(bars, i, f"s{i}") for i in (100, 200)]
    baseline = executor.run_baseline(bars, signals)
    ai = executor.run_with_decisions(
        bars,
        signals,
        [
            decision("s100", FinalAction.APPROVE, cost_usd=0.02),
            decision("s200", FinalAction.REJECT, cost_usd=0.02, blocking_agent="news"),
        ],
    )

    report = compare(
        config,
        baseline,
        ai,
        baseline_manifest=config.manifest("a", "baseline"),
        ai_manifest=config.manifest("b", "ai"),
        limitations=limitations_for("ai"),
    )
    assert report.manifests_verified
    assert report.ai_cost_usd > 0
    assert report.ai_net_profit_after_cost == pytest.approx(
        report.ai_metrics.net_profit - report.ai_cost_usd
    )
    assert report.ai_beats_baseline_after_cost is not None
    assert report.counterfactual is not None
    assert report.limitations


def test_comparison_markdown_names_the_two_arms_and_the_verification(tmp_path):
    from research.report.compare import write_reports as write_comparison

    bars = rising_bars()
    config = ExperimentConfig()
    executor = config.executor(make_settings())
    signals = [signal(bars, 100, "s1")]
    baseline = executor.run_baseline(bars, signals)
    ai = executor.run_with_decisions(bars, signals, [decision("s1", FinalAction.APPROVE)])
    report = compare(
        config, baseline, ai,
        baseline_manifest=config.manifest("a", "baseline"),
        ai_manifest=config.manifest("b", "ai"),
        limitations=limitations_for("ai"),
    )
    written = write_comparison(report, tmp_path)

    text = written["markdown"].read_text()
    assert "Experiment A vs B" in text
    assert "Manifest equality verified: yes" in text
    assert "Net of AI cost" in text
    assert "MODEL_KNOWLEDGE_LEAKAGE" in json.dumps(
        json.loads(written["json"].read_text())["limitations"]
    )
