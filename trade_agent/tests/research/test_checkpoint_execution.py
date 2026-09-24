from __future__ import annotations

import pytest

from app.services.risk_service import AccountState, RiskService
from research.ai.checkpoint import CheckpointStore
from research.ai.client import MockAgentClient, default_mock_verdicts
from research.ai.cost import CallRecord, TokenUsage
from research.ai.decision import SignalDecision
from research.ai.gate import DeterministicGate, replay_risk_settings
from research.ai.pit_store import NewsPitStore, SentimentPitStore
from research.ai.runner import AIBacktestRunner
from research.ai.schemas import FinalAction
from research.ai.settings import AISettings
from research.backtest.engine import BacktestEngine
from research.config import BacktestConfig
from research.experiment import ExperimentConfig, assert_manifests_match, ManifestMismatch
from research.manifest import DatasetVersion
from tests.conftest import make_settings
from tests.research.test_executor import rising_bars, signal

"""Resume must survive the execution join.

The requirement the checkpoint exists for: restarting a run must not re-issue
paid requests for signals already decided, AND the resumed decisions must
still produce the same trades when they reach the executor. A checkpoint that
restores decisions but yields a different equity curve is worse than none.
"""


def build_runner(bars, tmp_path, settings_overrides=None, budget=10.0):
    config = BacktestConfig()
    experiment = ExperimentConfig()
    ai_settings = AISettings(
        _env_file=None, ai_cost_limit_usd=budget, **(settings_overrides or {})
    )
    risk = RiskService(replay_risk_settings(make_settings()))
    engine = BacktestEngine(config)
    checkpoint = CheckpointStore(tmp_path / "checkpoint.sqlite")
    client = MockAgentClient(verdicts=default_mock_verdicts())
    runner = AIBacktestRunner(
        config=config,
        ai_settings=ai_settings,
        client=client,
        gate=DeterministicGate(config, risk, engine),
        risk_service=risk,
        engine=engine,
        checkpoint=checkpoint,
        news_store=NewsPitStore(None),
        sentiment_store=SentimentPitStore(None),
        policy_thresholds=experiment.policy_thresholds(),
    )
    return runner, checkpoint, client, experiment


@pytest.mark.asyncio
async def test_resume_skips_paid_calls_and_reproduces_the_same_trades(tmp_path):
    bars = rising_bars()
    signals = [signal(bars, i, f"s{i}") for i in (100, 160, 220)]
    account = AccountState(balance=100_000.0, market_open=True)

    runner, checkpoint, client, experiment = build_runner(bars, tmp_path)
    first = await runner.run(signals, bars, account)
    calls_first_run = len(client.calls)
    assert calls_first_run > 0

    executor = experiment.executor(make_settings())
    result_first = executor.run_with_decisions(bars, signals, first.decisions)

    # Restart: a new runner over the SAME checkpoint file.
    runner2, _, client2, _ = build_runner(bars, tmp_path)
    second = await runner2.run(signals, bars, account)

    assert second.resumed_count == len(signals)
    assert client2.calls == []  # nothing was paid for twice
    assert [d.signal_id for d in second.decisions] == [
        d.signal_id for d in first.decisions
    ]
    assert [d.action for d in second.decisions] == [d.action for d in first.decisions]

    # ...and the resumed decisions produce an identical run.
    executor2 = experiment.executor(make_settings())
    result_second = executor2.run_with_decisions(bars, signals, second.decisions)

    assert [t.signal_id for t in result_second.trades] == [
        t.signal_id for t in result_first.trades
    ]
    assert result_second.final_balance == pytest.approx(result_first.final_balance)
    assert len(result_second.skipped) == len(result_first.skipped)


@pytest.mark.asyncio
async def test_resumed_decisions_keep_their_chain_and_cost(tmp_path):
    bars = rising_bars()
    signals = [signal(bars, 100, "s1")]
    runner, checkpoint, _, experiment = build_runner(bars, tmp_path)
    first = await runner.run(signals, bars, AccountState(balance=100_000.0, market_open=True))
    original = first.decisions[0]

    runner2, _, _, _ = build_runner(bars, tmp_path)
    resumed = (
        await runner2.run(signals, bars, AccountState(balance=100_000.0, market_open=True))
    ).decisions[0]

    assert resumed.cost_usd == pytest.approx(original.cost_usd)
    assert resumed.deciding_rule == original.deciding_rule
    assert resumed.agents_called == original.agents_called
    assert resumed.chain()["final"] == original.chain()["final"]

    # The executor stamps the restored chain onto the trade.
    executor = experiment.executor(make_settings())
    result = executor.run_with_decisions(bars, signals, [resumed])
    if result.trades:
        assert result.trades[0].agent_chain["final"] == original.chain()["final"]
        assert result.trades[0].ai_cost_usd == pytest.approx(original.cost_usd)


@pytest.mark.asyncio
async def test_prior_spend_is_recovered_so_a_restart_cannot_double_the_budget(tmp_path):
    bars = rising_bars()
    signals = [signal(bars, 100, "s1")]
    runner, checkpoint, _, _ = build_runner(bars, tmp_path)
    await runner.run(signals, bars, AccountState(balance=100_000.0, market_open=True))
    spent = checkpoint.total_spend()
    assert spent > 0

    runner2, _, _, _ = build_runner(bars, tmp_path)
    assert runner2.ledger.spent_usd == pytest.approx(spent)


def test_a_failed_signal_is_retried_rather_than_treated_as_done(tmp_path):
    """A transient API error must not permanently exclude a signal."""
    from research.ai.checkpoint import SignalProgress

    store = CheckpointStore(tmp_path / "checkpoint.sqlite")
    store.save_progress(SignalProgress(signal_id="ok", status="done"))
    store.save_progress(SignalProgress(signal_id="skipped", status="skipped_deterministic"))
    store.save_progress(SignalProgress(signal_id="broken", status="failed"))

    completed = store.completed_signal_ids()
    assert completed == {"ok", "skipped"}
    assert "broken" not in completed


def test_call_log_survives_restart_for_the_cost_report(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.sqlite")
    store.log_call(
        CallRecord(
            signal_id="s1",
            agent="final",
            model="claude-sonnet-5",
            usage=TokenUsage(
                input_tokens=100, cache_creation_tokens=2000,
                cache_read_tokens=0, output_tokens=80,
            ),
            cost_usd=0.005,
            latency_seconds=1.2,
            prompt_version="abc",
        )
    )
    reopened = CheckpointStore(tmp_path / "checkpoint.sqlite")
    records = reopened.all_calls()

    assert len(records) == 1
    assert records[0].usage.cache_creation_tokens == 2000
    assert records[0].cost_usd == pytest.approx(0.005)
    # Enough to rebuild the cost report after a crash.
    from research.report.cost_report import build_cost_report

    report = build_cost_report(
        run_id="r", records=records, total_signals=1, processed_signals=1,
        skipped_deterministic=0, models_used={"final": "claude-sonnet-5"},
    )
    assert report.total_cost_usd == pytest.approx(0.005)


# --- the shared-dataset pin -------------------------------------------------
def test_arms_may_use_different_datasets_but_not_different_bars():
    config = ExperimentConfig()
    candles = DatasetVersion(
        name="candles", source="XAUUSD_M15.parquet", available=True,
        row_count=9000, content_hash="c" * 64,
    )
    news = DatasetVersion(name="news", source="wire", available=True, content_hash="n" * 64)

    baseline = config.manifest("a", "baseline", datasets=[candles])
    ai = config.manifest("b", "ai", datasets=[candles, news])
    # Different input sets are fine.
    assert_manifests_match(baseline, ai)

    # Different BARS are not.
    other_candles = candles.model_copy(update={"content_hash": "d" * 64})
    with pytest.raises(ManifestMismatch, match="datasets.candles"):
        assert_manifests_match(
            baseline, config.manifest("b", "ai", datasets=[other_candles, news])
        )


def test_a_different_row_count_on_shared_candles_is_caught():
    config = ExperimentConfig()
    candles = DatasetVersion(
        name="candles", source="f.parquet", available=True, row_count=9000,
        content_hash="c" * 64,
    )
    trimmed = candles.model_copy(update={"row_count": 8000})
    with pytest.raises(ManifestMismatch, match="row_count"):
        assert_manifests_match(
            config.manifest("a", "baseline", datasets=[candles]),
            config.manifest("b", "ai", datasets=[trimmed]),
        )
