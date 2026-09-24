from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.cli import (
    EXIT_LEAKAGE_REFUSED,
    EXIT_MISSING_INPUT,
    EXIT_OK,
    main,
)
from research.experiment import ExperimentConfig, save_experiment

UTC = timezone.utc
BASE = datetime(2021, 1, 4, tzinfo=UTC)

"""End-to-end CLI tests.

These build a seeded random-walk CANDLE FIXTURE so the command sequence can be
exercised without market data. Nothing here is a result: the fixture is not
XAUUSD and no performance claim is derived from it. What is being tested is
that the commands run in order, that each refuses to run before its inputs
exist, and that the seal genuinely gates out-of-sample access.

No command in these tests makes a network call or a paid API call: the pilot
runs with `--mock`.
"""


def fixture_candles(n: int = 9000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 1800 + np.cumsum(rng.normal(0.03, 1.4, n))
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [BASE + timedelta(minutes=15 * i) for i in range(n)], utc=True
            ),
            "open": closes,
            "high": closes + rng.uniform(0.5, 3.0, n),
            "low": closes - rng.uniform(0.5, 3.0, n),
            "close": closes,
            "volume": rng.uniform(50, 200, n),
            "tick_count": rng.integers(40, 200, n),
            "bid_close": closes - 0.15,
            "ask_close": closes + 0.15,
            "spread_mean": np.full(n, 0.3),
            "spread_max": np.full(n, 0.5),
            "is_partial": [False] * n,
        }
    )


@pytest.fixture
def workspace(tmp_path) -> dict:
    """A config whose data and results directories are inside tmp_path."""
    config = ExperimentConfig()
    config.backtest.data_dir = tmp_path / "data"
    config.backtest.results_dir = tmp_path / "results"
    # Eligibility is deliberately loosened so the fixture yields a selectable
    # candidate and the command SEQUENCE can be exercised. A random-walk
    # fixture should not produce robust parameters, and with the real filters
    # it does not -- that the strict filters reject it is asserted in
    # tests/research/test_optimizer.py, not worked around here.
    config.optimizer.folds = 3
    config.optimizer.purge_bars = 30
    config.optimizer.min_trades_per_fold = 2
    config.optimizer.min_trades_total = 8
    config.optimizer.min_profitable_fold_fraction = 0.3
    config.optimizer.max_fold_drawdown_pct = 60.0
    path = save_experiment(config, tmp_path / "experiment.json")
    return {"config": config, "config_path": path, "tmp": tmp_path}


def place_candles(workspace) -> Path:
    target = workspace["config"].backtest.candle_path
    target.parent.mkdir(parents=True, exist_ok=True)
    fixture_candles().to_parquet(target, index=False)
    return target


def run(workspace, *args) -> int:
    return main([*args, "--config", str(workspace["config_path"])])


def develop(workspace, limit: int = 6) -> int:
    """Run `develop` over a slice of the grid.

    The full 168-candidate search is right for a real run and far too slow for
    a test. What these tests check is the command sequence and the seal
    gating, and neither depends on how wide the grid is.
    """
    return run(workspace, "develop", "--limit-candidates", str(limit))


# --- ordering is enforced by artifacts, not documentation -------------------
def test_status_lists_what_is_missing(workspace, capsys):
    assert run(workspace, "status") == EXIT_OK
    output = capsys.readouterr().out
    assert "tick data" in output
    assert "M15 candles" in output
    assert "fetch" in output  # the next step


def test_commands_refuse_to_run_before_their_inputs_exist(workspace):
    assert run(workspace, "split") == EXIT_MISSING_INPUT
    assert run(workspace, "develop") == EXIT_MISSING_INPUT
    assert run(workspace, "baseline") == EXIT_MISSING_INPUT
    assert run(workspace, "freeze") == EXIT_MISSING_INPUT
    assert run(workspace, "compare") == EXIT_MISSING_INPUT


def test_candles_refuses_without_tick_data(workspace, capsys):
    assert run(workspace, "candles") == EXIT_MISSING_INPUT
    assert "Nothing is generated" in capsys.readouterr().err


def test_baseline_is_locked_until_the_strategy_is_frozen(workspace, capsys):
    place_candles(workspace)
    assert run(workspace, "baseline") == EXIT_MISSING_INPUT
    assert "out-of-sample data stays locked" in capsys.readouterr().err


# --- the happy path ---------------------------------------------------------
def test_split_reports_the_boundary_without_unlocking_it(workspace, capsys):
    place_candles(workspace)
    assert run(workspace, "split") == EXIT_OK
    output = capsys.readouterr().out
    summary = json.loads(output[: output.index("\n\n")] if "\n\n" in output else output)
    assert summary["development_fraction_actual"] == pytest.approx(0.70, abs=0.01)
    assert summary["sealed"] is False
    assert "LOCKED" in output


def test_develop_then_freeze_then_baseline(workspace, capsys):
    place_candles(workspace)
    config = workspace["config"]

    assert develop(workspace) == EXIT_OK
    report_path = config.backtest.results_dir / "optimization_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["candidates_evaluated"] > 0
    assert report["development_end"]
    # Every fold's evaluation window ended inside the development period.
    for candidate in report["candidates"]:
        for fold in candidate["folds"]:
            assert fold["end"] <= report["development_end"]

    assert run(workspace, "freeze") == EXIT_OK
    seal_path = config.backtest.seal_path
    assert seal_path.exists()
    seal = json.loads(seal_path.read_text())
    assert seal["params_hash"]
    assert seal["optimizer_config"]["folds"] == config.optimizer.folds
    assert seal["dataset_hash"]
    assert seal["selection_reason"]

    capsys.readouterr()
    assert run(workspace, "baseline") == EXIT_OK
    output = capsys.readouterr().out
    assert "Experiment A" in output
    assert (config.backtest.results_dir / "baseline_result.json").exists()
    assert (config.backtest.results_dir / "baseline_manifest.json").exists()


def test_freeze_refuses_a_second_time_without_explicit_permission(workspace, capsys):
    place_candles(workspace)
    assert develop(workspace) == EXIT_OK
    assert run(workspace, "freeze") == EXIT_OK
    capsys.readouterr()

    # A distinct exit code: this is not a missing input, it is a refusal.
    assert run(workspace, "freeze") == EXIT_LEAKAGE_REFUSED
    error = capsys.readouterr().err
    assert "already exists" in error
    assert "leakage guard" in error

    assert run(workspace, "freeze", "--allow-reseal", "--reason", "test") == EXIT_OK


def test_split_reports_sealed_after_freeze(workspace, capsys):
    place_candles(workspace)
    develop(workspace)
    run(workspace, "freeze")
    capsys.readouterr()
    assert run(workspace, "split") == EXIT_OK
    output = capsys.readouterr().out
    assert '"sealed": true' in output
    assert "LOCKED" not in output


# --- the pilot, with no API calls ------------------------------------------
def test_pilot_refuses_without_a_key_and_runs_with_mock(workspace, capsys, monkeypatch):
    place_candles(workspace)
    develop(workspace)
    run(workspace, "freeze")
    capsys.readouterr()

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    workspace["config"].ai = None
    assert run(workspace, "pilot", "--count", "5") == EXIT_MISSING_INPUT
    error = capsys.readouterr().err
    assert "no ANTHROPIC_API_KEY" in error
    assert "--mock" in error

    assert run(workspace, "pilot", "--count", "5", "--mock") == EXIT_OK
    output = capsys.readouterr().out
    assert "MOCK: synthetic costs" in output
    results = workspace["config"].backtest.results_dir
    assert (results / "ai_result.json").exists()
    assert (results / "cost_report.json").exists()
    assert (results / "cost_report.md").exists()
    assert (results / "ai_decisions.json").exists()


def test_full_sequence_through_compare(workspace, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    place_candles(workspace)
    assert develop(workspace) == EXIT_OK
    for command in ("freeze", "baseline"):
        assert run(workspace, command) == EXIT_OK
    assert run(workspace, "pilot", "--count", "10", "--mock") == EXIT_OK
    capsys.readouterr()

    assert run(workspace, "compare") == EXIT_OK
    output = capsys.readouterr().out
    assert "AI cost" in output

    results = workspace["config"].backtest.results_dir
    comparison = json.loads((results / "comparison.json").read_text())
    assert comparison["manifests_verified"] is True
    assert comparison["shared_signal_count"] >= 0
    assert comparison["restricted_to_shared_set"] is True
    assert comparison["deltas"]
    assert (results / "comparison.md").exists()
    text = (results / "comparison.md").read_text()
    assert "Experiment A vs B" in text
    assert "MODEL_KNOWLEDGE_LEAKAGE" in text or "knowledge" in text.lower()


# --- other commands ---------------------------------------------------------
def test_datasets_reports_unavailable_without_fabricating(workspace, capsys):
    assert run(workspace, "datasets") == EXIT_OK
    output = capsys.readouterr().out
    assert output.count("UNAVAILABLE") >= 3
    assert "No data is invented" in output
    assert (workspace["config"].backtest.results_dir / "dataset_report.json").exists()


def test_config_command_writes_and_validates(workspace, capsys):
    assert run(workspace, "config") == EXIT_OK
    output = capsys.readouterr().out
    assert "fingerprint" in output
    assert workspace["config_path"].exists()


def test_dashboard_is_explicitly_deferred(workspace, capsys):
    assert run(workspace, "dashboard") == EXIT_MISSING_INPUT
    assert "not implemented" in capsys.readouterr().err


def test_fetch_csv_reports_a_missing_file_rather_than_inventing_ticks(workspace, capsys):
    assert run(workspace, "fetch", "--csv", str(workspace["tmp"] / "nope.csv")) == (
        EXIT_MISSING_INPUT
    )
    assert "ERROR" in capsys.readouterr().err


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])
