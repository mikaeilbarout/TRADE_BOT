from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from research.data.ingest.normalize import write_dataset
from research.experiment import ExperimentConfig, save_experiment
from research_data.cli import (
    EXIT_MISSING_INPUT,
    EXIT_OK,
    EXIT_VALIDATION_FAILED,
    main,
)

UTC = timezone.utc

"""CLI tests for the ingestion pipeline.

No command in these tests makes a network request or a paid API call: the
commands that would fetch are checked for how they REFUSE (blocked host, missing
key), and the commands that read local data are given fixtures.
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> dict:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    config = ExperimentConfig()
    config.backtest.data_dir = tmp_path / "data"
    config.backtest.results_dir = tmp_path / "results"
    path = save_experiment(config, tmp_path / "experiment.json")
    return {"config": config, "config_path": path, "tmp": tmp_path}


def run(workspace, *args) -> int:
    return main([*args, "--config", str(workspace["config_path"])])


def news_rows(n: int = 5) -> pd.DataFrame:
    stamps = pd.date_range("2023-05-10T14:45Z", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "published_at": stamps,
            "discovered_at": stamps,
            "retrieved_at": pd.to_datetime(["2026-01-01T00:00Z"] * n),
            "source": [f"s{i}.com" for i in range(n)],
            "source_id": [str(i) for i in range(n)],
            "headline": [f"Gold headline {i}" for i in range(n)],
            "category": ["gold"] * n,
            "provenance": ["ORIGINAL_RELEASE"] * n,
        }
    )


# --- sources ---------------------------------------------------------------
def test_sources_lists_hosts_keys_and_rejected_options(capsys):
    assert main(["sources"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "data.gdeltproject.org" in output
    assert "api.stlouisfed.org" in output
    assert "FRED_API_KEY" in output
    assert "fredaccount.stlouisfed.org" in output  # how to get it
    assert "EVALUATED and NOT used" in output
    assert "investing" in output.lower()  # the licensing rejection is disclosed


def test_sources_json_is_machine_readable(capsys):
    """--json emits JSON and nothing else, so it can be piped."""
    assert main(["sources", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    keys = {row["key"] for row in payload["sources"]}
    assert {"gdelt_gkg", "fred_alfred"} <= keys
    assert all("evaluation" in row for row in payload["sources"])
    assert "data.gdeltproject.org" in payload["hosts_required"]
    assert {k["env_var"] for k in payload["api_keys"]} == {"FRED_API_KEY"}
    # Presence only -- never the value.
    assert all("value" not in entry for entry in payload["api_keys"])


def test_sources_never_prints_a_key_value(capsys, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRET-VALUE-1234")
    assert main(["sources"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "SECRET-VALUE-1234" not in output
    assert "SET" in output  # only presence is reported


# --- status ----------------------------------------------------------------
def test_status_reports_everything_unavailable_before_ingestion(workspace, capsys):
    assert run(workspace, "status") == EXIT_OK
    output = capsys.readouterr().out
    assert output.count("UNAVAILABLE") >= 3
    assert "no ingestion has run yet" in output
    assert "fetch-news" in output  # the next step


def test_status_reports_a_dataset_once_it_exists(workspace, capsys):
    write_dataset(news_rows(), workspace["config"].backtest.data_dir / "news" / "news.parquet")
    assert run(workspace, "status") == EXIT_OK
    output = capsys.readouterr().out
    assert "5 rows" in output.replace("      5 rows", "5 rows")
    assert "ORIGINAL_RELEASE" in output
    assert "fetch-calendar" in output  # advances to the next step


# --- fetch refusals --------------------------------------------------------
def test_fetch_calendar_refuses_without_a_key(workspace, capsys):
    assert run(workspace, "fetch-calendar") == EXIT_MISSING_INPUT
    error = capsys.readouterr().err
    assert "FRED_API_KEY" in error
    assert "fredaccount.stlouisfed.org" in error


def test_fetch_news_reports_a_blocked_host_rather_than_empty_data(workspace, capsys):
    """A denied host must be reported, never recorded as a quiet period."""
    code = run(workspace, "fetch-news", "--max-windows", "1", "--rate", "0")
    output = capsys.readouterr()
    # In a network-restricted environment this is a refusal, not a silent success.
    if code == EXIT_MISSING_INPUT:
        assert "STOPPED" in output.err or "STOPPED" in output.out
    else:
        assert code == EXIT_OK


def test_build_sentiment_retrospective_needs_news_first(workspace, capsys):
    assert run(workspace, "build-sentiment", "--mode", "retrospective") == (
        EXIT_MISSING_INPUT
    )
    assert "fetch news first" in capsys.readouterr().err


def test_retrospective_mode_is_a_dry_run_by_default(workspace, capsys):
    write_dataset(news_rows(), workspace["config"].backtest.data_dir / "news" / "news.parquet")
    assert run(workspace, "build-sentiment", "--mode", "retrospective") == EXIT_OK
    output = capsys.readouterr().out
    assert "RETROSPECTIVE" in output
    assert "labelled RETROSPECTIVE" in output or "labelled" in output
    assert "Nothing was scored and nothing was spent" in output
    estimate = json.loads(output[output.index("{") : output.rindex("}") + 1])
    assert estimate["batches_pending"] >= 1
    assert estimate["model"] == "claude-haiku-4-5"


def test_retrospective_execute_refuses_to_spend_as_a_side_effect(workspace, capsys):
    write_dataset(news_rows(), workspace["config"].backtest.data_dir / "news" / "news.parquet")
    code = run(workspace, "build-sentiment", "--mode", "retrospective", "--execute")
    assert code == EXIT_MISSING_INPUT
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# --- validate --------------------------------------------------------------
def test_validate_fails_closed_on_a_leaky_calendar(workspace, capsys):
    data = workspace["config"].backtest.data_dir
    write_dataset(news_rows(), data / "news" / "news.parquet")
    # A revision timestamped at the same moment as its original release.
    leaky = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-06-13T23:59:59Z"] * 2),
            "name": ["CPI"] * 2,
            "importance": ["HIGH"] * 2,
            "provenance": ["ORIGINAL_RELEASE", "REVISED"],
            "series_id": ["CPIAUCSL"] * 2,
            "reference_period": ["2023-05-01"] * 2,
            "source": ["fred_alfred"] * 2,
            "source_id": ["a", "b"],
        }
    )
    write_dataset(leaky, data / "calendar" / "calendar.parquet")

    assert run(workspace, "validate") == EXIT_VALIDATION_FAILED
    captured = capsys.readouterr()
    assert "revision_after_original" in captured.out + captured.err
    assert "blocked rather than" in captured.err


def test_validate_passes_on_clean_data_and_writes_a_report(workspace, capsys):
    data = workspace["config"].backtest.data_dir
    write_dataset(news_rows(), data / "news" / "news.parquet")
    clean = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-06-13T23:59:59Z", "2023-07-12T23:59:59Z"]),
            "name": ["CPI", "CPI"],
            "importance": ["HIGH", "HIGH"],
            "provenance": ["ORIGINAL_RELEASE", "REVISED"],
            "series_id": ["CPIAUCSL"] * 2,
            "reference_period": ["2023-05-01"] * 2,
            "source": ["fred_alfred"] * 2,
            "source_id": ["a", "b"],
        }
    )
    write_dataset(clean, data / "calendar" / "calendar.parquet")
    sentiment = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-05-10T14:45Z"]),
            "source": ["gdelt_gkg_tone"],
            "value": [0.2],
            "provenance": ["POINT_IN_TIME_CAPTURE"],
            "source_id": ["g1"],
            "method": ["tone"],
            "model": [None],
        }
    )
    write_dataset(sentiment, data / "sentiment" / "sentiment.parquet")

    code = run(workspace, "validate", "--max-gap-hours", "100000")
    output = capsys.readouterr().out
    assert "passed validation" in output
    assert "EMPTY" not in output  # all three datasets have rows
    assert code == EXIT_OK
    report = json.loads(
        (workspace["config"].backtest.results_dir / "ingest" / "validation.json").read_text()
    )
    assert {entry["dataset"] for entry in report} == {"news", "economic_calendar", "sentiment"}


def test_validate_rejects_retrospective_sentiment_in_the_main_dataset(workspace, capsys):
    data = workspace["config"].backtest.data_dir
    write_dataset(news_rows(), data / "news" / "news.parquet")
    retro = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-05-10T14:45Z"]),
            "source": ["llm_retrospective"],
            "value": [0.9],
            "provenance": ["RETROSPECTIVE"],
            "source_id": ["l1"],
            "method": ["llm"],
            "model": ["claude-haiku-4-5"],
        }
    )
    write_dataset(retro, data / "sentiment" / "sentiment.parquet")

    assert run(workspace, "validate") == EXIT_VALIDATION_FAILED
    captured = capsys.readouterr()
    assert "sentiment_point_in_time_only" in captured.out + captured.err


def test_validate_reports_missing_date_ranges(workspace, capsys):
    data = workspace["config"].backtest.data_dir
    sparse = news_rows(2)
    sparse.loc[1, "timestamp"] = pd.Timestamp("2024-01-01T00:00Z")
    sparse.loc[1, "published_at"] = pd.Timestamp("2024-01-01T00:00Z")
    sparse.loc[1, "discovered_at"] = pd.Timestamp("2024-01-01T00:00Z")
    write_dataset(sparse, data / "news" / "news.parquet")
    run(workspace, "validate", "--max-gap-hours", "48")
    output = capsys.readouterr().out
    assert "missing periods" in output


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
