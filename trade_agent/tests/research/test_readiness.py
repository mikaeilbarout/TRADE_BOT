from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from research.data.ingest.base import Provenance, TimePrecision
from research.data.ingest.fred import ReleaseTimePolicy
from research.data.ingest.normalize import write_dataset
from research.data.readiness import (
    STANDING_LEAKAGE_RISKS,
    assess_dataset,
    assess_split,
    assess_ticks,
    build_report,
    render_markdown,
    write_reports,
)
from research.data.ingest.validate import validate_calendar, validate_news, validate_sentiment
from research.experiment import ExperimentConfig
from research_data.cli import EXIT_MISSING_INPUT, EXIT_OK, main

UTC = timezone.utc
START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2023, 3, 1, tzinfo=UTC)


def news_frame(n: int = 60) -> pd.DataFrame:
    stamps = pd.date_range(START, periods=n, freq="24h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "published_at": stamps,
            "discovered_at": stamps,
            "retrieved_at": pd.to_datetime(["2026-01-01T00:00Z"] * n),
            "source": ["reuters.com"] * n,
            "source_id": [str(i) for i in range(n)],
            "headline": [f"Gold headline {i}" for i in range(n)],
            "category": ["gold"] * n,
            "provenance": ["ORIGINAL_RELEASE"] * n,
        }
    )


def calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-12T13:30Z", "2023-02-14T13:30Z", "2023-03-14T12:30Z"]
            ),
            "name": ["US CPI"] * 3,
            "importance": ["HIGH"] * 3,
            "provenance": ["ORIGINAL_RELEASE", "ORIGINAL_RELEASE", "REVISED"],
            "series_id": ["CPIAUCSL"] * 3,
            "reference_period": ["2022-12-01", "2023-01-01", "2022-12-01"],
            "source": ["fred_alfred"] * 3,
            "source_id": ["a", "b", "c"],
            "time_precision": [TimePrecision.IMPUTED_FROM_SCHEDULE.value] * 3,
            "retrieved_at": pd.to_datetime(["2026-01-01T00:00Z"] * 3),
        }
    )


def sentiment_frame() -> pd.DataFrame:
    stamps = pd.date_range(START, periods=40, freq="36h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "source": ["gdelt_gkg_tone"] * 40,
            "value": [0.1] * 40,
            "provenance": [Provenance.POINT_IN_TIME_CAPTURE.value] * 40,
            "source_id": [f"g{i}" for i in range(40)],
            "method": ["tone"] * 40,
            "model": [None] * 40,
            "retrieved_at": pd.to_datetime(["2026-01-01T00:00Z"] * 40),
        }
    )


# --- per-dataset assessment ------------------------------------------------
def test_absent_dataset_is_unavailable_and_says_nothing_is_substituted(tmp_path):
    readiness = assess_dataset(
        "news", tmp_path / "absent.parquet", START, END, validate_news, 72.0,
        (Provenance.ORIGINAL_RELEASE.value,),
    )
    assert readiness.status == "UNAVAILABLE"
    assert readiness.records == 0
    assert not readiness.usable
    assert "Nothing is substituted" in readiness.notes[0]


def test_complete_dataset_is_ready(tmp_path):
    path = write_dataset(news_frame(), tmp_path / "news.parquet")
    readiness = assess_dataset(
        "news", path, START, END, validate_news, 72.0,
        (Provenance.ORIGINAL_RELEASE.value,),
    )
    assert readiness.status == "READY"
    assert readiness.records == 60
    assert readiness.validation_passed is True
    assert readiness.provenance_counts == {"ORIGINAL_RELEASE": 60}
    assert readiness.sources == ["reuters.com"]
    assert readiness.coverage_fraction is not None


def test_a_gap_makes_a_dataset_partial_and_lists_the_range(tmp_path):
    frame = news_frame(10)
    frame.loc[9, ["timestamp", "published_at", "discovered_at"]] = pd.Timestamp(
        "2023-02-25T00:00Z"
    )
    path = write_dataset(frame.sort_values("timestamp"), tmp_path / "news.parquet")
    readiness = assess_dataset(
        "news", path, START, END, validate_news, 72.0,
        (Provenance.ORIGINAL_RELEASE.value,),
    )
    assert readiness.status == "PARTIAL"
    assert readiness.usable  # covered timestamps still work
    assert readiness.missing_ranges
    assert readiness.coverage_fraction < 1.0


def test_a_failing_dataset_is_invalid_not_merely_partial(tmp_path):
    frame = news_frame(5)
    frame.loc[0, "timestamp"] = pd.Timestamp("2030-01-01T00:00Z")  # future-dated
    path = write_dataset(frame.sort_values("timestamp"), tmp_path / "news.parquet")
    readiness = assess_dataset(
        "news", path, START, datetime(2030, 6, 1, tzinfo=UTC), validate_news, 100000.0,
        (Provenance.ORIGINAL_RELEASE.value,),
    )
    assert readiness.status == "INVALID"
    assert not readiness.usable
    assert any("future_dated" in error for error in readiness.validation_errors)


def test_imputed_from_schedule_records_are_counted(tmp_path):
    path = write_dataset(calendar_frame(), tmp_path / "calendar.parquet")
    readiness = assess_dataset(
        "economic_calendar", path, START, END, validate_calendar, 336.0,
        (Provenance.ORIGINAL_RELEASE.value, Provenance.REVISED.value),
    )
    assert readiness.time_precision_counts[TimePrecision.IMPUTED_FROM_SCHEDULE.value] == 3
    assert readiness.provenance_counts["ORIGINAL_RELEASE"] == 2
    assert readiness.provenance_counts["REVISED"] == 1


def test_retrospective_sentiment_in_the_main_dataset_is_invalid(tmp_path):
    frame = sentiment_frame()
    frame.loc[0, "provenance"] = Provenance.RETROSPECTIVE.value
    path = write_dataset(frame, tmp_path / "sentiment.parquet")
    readiness = assess_dataset(
        "sentiment", path, START, END, validate_sentiment, 72.0,
        (Provenance.POINT_IN_TIME_CAPTURE.value,), point_in_time_only=True,
    )
    assert readiness.status == "INVALID"
    assert any("point_in_time_only" in error for error in readiness.validation_errors)


# --- ticks and split -------------------------------------------------------
def test_ticks_unavailable_without_candles(tmp_path):
    ticks = assess_ticks(tmp_path / "ticks", tmp_path / "candles.parquet", START, END, 15)
    assert ticks.status == "UNAVAILABLE"
    assert "Nothing is synthesised" in ticks.notes[0]


def test_split_cannot_be_formed_without_bars(tmp_path):
    split = assess_split(tmp_path / "absent.parquet", 0.70, 200, tmp_path / "seal.json")
    assert split.ready is False
    assert "no candle dataset" in split.reason


def test_split_is_reported_once_bars_exist(tmp_path):
    n = 3000
    stamps = pd.date_range("2021-01-04", periods=n, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [1800.0] * n, "high": [1801.0] * n,
            "low": [1799.0] * n, "close": [1800.5] * n,
        }
    )
    path = write_dataset(bars, tmp_path / "candles.parquet")
    split = assess_split(path, 0.70, 200, tmp_path / "seal.json")
    assert split.ready
    assert split.total_bars == n
    assert split.development_bars == int(n * 0.70)
    assert split.out_of_sample_bars == n - int(n * 0.70)
    assert split.embargo_bars == 200
    assert split.sealed is False  # out-of-sample still locked


# --- whole report ----------------------------------------------------------
def _report(tmp_path, *, with_news=False, with_calendar=False, with_sentiment=False,
            host_probe=None):
    if with_news:
        write_dataset(news_frame(), tmp_path / "news.parquet")
    if with_calendar:
        write_dataset(calendar_frame(), tmp_path / "calendar.parquet")
    if with_sentiment:
        write_dataset(sentiment_frame(), tmp_path / "sentiment.parquet")
    return build_report(
        experiment="test",
        start=START,
        end=END,
        release_time_policy=ReleaseTimePolicy.SCHEDULED_LOCAL.value,
        tick_dir=tmp_path / "ticks",
        candle_path=tmp_path / "candles.parquet",
        news_path=tmp_path / "news.parquet",
        calendar_path=tmp_path / "calendar.parquet",
        sentiment_path=tmp_path / "sentiment.parquet",
        sentiment_retrospective_path=tmp_path / "retro.parquet",
        seal_path=tmp_path / "seal.json",
        checkpoint_path=tmp_path / "cp.sqlite",
        host_probe=host_probe or {},
    )


def test_verdict_is_not_ready_with_nothing_on_disk(tmp_path):
    report = _report(tmp_path)
    assert report.verdict == "NOT_READY"
    assert "Nothing was fabricated" in report.verdict_reason
    assert len(report.blockers) >= 4
    assert any("M15 candle dataset" in b for b in report.blockers)
    assert report.calendar_release_time_policy == "SCHEDULED_LOCAL"


def test_blocked_hosts_and_missing_keys_become_blockers(tmp_path):
    report = _report(
        tmp_path,
        host_probe={
            "data.gdeltproject.org": "BLOCKED (403) -- egress policy",
            "api.stlouisfed.org": "OK (200)",
        },
    )
    blockers = " ".join(report.blockers)
    assert "data.gdeltproject.org" in blockers
    assert "api.stlouisfed.org" not in blockers  # reachable hosts are not blockers
    assert "FRED_API_KEY" in blockers


def test_report_always_carries_the_standing_leakage_risks(tmp_path):
    report = _report(tmp_path)
    codes = {risk["risk"] for risk in report.leakage_risks}
    assert "MODEL_KNOWLEDGE_LEAKAGE" in codes
    assert "IMPUTED_RELEASE_TIMES" in codes
    assert "REVISION_LEAKAGE" in codes
    leakage = next(r for r in report.leakage_risks if r["risk"] == "MODEL_KNOWLEDGE_LEAKAGE")
    assert "UNRESOLVED" in leakage["status"]


def test_imputed_release_time_risk_is_disclosed_for_scheduled_local(tmp_path):
    report = _report(tmp_path, with_calendar=True)
    risk = next(r for r in report.leakage_risks if r["risk"] == "IMPUTED_RELEASE_TIMES")
    assert "SCHEDULED_LOCAL" in risk["status"]
    assert "END_OF_DAY eliminates" in risk["mitigation"]


def test_every_standing_risk_states_a_mitigation():
    for risk in STANDING_LEAKAGE_RISKS:
        assert risk["detail"] and risk["mitigation"] and risk["severity"]
        assert risk["status"]


def test_markdown_contains_every_requested_section(tmp_path):
    text = render_markdown(_report(tmp_path, with_news=True, with_calendar=True))
    for heading in (
        "# Data readiness report",
        "## Verdict",
        "## Coverage",
        "## Provenance",
        "### Calendar timestamp precision",
        "### Sentiment provenance",
        "## 70/30 split",
        "## Remaining data-quality problems",
        "## Look-ahead / leakage risks",
        "## Environment",
    ):
        assert heading in text, heading
    assert "IMPUTED_FROM_SCHEDULE" in text
    assert "POINT_IN_TIME_CAPTURE" in text
    assert "RETROSPECTIVE" in text


def test_reports_are_written_in_both_forms(tmp_path):
    written = write_reports(_report(tmp_path, with_news=True), tmp_path / "out")
    assert written["json"].exists() and written["markdown"].exists()
    import json

    payload = json.loads(written["json"].read_text())
    assert payload["verdict"] in ("READY", "PARTIAL", "NOT_READY")
    assert payload["calendar_release_time_policy"] == "SCHEDULED_LOCAL"
    assert payload["news"]["records"] == 60


# --- the SCHEDULED_LOCAL decision ------------------------------------------
def test_scheduled_local_is_the_configured_default():
    config = ExperimentConfig()
    assert config.calendar_release_time_policy == "SCHEDULED_LOCAL"
    # ...and it is part of the A/B fingerprint, so both arms share it.
    assert config.shared_fingerprint()["calendar_release_time_policy"] == "SCHEDULED_LOCAL"


def test_changing_the_policy_changes_the_fingerprint():
    baseline = ExperimentConfig()
    other = baseline.model_copy(deep=True)
    other.calendar_release_time_policy = "END_OF_DAY"
    assert other.fingerprint_hash() != baseline.fingerprint_hash()


def test_cli_defaults_the_calendar_policy_to_scheduled_local():
    from research_data.cli import build_parser

    args = build_parser().parse_args(["fetch-calendar"])
    assert args.release_time_policy == "SCHEDULED_LOCAL"


# --- the CLI command -------------------------------------------------------
def test_readiness_command_exits_non_zero_when_not_ready(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from research.experiment import save_experiment

    config = ExperimentConfig()
    config.backtest.data_dir = tmp_path / "data"
    config.backtest.results_dir = tmp_path / "results"
    path = save_experiment(config, tmp_path / "experiment.json")

    code = main(["readiness", "--no-probe", "--config", str(path)])
    assert code == EXIT_MISSING_INPUT
    assert (tmp_path / "results" / "ingest" / "data_readiness.json").exists()
    assert (tmp_path / "results" / "ingest" / "data_readiness.md").exists()


def test_validate_does_not_claim_success_on_empty_datasets(tmp_path, capsys, monkeypatch):
    """An empty dataset raises no ERROR, but it is not a validated dataset."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from research.experiment import save_experiment

    config = ExperimentConfig()
    config.backtest.data_dir = tmp_path / "data"
    config.backtest.results_dir = tmp_path / "results"
    path = save_experiment(config, tmp_path / "experiment.json")

    code = main(["validate", "--config", str(path)])
    output = capsys.readouterr().out
    assert code == EXIT_MISSING_INPUT
    assert "EMPTY and therefore not validated" in output
    assert "nothing was actually validated" in output
