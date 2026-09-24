from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from research.data.pit import (
    LookaheadError,
    PointInTimeDataset,
    get_information_available_at,
    load_point_in_time_dataset,
)

UTC = timezone.utc
T = datetime(2023, 5, 10, 15, 0, tzinfo=UTC)

"""Tests for the single door into historical information.

The property under test throughout: for a signal at time T, nothing that became
knowable after T can reach an agent.
"""


def news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-05-10T14:45Z", "2023-05-10T15:00Z", "2023-05-10T15:30Z"]
            ),
            "published_at": pd.to_datetime(
                ["2023-05-10T14:32Z", "2023-05-10T14:58Z", "2023-05-10T15:28Z"]
            ),
            "discovered_at": pd.to_datetime(
                ["2023-05-10T14:45Z", "2023-05-10T15:00Z", "2023-05-10T15:30Z"]
            ),
            "source": ["a.com", "b.com", "c.com"],
            "source_id": ["1", "2", "3"],
            "headline": ["Before", "Exactly at T", "After"],
            "category": ["gold"] * 3,
            "provenance": ["ORIGINAL_RELEASE"] * 3,
        }
    )


def calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-05-10T12:30Z", "2023-05-10T18:00Z", "2023-07-12T12:30Z"]
            ),
            "name": ["CPI", "FOMC decision", "CPI (revised)"],
            "importance": ["HIGH", "HIGH", "HIGH"],
            "provenance": ["ORIGINAL_RELEASE", "ORIGINAL_RELEASE", "REVISED"],
            "series_id": ["CPIAUCSL", "FEDFUNDS", "CPIAUCSL"],
            "reference_period": ["2023-04-01", "2023-05-01", "2023-04-01"],
            "released_value": ["4.9", "hold", "5.0"],
            "actual": ["4.9", "hold", "5.0"],
            "source": ["fred_alfred"] * 3,
            "source_id": ["a", "b", "c"],
        }
    )


def sentiment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-05-10T14:45Z", "2023-05-10T14:45Z"]),
            "source": ["gdelt_gkg_tone", "llm_retrospective"],
            "value": [0.2, 0.95],
            "provenance": ["POINT_IN_TIME_CAPTURE", "RETROSPECTIVE"],
            "source_id": ["g1", "l1"],
            "method": ["tone", "llm"],
            "model": [None, "claude-haiku-4-5"],
        }
    )


@pytest.fixture
def dataset() -> PointInTimeDataset:
    return PointInTimeDataset(
        news=news_frame(), calendar=calendar_frame(), sentiment=sentiment_frame()
    )


# --- the core filter -------------------------------------------------------
def test_only_information_available_at_or_before_t_is_returned(dataset):
    info = get_information_available_at(dataset, T)
    assert list(info.news["headline"]) == ["Before", "Exactly at T"]
    assert "After" not in list(info.news["headline"])


def test_the_boundary_is_inclusive(dataset):
    """A record available exactly at T was available at T."""
    assert "Exactly at T" in list(dataset.news_at(T)["headline"])


def test_an_ingest_lag_is_not_a_head_start(dataset):
    """Published 14:32, discovered 14:45: invisible at 14:40."""
    early = dataset.news_at(datetime(2023, 5, 10, 14, 40, tzinfo=UTC))
    assert early.empty


def test_revisions_are_dropped_at_load_and_unreachable(dataset):
    assert dataset.dropped_inadmissible["economic_calendar"] == 1
    # No lookback long enough brings the July revision back.
    far = dataset.calendar_at(
        datetime(2023, 8, 1, tzinfo=UTC), lookback=timedelta(days=365)
    )
    assert "CPI (revised)" not in list(far["name"])
    assert "5.0" not in list(far["actual"].astype(str))


def test_retrospective_sentiment_is_unreachable(dataset):
    assert dataset.dropped_inadmissible["sentiment"] == 1
    frame = dataset.sentiment_at(T, lookback=timedelta(days=365))
    assert list(frame["source"]) == ["gdelt_gkg_tone"]
    assert 0.95 not in list(frame["value"])


def test_schedule_may_look_ahead_but_the_value_may_not(dataset):
    """A release known to be due at 18:00 is public knowledge at 15:00.

    The outcome is not, so the value is blanked and the reason recorded.
    """
    schedule = dataset.scheduled_events_at(T, lookahead=timedelta(hours=6))
    fomc = schedule[schedule["name"] == "FOMC decision"].iloc[0]
    assert bool(fomc["already_released"]) is False
    assert pd.isna(fomc["released_value"]) or fomc["released_value"] is None
    assert fomc["withheld_reason"] == "not yet released at this timestamp"
    assert fomc["minutes_until"] == pytest.approx(180.0)


def test_an_already_printed_release_keeps_its_value(dataset):
    schedule = dataset.scheduled_events_at(T, lookback=timedelta(hours=6))
    cpi = schedule[schedule["name"] == "CPI"].iloc[0]
    assert bool(cpi["already_released"]) is True
    assert cpi["released_value"] == "4.9"


def test_availability_is_recomputed_from_the_component_columns():
    """A hand-edited timestamp cannot move availability earlier.

    The loader takes the max of timestamp, published_at and discovered_at, so
    editing only the timestamp column to an earlier value has no effect.
    """
    frame = news_frame()
    frame.loc[2, "timestamp"] = pd.Timestamp("2023-05-10T10:00Z")  # tampered
    dataset = PointInTimeDataset(news=frame.sort_values("timestamp"))
    visible = dataset.news_at(T)
    # discovered_at is still 15:30, so the row stays hidden at 15:00.
    assert "After" not in list(visible["headline"])


def test_unavailable_reasons_distinguish_absent_from_empty():
    dataset = PointInTimeDataset(news=news_frame(), calendar=None, sentiment=None)
    info = get_information_available_at(dataset, T)
    assert "economic_calendar" in info.unavailable
    assert "sentiment" in info.unavailable
    # News exists but has nothing in a one-minute window -- a different message.
    narrow = dataset.get_information_available_at(
        datetime(2023, 5, 11, tzinfo=UTC), news_lookback=timedelta(minutes=1)
    )
    assert "no news records in the lookback window" in narrow.unavailable["news"]


def test_a_dataset_of_only_inadmissible_rows_is_reported_unavailable():
    retro_only = sentiment_frame().iloc[[1]].reset_index(drop=True)
    dataset = PointInTimeDataset(sentiment=retro_only)
    info = get_information_available_at(dataset, T)
    assert "inadmissible provenance" in info.unavailable["sentiment"]
    assert info.sentiment.empty


def test_summary_and_coverage_report_what_is_loaded(dataset):
    summary = get_information_available_at(dataset, T).summary()
    assert summary["news"] == 2
    coverage = dataset.coverage()
    assert coverage["news"]["rows"] == 3
    assert coverage["dropped_inadmissible"]["sentiment"] == 1


# --- loading ---------------------------------------------------------------
def test_missing_files_yield_actionable_unavailable_reasons(tmp_path):
    dataset = load_point_in_time_dataset(
        news_path=tmp_path / "absent_news.parquet",
        calendar_path=None,
        sentiment_path=tmp_path / "absent_sentiment.parquet",
    )
    info = get_information_available_at(dataset, T)
    assert "research_data" in info.unavailable["news"]  # names the command to run
    assert "no data is invented" in info.unavailable["news"]
    assert "no economic_calendar dataset path configured" in info.unavailable[
        "economic_calendar"
    ]


def test_loading_round_trips_a_written_dataset(tmp_path):
    from research.data.ingest.normalize import write_dataset

    write_dataset(news_frame(), tmp_path / "news.parquet")
    dataset = load_point_in_time_dataset(news_path=tmp_path / "news.parquet")
    assert len(dataset.news_at(T)) == 2


# --- the 70/30 split ------------------------------------------------------
def test_information_respects_the_chronological_split():
    """A development-period signal cannot see out-of-sample information.

    The point-in-time filter enforces this without knowing the split exists: a
    signal inside the first 70% is asked about ITS moment, and everything from
    the final 30% is later than that moment by construction.
    """
    span = pd.date_range("2021-01-01", "2025-12-31", freq="7D", tz="UTC")
    boundary = span[int(len(span) * 0.70)]
    frame = pd.DataFrame(
        {
            "timestamp": span,
            "published_at": span,
            "discovered_at": span,
            "source": ["s.com"] * len(span),
            "source_id": [str(i) for i in range(len(span))],
            "headline": [f"Story {i}" for i in range(len(span))],
            "category": ["gold"] * len(span),
            "provenance": ["ORIGINAL_RELEASE"] * len(span),
        }
    )
    dataset = PointInTimeDataset(news=frame)

    development_signal = boundary - pd.Timedelta(days=30)
    visible = dataset.news_at(
        development_signal.to_pydatetime(), lookback=timedelta(days=3650), limit=None
    )
    assert not visible.empty
    assert visible["timestamp"].max() <= development_signal
    # Nothing from the out-of-sample period leaked into a development query.
    assert (visible["timestamp"] < boundary).all()


def test_lookahead_error_exists_for_the_invariant_breach():
    """The guard is present even though normal operation cannot trigger it."""
    assert issubclass(LookaheadError, Exception)


# --- the offline guarantee -------------------------------------------------
def test_the_pit_layer_reads_only_local_files():
    """No network client may appear in the point-in-time read path.

    The agents must not reach the internet during a historical run. The
    ingestion layer is the only component allowed to make requests, so this
    asserts the read path has no HTTP dependency at all.
    """
    import research.data.pit as module

    source = module.__file__
    with open(source) as handle:
        text = handle.read()
    for forbidden in ("httpx", "requests", "urllib.request", "aiohttp", "socket"):
        assert forbidden not in text, f"{forbidden} must not be reachable from the PIT layer"


def test_the_ai_runner_has_no_http_client_in_its_read_path():
    """The agent runner reads through stores, never over the network."""
    import research.ai.runner as runner

    with open(runner.__file__) as handle:
        text = handle.read()
    for forbidden in ("httpx.", "requests.", "urllib.request"):
        assert forbidden not in text, (
            f"{forbidden} appears in the AI runner; historical runs must read only "
            "from local point-in-time stores"
        )
