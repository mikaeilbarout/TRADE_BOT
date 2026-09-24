from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from research.ai.pit_store import CalendarPitStore
from research.data.datasets import (
    Availability,
    Provenance,
    candle_dataset_report,
    inspect_all,
    inspect_dataset,
    NEWS_SCHEMA,
    NEWS_SEMANTICS,
    SENTIMENT_SCHEMA,
    SENTIMENT_SEMANTICS,
)

UTC = timezone.utc
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 3, 1, tzinfo=UTC)


def write_csv(path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def news_rows(count: int = 40, start: datetime = START, spacing_hours: int = 24) -> list[dict]:
    return [
        {
            "timestamp": (start + timedelta(hours=spacing_hours * i)).isoformat(),
            "source": "fixture-wire",
            "headline": f"headline {i}",
            "category": "macro",
        }
        for i in range(count)
    ]


# --- availability states ----------------------------------------------------
def test_absent_dataset_is_unavailable_not_empty(tmp_path):
    report, frame = inspect_dataset(
        "news", tmp_path / "nothing.csv", NEWS_SCHEMA, NEWS_SEMANTICS
    )
    assert report.availability == Availability.UNAVAILABLE
    assert not report.usable
    assert frame is None
    assert "not found" in report.notes[0]


def test_no_path_configured_is_unavailable():
    report, frame = inspect_dataset("news", None, NEWS_SCHEMA, NEWS_SEMANTICS)
    assert report.availability == Availability.UNAVAILABLE
    assert frame is None


def test_complete_dataset_is_available(tmp_path):
    path = write_csv(tmp_path / "news.csv", news_rows(60, spacing_hours=24))
    report, frame = inspect_dataset(
        "news", path, NEWS_SCHEMA, NEWS_SEMANTICS, START, END
    )
    assert report.availability == Availability.AVAILABLE
    assert report.usable
    assert report.row_count == 60
    assert report.content_hash
    assert report.point_in_time_semantics == NEWS_SEMANTICS
    assert len(frame) == 60


def test_dataset_with_a_hole_is_partial_and_lists_the_gap(tmp_path):
    rows = news_rows(10) + news_rows(10, start=START + timedelta(days=40))
    path = write_csv(tmp_path / "news.csv", rows)
    report, _ = inspect_dataset("news", path, NEWS_SCHEMA, NEWS_SEMANTICS, START, END)

    assert report.availability == Availability.PARTIAL
    assert report.usable  # covered timestamps still work
    assert report.missing_periods
    gap = report.missing_periods[0]
    assert gap.days > 7
    assert report.coverage_fraction is not None and report.coverage_fraction < 1.0


def test_missing_required_column_is_refused(tmp_path):
    rows = [{"timestamp": START.isoformat(), "source": "x", "headline": "y"}]  # no category
    path = write_csv(tmp_path / "news.csv", rows)
    report, frame = inspect_dataset("news", path, NEWS_SCHEMA, NEWS_SEMANTICS)

    assert report.availability == Availability.UNAVAILABLE
    assert frame is None
    assert "category" in report.notes[0]


def test_non_overlapping_coverage_is_refused(tmp_path):
    path = write_csv(
        tmp_path / "news.csv", news_rows(10, start=datetime(2019, 1, 1, tzinfo=UTC))
    )
    report, frame = inspect_dataset(
        "news", path, NEWS_SCHEMA, NEWS_SEMANTICS, START, END
    )
    assert report.availability == Availability.UNAVAILABLE
    assert "does not overlap" in report.notes[0]
    assert frame is None


# --- timestamp precision ----------------------------------------------------
@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2025-01-02", "day"),
        ("2025-01-02T13:00:00", "hour"),
        ("2025-01-02T13:45:00", "minute"),
        ("2025-01-02T13:45:07", "second"),
    ],
)
def test_precision_is_reported(tmp_path, stamp, expected):
    """A day-stamped feed cannot support a 15-minute blackout window, so the
    precision has to be stated rather than assumed."""
    rows = [
        {"timestamp": stamp, "source": "x", "headline": "y", "category": "macro"}
    ]
    path = write_csv(tmp_path / "news.csv", rows)
    report, _ = inspect_dataset("news", path, NEWS_SCHEMA, NEWS_SEMANTICS)
    assert report.timestamp_precision == expected


# --- provenance -------------------------------------------------------------
def test_revised_rows_are_excluded(tmp_path):
    rows = news_rows(10)
    for i, row in enumerate(rows):
        row["provenance"] = "REVISED" if i < 4 else "ORIGINAL_RELEASE"
    path = write_csv(tmp_path / "news.csv", rows)
    report, frame = inspect_dataset("news", path, NEWS_SCHEMA, NEWS_SEMANTICS, START, END)

    assert report.rows_excluded == 4
    assert "REVISED" in report.exclusion_reason
    assert len(frame) == 6
    assert (frame["provenance"] == "ORIGINAL_RELEASE").all()


def test_a_dataset_of_only_revised_rows_is_unavailable(tmp_path):
    rows = news_rows(5)
    for row in rows:
        row["provenance"] = "REVISED"
    path = write_csv(tmp_path / "news.csv", rows)
    report, frame = inspect_dataset("news", path, NEWS_SCHEMA, NEWS_SEMANTICS, START, END)

    assert report.availability == Availability.UNAVAILABLE
    assert frame is None


def test_retrospectively_scored_sentiment_is_refused(tmp_path):
    """Scoring old text with a present-day model is not point-in-time data."""
    rows = [
        {
            "timestamp": (START + timedelta(hours=i)).isoformat(),
            "source": "archive",
            "value": 0.4,
            "provenance": "RETROSPECTIVE_SCORING",
        }
        for i in range(50)
    ]
    path = write_csv(tmp_path / "sentiment.csv", rows)
    report, frame = inspect_dataset(
        "sentiment",
        path,
        SENTIMENT_SCHEMA,
        SENTIMENT_SEMANTICS,
        START,
        END,
        require_point_in_time_provenance=True,
    )
    assert report.availability == Availability.UNAVAILABLE
    assert frame is None


def test_sentiment_without_any_provenance_marker_is_refused(tmp_path):
    """Absent evidence of live capture, the dataset cannot be trusted as PIT."""
    rows = [
        {
            "timestamp": (START + timedelta(hours=i)).isoformat(),
            "source": "feed",
            "value": 0.5,
        }
        for i in range(50)
    ]
    path = write_csv(tmp_path / "sentiment.csv", rows)
    report, frame = inspect_dataset(
        "sentiment", path, SENTIMENT_SCHEMA, SENTIMENT_SEMANTICS, START, END,
        require_point_in_time_provenance=True,
    )
    assert report.availability == Availability.UNAVAILABLE
    assert "not point-in-time data" in report.notes[0]
    assert frame is None


def test_genuinely_captured_sentiment_is_accepted(tmp_path):
    rows = [
        {
            "timestamp": (START + timedelta(hours=6 * i)).isoformat(),
            "source": "positioning-feed",
            "value": 0.3 + (i % 5) * 0.1,
            "provenance": Provenance.POINT_IN_TIME_CAPTURE.value,
        }
        for i in range(200)
    ]
    path = write_csv(tmp_path / "sentiment.csv", rows)
    report, frame = inspect_dataset(
        "sentiment", path, SENTIMENT_SCHEMA, SENTIMENT_SEMANTICS, START, END,
        require_point_in_time_provenance=True,
    )
    assert report.usable
    assert report.provenance == Provenance.POINT_IN_TIME_CAPTURE.value
    assert len(frame) == 200


def test_mixed_sentiment_keeps_only_the_captured_rows(tmp_path):
    rows = []
    for i in range(100):
        rows.append(
            {
                "timestamp": (START + timedelta(hours=6 * i)).isoformat(),
                "source": "feed",
                "value": 0.5,
                "provenance": (
                    Provenance.POINT_IN_TIME_CAPTURE.value
                    if i % 2 == 0
                    else Provenance.UNKNOWN.value
                ),
            }
        )
    path = write_csv(tmp_path / "sentiment.csv", rows)
    report, frame = inspect_dataset(
        "sentiment", path, SENTIMENT_SCHEMA, SENTIMENT_SEMANTICS, START, END,
        require_point_in_time_provenance=True,
    )
    assert len(frame) == 50
    assert any("without point-in-time provenance" in note for note in report.notes)


# --- calendar ---------------------------------------------------------------
def test_calendar_withholds_a_revised_value_even_after_release():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-06-01 12:00", "2025-06-01 13:00"], utc=True
            ),
            "name": ["CPI", "NFP"],
            "importance": ["HIGH", "HIGH"],
            "released_value": ["3.1%", "210k"],
            "forecast_value": ["3.0%", "200k"],
            "provenance": ["REVISED", "ORIGINAL_RELEASE"],
        }
    )
    # A window wide enough to contain both the printed and the pending event.
    events = CalendarPitStore(frame).query_events(
        datetime(2025, 6, 1, 15, 0, tzinfo=UTC), 400
    ).events
    by_name = {event.name: event for event in events}

    assert by_name["NFP"].released_value == "210k"
    assert by_name["CPI"].released_value is None
    assert "later revision" in by_name["CPI"].withheld_reason


def test_calendar_serves_the_schedule_but_not_a_future_value():
    """A scheduled event is public knowledge; its outcome is not.

    The earlier row is there so the dataset demonstrably covers the query
    time -- a store whose first entry is after `as_of` reports "does not
    cover" rather than guessing, which is a separate behaviour tested below.
    """
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-06-01 09:00", "2025-06-01 18:00"], utc=True
            ),
            "name": ["Claims", "FOMC"],
            "importance": ["MEDIUM", "HIGH"],
            "released_value": ["220k", "hold"],
            "forecast_value": ["215k", "hold"],
            "provenance": ["ORIGINAL_RELEASE", "ORIGINAL_RELEASE"],
        }
    )
    # A window wide enough to contain both the printed and the pending event.
    events = CalendarPitStore(frame).query_events(
        datetime(2025, 6, 1, 15, 0, tzinfo=UTC), 400
    ).events
    by_name = {event.name: event for event in events}

    # The schedule is known in advance...
    assert by_name["FOMC"].scheduled_at.hour == 18
    # ...the outcome is not.
    assert by_name["FOMC"].released_value is None
    assert "not yet released" in by_name["FOMC"].withheld_reason
    assert by_name["FOMC"].forecast_value == "hold"
    # An event that already printed does report its value.
    assert by_name["Claims"].released_value == "220k"


def test_calendar_reports_no_coverage_rather_than_guessing():
    """A dataset that starts after the query time is not evidence about it."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-06-10 18:00"], utc=True),
            "name": ["FOMC"],
            "importance": ["HIGH"],
        }
    )
    result = CalendarPitStore(frame).query_events(
        datetime(2025, 6, 1, 15, 0, tzinfo=UTC), 300
    )
    assert not result.available
    assert "does not cover" in result.reason


# --- bundle -----------------------------------------------------------------
def test_bundle_reports_all_three_as_unavailable_when_nothing_is_supplied():
    bundle, frames = inspect_all(None, None, None, START, END)
    assert set(bundle.unavailable_names()) == {"news", "sentiment", "economic_calendar"}
    assert all(frame is None for frame in frames.values())
    versions = bundle.dataset_versions()
    assert all(not version.available for version in versions)
    assert all(version.note for version in versions)


def test_bundle_summary_rows_state_every_required_attribute(tmp_path):
    path = write_csv(tmp_path / "news.csv", news_rows(60))
    bundle, _ = inspect_all(path, None, None, START, END)
    row = next(r for r in bundle.summary_rows() if r["dataset"] == "news")
    for key in (
        "availability",
        "source",
        "rows",
        "coverage",
        "precision",
        "provenance",
        "rows_excluded",
        "gaps",
    ):
        assert key in row


def test_candle_report_is_unavailable_without_a_file():
    report = candle_dataset_report(None, 15)
    assert report.availability == Availability.UNAVAILABLE
    assert "No synthetic series is substituted" in report.notes[0]
