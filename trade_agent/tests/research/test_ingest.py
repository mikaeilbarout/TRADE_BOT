from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

from research.data.ingest.base import (
    INADMISSIBLE_PROVENANCE,
    CalendarRecord,
    FetchWindow,
    NewsRecord,
    Provenance,
    SentimentRecord,
    SourceUnavailable,
    TimePrecision,
    now_utc,
)
from research.data.ingest.cache import ArtifactCache
from research.data.ingest.checkpoint import (
    STATUS_DONE,
    STATUS_EMPTY,
    STATUS_FAILED,
    IngestCheckpoint,
    WindowProgress,
)
from research.data.ingest.fred import (
    DEFAULT_SERIES,
    OUTPUT_INITIAL_RELEASE_ONLY,
    OUTPUT_NEW_AND_REVISED_ONLY,
    FredCalendarSource,
    ReleaseTimePolicy,
)
from research.data.ingest.gdelt import (
    COL_DATE,
    COL_DOCUMENT_IDENTIFIER,
    COL_RECORD_ID,
    COL_SOURCE_COMMON_NAME,
    COL_V15_TONE,
    COL_V1_THEMES,
    GDELT_2_START,
    GKG_EXPECTED_COLUMNS,
    GdeltGkgSource,
    GkgLayoutError,
    gkg_url,
    headline_from_url,
    parse_tone,
    score_relevance,
    validate_gkg_row,
)
from research.data.ingest.normalize import (
    normalise_calendar,
    normalise_news,
    read_dataset,
    to_frame,
    write_dataset,
)
from research.data.ingest.pipeline import IngestPipeline, coverage_gaps
from research.data.ingest.registry import IMPLEMENTED_SOURCES, hosts_required, required_keys
from research.data.ingest.sentiment import (
    PointInTimeToneBuilder,
    RetrospectiveLlmBuilder,
    split_by_provenance,
)

UTC = timezone.utc


# --- fixtures --------------------------------------------------------------
def gkg_row(
    timestamp: str = "20230510143200",
    domain: str = "reuters.com",
    url: str = "https://www.reuters.com/markets/gold-climbs-on-fed-rate-cut-bets-2023-05-10/",
    themes: str = "ECON_INFLATION;ECON_INTEREST_RATE",
    tone: str = "-2.5,1.0,3.5,4.5,20.0,1.0,0",
    record_id: str = "20230510143200-1",
) -> list[str]:
    """A GKG 2.1 row shaped as the published codebook describes."""
    row = [""] * GKG_EXPECTED_COLUMNS
    row[COL_RECORD_ID] = record_id
    row[COL_DATE] = timestamp
    row[COL_SOURCE_COMMON_NAME] = domain
    row[COL_DOCUMENT_IDENTIFIER] = url
    row[COL_V1_THEMES] = themes
    row[COL_V15_TONE] = tone
    return row


def gkg_zip(rows: list[list[str]]) -> bytes:
    payload = "\n".join("\t".join(row) for row in rows).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("20230510144500.gkg.csv", payload)
    return buffer.getvalue()


def transport(routes: dict[str, tuple[int, bytes]]) -> httpx.Client:
    """A client that serves canned bytes, so nothing touches the network."""

    def handler(request: httpx.Request) -> httpx.Response:
        for url, (status, body) in routes.items():
            if str(request.url) == url:
                return httpx.Response(status, content=body)
        return httpx.Response(404, content=b"")

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- availability semantics ------------------------------------------------
def test_available_at_is_the_later_of_publication_and_discovery():
    """An aggregator's ingest lag must never become a head start."""
    record = NewsRecord(
        source="r.com",
        source_id="1",
        headline="Gold rises",
        published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC),
        discovered_at=datetime(2023, 5, 10, 14, 45, tzinfo=UTC),
        retrieved_at=now_utc(),
        provenance=Provenance.ORIGINAL_RELEASE,
    )
    assert record.available_at == datetime(2023, 5, 10, 14, 45, tzinfo=UTC)
    assert record.available_at >= record.published_at


def test_available_at_falls_back_to_publication_when_undiscovered():
    record = NewsRecord(
        source="r.com", source_id="1", headline="H",
        published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC),
        retrieved_at=now_utc(),
    )
    assert record.available_at == record.published_at


def test_naive_timestamps_are_treated_as_utc():
    record = NewsRecord(
        source="r.com", source_id="1", headline="H",
        published_at=datetime(2023, 5, 10, 14, 32),
        retrieved_at=datetime(2026, 1, 1),
    )
    assert record.published_at.tzinfo is not None
    assert record.published_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "provenance,admissible",
    [
        (Provenance.ORIGINAL_RELEASE, True),
        (Provenance.POINT_IN_TIME_CAPTURE, True),
        (Provenance.REVISED, False),
        (Provenance.RETROSPECTIVE, False),
    ],
)
def test_admissibility_follows_provenance(provenance, admissible):
    record = NewsRecord(
        source="s", source_id="1", headline="H",
        published_at=datetime(2023, 1, 1, tzinfo=UTC),
        retrieved_at=now_utc(), provenance=provenance,
    )
    assert record.admissible is admissible


def test_retrospective_labels_are_both_inadmissible():
    """The current label and the older alias must both be refused."""
    assert "RETROSPECTIVE" in INADMISSIBLE_PROVENANCE
    assert "RETROSPECTIVE_SCORING" in INADMISSIBLE_PROVENANCE
    assert "REVISED" in INADMISSIBLE_PROVENANCE


# --- cache -----------------------------------------------------------------
def test_cache_path_is_deterministic(tmp_path):
    cache = ArtifactCache(tmp_path)
    url = "http://data.gdeltproject.org/gdeltv2/20230510144500.gkg.csv.zip"
    assert cache.path_for(url, ".zip") == cache.path_for(url, ".zip")
    assert cache.path_for(url, ".zip") != cache.path_for(url + "x", ".zip")


def test_cache_downloads_once_then_serves_from_disk(tmp_path):
    url = "http://example.test/a.zip"
    body = b"payload"
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cache = ArtifactCache(tmp_path, client=client)

    first = cache.fetch(url, ".zip")
    second = cache.fetch(url, ".zip")

    assert calls["n"] == 1  # never re-downloaded
    assert first.from_cache is False and second.from_cache is True
    assert second.sha256 == hashlib.sha256(body).hexdigest()


def test_cache_refuses_a_payload_failing_the_provider_checksum(tmp_path):
    url = "http://example.test/a.zip"
    client = transport({url: (200, b"tampered")})
    cache = ArtifactCache(tmp_path, client=client)

    with pytest.raises(SourceUnavailable, match="checksum mismatch"):
        cache.fetch(url, ".zip", expected_checksum=hashlib.md5(b"original").hexdigest())
    assert cache.peek(url, ".zip") is None  # nothing was stored


def test_cache_verify_detects_tampering(tmp_path):
    url = "http://example.test/a.zip"
    cache = ArtifactCache(tmp_path, client=transport({url: (200, b"good")}))
    artifact = cache.fetch(url, ".zip")
    artifact.path.write_bytes(b"edited by hand")
    problems = cache.verify()
    assert problems and "on-disk bytes hash to" in problems[0]


def test_blocked_host_raises_rather_than_returning_nothing(tmp_path):
    """A 403 must never look like 'there was no news that day'."""
    url = "http://example.test/a.zip"
    cache = ArtifactCache(tmp_path, client=transport({url: (403, b"denied")}))
    with pytest.raises(SourceUnavailable, match="egress policy|rejected the request"):
        cache.fetch(url, ".zip")


def test_missing_file_is_allowed_when_the_source_has_gaps(tmp_path):
    cache = ArtifactCache(tmp_path, client=transport({}))
    assert cache.fetch("http://example.test/absent.zip", ".zip", allow_missing=True) is None


# --- GDELT -----------------------------------------------------------------
def test_gkg_url_requires_a_quarter_hour_boundary():
    assert gkg_url(datetime(2023, 5, 10, 14, 45, tzinfo=UTC)).endswith(
        "20230510144500.gkg.csv.zip"
    )
    with pytest.raises(ValueError, match="15-minute boundaries"):
        gkg_url(datetime(2023, 5, 10, 14, 47, tzinfo=UTC))


def test_layout_check_rejects_a_wrong_column_map():
    """The guard that turns a mis-indexed parser into a loud failure."""
    assert validate_gkg_row(gkg_row()) is None
    short = gkg_row()[:10]
    assert "expected 27 columns" in validate_gkg_row(short)
    bad_date = gkg_row(timestamp="not-a-date")
    assert "YYYYMMDDHHMMSS" in validate_gkg_row(bad_date)
    bad_tone = gkg_row(tone="1.0,2.0")
    assert "7-field tone vector" in validate_gkg_row(bad_tone)


def test_a_file_that_fails_the_layout_check_raises(tmp_path):
    url = gkg_url(datetime(2023, 5, 10, 14, 45, tzinfo=UTC))
    garbage = gkg_zip([["only", "three", "columns"]])
    source = GdeltGkgSource(ArtifactCache(tmp_path, client=transport({url: (200, garbage)})))
    result = source.fetch_window(
        FetchWindow("x", datetime(2023, 5, 10, 14, 45, tzinfo=UTC),
                    datetime(2023, 5, 10, 15, 0, tzinfo=UTC))
    )
    assert result.failed
    assert "layout check" in result.error
    assert "codebook" in result.error


def test_headline_is_derived_from_the_url_and_says_so():
    headline, how = headline_from_url(
        "https://www.reuters.com/markets/gold-climbs-on-fed-rate-cut-bets-2023-05-10/"
    )
    assert "gold" in headline.lower()
    assert how == "derived_from_url_slug"
    # No usable slug: the URL is kept rather than words being invented.
    headline, how = headline_from_url("https://x.test/a/12345678.html")
    assert how == "url_only"


def test_relevance_filter_keeps_gold_macro_and_drops_the_rest():
    high, terms = score_relevance("Gold climbs as Fed rate cut bets grow", "ECON_INFLATION")
    low, _ = score_relevance("Local bakery wins award for sourdough", "")
    assert high > 2.0 and low < 2.0
    assert "gold" in terms


def test_gdelt_window_yields_records_with_full_provenance(tmp_path):
    slot = datetime(2023, 5, 10, 14, 45, tzinfo=UTC)
    url = gkg_url(slot)
    # The irrelevant row carries NO theme codes: GDELT's own economic themes are
    # a stronger relevance signal than the URL slug, so an article tagged
    # ECON_INFLATION is kept whatever its slug looks like.
    body = gkg_zip([
        gkg_row(),
        gkg_row(record_id="20230510143300-2",
                url="https://x.test/cat-video-here-today", themes=""),
    ])
    source = GdeltGkgSource(ArtifactCache(tmp_path, client=transport({url: (200, body)})))
    result = source.fetch_window(FetchWindow("k", slot, slot + timedelta(minutes=15)))

    assert result.ok and len(result.records) == 1  # the cat video is filtered out
    record = result.records[0]
    assert record.source == "reuters.com"
    assert record.provenance == Provenance.ORIGINAL_RELEASE
    assert record.published_at == datetime(2023, 5, 10, 14, 32, tzinfo=UTC)
    assert record.discovered_at == slot
    assert record.available_at == slot
    assert record.source_checksum and record.source_artifact
    assert record.matched_terms


def test_provider_theme_codes_outrank_the_url_slug(tmp_path):
    """GDELT's own classification is a stronger signal than a URL slug.

    An article whose slug says nothing about markets but which GDELT tagged
    ECON_INFLATION is macro-relevant, and is kept.
    """
    slot = datetime(2023, 5, 10, 14, 45, tzinfo=UTC)
    url = gkg_url(slot)
    body = gkg_zip([
        gkg_row(url="https://x.test/some-opaque-permalink-abc",
                themes="ECON_INFLATION;ECON_INTEREST_RATE"),
    ])
    source = GdeltGkgSource(ArtifactCache(tmp_path, client=transport({url: (200, body)})))
    result = source.fetch_window(FetchWindow("k", slot, slot + timedelta(minutes=15)))
    assert len(result.records) == 1
    assert any(term.startswith("theme:") for term in result.records[0].matched_terms)


def test_gdelt_reports_a_missing_slot_as_empty_not_failed(tmp_path):
    slot = datetime(2023, 5, 10, 14, 45, tzinfo=UTC)
    source = GdeltGkgSource(ArtifactCache(tmp_path, client=transport({})))
    result = source.fetch_window(FetchWindow("k", slot, slot + timedelta(minutes=15)))
    assert result.empty and result.ok and not result.records


def test_gdelt_windows_are_quarter_hourly_and_clamped_to_coverage():
    source = GdeltGkgSource(ArtifactCache(Path("/tmp")))
    windows = source.windows(
        datetime(2023, 5, 10, 14, 0, tzinfo=UTC), datetime(2023, 5, 10, 15, 0, tzinfo=UTC)
    )
    assert [w.key for w in windows] == [
        "20230510140000", "20230510141500", "20230510143000", "20230510144500",
    ]
    # A range starting before GDELT 2.0 is clamped, and the shortfall reported.
    early = source.windows(datetime(2010, 1, 1, tzinfo=UTC), GDELT_2_START + timedelta(hours=1))
    assert all(w.start >= GDELT_2_START for w in early)
    assert "UNAVAILABLE" in source.coverage_shortfall(
        datetime(2010, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)
    )


def test_gdelt_tone_is_point_in_time_and_from_the_same_file(tmp_path):
    slot = datetime(2023, 5, 10, 14, 45, tzinfo=UTC)
    url = gkg_url(slot)
    body = gkg_zip([gkg_row(tone="-4.0,1.0,5.0,6.0,20.0,1.0,0"), gkg_row(record_id="b")])
    source = GdeltGkgSource(ArtifactCache(tmp_path, client=transport({url: (200, body)})))
    result = source.sentiment_from_window(FetchWindow("k", slot, slot + timedelta(minutes=15)))

    record = result.records[0]
    assert record.provenance == Provenance.POINT_IN_TIME_CAPTURE
    assert record.published_at == slot
    assert record.article_count == 2
    assert -1.0 <= record.value <= 1.0
    assert record.model is None  # no model produced this


# --- FRED ------------------------------------------------------------------
def test_fred_requires_a_key_and_says_how_to_get_one(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key=None)
    message = source.preflight()
    assert "FRED_API_KEY" in message
    assert "fredaccount.stlouisfed.org" in message
    assert "no cost" in message


def test_fred_requests_both_vintage_streams(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="K")
    windows = source.windows(
        datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
    )
    keys = {w.key for w in windows}
    assert "CPIAUCSL:2023:initial" in keys
    assert "CPIAUCSL:2023:revised" in keys

    initial = source._observations_url(DEFAULT_SERIES[0], windows[0], OUTPUT_INITIAL_RELEASE_ONLY)
    revised = source._observations_url(DEFAULT_SERIES[0], windows[0], OUTPUT_NEW_AND_REVISED_ONLY)
    assert "output_type=4" in initial   # initial release only
    assert "output_type=3" in revised   # new and revised only
    # The window's own start, not FRED's 1776-07-04 archive floor: a value
    # for this window can't have been revised before the window itself
    # starts, and the archive floor hits ALFRED's 2000-vintage-date cap for
    # any daily series spanning more than a few years (confirmed against the
    # live API: DFF/DGS10/DGS2 all hit it with the old floor).
    assert "realtime_start=2023-01-01" in initial
    assert "realtime_end=9999-12-31" in initial  # still open-ended going forward


def test_fred_key_is_never_written_into_a_record(tmp_path):
    """A secret in the URL must not reach the dataset."""
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="SUPERSECRET")
    url = source._observations_url(
        DEFAULT_SERIES[0],
        FetchWindow("k", datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
        4,
    )
    assert "SUPERSECRET" in url  # it is needed in the request...
    artifact = type("A", (), {"relative_name": "x.json", "sha256": "abc", "retrieved_at": now_utc()})()
    records = source._to_records(
        DEFAULT_SERIES[0],
        [{"date": "2023-05-01", "value": "301.8", "realtime_start": "2023-06-13"}],
        "initial",
        artifact,
    )
    serialised = json.dumps([r.to_row() for r in records], default=str)
    assert "SUPERSECRET" not in serialised  # ...and never in the data


def test_initial_and_revised_become_separate_records(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="K")
    artifact = type("A", (), {"relative_name": "x.json", "sha256": "abc", "retrieved_at": now_utc()})()
    observations = [{"date": "2023-05-01", "value": "301.8", "realtime_start": "2023-06-13"}]

    initial = source._to_records(DEFAULT_SERIES[0], observations, "initial", artifact)[0]
    revised = source._to_records(
        DEFAULT_SERIES[0],
        [{"date": "2023-05-01", "value": "302.1", "realtime_start": "2023-07-12"}],
        "revised",
        artifact,
    )[0]

    assert initial.provenance == Provenance.ORIGINAL_RELEASE
    assert initial.original_release is True
    assert initial.revision_timestamp is None
    assert revised.provenance == Provenance.REVISED
    assert revised.original_release is False
    assert revised.revision_timestamp is not None
    assert revised.available_at > initial.available_at
    # Two records, not one merged value.
    kept, stats = normalise_calendar([initial, revised])
    assert len(kept) == 2
    assert stats["original_releases"] == 1 and stats["revisions"] == 1


def test_fred_records_that_no_forecast_is_available(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="K")
    artifact = type("A", (), {"relative_name": "x", "sha256": "a", "retrieved_at": now_utc()})()
    record = source._to_records(
        DEFAULT_SERIES[0],
        [{"date": "2023-05-01", "value": "301.8", "realtime_start": "2023-06-13"}],
        "initial",
        artifact,
    )[0]
    assert record.forecast is None
    assert "not consensus forecasts" in record.forecast_unavailable_reason


def test_previous_value_comes_from_the_preceding_published_observation(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="K")
    artifact = type("A", (), {"relative_name": "x", "sha256": "a", "retrieved_at": now_utc()})()
    records = source._to_records(
        DEFAULT_SERIES[0],
        [
            {"date": "2023-04-01", "value": "300.0", "realtime_start": "2023-05-10"},
            {"date": "2023-05-01", "value": "301.8", "realtime_start": "2023-06-13"},
        ],
        "initial",
        artifact,
    )
    assert records[0].previous is None
    assert records[1].previous == "300.0"
    assert records[1].previous_basis  # how it was obtained is stated


def test_missing_observations_are_skipped_not_zeroed(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="K")
    artifact = type("A", (), {"relative_name": "x", "sha256": "a", "retrieved_at": now_utc()})()
    records = source._to_records(
        DEFAULT_SERIES[0],
        [{"date": "2023-05-01", "value": ".", "realtime_start": "2023-06-13"}],
        "initial",
        artifact,
    )
    assert records == []


@pytest.mark.parametrize("month,expected_hour", [(1, 13), (7, 12)])
def test_scheduled_local_policy_handles_dst(tmp_path, month, expected_hour):
    """08:30 New York is 13:30 UTC in winter and 12:30 UTC in summer."""
    source = FredCalendarSource(
        ArtifactCache(tmp_path), api_key="K",
        release_time_policy=ReleaseTimePolicy.SCHEDULED_LOCAL,
    )
    moment, precision = source._availability(DEFAULT_SERIES[0], date(2023, month, 13))
    assert moment.hour == expected_hour
    assert precision == TimePrecision.IMPUTED_FROM_SCHEDULE


def test_end_of_day_policy_is_the_leakage_safe_default(tmp_path):
    """The default must never reveal a figure before its release day ends."""
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="K")
    moment, precision = source._availability(DEFAULT_SERIES[0], date(2023, 6, 13))
    assert moment == datetime(2023, 6, 13, 23, 59, 59, tzinfo=UTC)
    assert precision == TimePrecision.DATE_ONLY
    # A signal at 10:00 on release day cannot see it.
    assert moment > datetime(2023, 6, 13, 10, 0, tzinfo=UTC)


def test_fred_reports_an_error_object_rather_than_parsing_it(tmp_path):
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key="BAD")
    window = source.windows(
        datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
    )[0]
    url = source._observations_url(DEFAULT_SERIES[0], window, OUTPUT_INITIAL_RELEASE_ONLY)
    body = json.dumps({"error_code": 400, "error_message": "Bad Request. Invalid api_key"}).encode()
    source = FredCalendarSource(
        ArtifactCache(tmp_path, client=transport({url: (200, body)})), api_key="BAD"
    )
    result = source.fetch_window(window)
    assert result.failed
    assert "no 'observations' key" in result.error


# --- deduplication ---------------------------------------------------------
def test_duplicate_news_keeps_the_earliest_availability():
    late = NewsRecord(
        source="r.com", source_id="a", headline="Gold up", url="https://r.com/x",
        published_at=datetime(2023, 5, 10, 15, 0, tzinfo=UTC), retrieved_at=now_utc(),
        provenance=Provenance.ORIGINAL_RELEASE,
    )
    early = late.model_copy(
        update={"source_id": "b", "published_at": datetime(2023, 5, 10, 14, 32, tzinfo=UTC)}
    )
    kept, stats = normalise_news([late, early])
    assert len(kept) == 1 and stats["duplicates_removed"] == 1
    assert kept[0].available_at == datetime(2023, 5, 10, 14, 32, tzinfo=UTC)


def test_the_same_story_on_two_sites_is_not_a_duplicate():
    base = dict(
        headline="Gold up", published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC),
        retrieved_at=now_utc(), provenance=Provenance.ORIGINAL_RELEASE,
    )
    a = NewsRecord(source="r.com", source_id="1", url="https://r.com/x", **base)
    b = NewsRecord(source="b.com", source_id="2", url="https://b.com/x", **base)
    kept, _ = normalise_news([a, b])
    assert len(kept) == 2


# --- sentiment provenance --------------------------------------------------
def test_point_in_time_builder_refuses_retrospective_records():
    record = SentimentRecord(
        source="llm", source_id="x", value=0.4, method="llm",
        published_at=datetime(2023, 5, 10, tzinfo=UTC), retrieved_at=now_utc(),
        provenance=Provenance.RETROSPECTIVE,
    )
    with pytest.raises(ValueError, match="POINT_IN_TIME_CAPTURE"):
        PointInTimeToneBuilder().from_tone_records([record])


def test_retrospective_builder_cannot_emit_point_in_time():
    """The label is a class attribute, not a caller-supplied argument."""
    news = [
        NewsRecord(
            source="r.com", source_id=f"n{i}", headline=f"Gold headline {i}",
            published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC), retrieved_at=now_utc(),
            provenance=Provenance.ORIGINAL_RELEASE,
        )
        for i in range(3)
    ]
    builder = RetrospectiveLlmBuilder()
    batches = builder.batches(news)
    builder.store_scores(
        {batches[0].custom_id: {"score": 40, "confidence": 60, "model": "claude-haiku-4-5"}}
    )
    records = builder.to_records(batches)
    assert records[0].provenance == Provenance.RETROSPECTIVE
    assert records[0].model == "claude-haiku-4-5"
    assert not records[0].admissible


def test_retrospective_scoring_is_cached_so_a_rerun_costs_nothing(tmp_path):
    news = [
        NewsRecord(
            source="r.com", source_id="n1", headline="Gold rises on Fed bets",
            published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC), retrieved_at=now_utc(),
        )
    ]
    builder = RetrospectiveLlmBuilder(cache_path=tmp_path / "scores.json")
    batches = builder.batches(news)
    assert len(builder.pending(batches)) == 1
    first_estimate = builder.cost_estimate(batches)
    builder.store_scores({batches[0].custom_id: {"score": 10, "confidence": 50}})

    # A second builder over the same cache re-reads it from disk.
    again = RetrospectiveLlmBuilder(cache_path=tmp_path / "scores.json")
    assert again.pending(again.batches(news)) == []
    assert again.cost_estimate(again.batches(news))["estimated_usd"] == 0.0
    assert first_estimate["batches_pending"] == 1


def test_batching_is_one_call_per_slot_not_per_article():
    news = [
        NewsRecord(
            source="r.com", source_id=f"n{i}", headline=f"Gold story {i}",
            published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC) + timedelta(seconds=i),
            retrieved_at=now_utc(),
        )
        for i in range(12)
    ]
    batches = RetrospectiveLlmBuilder().batches(news)
    assert len(batches) == 1
    assert len(batches[0].headlines) == 12


def test_sentiment_datasets_are_split_by_provenance():
    common = dict(
        source_id="x", value=0.1, method="m",
        published_at=datetime(2023, 5, 10, tzinfo=UTC), retrieved_at=now_utc(),
    )
    pit = SentimentRecord(source="gdelt", provenance=Provenance.POINT_IN_TIME_CAPTURE, **common)
    retro = SentimentRecord(source="llm", provenance=Provenance.RETROSPECTIVE, **common)
    grouped = split_by_provenance([pit, retro])
    assert set(grouped) == {"POINT_IN_TIME_CAPTURE", "RETROSPECTIVE"}


# --- resumability ----------------------------------------------------------
class StubSource:
    """A source that fails once then succeeds, to exercise resume."""

    spec = IMPLEMENTED_SOURCES[0]

    def __init__(self, fail_keys: set[str]) -> None:
        self._fail = fail_keys
        self.fetched: list[str] = []

    def windows(self, start, end):
        return [
            FetchWindow(f"w{i}", start + timedelta(minutes=15 * i),
                        start + timedelta(minutes=15 * (i + 1)))
            for i in range(4)
        ]

    def preflight(self):
        return None

    def fetch_window(self, window):
        from research.data.ingest.base import FetchResult

        self.fetched.append(window.key)
        if window.key in self._fail:
            return FetchResult(window=window, failed=True, error="transient")
        return FetchResult(
            window=window,
            records=[
                NewsRecord(
                    source="s.com", source_id=window.key, headline=f"Gold {window.key}",
                    published_at=window.start, discovered_at=window.start,
                    retrieved_at=now_utc(), provenance=Provenance.ORIGINAL_RELEASE,
                )
            ],
        )


def test_a_resumed_run_skips_settled_windows_and_retries_failures(tmp_path):
    start = datetime(2023, 5, 10, 14, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    checkpoint = IngestCheckpoint(tmp_path / "cp.sqlite")
    dataset = tmp_path / "news.parquet"

    first_source = StubSource(fail_keys={"w2"})
    first = IngestPipeline(first_source, checkpoint, "news", requests_per_second=0).run(
        start, end, dataset
    )
    assert first.windows_fetched == 3 and first.windows_failed == 1
    assert len(first_source.fetched) == 4

    second_source = StubSource(fail_keys=set())
    second = IngestPipeline(second_source, checkpoint, "news", requests_per_second=0).run(
        start, end, dataset
    )
    # Only the failed window is retried; the three settled ones are not.
    assert second_source.fetched == ["w2"]
    assert second.windows_settled_before == 3
    assert second.windows_fetched == 1

    frame = read_dataset(dataset)
    assert len(frame) == 4  # the resumed run converged on the full dataset
    assert frame["timestamp"].is_monotonic_increasing


def test_an_empty_window_is_settled_and_never_refetched(tmp_path):
    checkpoint = IngestCheckpoint(tmp_path / "cp.sqlite")
    checkpoint.record(
        WindowProgress(
            source="gdelt_gkg", window_key="quiet", kind="news", status=STATUS_EMPTY,
            window_start=datetime(2023, 1, 1, tzinfo=UTC),
            window_end=datetime(2023, 1, 1, 0, 15, tzinfo=UTC),
        )
    )
    assert "quiet" in checkpoint.settled_windows("gdelt_gkg", "news")


def test_failed_windows_are_reported_as_gaps(tmp_path):
    checkpoint = IngestCheckpoint(tmp_path / "cp.sqlite")
    checkpoint.record(
        WindowProgress(
            source="gdelt_gkg", window_key="broken", kind="news", status=STATUS_FAILED,
            window_start=datetime(2023, 3, 1, tzinfo=UTC),
            window_end=datetime(2023, 3, 1, 0, 15, tzinfo=UTC), error="503",
        )
    )
    assert checkpoint.gaps("gdelt_gkg", "news")


def test_source_level_failure_stops_the_run_with_its_reason(tmp_path):
    class Blocked(StubSource):
        def fetch_window(self, window):
            raise SourceUnavailable("host not in allowlist: data.gdeltproject.org")

    outcome = IngestPipeline(
        Blocked(set()), IngestCheckpoint(tmp_path / "cp.sqlite"), "news",
        requests_per_second=0,
    ).run(datetime(2023, 5, 10, tzinfo=UTC), datetime(2023, 5, 10, 1, 0, tzinfo=UTC),
          tmp_path / "n.parquet")
    assert outcome.stopped_early
    assert "allowlist" in outcome.stopped_reason


def test_missing_key_stops_before_any_request(tmp_path):
    class NoKey(StubSource):
        def preflight(self):
            return "FRED_API_KEY is not set"

    source = NoKey(set())
    outcome = IngestPipeline(
        source, IngestCheckpoint(tmp_path / "cp.sqlite"), "calendar", requests_per_second=0
    ).run(datetime(2023, 5, 10, tzinfo=UTC), datetime(2023, 5, 10, 1, 0, tzinfo=UTC),
          tmp_path / "c.parquet")
    assert outcome.stopped_early and source.fetched == []


# --- gaps and reproducibility ---------------------------------------------
def test_coverage_gaps_report_missing_ranges():
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2023-01-01T00:00Z", "2023-03-01T00:00Z"])}
    )
    gaps = coverage_gaps(
        frame, datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 3, 2, tzinfo=UTC),
        max_gap_hours=48,
    )
    assert gaps and gaps[0]["hours"] > 48


def test_empty_dataset_reports_the_whole_range_as_missing():
    gaps = coverage_gaps(
        pd.DataFrame(), datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 2, 1, tzinfo=UTC)
    )
    assert gaps[0]["reason"] == "no records at all"


def test_dataset_round_trips_preserving_utc(tmp_path):
    records = [
        NewsRecord(
            source="r.com", source_id="1", headline="Gold up",
            published_at=datetime(2023, 5, 10, 14, 32, tzinfo=UTC),
            discovered_at=datetime(2023, 5, 10, 14, 45, tzinfo=UTC),
            retrieved_at=now_utc(), provenance=Provenance.ORIGINAL_RELEASE,
        )
    ]
    path = write_dataset(to_frame(records), tmp_path / "news.parquet")
    frame = read_dataset(path)
    assert frame["timestamp"].dt.tz is not None
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2023-05-10T14:45Z")
    assert frame["provenance"].iloc[0] == "ORIGINAL_RELEASE"
    assert frame["source_checksum"].isna().all() or True  # column present


def test_registry_names_the_hosts_and_keys_a_person_must_provide():
    assert "data.gdeltproject.org" in hosts_required()
    assert "api.stlouisfed.org" in hosts_required()
    keys = {k["env_var"] for k in required_keys()}
    assert keys == {"FRED_API_KEY"}  # GDELT needs none


def test_no_api_key_value_is_ever_stored_in_the_registry_output():
    for entry in required_keys():
        assert set(entry) >= {"env_var", "present"}
        assert "value" not in entry
        assert isinstance(entry["present"], bool)


# --- secret redaction ------------------------------------------------------
def test_redaction_masks_credentialed_query_parameters():
    from research.data.ingest.redact import contains_secret, redact_url

    url = (
        "https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL"
        "&api_key=deadbeefdeadbeefdeadbeefdeadbeef&file_type=json"
    )
    redacted = redact_url(url)
    assert "deadbeef" not in redacted
    assert "api_key=<REDACTED>" in redacted
    # The shape is preserved, so the message still aids diagnosis.
    assert "series_id=CPIAUCSL" in redacted
    assert contains_secret(url) and not contains_secret(redacted)


@pytest.mark.parametrize(
    "param", ["api_key", "apikey", "token", "access_token", "password", "secret", "key"]
)
def test_every_credential_parameter_name_is_covered(param):
    from research.data.ingest.redact import redact

    assert "s3cr3t" not in redact(f"https://x.test/a?{param}=s3cr3t&b=1")


def test_a_cache_error_never_contains_the_key(tmp_path):
    """The failure path is where a URL most often becomes a log line."""
    secret = "deadbeefdeadbeefdeadbeefdeadbeef"
    url = f"https://api.test/data?api_key={secret}"
    cache = ArtifactCache(tmp_path, client=transport({url: (403, b"denied")}))
    with pytest.raises(SourceUnavailable) as error:
        cache.fetch(url, ".json")
    assert secret not in str(error.value)
    assert "<REDACTED>" in str(error.value)


def test_cache_metadata_on_disk_never_contains_the_key(tmp_path):
    """The sidecar file is written to disk and read back by reports."""
    secret = "deadbeefdeadbeefdeadbeefdeadbeef"
    url = f"https://api.test/data?api_key={secret}"
    cache = ArtifactCache(tmp_path, client=transport({url: (200, b'{"observations":[]}')}))
    artifact = cache.fetch(url, ".json")

    meta = cache.meta_path(artifact.path).read_text()
    assert secret not in meta
    assert "<REDACTED>" in meta
    # ...and the cache still resolves the same URL from disk.
    assert cache.peek(url, ".json") is not None


def test_a_fred_failure_message_is_redacted(tmp_path):
    secret = "deadbeefdeadbeefdeadbeefdeadbeef"
    source = FredCalendarSource(ArtifactCache(tmp_path), api_key=secret)
    window = source.windows(
        datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
    )[0]
    url = source._observations_url(DEFAULT_SERIES[0], window, OUTPUT_INITIAL_RELEASE_ONLY)
    body = json.dumps({"error_code": 400, "error_message": "Bad Request"}).encode()
    source = FredCalendarSource(
        ArtifactCache(tmp_path, client=transport({url: (200, body)})), api_key=secret
    )
    result = source.fetch_window(window)
    assert result.failed
    assert secret not in (result.error or "")
    assert "<REDACTED>" in (result.error or "")


def test_the_checkpoint_never_persists_an_unredacted_key(tmp_path):
    """The checkpoint is durable and is read back by the status report."""
    secret = "deadbeefdeadbeefdeadbeefdeadbeef"

    class Leaky(StubSource):
        def fetch_window(self, window):
            from research.data.ingest.base import FetchResult

            self.fetched.append(window.key)
            return FetchResult(
                window=window,
                failed=True,
                error=f"https://api.test/x?api_key={secret} returned 403",
            )

    checkpoint = IngestCheckpoint(tmp_path / "cp.sqlite")
    outcome = IngestPipeline(Leaky(set()), checkpoint, "calendar", requests_per_second=0).run(
        datetime(2023, 5, 10, tzinfo=UTC),
        datetime(2023, 5, 10, 1, 0, tzinfo=UTC),
        tmp_path / "c.parquet",
    )
    stored = checkpoint.failed_windows(StubSource.spec.key)
    assert stored
    for window in stored:
        assert secret not in (window.error or "")
        assert "<REDACTED>" in (window.error or "")
    # The in-memory outcome, which reaches the JSON reports, is clean too.
    assert all(secret not in message for message in outcome.errors)
    # And the raw bytes on disk carry no secret.
    assert secret.encode() not in (tmp_path / "cp.sqlite").read_bytes()


def test_env_file_is_loaded_but_an_exported_value_wins(tmp_path, monkeypatch):
    from research.data.ingest.env import describe_presence, load_env_file

    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nFRED_API_KEY=from-file\nexport OTHER_KEY='quoted-value'\nBAD_LINE\n"
    )
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)

    applied = load_env_file(env)
    assert set(applied) == {"FRED_API_KEY", "OTHER_KEY"}
    assert os.environ["FRED_API_KEY"] == "from-file"
    assert os.environ["OTHER_KEY"] == "quoted-value"  # quotes stripped
    assert describe_presence(["FRED_API_KEY"]) == {"FRED_API_KEY": True}

    # An explicitly exported value is a deliberate override and must survive.
    monkeypatch.setenv("FRED_API_KEY", "from-environment")
    load_env_file(env)
    assert os.environ["FRED_API_KEY"] == "from-environment"


def test_describe_presence_never_returns_a_value(monkeypatch):
    from research.data.ingest.env import describe_presence

    monkeypatch.setenv("FRED_API_KEY", "super-secret-value")
    result = describe_presence(["FRED_API_KEY"])
    assert result == {"FRED_API_KEY": True}
    assert "super-secret-value" not in json.dumps(result)
