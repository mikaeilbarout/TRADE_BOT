from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from zoneinfo import ZoneInfo

import httpx

from research.data.ingest.base import (
    CalendarRecord,
    FetchResult,
    FetchWindow,
    HistoricalSource,
    Provenance,
    SourceCost,
    SourceSpec,
    SourceUnavailable,
    TimePrecision,
    now_utc,
    stable_id,
)
from research.data.ingest.cache import ArtifactCache
from research.data.ingest.redact import redact_url

"""Historical economic calendar from FRED / ALFRED, vintage-aware.

Why this source. The hard requirement on the calendar is that a backtest must
see the figure that was public at the time, never a later revision. Almost no
free calendar feed preserves that distinction -- most serve the current value
with a historical date attached, which is precisely the leak. FRED's archival
side (ALFRED) does preserve it, and exposes it directly:

* `output_type=4` -- "Observations, Initial Release Only". The first published
  value for each period, with `realtime_start` = the date it became available.
  These become `provenance=ORIGINAL_RELEASE`.
* `output_type=3` -- "New and Revised Observations Only". Later corrections.
  These become `provenance=REVISED`, stored as SEPARATE records, and never
  served as the value known at the original release.

Both are fetched, so the dataset can answer "what was published then" and "what
do we know now" without ever confusing the two.

The honest limitation, stated up front because it shapes how the data may be
used: **FRED's real-time fields are dates, not timestamps.** ALFRED records that
CPI for May became available on 2023-06-13; it does not record 08:30. A
15-minute backtest needs a clock time, and there are only two defensible
options, both provided and both labelled:

* `END_OF_DAY` (default) -- the release becomes available at 23:59:59 UTC on its
  release date. Never reveals a figure before it was published, at the cost of
  hiding it for the rest of the release day. Leakage-safe.
* `SCHEDULED_LOCAL` -- the release becomes available at its publisher's
  long-standing scheduled clock time (08:30 America/New_York for BLS releases,
  14:00 for FOMC), converted to UTC with DST handled. Realistic, but an
  assumption about the schedule rather than an observation, so every such record
  carries `time_precision=IMPUTED_FROM_SCHEDULE`.

Neither option invents a value; they differ only in when an existing value is
allowed to become visible.
"""

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_API_KEY_ENV = "FRED_API_KEY"
EASTERN = ZoneInfo("America/New_York")

# FRED output types, from the published API documentation.
OUTPUT_INITIAL_RELEASE_ONLY = 4
OUTPUT_NEW_AND_REVISED_ONLY = 3


class ReleaseTimePolicy(StrEnum):
    END_OF_DAY = "END_OF_DAY"
    SCHEDULED_LOCAL = "SCHEDULED_LOCAL"


@dataclass(frozen=True)
class SeriesSpec:
    """One tracked release.

    `scheduled_local_time` is the publisher's long-standing release clock time
    in US Eastern. It is used only under SCHEDULED_LOCAL, and only ever labelled
    as imputed.
    """

    series_id: str
    name: str
    importance: str                  # HIGH | MEDIUM | LOW
    scheduled_local_time: time
    units_hint: str | None = None
    note: str = ""


# Releases that actually move gold and the dollar. Curated deliberately rather
# than pulling FRED's full catalogue: every extra series is requests spent and
# noise added, and the agents need the handful of prints that matter.
DEFAULT_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("CPIAUCSL", "US CPI (all items, SA)", "HIGH", time(8, 30),
               "index", "BLS, 08:30 ET"),
    SeriesSpec("CPILFESL", "US Core CPI (ex food & energy, SA)", "HIGH", time(8, 30),
               "index", "BLS, 08:30 ET"),
    SeriesSpec("PAYEMS", "US Nonfarm Payrolls", "HIGH", time(8, 30),
               "thousands of persons", "BLS employment situation, 08:30 ET"),
    SeriesSpec("UNRATE", "US Unemployment Rate", "HIGH", time(8, 30),
               "percent", "BLS employment situation, 08:30 ET"),
    SeriesSpec("PCEPILFE", "US Core PCE Price Index", "HIGH", time(8, 30),
               "index", "BEA, 08:30 ET"),
    SeriesSpec("FEDFUNDS", "US Effective Federal Funds Rate (monthly)", "MEDIUM",
               time(16, 0), "percent", "monthly average, published after month end"),
    SeriesSpec("DFF", "US Effective Federal Funds Rate (daily)", "LOW", time(16, 0),
               "percent", "daily, published next business day"),
    SeriesSpec("GDPC1", "US Real GDP (quarterly)", "HIGH", time(8, 30),
               "billions chained", "BEA, 08:30 ET"),
    SeriesSpec("RSAFS", "US Retail Sales", "MEDIUM", time(8, 30),
               "millions USD", "Census, 08:30 ET"),
    SeriesSpec("INDPRO", "US Industrial Production", "MEDIUM", time(9, 15),
               "index", "Federal Reserve, 09:15 ET"),
    SeriesSpec("DGS10", "US 10-Year Treasury Yield", "MEDIUM", time(16, 15),
               "percent", "Treasury H.15, close"),
    SeriesSpec("DGS2", "US 2-Year Treasury Yield", "LOW", time(16, 15),
               "percent", "Treasury H.15, close"),
    SeriesSpec("T10Y2Y", "US 10Y-2Y Treasury Spread", "LOW", time(16, 15),
               "percent", "derived, Treasury H.15"),
    SeriesSpec("DTWEXBGS", "US Dollar Index (broad, goods & services)", "MEDIUM",
               time(16, 15), "index", "Federal Reserve H.10"),
    SeriesSpec("UMCSENT", "University of Michigan Consumer Sentiment", "MEDIUM",
               time(10, 0), "index", "final release, 10:00 ET"),
    SeriesSpec("ICSA", "US Initial Jobless Claims", "MEDIUM", time(8, 30),
               "persons", "DOL, Thursday 08:30 ET"),
    SeriesSpec("PPIACO", "US Producer Price Index (all commodities)", "MEDIUM",
               time(8, 30), "index", "BLS, 08:30 ET"),
    SeriesSpec("HOUST", "US Housing Starts", "LOW", time(8, 30),
               "thousands of units", "Census, 08:30 ET"),
)


class FredCalendarSource(HistoricalSource):
    """Vintage-aware economic calendar from FRED / ALFRED."""

    spec = SourceSpec(
        key="fred_alfred",
        name="FRED / ALFRED (Federal Reserve Bank of St. Louis)",
        kinds=("calendar",),
        hosts=("api.stlouisfed.org",),
        coverage_start="varies by series; most tracked series cover 1950 onward",
        coverage_note=(
            "Vintage (real-time) history is available from 1996-ish for most "
            "series -- ALFRED's archive begins when the Fed started retaining "
            "vintages. Series-level coverage is reported per series."
        ),
        timestamp_granularity=(
            "DATE only. ALFRED records the release DATE, never the clock time; "
            "the clock time comes from a release-time policy and is labelled as "
            "imputed when used."
        ),
        cost=SourceCost.FREE_WITH_KEY,
        api_key_env=FRED_API_KEY_ENV,
        key_signup_url="https://fredaccount.stlouisfed.org/apikeys",
        rate_limit=(
            "~120 requests/minute with a key (30 without). HTTP 429 when "
            "exceeded; the run is resumable so a 429 costs nothing already "
            "fetched."
        ),
        licensing=(
            "Free for any use including commercial; most series are US "
            "government data. Some third-party series in FRED carry their own "
            "terms -- the series tracked here are US federal statistics."
        ),
        reproducible=True,
        provenance_support=(
            "Strongest available. ALFRED distinguishes the initial release from "
            "every later revision, and dates each one."
        ),
        evaluation=(
            "Chosen as the calendar source specifically because it preserves "
            "original-release vintages. Free with a key, generous limits, "
            "authoritative. Weaknesses: no intraday release time, and no "
            "consensus forecast (FRED publishes actuals, not expectations)."
        ),
    )

    def __init__(
        self,
        cache: ArtifactCache,
        api_key: str | None = None,
        series: tuple[SeriesSpec, ...] = DEFAULT_SERIES,
        release_time_policy: ReleaseTimePolicy = ReleaseTimePolicy.END_OF_DAY,
        client: httpx.Client | None = None,
    ) -> None:
        self._cache = cache
        self._api_key = api_key or os.environ.get(FRED_API_KEY_ENV) or None
        self._series = series
        self._policy = release_time_policy
        self._client = client

    def preflight(self) -> str | None:
        if not self._api_key:
            return (
                f"{FRED_API_KEY_ENV} is not set. FRED requires a free API key: "
                "register at https://fredaccount.stlouisfed.org/apikeys (no cost, "
                "no card, issued immediately) and put it in .env as "
                f"{FRED_API_KEY_ENV}=... . Without it no calendar data can be "
                "fetched, and none will be invented."
            )
        return None

    # --- windows ----------------------------------------------------------
    def windows(self, start: datetime, end: datetime) -> list[FetchWindow]:
        """One window per (series, calendar year, vintage stream).

        Series-year granularity keeps a failure cheap: a 429 halfway through
        costs one series-year, not the backfill.
        """
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        windows: list[FetchWindow] = []
        for spec in self._series:
            for year in range(start.year, end.year + 1):
                year_start = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
                year_end = min(end, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
                if year_start >= year_end:
                    continue
                for stream in ("initial", "revised"):
                    windows.append(
                        FetchWindow(
                            key=f"{spec.series_id}:{year}:{stream}",
                            start=year_start,
                            end=year_end,
                        )
                    )
        return windows

    # --- fetching ---------------------------------------------------------
    def fetch_window(self, window: FetchWindow) -> FetchResult:
        missing = self.preflight()
        if missing:
            return FetchResult(window=window, failed=True, error=missing)

        series_id, _year, stream = window.key.split(":")
        spec = next((s for s in self._series if s.series_id == series_id), None)
        if spec is None:
            return FetchResult(
                window=window, failed=True, error=f"unknown series {series_id}"
            )

        output_type = (
            OUTPUT_INITIAL_RELEASE_ONLY if stream == "initial"
            else OUTPUT_NEW_AND_REVISED_ONLY
        )
        url = self._observations_url(spec, window, output_type)
        try:
            artifact = self._cache.fetch(url, suffix=".json")
        except SourceUnavailable as exc:
            return FetchResult(window=window, failed=True, error=str(exc))
        if artifact is None:
            return FetchResult(window=window, empty=True, requests_made=1)

        try:
            payload = json.loads(artifact.path.read_text())
        except json.JSONDecodeError as exc:
            return FetchResult(
                window=window,
                failed=True,
                error=f"{redact_url(url)}: invalid JSON ({exc})",
            )
        if "observations" not in payload:
            return FetchResult(
                window=window,
                failed=True,
                error=(
                    f"{redact_url(url)}: response has no 'observations' key "
                    f"(keys: {sorted(payload)[:6]}). FRED returns an error object "
                    "when the key is rejected or the series is unknown."
                ),
            )

        records = self._to_records(spec, payload["observations"], stream, artifact)
        return FetchResult(
            window=window,
            records=records,
            bytes_fetched=0 if artifact.from_cache else artifact.bytes_len,
            requests_made=0 if artifact.from_cache else 1,
            from_cache=artifact.from_cache,
            empty=not records,
        )

    def _observations_url(
        self, spec: SeriesSpec, window: FetchWindow, output_type: int
    ) -> str:
        """Build the observations URL.

        `realtime_start` is the window's own observation start, not FRED's
        archive floor (1776-07-04) -- confirmed against the live API
        (2026-09-16): a value for a period starting on `window.start` cannot
        have been revised before that date, so nothing is lost by starting
        there, but the archive floor forces ALFRED to count vintage dates
        across the series' ENTIRE history, which hits ALFRED's 2000-vintage-
        date cap for any daily series spanning more than a few years (real
        error hit: "5124 vintage dates ... exceeds the maximum ... (2000)"
        for DFF/DGS10/DGS2 -- daily series accumulate far more vintages per
        year than the monthly/quarterly series this never affected).
        `realtime_end` stays open-ended so future revisions are still
        captured, which is the entire point of using this endpoint.
        """
        params = {
            "series_id": spec.series_id,
            "api_key": self._api_key or "",
            "file_type": "json",
            "output_type": str(output_type),
            "observation_start": f"{window.start:%Y-%m-%d}",
            "observation_end": f"{window.end:%Y-%m-%d}",
            "realtime_start": f"{window.start:%Y-%m-%d}",
            "realtime_end": "9999-12-31",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{FRED_BASE}/series/observations?{query}"

    def _to_records(
        self, spec: SeriesSpec, observations: list[dict], stream: str, artifact
    ) -> list[CalendarRecord]:
        records: list[CalendarRecord] = []
        retrieved_at = artifact.retrieved_at or now_utc()
        # `previous` is taken from the preceding observation in this same
        # vintage stream, in order, so it is a value that was already public.
        previous_value: str | None = None

        for observation in observations:
            value = str(observation.get("value", "")).strip()
            realtime_start = observation.get("realtime_start")
            reference = observation.get("date")
            if not realtime_start or not reference:
                continue
            # FRED writes "." for a missing observation. Skipped, never zeroed.
            if value in ("", "."):
                continue

            try:
                release_date = date.fromisoformat(realtime_start)
            except ValueError:
                continue

            available_at, precision = self._availability(spec, release_date)
            is_initial = stream == "initial"

            records.append(
                CalendarRecord(
                    name=spec.name,
                    importance=spec.importance,
                    series_id=spec.series_id,
                    reference_period=reference,
                    actual=value,
                    forecast=None,
                    forecast_unavailable_reason=(
                        "FRED publishes realised statistics, not consensus "
                        "forecasts. No forecast is available from this source and "
                        "none is invented; a forecast feed would have to be added "
                        "separately."
                    ),
                    previous=previous_value,
                    previous_basis=(
                        "preceding observation in the same vintage stream"
                        if previous_value is not None
                        else None
                    ),
                    units=spec.units_hint,
                    source="fred_alfred",
                    source_id=stable_id(spec.series_id, reference, realtime_start, stream),
                    published_at=available_at,
                    retrieved_at=retrieved_at,
                    time_precision=precision,
                    provenance=(
                        Provenance.ORIGINAL_RELEASE if is_initial else Provenance.REVISED
                    ),
                    original_release=is_initial,
                    revision_timestamp=None if is_initial else available_at,
                    source_artifact=artifact.relative_name,
                    source_checksum=artifact.sha256,
                )
            )
            previous_value = value

        return records

    def _availability(
        self, spec: SeriesSpec, release_date: date
    ) -> tuple[datetime, TimePrecision]:
        """When a release dated `release_date` becomes visible, and how exact that is."""
        if self._policy == ReleaseTimePolicy.SCHEDULED_LOCAL:
            local = datetime.combine(release_date, spec.scheduled_local_time, EASTERN)
            return local.astimezone(timezone.utc), TimePrecision.IMPUTED_FROM_SCHEDULE
        # END_OF_DAY: the safe direction. A figure released at 08:30 stays hidden
        # until the day ends, so no signal can ever see it early.
        end_of_day = datetime.combine(
            release_date, time(23, 59, 59), timezone.utc
        )
        return end_of_day, TimePrecision.DATE_ONLY

    # --- schedule (known in advance) --------------------------------------
    def release_dates_url(self, release_id: int, start: date, end: date) -> str:
        params = {
            "release_id": str(release_id),
            "api_key": self._api_key or "",
            "file_type": "json",
            "realtime_start": start.isoformat(),
            "realtime_end": end.isoformat(),
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{FRED_BASE}/release/dates?{query}"

    def series_catalog(self) -> list[dict]:
        return [
            {
                "series_id": s.series_id,
                "name": s.name,
                "importance": s.importance,
                "scheduled_local_time_et": s.scheduled_local_time.isoformat(),
                "units": s.units_hint,
                "note": s.note,
            }
            for s in self._series
        ]

    def config_fingerprint(self) -> dict:
        """What this dataset was built with, for reproducibility."""
        return {
            "source": self.spec.key,
            "release_time_policy": self._policy.value,
            "series": [s.series_id for s in self._series],
            "streams": ["initial", "revised"],
            "output_types": {
                "initial": OUTPUT_INITIAL_RELEASE_ONLY,
                "revised": OUTPUT_NEW_AND_REVISED_ONLY,
            },
        }
