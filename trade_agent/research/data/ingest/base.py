from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

"""Canonical record shapes for automated historical ingestion.

Three record types, one rule: every record carries enough provenance to answer,
without guessing, when its information became knowable. That is the only
property the backtest's leakage guarantee rests on.

The distinction that runs through this module:

* `published_at` -- when the source says the information was published or
  released.
* `discovered_at` -- when the aggregator we actually fetched from first made it
  retrievable.
* `available_at` -- `max(published_at, discovered_at)`, and the ONLY timestamp
  used for point-in-time filtering.

Using the maximum is deliberate. An article published at 14:32 that appeared in
GDELT's 14:45 file could not have been read through GDELT at 14:33, so treating
14:32 as its availability would grant the backtest a lookahead of up to fifteen
minutes. The maximum is never earlier than the publication time, so it also
satisfies the stricter reading of "publication_time <= T".
"""


class Provenance(StrEnum):
    """Where a value came from, and whether it is what was public at the time."""

    # An article as published, or an economic figure as first released.
    ORIGINAL_RELEASE = "ORIGINAL_RELEASE"
    # A later correction to a figure. Preserved, never served as the value that
    # was public at the original release time.
    REVISED = "REVISED"
    # A sentiment value that was itself computed and published at its
    # timestamp -- not recomputed later.
    POINT_IN_TIME_CAPTURE = "POINT_IN_TIME_CAPTURE"
    # A value produced now from an old document. Honest, useful for
    # diagnostics, and inadmissible as point-in-time evidence.
    RETROSPECTIVE = "RETROSPECTIVE"
    UNKNOWN = "UNKNOWN"


# Provenance values that must never reach a point-in-time query. Kept as a set
# so the check is one membership test wherever it appears.
INADMISSIBLE_PROVENANCE: frozenset[str] = frozenset(
    {Provenance.REVISED.value, Provenance.RETROSPECTIVE.value, "RETROSPECTIVE_SCORING"}
)


class TimePrecision(StrEnum):
    """How exact a record's timestamp actually is.

    Recorded because a 15-minute backtest cannot honestly use a date-only
    release as if it were an intraday event, and the difference must be visible
    rather than buried in a parser.
    """

    EXACT = "EXACT"  # the source published a real timestamp
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    # The source gave a date; the clock time comes from a published release
    # schedule. Realistic, but an assumption -- labelled so it cannot be
    # mistaken for an observation.
    IMPUTED_FROM_SCHEDULE = "IMPUTED_FROM_SCHEDULE"
    # The source gave a date and we refuse to invent a time, so the record
    # becomes available at the end of that UTC day.
    DATE_ONLY = "DATE_ONLY"


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class IngestRecord(BaseModel):
    """Fields every ingested record carries, whatever its kind."""

    # --- identity ----------------------------------------------------------
    source: str                      # provider or publication domain
    source_id: str                   # stable id within that source
    # --- timing ------------------------------------------------------------
    published_at: datetime           # when the source says it was published
    discovered_at: datetime | None = None  # when the aggregator exposed it
    retrieved_at: datetime           # when WE fetched it
    time_precision: TimePrecision = TimePrecision.EXACT
    # --- provenance --------------------------------------------------------
    provenance: Provenance = Provenance.UNKNOWN
    original_release: bool = True
    revision_timestamp: datetime | None = None
    # --- reproducibility ---------------------------------------------------
    # The cached artifact this record was parsed out of, and its checksum, so
    # the record can be traced back to bytes on disk and re-derived.
    source_artifact: str | None = None
    source_checksum: str | None = None

    @field_validator("published_at", "discovered_at", "retrieved_at", "revision_timestamp")
    @classmethod
    def to_utc(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value else None

    @property
    def available_at(self) -> datetime:
        """When this information first became knowable through our source.

        The later of publication and discovery, so an aggregator's ingest lag
        can never be used as a head start.
        """
        if self.discovered_at is None:
            return self.published_at
        return max(self.published_at, self.discovered_at)

    @property
    def admissible(self) -> bool:
        return self.provenance.value not in INADMISSIBLE_PROVENANCE


class NewsRecord(IngestRecord):
    """One historical news item."""

    headline: str
    category: str = "general"
    url: str | None = None
    language: str | None = None
    # Free-text relevance trace: which keyword rule matched. Kept so a
    # filtered corpus can be audited rather than trusted.
    matched_terms: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0

    def to_row(self) -> dict:
        """Flat row for the point-in-time store (news schema)."""
        return {
            "timestamp": self.available_at,
            "source": self.source,
            "headline": self.headline,
            "category": self.category,
            "provenance": self.provenance.value,
            "source_id": self.source_id,
            "published_at": self.published_at,
            "discovered_at": self.discovered_at,
            "retrieved_at": self.retrieved_at,
            "time_precision": self.time_precision.value,
            "original_release": self.original_release,
            "url": self.url,
            "matched_terms": ",".join(self.matched_terms),
            "relevance_score": self.relevance_score,
            "source_artifact": self.source_artifact,
            "source_checksum": self.source_checksum,
        }


class CalendarRecord(IngestRecord):
    """One economic release, at one vintage.

    The same event appears twice when it is revised: once as
    ORIGINAL_RELEASE and once per REVISED vintage. They are never merged --
    merging is exactly how a revised figure ends up in a backtest.
    """

    name: str
    importance: str = "MEDIUM"        # LOW | MEDIUM | HIGH
    series_id: str | None = None
    reference_period: str | None = None  # the period the figure describes
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None
    # How `previous` was obtained, when it was. Sources rarely publish it, so
    # a derived value says so instead of implying the source supplied it.
    previous_basis: str | None = None
    units: str | None = None
    # Set when the source publishes no consensus forecast, which most public
    # statistical sources do not. Absence is recorded, never filled in.
    forecast_unavailable_reason: str | None = None

    def to_row(self) -> dict:
        return {
            "timestamp": self.available_at,
            "name": self.name,
            "importance": self.importance,
            "provenance": self.provenance.value,
            "source": self.source,
            "source_id": self.source_id,
            "series_id": self.series_id,
            "reference_period": self.reference_period,
            "released_value": self.actual,
            "actual": self.actual,
            "forecast_value": self.forecast,
            "forecast": self.forecast,
            "previous": self.previous,
            "previous_basis": self.previous_basis,
            "units": self.units,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "time_precision": self.time_precision.value,
            "original_release": self.original_release,
            "revision_timestamp": self.revision_timestamp,
            "forecast_unavailable_reason": self.forecast_unavailable_reason,
            "source_artifact": self.source_artifact,
            "source_checksum": self.source_checksum,
        }


class SentimentRecord(IngestRecord):
    """One sentiment observation.

    `provenance` is the whole point of this record. POINT_IN_TIME_CAPTURE means
    the VALUE itself existed at `published_at` -- it was computed and published
    then, not derived later from an archived document. RETROSPECTIVE means a
    model scored old text today; useful for diagnostics, inadmissible as
    point-in-time evidence, and refused by the leakage-safe experiment.
    """

    value: float                      # normalized to [-1, 1]
    raw_value: float | None = None     # as the source expressed it
    scale: str = "normalized_-1_1"
    method: str = "unknown"            # how the value was produced
    model: str | None = None           # set only for model-generated values
    article_count: int = 1
    confidence: float | None = None

    def to_row(self) -> dict:
        return {
            "timestamp": self.available_at,
            "source": self.source,
            "value": self.value,
            "provenance": self.provenance.value,
            "source_id": self.source_id,
            "raw_value": self.raw_value,
            "scale": self.scale,
            "method": self.method,
            "model": self.model,
            "article_count": self.article_count,
            "confidence": self.confidence,
            "published_at": self.published_at,
            "discovered_at": self.discovered_at,
            "retrieved_at": self.retrieved_at,
            "time_precision": self.time_precision.value,
            "original_release": self.original_release,
            "source_artifact": self.source_artifact,
            "source_checksum": self.source_checksum,
        }


# --- source description ----------------------------------------------------
class SourceCost(StrEnum):
    FREE = "FREE"
    FREE_WITH_KEY = "FREE_WITH_KEY"
    FREE_TIER_LIMITED = "FREE_TIER_LIMITED"
    PAID = "PAID"


@dataclass(frozen=True)
class SourceSpec:
    """What a source is, and what using it costs and requires.

    Every field here is something a person needs in order to decide whether to
    enable the source. Written down rather than discovered at runtime so the
    status report can list options the pipeline is not currently using.
    """

    key: str
    name: str
    kinds: tuple[str, ...]            # "news" | "calendar" | "sentiment"
    hosts: tuple[str, ...]            # hosts that must be reachable
    coverage_start: str               # earliest data, as documented
    coverage_note: str
    timestamp_granularity: str
    cost: SourceCost
    api_key_env: str | None = None
    key_signup_url: str | None = None
    rate_limit: str = "unknown"
    licensing: str = "see provider terms"
    reproducible: bool = True
    implemented: bool = True
    not_implemented_reason: str | None = None
    provenance_support: str = ""
    evaluation: str = ""

    def requires_key(self) -> bool:
        return self.api_key_env is not None


@dataclass
class FetchWindow:
    """One unit of work, so a five-year backfill is resumable.

    Windows are the granularity of both the cache and the checkpoint: a window
    either completed and is never re-fetched, or it did not and is retried.
    """

    key: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        self.start = _utc(self.start)
        self.end = _utc(self.end)


@dataclass
class FetchResult:
    """What one window produced, including why it produced nothing."""

    window: FetchWindow
    records: list[IngestRecord] = field(default_factory=list)
    bytes_fetched: int = 0
    requests_made: int = 0
    from_cache: bool = False
    # A window can legitimately have no data (a quiet 15 minutes, a day with no
    # releases). That is not a failure and must not look like one.
    empty: bool = False
    failed: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failed


class SourceUnavailable(Exception):
    """A source cannot be reached or used, with the reason a person needs.

    Raised rather than returning empty so a network denial can never be
    mistaken for "there was no news that week".
    """


class HistoricalSource(ABC):
    """A source of historical records, fetched window by window."""

    spec: SourceSpec

    @abstractmethod
    def windows(self, start: datetime, end: datetime) -> list[FetchWindow]:
        """Split a date range into resumable units of work."""

    @abstractmethod
    def fetch_window(self, window: FetchWindow) -> FetchResult:
        """Fetch one window, using the cache when it already holds the bytes."""

    def preflight(self) -> str | None:
        """Return a human-readable reason this source cannot run, or None.

        Checked before any window is attempted so a missing API key is one
        clear message rather than a thousand identical failures.
        """
        return None


def stable_id(*parts: str) -> str:
    """Deterministic id from the parts that identify a record.

    Deterministic so re-running ingestion produces the same ids, which is what
    makes deduplication and cache keys stable across runs.
    """
    return hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()[:20]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
