from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from research.manifest import DatasetVersion, sha256_file

"""Dataset availability, coverage and provenance.

The rule this module enforces, from the experiment's own terms: missing data
is reported as missing. Nothing is invented, nothing is back-filled, and a
dataset that cannot support a point-in-time claim is reported UNAVAILABLE
rather than used with a caveat in a footnote.

Three distinct states, never collapsed:

* **AVAILABLE** -- the dataset covers the period and satisfies its
  point-in-time contract.
* **UNAVAILABLE** -- there is no dataset, or it fails its contract. The agent
  that depends on it answers UNAVAILABLE and the fail-closed policy applies.
* **PARTIAL** -- a real dataset with real gaps. Covered timestamps are used;
  uncovered ones are UNAVAILABLE individually. The gaps are listed, because
  "available" over 60% of the period is a materially different claim from
  "available".

The sentiment rule deserves stating plainly: scoring historical text with a
model that exists today does NOT produce point-in-time sentiment. It produces
today's reading of old text, which is hindsight wearing a timestamp. Such a
dataset is refused (`RETROSPECTIVE_SCORING`) unless it carries evidence that
the sentiment value itself was observed at the time.
"""


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class Provenance(StrEnum):
    """Where a value came from, and whether it is the value that was public
    at the time."""

    ORIGINAL_RELEASE = "ORIGINAL_RELEASE"
    # A later correction. Never admissible: the market at the time traded the
    # original print, and the revision did not exist yet.
    REVISED = "REVISED"
    # Captured live at the timestamp (a sentiment or positioning snapshot).
    POINT_IN_TIME_CAPTURE = "POINT_IN_TIME_CAPTURE"
    # Scored after the fact by a model. Inadmissible as point-in-time data.
    # RETROSPECTIVE is the label the ingestion pipeline writes;
    # RETROSPECTIVE_SCORING is kept as an accepted alias so datasets written
    # before the rename still load, and both are inadmissible.
    RETROSPECTIVE = "RETROSPECTIVE"
    RETROSPECTIVE_SCORING = "RETROSPECTIVE_SCORING"
    UNKNOWN = "UNKNOWN"


# Required columns per dataset kind.
NEWS_SCHEMA = ("timestamp", "source", "headline", "category")
SENTIMENT_SCHEMA = ("timestamp", "source", "value")
CALENDAR_SCHEMA = ("timestamp", "name", "importance")

INADMISSIBLE = {
    Provenance.REVISED,
    Provenance.RETROSPECTIVE,
    Provenance.RETROSPECTIVE_SCORING,
}


class CoverageGap(BaseModel):
    start: datetime
    end: datetime
    days: float


class DatasetReport(BaseModel):
    """One dataset's full status, as the experiment must disclose it."""

    name: str
    availability: Availability
    source: str = "not supplied"
    path: str | None = None
    row_count: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    timestamp_precision: str = "unknown"
    point_in_time_semantics: str = ""
    provenance: str = Provenance.UNKNOWN.value
    provenance_counts: dict[str, int] = Field(default_factory=dict)
    rows_excluded: int = 0
    exclusion_reason: str | None = None
    missing_periods: list[CoverageGap] = Field(default_factory=list)
    coverage_fraction: float | None = None
    content_hash: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.availability in (Availability.AVAILABLE, Availability.PARTIAL)

    def to_dataset_version(self) -> DatasetVersion:
        return DatasetVersion(
            name=self.name,
            source=self.source,
            available=self.usable,
            row_count=self.row_count or None,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            content_hash=self.content_hash,
            note="; ".join([self.availability.value, *self.notes]) or None,
        )


def _precision(series: pd.Series) -> str:
    """Infer the finest unit the timestamps actually resolve.

    Reported because a news dataset stamped to the day cannot support a
    15-minute blackout window, and claiming otherwise would be the whole
    experiment's undoing.
    """
    if series.empty:
        return "unknown"
    values = pd.to_datetime(series, utc=True)
    if (values.dt.nanosecond != 0).any() or (values.dt.microsecond != 0).any():
        return "sub-millisecond"
    if (values.dt.second != 0).any():
        return "second"
    if (values.dt.minute != 0).any():
        return "minute"
    if (values.dt.hour != 0).any():
        return "hour"
    return "day"


def _gaps(series: pd.Series, max_gap_days: float) -> list[CoverageGap]:
    values = pd.to_datetime(series, utc=True).sort_values()
    gaps: list[CoverageGap] = []
    previous = None
    for value in values:
        if previous is not None:
            span = (value - previous).total_seconds() / 86400.0
            if span > max_gap_days:
                gaps.append(
                    CoverageGap(
                        start=previous.to_pydatetime(),
                        end=value.to_pydatetime(),
                        days=round(span, 2),
                    )
                )
        previous = value
    return gaps


def _coverage_fraction(
    series: pd.Series, period_start: datetime, period_end: datetime, gaps: list[CoverageGap]
) -> float | None:
    total = (period_end - period_start).total_seconds() / 86400.0
    if total <= 0:
        return None
    missing = sum(gap.days for gap in gaps)
    first = pd.to_datetime(series, utc=True).min().to_pydatetime()
    last = pd.to_datetime(series, utc=True).max().to_pydatetime()
    if first > period_start:
        missing += (first - period_start).total_seconds() / 86400.0
    if last < period_end:
        missing += (period_end - last).total_seconds() / 86400.0
    return round(max(0.0, min(1.0, 1.0 - missing / total)), 4)


def unavailable(name: str, reason: str, semantics: str = "") -> DatasetReport:
    """The honest answer when a dataset is absent."""
    return DatasetReport(
        name=name,
        availability=Availability.UNAVAILABLE,
        point_in_time_semantics=semantics,
        notes=[reason],
    )


def inspect_dataset(
    name: str,
    path: Path | None,
    required_columns: tuple[str, ...],
    semantics: str,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    max_gap_days: float = 7.0,
    require_point_in_time_provenance: bool = False,
    default_provenance: Provenance = Provenance.UNKNOWN,
) -> tuple[DatasetReport, pd.DataFrame | None]:
    """Load and audit one point-in-time dataset.

    Returns the report and the ADMISSIBLE rows -- rows whose provenance makes
    them inadmissible are dropped here, before any store can serve them, so
    an inadmissible row cannot reach an agent by some other path.
    """
    if path is None:
        return unavailable(name, f"no {name} dataset path configured", semantics), None
    path = Path(path)
    if not path.exists():
        return unavailable(name, f"{name} dataset not found at {path}", semantics), None

    frame = _read(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        return (
            unavailable(
                name,
                f"{path.name} is missing required column(s) {missing}; expected "
                f"{list(required_columns)}",
                semantics,
            ),
            None,
        )

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    unparseable = int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        return unavailable(name, f"{path.name} contains no parseable timestamps", semantics), None

    # --- provenance ----------------------------------------------------
    if "provenance" in frame.columns:
        provenance = frame["provenance"].fillna(Provenance.UNKNOWN.value).astype(str).str.upper()
    else:
        provenance = pd.Series(
            [default_provenance.value] * len(frame), index=frame.index, dtype=object
        )
    frame["provenance"] = provenance
    counts = {str(k): int(v) for k, v in provenance.value_counts().items()}

    inadmissible_mask = provenance.isin({p.value for p in INADMISSIBLE})
    excluded = int(inadmissible_mask.sum())
    exclusion_reason = None
    if excluded:
        kinds = sorted(set(provenance[inadmissible_mask]))
        exclusion_reason = (
            f"{excluded} row(s) excluded as inadmissible provenance {kinds}: a revised "
            "figure or a retrospectively scored value was not public at the timestamp "
            "it carries"
        )
        frame = frame[~inadmissible_mask].reset_index(drop=True)

    notes: list[str] = []
    if unparseable:
        notes.append(f"{unparseable} row(s) dropped for unparseable timestamps")
    if exclusion_reason:
        notes.append(exclusion_reason)

    if frame.empty:
        report = unavailable(
            name,
            "every row was inadmissible after the provenance filter",
            semantics,
        )
        report.provenance_counts = counts
        report.rows_excluded = excluded
        report.exclusion_reason = exclusion_reason
        report.path = str(path)
        return report, None

    # A dataset that cannot evidence point-in-time capture is refused for the
    # roles that require it, however much data it contains.
    if require_point_in_time_provenance:
        admissible = frame["provenance"] == Provenance.POINT_IN_TIME_CAPTURE.value
        if not admissible.any():
            report = unavailable(
                name,
                f"{path.name} carries no rows marked "
                f"{Provenance.POINT_IN_TIME_CAPTURE.value}. A value scored after the "
                "fact is not point-in-time data, so this dataset cannot be used for "
                f"{name}; supply a feed captured at the time, with a provenance column",
                semantics,
            )
            report.provenance_counts = counts
            report.path = str(path)
            report.row_count = len(frame)
            return report, None
        if not admissible.all():
            notes.append(
                f"{int((~admissible).sum())} row(s) without point-in-time provenance "
                "were dropped"
            )
            frame = frame[admissible].reset_index(drop=True)

    first = frame["timestamp"].iloc[0].to_pydatetime()
    last = frame["timestamp"].iloc[-1].to_pydatetime()
    gaps = _gaps(frame["timestamp"], max_gap_days)
    fraction = (
        _coverage_fraction(frame["timestamp"], period_start, period_end, gaps)
        if period_start and period_end
        else None
    )

    availability = Availability.AVAILABLE
    if gaps or (fraction is not None and fraction < 0.98):
        availability = Availability.PARTIAL
        notes.append(
            f"{len(gaps)} gap(s) longer than {max_gap_days} days; timestamps inside a "
            "gap are served as UNAVAILABLE individually"
        )
    if period_start and period_end and (first > period_end or last < period_start):
        return (
            unavailable(
                name,
                f"{path.name} covers {first.date()}..{last.date()}, which does not "
                f"overlap the experiment period {period_start.date()}..{period_end.date()}",
                semantics,
            ),
            None,
        )

    dominant = max(counts, key=counts.get) if counts else Provenance.UNKNOWN.value
    report = DatasetReport(
        name=name,
        availability=availability,
        source=str(frame["source"].iloc[0]) if "source" in frame.columns else str(path.name),
        path=str(path),
        row_count=len(frame),
        first_timestamp=first,
        last_timestamp=last,
        timestamp_precision=_precision(frame["timestamp"]),
        point_in_time_semantics=semantics,
        provenance=dominant,
        provenance_counts=counts,
        rows_excluded=excluded,
        exclusion_reason=exclusion_reason,
        missing_periods=gaps[:50],
        coverage_fraction=fraction,
        content_hash=sha256_file(path),
        notes=notes,
    )
    return report, frame


NEWS_SEMANTICS = (
    "A headline is served only for timestamps at or after its publication time. "
    "Nothing is served from the future."
)
SENTIMENT_SEMANTICS = (
    "A sentiment value is served only if it was OBSERVED at or before the query "
    "timestamp. Scoring archived text with a present-day model is not admissible."
)
CALENDAR_SEMANTICS = (
    "The SCHEDULE is known in advance, so events scheduled after the query time are "
    "served -- schedule only. A released value is withheld until its release time, "
    "and revised figures are excluded entirely."
)


class DatasetBundle(BaseModel):
    """The three point-in-time datasets, with their status.

    Carried into the run manifest and the final report, so results are never
    presented without the availability of the data behind them.
    """

    news: DatasetReport
    sentiment: DatasetReport
    calendar: DatasetReport
    candles: DatasetReport | None = None

    def reports(self) -> list[DatasetReport]:
        return [r for r in (self.candles, self.news, self.sentiment, self.calendar) if r]

    def dataset_versions(self) -> list[DatasetVersion]:
        return [report.to_dataset_version() for report in self.reports()]

    def unavailable_names(self) -> list[str]:
        return [r.name for r in self.reports() if not r.usable]

    def summary_rows(self) -> list[dict]:
        return [
            {
                "dataset": r.name,
                "availability": r.availability.value,
                "source": r.source,
                "rows": r.row_count,
                "coverage": (
                    f"{r.first_timestamp:%Y-%m-%d} to {r.last_timestamp:%Y-%m-%d}"
                    if r.first_timestamp and r.last_timestamp
                    else "none"
                ),
                "coverage_fraction": r.coverage_fraction,
                "precision": r.timestamp_precision,
                "provenance": r.provenance,
                "rows_excluded": r.rows_excluded,
                "gaps": len(r.missing_periods),
            }
            for r in self.reports()
        ]


def inspect_all(
    news_path: Path | None,
    sentiment_path: Path | None,
    calendar_path: Path | None,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> tuple[DatasetBundle, dict[str, pd.DataFrame | None]]:
    """Audit all three point-in-time datasets at once."""
    news_report, news_frame = inspect_dataset(
        "news", news_path, NEWS_SCHEMA, NEWS_SEMANTICS, period_start, period_end,
        default_provenance=Provenance.ORIGINAL_RELEASE,
    )
    sentiment_report, sentiment_frame = inspect_dataset(
        "sentiment", sentiment_path, SENTIMENT_SCHEMA, SENTIMENT_SEMANTICS,
        period_start, period_end,
        # The strict one: without evidence of live capture, refuse it.
        require_point_in_time_provenance=True,
        default_provenance=Provenance.UNKNOWN,
    )
    calendar_report, calendar_frame = inspect_dataset(
        "economic_calendar", calendar_path, CALENDAR_SCHEMA, CALENDAR_SEMANTICS,
        period_start, period_end, max_gap_days=14.0,
        default_provenance=Provenance.ORIGINAL_RELEASE,
    )
    bundle = DatasetBundle(
        news=news_report, sentiment=sentiment_report, calendar=calendar_report
    )
    return bundle, {
        "news": news_frame,
        "sentiment": sentiment_frame,
        "economic_calendar": calendar_frame,
    }


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        import json

        return pd.DataFrame(
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        )
    return pd.read_csv(path)


def candle_dataset_report(path: Path | None, timeframe_minutes: int) -> DatasetReport:
    """Status of the M15 candle dataset the whole experiment rests on."""
    if path is None or not Path(path).exists():
        return unavailable(
            "candles",
            f"no M{timeframe_minutes} candle file found; build it from tick data first "
            "(research.cli candles). No synthetic series is substituted.",
            "Bars are left-closed/right-open on the timeframe boundary; a signal on the "
            "close of bar i is executed at the open of bar i+1.",
        )
    frame = _read(Path(path))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    expected_gap = timedelta(minutes=timeframe_minutes)
    gaps = _gaps(frame["timestamp"], max_gap_days=expected_gap.total_seconds() / 86400.0 * 96)
    return DatasetReport(
        name="candles",
        availability=Availability.PARTIAL if gaps else Availability.AVAILABLE,
        source=str(Path(path).name),
        path=str(path),
        row_count=len(frame),
        first_timestamp=frame["timestamp"].iloc[0].to_pydatetime(),
        last_timestamp=frame["timestamp"].iloc[-1].to_pydatetime(),
        timestamp_precision=_precision(frame["timestamp"]),
        point_in_time_semantics=(
            "Bars are left-closed/right-open; execution is one bar later than the signal."
        ),
        provenance=Provenance.ORIGINAL_RELEASE.value,
        missing_periods=gaps[:50],
        content_hash=sha256_file(Path(path)),
        notes=(
            [f"{len(gaps)} gap(s) beyond a normal weekend break"] if gaps else []
        ),
    )
