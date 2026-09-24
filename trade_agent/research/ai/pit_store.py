from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

"""Point-in-time news / sentiment / economic-calendar stores.

The contract every store here obeys: given a timestamp, return ONLY records
whose own publication (or scheduled-release) time is at or before it, and
report `available=False` when the dataset genuinely has no coverage for that
period. Absence is data. Nothing is generated to fill a gap.

A scheduled economic event is the one deliberate exception to "at or before":
a release KNOWN to be scheduled for 14:30 is public information at 13:00, so
the calendar store may return future release times -- but only the schedule,
never the released value. `released_value` is withheld until the release
timestamp passes.
"""


class PitRecord(BaseModel):
    timestamp: datetime  # when this became public knowledge
    source: str
    headline: str
    category: str = "general"
    payload: dict = Field(default_factory=dict)


class CalendarEvent(BaseModel):
    scheduled_at: datetime
    name: str
    importance: str  # LOW | MEDIUM | HIGH
    currency: str = "USD"
    # Withheld until scheduled_at has passed, so a backtest cannot see the
    # print before it printed.
    released_value: str | None = None
    forecast_value: str | None = None
    # ORIGINAL_RELEASE | REVISED | UNKNOWN. A revised figure is never served
    # as a released value: the market at the time traded the original print.
    provenance: str = "UNKNOWN"
    withheld_reason: str | None = None


class PitQueryResult(BaseModel):
    """What a store returns for one timestamp. `available=False` is a
    first-class answer that the runner turns into UNAVAILABLE."""

    available: bool
    reason: str | None = None
    records: list[PitRecord] = Field(default_factory=list)
    events: list[CalendarEvent] = Field(default_factory=list)
    dataset_name: str = "unknown"
    dataset_version: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.records and not self.events


class LookaheadError(Exception):
    """Raised when a store would return a record newer than the as-of
    timestamp. Fatal: this is the bias the whole experiment exists to avoid."""


class PitStore:
    """Base store over a table of timestamped records.

    Subclasses supply loading; this class owns the as-of filtering and the
    look-ahead assertion, so no subclass can accidentally skip it.
    """

    dataset_name = "base"

    def __init__(self, frame: pd.DataFrame | None, dataset_version: str | None = None) -> None:
        self._frame = frame
        self._dataset_version = dataset_version
        if frame is not None and not frame.empty:
            if "timestamp" not in frame.columns:
                raise ValueError(f"{self.dataset_name}: store frame needs a 'timestamp' column")
            self._frame = frame.sort_values("timestamp").reset_index(drop=True)

    @property
    def has_data(self) -> bool:
        return self._frame is not None and not self._frame.empty

    def coverage(self) -> tuple[datetime, datetime] | None:
        if not self.has_data:
            return None
        return (
            self._frame["timestamp"].iloc[0].to_pydatetime(),
            self._frame["timestamp"].iloc[-1].to_pydatetime(),
        )

    def _covers(self, as_of: datetime) -> bool:
        span = self.coverage()
        if span is None:
            return False
        return span[0] <= as_of <= span[1] + timedelta(days=1)

    def query(self, as_of: datetime, lookback_minutes: int, max_items: int) -> PitQueryResult:
        if not self.has_data:
            return PitQueryResult(
                available=False,
                reason=f"{self.dataset_name} dataset not loaded",
                dataset_name=self.dataset_name,
                dataset_version=self._dataset_version,
            )
        if not self._covers(as_of):
            return PitQueryResult(
                available=False,
                reason=(
                    f"{self.dataset_name} dataset does not cover {as_of.isoformat()} "
                    f"(coverage: {self.coverage()})"
                ),
                dataset_name=self.dataset_name,
                dataset_version=self._dataset_version,
            )

        as_of_ts = pd.Timestamp(as_of)
        window_start = as_of_ts - pd.Timedelta(minutes=lookback_minutes)
        frame = self._frame
        selected = frame[(frame["timestamp"] <= as_of_ts) & (frame["timestamp"] >= window_start)]
        selected = selected.tail(max_items)

        # `headline`/`category` are the news shape; a sentiment row has
        # neither (see SentimentRecord: it is a numeric GDELT tone average,
        # never article text) but DOES carry `value`/`raw_value`/
        # `article_count`. Passing every other column through in `payload`
        # means a numeric-only dataset's actual content reaches its
        # consumer instead of silently vanishing behind an empty headline.
        known = {"timestamp", "source", "headline", "category"}
        records = [
            PitRecord(
                timestamp=row["timestamp"].to_pydatetime(),
                source=str(row.get("source", "unknown")),
                headline=str(row.get("headline", "")),
                category=str(row.get("category", "general")),
                payload={k: row[k] for k in row.index if k not in known and pd.notna(row[k])},
            )
            for _, row in selected.iterrows()
        ]
        for record in records:
            if record.timestamp > as_of:
                raise LookaheadError(
                    f"{self.dataset_name}: record at {record.timestamp} is after as_of {as_of}"
                )

        return PitQueryResult(
            available=True,
            records=records,
            dataset_name=self.dataset_name,
            dataset_version=self._dataset_version,
            reason=None if records else "no items in lookback window",
        )


class NewsPitStore(PitStore):
    dataset_name = "news"


class SentimentPitStore(PitStore):
    dataset_name = "sentiment"


class CalendarPitStore(PitStore):
    """Economic calendar. Returns the SCHEDULE around the timestamp (known in
    advance) but withholds released values until their release time."""

    dataset_name = "economic_calendar"

    def query_events(
        self, as_of: datetime, window_minutes: int
    ) -> PitQueryResult:
        if not self.has_data:
            return PitQueryResult(
                available=False,
                reason="economic calendar dataset not loaded",
                dataset_name=self.dataset_name,
                dataset_version=self._dataset_version,
            )
        if not self._covers(as_of):
            return PitQueryResult(
                available=False,
                reason=f"calendar does not cover {as_of.isoformat()}",
                dataset_name=self.dataset_name,
                dataset_version=self._dataset_version,
            )

        as_of_ts = pd.Timestamp(as_of)
        frame = self._frame
        window = frame[
            (frame["timestamp"] >= as_of_ts - pd.Timedelta(minutes=window_minutes))
            & (frame["timestamp"] <= as_of_ts + pd.Timedelta(minutes=window_minutes))
        ]

        events = []
        for _, row in window.iterrows():
            scheduled = row["timestamp"].to_pydatetime()
            already_released = scheduled <= as_of
            provenance = str(row.get("provenance") or "UNKNOWN").upper()
            has_value = row.get("released_value") is not None
            # Two independent reasons to withhold a released value, reported
            # separately so the audit trail says which one applied.
            if not already_released:
                withheld = "not yet released at this timestamp"
            elif provenance == "REVISED":
                withheld = (
                    "value is a later revision; the original print is what was "
                    "public at this timestamp"
                )
            else:
                withheld = None
            events.append(
                CalendarEvent(
                    scheduled_at=scheduled,
                    name=str(row.get("name", row.get("headline", "event"))),
                    importance=str(row.get("importance", "MEDIUM")),
                    currency=str(row.get("currency", "USD")),
                    forecast_value=(
                        str(row["forecast_value"]) if row.get("forecast_value") is not None else None
                    ),
                    # The actual print is knowledge only after it prints, and
                    # only in the form it printed in.
                    released_value=(
                        str(row["released_value"])
                        if has_value and withheld is None
                        else None
                    ),
                    provenance=provenance,
                    withheld_reason=withheld if has_value else None,
                )
            )

        return PitQueryResult(
            available=True,
            events=events,
            dataset_name=self.dataset_name,
            dataset_version=self._dataset_version,
            reason=None if events else "no scheduled events in window",
        )


def load_pit_frame(path: Path | None) -> pd.DataFrame | None:
    """Load a point-in-time dataset from Parquet/CSV/JSONL.

    Returns None when the path is absent, which propagates as UNAVAILABLE
    rather than as an empty-but-available dataset -- the difference matters:
    "no news happened" and "we have no news data" are not the same claim.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None

    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])
    else:
        frame = pd.read_csv(path)

    if "timestamp" not in frame.columns:
        raise ValueError(f"{path}: point-in-time dataset needs a 'timestamp' column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path}: unparseable timestamps in point-in-time dataset")
    return frame


def utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
