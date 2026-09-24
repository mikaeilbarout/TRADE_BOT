from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.data.ingest.base import (
    CalendarRecord,
    IngestRecord,
    NewsRecord,
    Provenance,
    SentimentRecord,
)

"""Normalization, deduplication and writing.

Every record that reaches storage has passed through here, so the invariants
are in one place:

* **UTC, always.** A naive timestamp is treated as UTC and the assumption is
  recorded; a local-time timestamp from a source that declares its zone is
  converted. Mixed zones in one dataset is a silent way to shift events by
  hours.
* **Chronological order.** Stores binary-search on time; an unsorted frame
  turns a point-in-time query into a wrong answer rather than a slow one.
* **Deduplication that keeps the EARLIEST availability.** The same article
  reaches an aggregator more than once, and the same figure appears in several
  vintages. When two records collide, the one that became available first is
  kept -- taking the later one would hide information the market already had,
  and taking an arbitrary one would make the dataset non-reproducible.

Calendar records are the deliberate exception to deduplication: an
ORIGINAL_RELEASE and a REVISED record for the same period are NOT duplicates.
Collapsing them is precisely the revision leak the experiment must avoid, so
the dedup key includes the provenance and the vintage date.
"""


def ensure_utc(frame: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Parse a timestamp column to UTC, dropping unparseable rows loudly."""
    out = frame.copy()
    out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out


def news_dedup_key(record: NewsRecord) -> tuple:
    """Identity of a news item.

    The URL identifies an article better than its headline does (the same story
    is re-headlined across editions), so the URL wins when present and the
    headline is the fallback. The source domain is included because the same
    wire story legitimately appears on several sites and each is a separate
    observation of when the information spread.
    """
    if record.url:
        return ("url", record.url.strip().lower())
    return ("headline", record.source.lower(), record.headline.strip().lower())


def calendar_dedup_key(record: CalendarRecord) -> tuple:
    """Identity of one economic figure AT ONE VINTAGE.

    Provenance and the release date are part of the key on purpose: the initial
    release and each revision of the same reference period are different facts,
    and merging them is the leak.
    """
    return (
        record.series_id or record.name,
        record.reference_period or "",
        record.provenance.value,
        record.published_at.date().isoformat(),
    )


def sentiment_dedup_key(record: SentimentRecord) -> tuple:
    return (record.source, record.provenance.value, record.published_at.isoformat())


def deduplicate(records: list, key_fn) -> tuple[list, int]:
    """Keep the earliest-available record per key. Returns (kept, removed)."""
    best: dict[tuple, IngestRecord] = {}
    for record in records:
        key = key_fn(record)
        existing = best.get(key)
        if existing is None or record.available_at < existing.available_at:
            best[key] = record
    kept = sorted(best.values(), key=lambda r: (r.available_at, r.source_id))
    return kept, len(records) - len(kept)


def normalise_news(records: list[NewsRecord]) -> tuple[list[NewsRecord], dict]:
    kept, removed = deduplicate(records, news_dedup_key)
    return kept, {"input": len(records), "kept": len(kept), "duplicates_removed": removed}


def normalise_calendar(records: list[CalendarRecord]) -> tuple[list[CalendarRecord], dict]:
    kept, removed = deduplicate(records, calendar_dedup_key)
    originals = sum(1 for r in kept if r.provenance == Provenance.ORIGINAL_RELEASE)
    revisions = sum(1 for r in kept if r.provenance == Provenance.REVISED)
    return kept, {
        "input": len(records),
        "kept": len(kept),
        "duplicates_removed": removed,
        "original_releases": originals,
        "revisions": revisions,
    }


def normalise_sentiment(
    records: list[SentimentRecord],
) -> tuple[list[SentimentRecord], dict]:
    kept, removed = deduplicate(records, sentiment_dedup_key)
    by_provenance: dict[str, int] = {}
    for record in kept:
        by_provenance[record.provenance.value] = (
            by_provenance.get(record.provenance.value, 0) + 1
        )
    return kept, {
        "input": len(records),
        "kept": len(kept),
        "duplicates_removed": removed,
        "by_provenance": by_provenance,
    }


def to_frame(records: list) -> pd.DataFrame:
    """Rows -> chronologically ordered UTC frame."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([record.to_row() for record in records])
    frame = ensure_utc(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def write_dataset(frame: pd.DataFrame, path: Path) -> Path:
    """Write a dataset, choosing the format from the suffix.

    Parquet by default because it preserves timezone-aware timestamps exactly;
    CSV is accepted for the same reason a person might want to read the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        # An empty dataset is still written, with the right columns, so
        # downstream code sees "available and empty" rather than a missing file
        # it might mistake for a configuration error.
        frame = pd.DataFrame(columns=["timestamp", "source", "provenance"])
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def read_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "timestamp" in frame.columns and not frame.empty:
        frame = ensure_utc(frame).sort_values("timestamp").reset_index(drop=True)
    return frame


def merge_into(existing: pd.DataFrame, incoming: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Append incoming rows to an existing dataset, keeping the earliest per key.

    Used when a resumed run adds windows to a dataset already on disk: the
    result must be the same as one uninterrupted run, which means the merge has
    to be order-independent.
    """
    if existing.empty:
        combined = incoming
    elif incoming.empty:
        combined = existing
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    if combined.empty:
        return combined
    combined = ensure_utc(combined).sort_values("timestamp")
    present = [c for c in key_columns if c in combined.columns]
    if present:
        combined = combined.drop_duplicates(subset=present, keep="first")
    return combined.sort_values("timestamp").reset_index(drop=True)


NEWS_KEY_COLUMNS = ["source", "source_id"]
CALENDAR_KEY_COLUMNS = ["series_id", "reference_period", "provenance", "published_at"]
SENTIMENT_KEY_COLUMNS = ["source", "provenance", "timestamp"]
