from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.data.ingest.base import (
    CalendarRecord,
    FetchWindow,
    HistoricalSource,
    NewsRecord,
    SentimentRecord,
    SourceUnavailable,
)
from research.data.ingest.redact import redact
from research.data.ingest.checkpoint import (
    STATUS_DONE,
    STATUS_EMPTY,
    STATUS_FAILED,
    IngestCheckpoint,
    WindowProgress,
)
from research.data.ingest.normalize import (
    CALENDAR_KEY_COLUMNS,
    NEWS_KEY_COLUMNS,
    SENTIMENT_KEY_COLUMNS,
    merge_into,
    normalise_calendar,
    normalise_news,
    normalise_sentiment,
    read_dataset,
    to_frame,
    write_dataset,
)

"""The orchestrator: run a source over a date range, resumably.

The loop is deliberately dull -- ask the checkpoint which windows are settled,
fetch the rest, commit each one before moving on. What matters is what it
refuses to do:

* It never treats a failure as an empty period. A window that errored is
  recorded `failed` and retried next run; only a window the source confirmed
  had nothing becomes `empty`.
* It never partially writes. Records are accumulated, normalised, merged with
  whatever is already on disk, and written once per flush, so an interrupted
  run leaves a valid dataset rather than a truncated one.
* It stops on a source-level problem. A missing API key or a blocked host fails
  the whole run immediately with that reason, instead of producing thousands of
  identical failures and a dataset that looks merely sparse.
"""


@dataclass
class IngestOutcome:
    source: str
    kind: str
    windows_total: int = 0
    windows_settled_before: int = 0
    windows_fetched: int = 0
    windows_empty: int = 0
    windows_failed: int = 0
    records_written: int = 0
    duplicates_removed: int = 0
    bytes_fetched: int = 0
    requests_made: int = 0
    stopped_early: bool = False
    stopped_reason: str | None = None
    dataset_path: Path | None = None
    normalisation: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["dataset_path"] = str(self.dataset_path) if self.dataset_path else None
        payload["errors"] = self.errors[:10]
        return payload


class IngestPipeline:
    """Runs one source over a range, writing a validated dataset."""

    def __init__(
        self,
        source: HistoricalSource,
        checkpoint: IngestCheckpoint,
        kind: str,
        requests_per_second: float = 4.0,
        flush_every: int = 500,
    ) -> None:
        self._source = source
        self._checkpoint = checkpoint
        self._kind = kind
        # Self-imposed politeness. GDELT publishes no quota and FRED allows ~120
        # requests/minute; staying well inside both is cheaper than being
        # throttled, and a backfill is not urgent.
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._flush_every = flush_every

    def run(
        self,
        start: datetime,
        end: datetime,
        dataset_path: Path,
        max_windows: int | None = None,
        sentiment_mode: bool = False,
        progress_every: int = 200,
    ) -> IngestOutcome:
        outcome = IngestOutcome(source=self._source.spec.key, kind=self._kind)

        blocked = self._source.preflight()
        if blocked:
            outcome.stopped_early = True
            outcome.stopped_reason = redact(blocked)
            outcome.errors.append(redact(blocked))
            return outcome

        windows = self._source.windows(start, end)
        outcome.windows_total = len(windows)
        settled = self._checkpoint.settled_windows(self._source.spec.key, self._kind)
        pending = [w for w in windows if w.key not in settled]
        outcome.windows_settled_before = len(windows) - len(pending)
        if max_windows is not None:
            pending = pending[:max_windows]

        self._checkpoint.record_source_spec(
            self._source.spec.key,
            {
                "kind": self._kind,
                "range_start": start.isoformat(),
                "range_end": end.isoformat(),
                "windows": len(windows),
                **(
                    self._source.config_fingerprint()
                    if hasattr(self._source, "config_fingerprint")
                    else {}
                ),
            },
        )

        buffer: list = []
        last_request = 0.0

        for index, window in enumerate(pending, start=1):
            elapsed = time.monotonic() - last_request
            if self._min_interval and elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            last_request = time.monotonic()

            try:
                result = (
                    self._source.sentiment_from_window(window)  # type: ignore[attr-defined]
                    if sentiment_mode
                    else self._source.fetch_window(window)
                )
            except SourceUnavailable as exc:
                # A source-level failure (blocked host, bad key, rate limit) is
                # not a per-window problem: stop, keep everything already
                # fetched, and report it once.
                self._record(window, STATUS_FAILED, 0, 0, 0, str(exc))
                outcome.windows_failed += 1
                outcome.stopped_early = True
                outcome.stopped_reason = redact(str(exc))
                outcome.errors.append(redact(str(exc)))
                break

            if result.failed:
                self._record(window, STATUS_FAILED, 0, 0, result.requests_made, result.error)
                outcome.windows_failed += 1
                if result.error:
                    outcome.errors.append(redact(f"{window.key}: {result.error}"))
                continue

            outcome.bytes_fetched += result.bytes_fetched
            outcome.requests_made += result.requests_made

            if result.empty and not result.records:
                self._record(window, STATUS_EMPTY, 0, result.bytes_fetched, result.requests_made)
                outcome.windows_empty += 1
                continue

            buffer.extend(result.records)
            outcome.windows_fetched += 1
            self._record(
                window,
                STATUS_DONE,
                len(result.records),
                result.bytes_fetched,
                result.requests_made,
            )

            if len(buffer) >= self._flush_every:
                written, stats = self._flush(buffer, dataset_path)
                outcome.records_written += written
                outcome.duplicates_removed += stats.get("duplicates_removed", 0)
                buffer = []

            if progress_every and index % progress_every == 0:
                print(
                    f"  {self._source.spec.key}: {index}/{len(pending)} windows, "
                    f"{outcome.records_written + len(buffer)} records",
                    flush=True,
                )

        if buffer:
            written, stats = self._flush(buffer, dataset_path)
            outcome.records_written += written
            outcome.duplicates_removed += stats.get("duplicates_removed", 0)
            outcome.normalisation = stats

        outcome.dataset_path = dataset_path
        return outcome

    # --- helpers ----------------------------------------------------------
    def _record(
        self,
        window: FetchWindow,
        status: str,
        records: int,
        bytes_fetched: int,
        requests: int,
        error: str | None = None,
    ) -> None:
        # Redact before the error is committed: the checkpoint is durable and is
        # read back by the status and readiness reports.
        error = redact(error)
        self._checkpoint.record(
            WindowProgress(
                source=self._source.spec.key,
                window_key=window.key,
                kind=self._kind,
                status=status,
                window_start=window.start,
                window_end=window.end,
                record_count=records,
                bytes_fetched=bytes_fetched,
                requests_made=requests,
                error=error,
            )
        )

    def _flush(self, records: list, dataset_path: Path) -> tuple[int, dict]:
        """Normalise, merge with what is on disk, and write atomically-ish.

        Merging rather than appending means a resumed run converges on the same
        dataset an uninterrupted run would have produced, regardless of the
        order windows completed in.
        """
        if not records:
            return 0, {}

        first = records[0]
        if isinstance(first, NewsRecord):
            kept, stats = normalise_news(records)
            keys = NEWS_KEY_COLUMNS
        elif isinstance(first, CalendarRecord):
            kept, stats = normalise_calendar(records)
            keys = CALENDAR_KEY_COLUMNS
        elif isinstance(first, SentimentRecord):
            kept, stats = normalise_sentiment(records)
            keys = SENTIMENT_KEY_COLUMNS
        else:
            raise TypeError(f"unsupported record type {type(first).__name__}")

        incoming = to_frame(kept)
        existing = read_dataset(dataset_path)
        merged = merge_into(existing, incoming, list(keys))
        write_dataset(merged, dataset_path)
        return len(merged), stats


def coverage_gaps(
    frame: pd.DataFrame, start: datetime, end: datetime, max_gap_hours: float = 48.0
) -> list[dict]:
    """Periods inside [start, end] with no records at all.

    Reported so "missing date ranges" is a concrete list rather than an
    impression, and so a dataset that silently covers half the requested range
    cannot pass for a complete one.
    """
    if frame.empty:
        return [
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "hours": round((end - start).total_seconds() / 3600, 1),
                "reason": "no records at all",
            }
        ]
    stamps = pd.to_datetime(frame["timestamp"], utc=True).sort_values()
    gaps: list[dict] = []
    cursor = pd.Timestamp(start)
    for stamp in stamps:
        delta = (stamp - cursor).total_seconds() / 3600
        if delta > max_gap_hours:
            gaps.append(
                {
                    "start": cursor.isoformat(),
                    "end": stamp.isoformat(),
                    "hours": round(delta, 1),
                    "reason": "no records in this period",
                }
            )
        cursor = max(cursor, stamp)
    tail = (pd.Timestamp(end) - cursor).total_seconds() / 3600
    if tail > max_gap_hours:
        gaps.append(
            {
                "start": cursor.isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "hours": round(tail, 1),
                "reason": "no records after this point",
            }
        )
    return gaps
