from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

"""Resumable progress for a multi-year backfill.

The requirement: if downloading five years of data fails halfway, the next run
continues instead of starting over. That means every completed window is
committed to durable storage before the next one starts, and a window's state
distinguishes the three outcomes that matter:

* `done`    -- fetched and parsed; never re-fetched.
* `empty`   -- fetched, and the source genuinely had nothing. Also never
               re-fetched: "no news in those fifteen minutes" is an answer.
* `failed`  -- retried on the next run. A transient network error must not
               permanently blank a period, because a blank period is
               indistinguishable from a quiet one once it is recorded.

SQLite rather than a progress file: resume needs random access by window key
and an atomic commit per window, and a half-written line at the end of a
crashed run is exactly the corruption this must not have.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS window_progress (
    source          TEXT NOT NULL,
    window_key      TEXT NOT NULL,
    kind            TEXT NOT NULL,
    status          TEXT NOT NULL,
    window_start    TEXT NOT NULL,
    window_end      TEXT NOT NULL,
    record_count    INTEGER NOT NULL DEFAULT 0,
    bytes_fetched   INTEGER NOT NULL DEFAULT 0,
    requests_made   INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (source, kind, window_key)
);

CREATE INDEX IF NOT EXISTS idx_window_status ON window_progress(source, kind, status);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

STATUS_DONE = "done"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"
# Statuses that must never be re-fetched.
SETTLED = (STATUS_DONE, STATUS_EMPTY)


class WindowProgress(BaseModel):
    source: str
    window_key: str
    kind: str
    status: str
    window_start: datetime
    window_end: datetime
    record_count: int = 0
    bytes_fetched: int = 0
    requests_made: int = 0
    error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestCheckpoint:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, progress: WindowProgress) -> None:
        """Commit one window's outcome atomically."""
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO window_progress (
                    source, window_key, kind, status, window_start, window_end,
                    record_count, bytes_fetched, requests_made, error, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source, kind, window_key) DO UPDATE SET
                    status=excluded.status,
                    record_count=excluded.record_count,
                    bytes_fetched=excluded.bytes_fetched,
                    requests_made=excluded.requests_made,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (
                    progress.source,
                    progress.window_key,
                    progress.kind,
                    progress.status,
                    progress.window_start.isoformat(),
                    progress.window_end.isoformat(),
                    progress.record_count,
                    progress.bytes_fetched,
                    progress.requests_made,
                    progress.error,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def settled_windows(self, source: str, kind: str) -> set[str]:
        """Windows that must not be re-fetched.

        Scoped by (source, kind): news and sentiment are both built from the
        same GDELT windows under the same source key, so a source-only lookup
        would see news's "done" rows and wrongly skip sentiment's fetch of
        the identical window keys.

        A failed window is deliberately absent: it is retried next run, so a
        transient error never becomes a permanent hole in the dataset.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT window_key FROM window_progress "
                "WHERE source = ? AND kind = ? AND status IN (?, ?)",
                (source, kind, STATUS_DONE, STATUS_EMPTY),
            ).fetchall()
        return {row["window_key"] for row in rows}

    def failed_windows(self, source: str, kind: str | None = None) -> list[WindowProgress]:
        """Failed windows for a source, optionally narrowed to one kind.

        `kind=None` reports across every kind that shares this source (used
        by the aggregate status view); pipeline resumability always passes
        a specific kind.
        """
        with closing(self._connect()) as conn:
            if kind is None:
                rows = conn.execute(
                    "SELECT * FROM window_progress WHERE source = ? AND status = ? "
                    "ORDER BY window_start",
                    (source, STATUS_FAILED),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM window_progress WHERE source = ? AND kind = ? "
                    "AND status = ? ORDER BY window_start",
                    (source, kind, STATUS_FAILED),
                ).fetchall()
        return [_row(row) for row in rows]

    def all_windows(self, source: str | None = None) -> list[WindowProgress]:
        query = "SELECT * FROM window_progress"
        params: tuple = ()
        if source:
            query += " WHERE source = ?"
            params = (source,)
        query += " ORDER BY source, window_start"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row(row) for row in rows]

    def stats(self, source: str | None = None) -> dict:
        windows = self.all_windows(source)
        by_status: dict[str, int] = {}
        for window in windows:
            by_status[window.status] = by_status.get(window.status, 0) + 1
        return {
            "windows": len(windows),
            "by_status": by_status,
            "records": sum(w.record_count for w in windows),
            "bytes_fetched": sum(w.bytes_fetched for w in windows),
            "requests_made": sum(w.requests_made for w in windows),
            "first_window": min((w.window_start for w in windows), default=None),
            "last_window": max((w.window_end for w in windows), default=None),
        }

    def gaps(self, source: str, kind: str) -> list[tuple[datetime, datetime]]:
        """Windows that failed or were never attempted, as date ranges.

        This is what the status report turns into "missing date ranges" -- a
        gap the operator can see and act on, rather than a silently short
        dataset.
        """
        failed = [(w.window_start, w.window_end) for w in self.failed_windows(source, kind)]
        return sorted(failed)

    def set_meta(self, key: str, value: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO ingest_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()

    def get_meta(self, key: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT value FROM ingest_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def record_source_spec(self, source: str, spec: dict) -> None:
        """Store the source configuration a dataset was built with.

        Part of reproducibility: the dataset on disk is only meaningful
        alongside the query parameters and filters that produced it.
        """
        self.set_meta(f"spec::{source}", json.dumps(spec, sort_keys=True, default=str))

    def source_spec(self, source: str) -> dict | None:
        raw = self.get_meta(f"spec::{source}")
        return json.loads(raw) if raw else None


def _row(row: sqlite3.Row) -> WindowProgress:
    return WindowProgress(
        source=row["source"],
        window_key=row["window_key"],
        kind=row["kind"],
        status=row["status"],
        window_start=datetime.fromisoformat(row["window_start"]),
        window_end=datetime.fromisoformat(row["window_end"]),
        record_count=row["record_count"],
        bytes_fetched=row["bytes_fetched"],
        requests_made=row["requests_made"],
        error=row["error"],
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
