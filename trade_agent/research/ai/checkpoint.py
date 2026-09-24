from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from research.ai.cost import CallRecord, TokenUsage

"""Checkpoint / resume store.

Design requirement: restarting after a crash at signal 1,237 of 5,000 must
resume at 1,238 without re-issuing 1,237 paid requests. That means every
completed signal is committed to durable storage BEFORE the next one starts,
and spend is recovered from the store on resume so the budget guard isn't
reset by a restart.

SQLite (not JSONL) because resume needs random access by signal_id and an
atomic commit per signal; a half-written JSON line at the end of a crashed
run is exactly the corruption this must not have.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_progress (
    signal_id           TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    decision            TEXT,
    confidence          REAL,
    payload             TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    prompt_versions     TEXT,
    models              TEXT,
    run_id              TEXT,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS call_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id           TEXT NOT NULL,
    agent               TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL,
    cache_creation_tokens INTEGER NOT NULL,
    cache_read_tokens   INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    latency_seconds     REAL NOT NULL,
    batch               INTEGER NOT NULL DEFAULT 0,
    prompt_version      TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_call_log_signal ON call_log(signal_id);
CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SignalProgress(BaseModel):
    signal_id: str
    status: str  # pending | done | failed | skipped_deterministic | budget_stopped
    decision: str | None = None
    confidence: float | None = None
    payload: dict = Field(default_factory=dict)
    cost_usd: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    prompt_versions: dict = Field(default_factory=dict)
    models: dict = Field(default_factory=dict)
    run_id: str | None = None


class CheckpointStore:
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

    # --- progress ---------------------------------------------------------
    def save_progress(self, progress: SignalProgress) -> None:
        """Commit one signal's outcome atomically. Called after each signal so
        a crash loses at most the in-flight signal."""
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO signal_progress (
                    signal_id, status, decision, confidence, payload, cost_usd,
                    input_tokens, cache_creation_tokens, cache_read_tokens,
                    output_tokens, prompt_versions, models, run_id, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    status=excluded.status,
                    decision=excluded.decision,
                    confidence=excluded.confidence,
                    payload=excluded.payload,
                    cost_usd=excluded.cost_usd,
                    input_tokens=excluded.input_tokens,
                    cache_creation_tokens=excluded.cache_creation_tokens,
                    cache_read_tokens=excluded.cache_read_tokens,
                    output_tokens=excluded.output_tokens,
                    prompt_versions=excluded.prompt_versions,
                    models=excluded.models,
                    run_id=excluded.run_id,
                    updated_at=excluded.updated_at
                """,
                (
                    progress.signal_id,
                    progress.status,
                    progress.decision,
                    progress.confidence,
                    json.dumps(progress.payload, default=str),
                    progress.cost_usd,
                    progress.usage.input_tokens,
                    progress.usage.cache_creation_tokens,
                    progress.usage.cache_read_tokens,
                    progress.usage.output_tokens,
                    json.dumps(progress.prompt_versions),
                    json.dumps(progress.models),
                    progress.run_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_progress(self, signal_id: str) -> SignalProgress | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM signal_progress WHERE signal_id = ?", (signal_id,)
            ).fetchone()
        return _row_to_progress(row) if row else None

    def completed_signal_ids(self) -> set[str]:
        """Signals that must NOT be re-requested on resume.

        A failed signal is deliberately NOT included: a transient API error
        should be retried on the next run, while a completed or
        deterministically-skipped one should never be paid for twice.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT signal_id FROM signal_progress WHERE status IN "
                "('done','skipped_deterministic')"
            ).fetchall()
        return {row["signal_id"] for row in rows}

    def all_progress(self) -> list[SignalProgress]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM signal_progress ORDER BY updated_at"
            ).fetchall()
        return [_row_to_progress(row) for row in rows]

    # --- call log ---------------------------------------------------------
    def log_call(self, record: CallRecord) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO call_log (
                    signal_id, agent, model, input_tokens, cache_creation_tokens,
                    cache_read_tokens, output_tokens, cost_usd, latency_seconds,
                    batch, prompt_version, error, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.signal_id,
                    record.agent,
                    record.model,
                    record.usage.input_tokens,
                    record.usage.cache_creation_tokens,
                    record.usage.cache_read_tokens,
                    record.usage.output_tokens,
                    record.cost_usd,
                    record.latency_seconds,
                    int(record.batch),
                    record.prompt_version,
                    record.error,
                    record.created_at.isoformat(),
                ),
            )
            conn.commit()

    def all_calls(self) -> list[CallRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM call_log ORDER BY id").fetchall()
        return [
            CallRecord(
                signal_id=row["signal_id"],
                agent=row["agent"],
                model=row["model"],
                usage=TokenUsage(
                    input_tokens=row["input_tokens"],
                    cache_creation_tokens=row["cache_creation_tokens"],
                    cache_read_tokens=row["cache_read_tokens"],
                    output_tokens=row["output_tokens"],
                ),
                cost_usd=row["cost_usd"],
                latency_seconds=row["latency_seconds"],
                batch=bool(row["batch"]),
                prompt_version=row["prompt_version"],
                error=row["error"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def total_spend(self) -> float:
        """Recover spend from disk so the budget survives a restart."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) AS total FROM call_log").fetchone()
        return float(row["total"])

    # --- run metadata -----------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO run_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()

    def get_meta(self, key: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM run_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def _row_to_progress(row: sqlite3.Row) -> SignalProgress:
    return SignalProgress(
        signal_id=row["signal_id"],
        status=row["status"],
        decision=row["decision"],
        confidence=row["confidence"],
        payload=json.loads(row["payload"]) if row["payload"] else {},
        cost_usd=row["cost_usd"],
        usage=TokenUsage(
            input_tokens=row["input_tokens"],
            cache_creation_tokens=row["cache_creation_tokens"],
            cache_read_tokens=row["cache_read_tokens"],
            output_tokens=row["output_tokens"],
        ),
        prompt_versions=json.loads(row["prompt_versions"]) if row["prompt_versions"] else {},
        models=json.loads(row["models"]) if row["models"] else {},
        run_id=row["run_id"],
    )
