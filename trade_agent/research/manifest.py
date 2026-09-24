from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def sha256_dir(path: Path, pattern: str = "**/*") -> str:
    """Stable hash over a directory's file contents, so a prompt set or a
    dataset directory can be versioned by content rather than by mtime."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.glob(pattern) if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class DatasetVersion(BaseModel):
    """Identity of one input dataset. `available=False` records that a
    dataset was genuinely missing for this run -- the spec forbids
    substituting fabricated data, so absence is recorded, not filled in."""

    name: str
    source: str
    available: bool
    row_count: int | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    content_hash: str | None = None
    note: str | None = None


class RunManifest(BaseModel):
    """Everything needed to reproduce one backtest run (spec: Reproducibility).

    Written next to the results of every run. Two runs with the same manifest
    hash used identical data, parameters, prompts and model.
    """

    run_id: str
    run_kind: str  # "baseline" | "ai" | "optimization"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    backtest_config: dict
    strategy_name: str
    strategy_params: dict
    strategy_seal_hash: str | None = None

    datasets: list[DatasetVersion] = Field(default_factory=list)

    # AI-run provenance (None for the baseline run).
    llm_model: str | None = None
    llm_provider: str | None = None
    prompts_hash: str | None = None
    agent_settings: dict | None = None

    # Known methodological limitations carried WITH the results, so a
    # reader of the manifest cannot see the numbers without seeing what
    # qualifies them (e.g. LLM knowledge of the test period).
    known_limitations: list[dict] = Field(default_factory=list)

    random_seed: int
    git_revision: str | None = Field(default_factory=git_revision)
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)

    def manifest_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"run_id", "created_at"})
        return sha256_obj(payload)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        data["manifest_hash"] = self.manifest_hash()
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path
