from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from research.data.ingest.base import SourceUnavailable
from research.data.ingest.redact import redact_url

"""Content-addressed HTTP cache for historical downloads.

Three properties this has to have, all of them for the same reason -- a
five-year backfill that cannot be re-run identically is not reproducible:

1. **Deterministic paths.** A URL maps to exactly one file on disk, derived
   from the URL itself, so two runs agree on where the bytes live.
2. **Checksums.** Every artifact is stored with the SHA-256 of its bytes, and
   when the provider publishes its own checksum (GDELT ships an MD5 per file)
   that is verified on download and recorded. A record can then be traced to
   bytes, and the bytes to the provider's own hash.
3. **Never re-download.** A cached artifact is returned from disk. Historical
   data does not change; re-fetching it wastes the provider's bandwidth and
   risks getting a different answer.

The cache stores raw provider bytes, unparsed. Keeping the original payload
means a parser bug is fixable without re-downloading five years of data, and
the exact vendor response stays available for verification.
"""

SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _slug(value: str, limit: int = 60) -> str:
    cleaned = "".join(c if c in SAFE_CHARS else "-" for c in value)
    return cleaned[:limit].strip("-") or "item"


@dataclass
class CachedArtifact:
    path: Path
    sha256: str
    bytes_len: int
    from_cache: bool
    url: str
    retrieved_at: datetime
    provider_checksum: str | None = None
    provider_checksum_algorithm: str | None = None

    @property
    def relative_name(self) -> str:
        return self.path.name


class ArtifactCache:
    """Disk cache keyed by URL, with a sidecar metadata file per artifact."""

    def __init__(
        self,
        root: Path,
        client: httpx.Client | None = None,
        timeout_seconds: float = 60.0,
        user_agent: str = "trade-agent-research/1.0 (historical backtest ingestion)",
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._timeout = timeout_seconds
        self._user_agent = user_agent

    # --- paths ------------------------------------------------------------
    def path_for(self, url: str, suffix: str | None = None) -> Path:
        """Deterministic cache path for a URL.

        The directory mirrors the host and the leading path segments so a
        person can find an artifact by eye; the filename ends in a hash of the
        full URL so two URLs cannot collide.
        """
        parsed = urlparse(url)
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        segments = [s for s in parsed.path.split("/") if s][:-1]
        leaf = parsed.path.rsplit("/", 1)[-1] or "index"
        name = f"{_slug(leaf)}.{digest}{suffix or ''}"
        return self._root.joinpath(_slug(parsed.netloc), *[_slug(s, 30) for s in segments], name)

    def meta_path(self, artifact: Path) -> Path:
        return artifact.with_suffix(artifact.suffix + ".meta.json")

    # --- reads and writes -------------------------------------------------
    def peek(self, url: str, suffix: str | None = None) -> CachedArtifact | None:
        path = self.path_for(url, suffix)
        meta = self.meta_path(path)
        if not path.exists() or not meta.exists():
            return None
        payload = json.loads(meta.read_text())
        return CachedArtifact(
            path=path,
            sha256=payload["sha256"],
            bytes_len=payload["bytes_len"],
            from_cache=True,
            url=payload["url"],
            retrieved_at=datetime.fromisoformat(payload["retrieved_at"]),
            provider_checksum=payload.get("provider_checksum"),
            provider_checksum_algorithm=payload.get("provider_checksum_algorithm"),
        )

    def fetch(
        self,
        url: str,
        suffix: str | None = None,
        expected_checksum: str | None = None,
        checksum_algorithm: str = "md5",
        allow_missing: bool = False,
    ) -> CachedArtifact | None:
        """Return the artifact for a URL, downloading only if not cached.

        `expected_checksum` is the provider's own hash when it publishes one.
        A mismatch raises rather than storing the bytes: a corrupted or
        substituted artifact must not silently become part of the dataset.

        `allow_missing` turns a 404 into None, for sources that legitimately
        have no file for a period (GDELT has gaps; a quiet 15 minutes has no
        file at all).
        """
        cached = self.peek(url, suffix)
        if cached is not None:
            return cached

        payload = self._download(url, allow_missing=allow_missing)
        if payload is None:
            return None

        digest = hashlib.sha256(payload).hexdigest()
        if expected_checksum:
            actual = hashlib.new(checksum_algorithm, payload).hexdigest()
            if actual.lower() != expected_checksum.lower():
                raise SourceUnavailable(
                    f"checksum mismatch for {redact_url(url)}: provider published "
                    f"{checksum_algorithm} {expected_checksum}, downloaded bytes hash "
                    f"to {actual}. Refusing to cache a payload that is not the one "
                    "the provider vouched for."
                )

        path = self.path_for(url, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        retrieved_at = datetime.now(timezone.utc)
        self.meta_path(path).write_text(
            json.dumps(
                {
                    # Redacted: this file is written to disk and read by reports.
                    # The cache path comes from a hash of the full URL, so
                    # redacting here cannot break lookups.
                    "url": redact_url(url),
                    "sha256": digest,
                    "bytes_len": len(payload),
                    "retrieved_at": retrieved_at.isoformat(),
                    "provider_checksum": expected_checksum,
                    "provider_checksum_algorithm": (
                        checksum_algorithm if expected_checksum else None
                    ),
                },
                indent=2,
            )
        )
        return CachedArtifact(
            path=path,
            sha256=digest,
            bytes_len=len(payload),
            from_cache=False,
            url=url,
            retrieved_at=retrieved_at,
            provider_checksum=expected_checksum,
            provider_checksum_algorithm=checksum_algorithm if expected_checksum else None,
        )

    def _download(self, url: str, allow_missing: bool) -> bytes | None:
        client = self._client or httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": self._user_agent},
        )
        try:
            response = client.get(url)
            if response.status_code == 404 and allow_missing:
                return None
            if response.status_code in (401, 403):
                raise SourceUnavailable(
                    f"{redact_url(url)} returned {response.status_code}. Either this "
                    "host is not "
                    "permitted by the environment's network egress policy, or the "
                    "provider rejected the request (missing or invalid API key). "
                    "No data is substituted."
                )
            if response.status_code == 429:
                raise SourceUnavailable(
                    f"{redact_url(url)} returned 429 (rate limited). Lower the request "
                    "rate or "
                    "resume later -- the run is resumable, so nothing already "
                    "downloaded is lost."
                )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            raise SourceUnavailable(
                f"{redact_url(url)} failed: {redact_url(str(exc))}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SourceUnavailable(
                f"{redact_url(url)} unreachable: {redact_url(str(exc))}. If this host "
                "is blocked by the "
                "environment's egress policy, allow it there; no synthetic data "
                "will be substituted."
            ) from exc
        finally:
            if self._client is None:
                client.close()

    # --- inventory --------------------------------------------------------
    def artifacts(self) -> list[CachedArtifact]:
        found: list[CachedArtifact] = []
        for meta in sorted(self._root.rglob("*.meta.json")):
            payload = json.loads(meta.read_text())
            artifact = meta.with_suffix("")
            artifact = artifact.with_suffix("")  # strip .meta then .json
            real = Path(str(meta)[: -len(".meta.json")])
            if not real.exists():
                continue
            found.append(
                CachedArtifact(
                    path=real,
                    sha256=payload["sha256"],
                    bytes_len=payload["bytes_len"],
                    from_cache=True,
                    url=payload["url"],
                    retrieved_at=datetime.fromisoformat(payload["retrieved_at"]),
                    provider_checksum=payload.get("provider_checksum"),
                    provider_checksum_algorithm=payload.get("provider_checksum_algorithm"),
                )
            )
        return found

    def summary(self) -> dict:
        artifacts = self.artifacts()
        return {
            "root": str(self._root),
            "artifacts": len(artifacts),
            "bytes": sum(a.bytes_len for a in artifacts),
            "with_provider_checksum": sum(1 for a in artifacts if a.provider_checksum),
        }

    def verify(self) -> list[str]:
        """Re-hash every cached artifact against its recorded SHA-256.

        Catches disk corruption and hand-editing of the cache, either of which
        would make the dataset something other than what was downloaded.
        """
        problems: list[str] = []
        for artifact in self.artifacts():
            actual = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
            if actual != artifact.sha256:
                problems.append(
                    f"{artifact.path}: recorded sha256 {artifact.sha256[:12]}, "
                    f"on-disk bytes hash to {actual[:12]}"
                )
        return problems
