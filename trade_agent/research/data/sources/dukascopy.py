from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

from research.data.sources.base import (
    TickDataUnavailableError,
    TickSource,
    empty_tick_frame,
    validate_tick_frame,
)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Each tick is 20 bytes, big-endian:
#   uint32 milliseconds offset from the hour
#   uint32 ask in integer points
#   uint32 bid in integer points
#   float32 ask volume
#   float32 bid volume
_TICK_STRUCT = struct.Struct(">IIIff")
_TICK_SIZE = _TICK_STRUCT.size


def hour_url(symbol: str, hour_start: datetime) -> str:
    """Build the bi5 URL for one hour.

    Dukascopy's month component is ZERO-INDEXED (January is 00) while day and
    hour are one/zero-based as you'd expect. Getting this wrong silently
    fetches the wrong month's data, so it is asserted in tests.
    """
    hour_start = hour_start.astimezone(timezone.utc)
    return (
        f"{BASE_URL}/{symbol.upper()}/{hour_start.year:04d}/"
        f"{hour_start.month - 1:02d}/{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"
    )


def decode_bi5(payload: bytes, hour_start: datetime, point_divisor: float) -> pd.DataFrame:
    """Decode one LZMA-compressed bi5 hour into a tick frame.

    An empty payload means the vendor has no ticks for that hour (weekend,
    holiday, halt) -- that is valid and returns an empty frame.
    """
    if not payload:
        return empty_tick_frame()

    try:
        raw = lzma.decompress(payload)
    except lzma.LZMAError as exc:
        raise TickDataUnavailableError(
            f"corrupt bi5 payload for {hour_start.isoformat()}: {exc}"
        ) from exc

    if len(raw) % _TICK_SIZE != 0:
        raise TickDataUnavailableError(
            f"bi5 payload for {hour_start.isoformat()} is {len(raw)} bytes, "
            f"not a multiple of the {_TICK_SIZE}-byte tick record"
        )

    count = len(raw) // _TICK_SIZE
    if count == 0:
        return empty_tick_frame()

    ms, ask_points, bid_points, ask_volume, bid_volume = zip(
        *_TICK_STRUCT.iter_unpack(raw)
    )
    hour_start = hour_start.astimezone(timezone.utc)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(ms), unit="ms", utc=True)
            + (pd.Timestamp(hour_start) - pd.Timestamp("1970-01-01", tz="UTC")),
            "bid": [p / point_divisor for p in bid_points],
            "ask": [p / point_divisor for p in ask_points],
            "bid_volume": list(bid_volume),
            "ask_volume": list(ask_volume),
        }
    )
    return validate_tick_frame(frame, f"dukascopy {hour_start.isoformat()}")


class DukascopyTickSource(TickSource):
    """Dukascopy historical tick feed (free, no account required).

    Data is fetched one hour at a time and cached on disk as the raw bi5
    bytes, so a re-run never re-downloads and the exact vendor payload stays
    available for verification.
    """

    name = "dukascopy"

    def __init__(
        self,
        cache_dir: Path,
        point_divisor: float = 1000.0,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._point_divisor = point_divisor
        self._timeout = timeout_seconds
        self._client = client

    def _cache_path(self, symbol: str, hour_start: datetime) -> Path:
        hour_start = hour_start.astimezone(timezone.utc)
        return (
            self._cache_dir
            / symbol.upper()
            / f"{hour_start.year:04d}"
            / f"{hour_start.month:02d}"
            / f"{hour_start.day:02d}"
            / f"{hour_start.hour:02d}h_ticks.bi5"
        )

    def _download(self, url: str) -> bytes:
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        try:
            response = client.get(url)
            if response.status_code == 404:
                # Dukascopy returns 404 for hours it has no file for.
                return b""
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            raise TickDataUnavailableError(
                f"Dukascopy unreachable for {url}: {exc}. "
                "Allow datafeed.dukascopy.com in the environment's network policy, "
                "or import a local copy with the CSV source. No synthetic data "
                "will be substituted."
            ) from exc
        finally:
            if self._client is None:
                client.close()

    def fetch_hour(self, symbol: str, hour_start: datetime) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, hour_start)
        if cache_path.exists():
            payload = cache_path.read_bytes()
        else:
            payload = self._download(hour_url(symbol, hour_start))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
        return decode_bi5(payload, hour_start, self._point_divisor)

    def iter_hours(self, start: datetime, end: datetime):
        cursor = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        end = end.astimezone(timezone.utc)
        while cursor < end:
            yield cursor
            cursor += timedelta(hours=1)
