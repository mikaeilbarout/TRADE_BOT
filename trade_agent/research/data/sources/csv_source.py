from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from research.data.sources.base import (
    TickDataUnavailableError,
    TickSource,
    empty_tick_frame,
    validate_tick_frame,
)


class CsvTickSource(TickSource):
    """Reads tick data you already have on disk (any vendor, any export).

    This is the escape hatch for environments where the vendor hosts are
    blocked: point it at your own CSV/Parquet export and the rest of the
    pipeline is identical. Column names are configurable because every
    vendor spells them differently; timestamps are parsed as UTC unless a
    timezone is given.

    Expected shape (after mapping): timestamp, bid, ask, and optionally
    bid_volume / ask_volume. A single `price` column is also accepted, in
    which case the spread is applied symmetrically from `assumed_spread`.
    """

    name = "csv"

    def __init__(
        self,
        path: Path,
        timestamp_column: str = "timestamp",
        bid_column: str = "bid",
        ask_column: str = "ask",
        bid_volume_column: str | None = "bid_volume",
        ask_volume_column: str | None = "ask_volume",
        price_column: str | None = None,
        assumed_spread: float | None = None,
        source_timezone: str = "UTC",
    ) -> None:
        self._path = Path(path)
        self._timestamp_column = timestamp_column
        self._bid_column = bid_column
        self._ask_column = ask_column
        self._bid_volume_column = bid_volume_column
        self._ask_volume_column = ask_volume_column
        self._price_column = price_column
        self._assumed_spread = assumed_spread
        self._source_timezone = source_timezone
        self._frame: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._frame is not None:
            return self._frame

        if not self._path.exists():
            raise TickDataUnavailableError(
                f"tick file not found: {self._path}. Provide your XAUUSD tick export "
                "here, or configure a vendor source."
            )

        if self._path.suffix.lower() in {".parquet", ".pq"}:
            raw = pd.read_parquet(self._path)
        else:
            raw = pd.read_csv(self._path)

        if self._timestamp_column not in raw.columns:
            raise TickDataUnavailableError(
                f"{self._path}: timestamp column {self._timestamp_column!r} not found; "
                f"available columns: {list(raw.columns)}"
            )

        timestamps = pd.to_datetime(raw[self._timestamp_column], utc=False, errors="coerce")
        if timestamps.isna().any():
            bad = int(timestamps.isna().sum())
            raise TickDataUnavailableError(
                f"{self._path}: {bad} rows have unparseable timestamps"
            )
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(self._source_timezone).dt.tz_convert("UTC")
        else:
            timestamps = timestamps.dt.tz_convert("UTC")

        if self._price_column and self._price_column in raw.columns:
            if self._assumed_spread is None:
                raise TickDataUnavailableError(
                    f"{self._path}: only a single price column was provided, so "
                    "assumed_spread must be configured to derive bid/ask. Refusing to "
                    "invent a spread."
                )
            mid = raw[self._price_column].astype(float)
            bid = mid - self._assumed_spread / 2
            ask = mid + self._assumed_spread / 2
        else:
            for column in (self._bid_column, self._ask_column):
                if column not in raw.columns:
                    raise TickDataUnavailableError(
                        f"{self._path}: column {column!r} not found; "
                        f"available columns: {list(raw.columns)}"
                    )
            bid = raw[self._bid_column].astype(float)
            ask = raw[self._ask_column].astype(float)

        def volume(column: str | None) -> pd.Series:
            if column and column in raw.columns:
                return raw[column].astype(float)
            return pd.Series([0.0] * len(raw), dtype="float64")

        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bid": bid.to_numpy(),
                "ask": ask.to_numpy(),
                "bid_volume": volume(self._bid_volume_column).to_numpy(),
                "ask_volume": volume(self._ask_volume_column).to_numpy(),
            }
        )
        self._frame = validate_tick_frame(frame, str(self._path))
        return self._frame

    def fetch_hour(self, symbol: str, hour_start: datetime) -> pd.DataFrame:
        frame = self._load()
        hour_start = hour_start.astimezone(timezone.utc)
        hour_end = hour_start + timedelta(hours=1)
        mask = (frame["timestamp"] >= hour_start) & (frame["timestamp"] < hour_end)
        selected = frame.loc[mask]
        return selected.reset_index(drop=True) if not selected.empty else empty_tick_frame()

    def load_all(self) -> pd.DataFrame:
        return self._load()
