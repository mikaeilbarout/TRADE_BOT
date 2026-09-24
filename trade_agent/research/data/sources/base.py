from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

TICK_COLUMNS = ["timestamp", "bid", "ask", "bid_volume", "ask_volume"]


class TickDataUnavailableError(Exception):
    """Raised when real tick data cannot be obtained for a requested period.

    This is deliberately fatal. The system never substitutes synthetic or
    interpolated ticks for missing market data -- a backtest built on
    invented prices is worse than no backtest, because it looks credible.
    """


class TickSource(ABC):
    """A source of real historical tick data.

    Implementations must return actual vendor data or raise
    TickDataUnavailableError. They must never generate, interpolate or
    forward-fill ticks that the vendor did not provide.
    """

    name: str = "base"

    @abstractmethod
    def fetch_hour(self, symbol: str, hour_start: datetime) -> pd.DataFrame:
        """Return ticks for one UTC hour as a DataFrame with TICK_COLUMNS.

        An empty DataFrame is a legitimate result (market closed, no ticks);
        an unreachable vendor is not, and must raise.
        """
        raise NotImplementedError

    def available_range(self, symbol: str) -> tuple[date, date] | None:
        """Optional: the vendor's coverage, when it can be determined."""
        return None


def empty_tick_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
            "bid": pd.Series([], dtype="float64"),
            "ask": pd.Series([], dtype="float64"),
            "bid_volume": pd.Series([], dtype="float64"),
            "ask_volume": pd.Series([], dtype="float64"),
        }
    )


def validate_tick_frame(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Reject structurally impossible tick data rather than trading on it."""
    missing = [c for c in TICK_COLUMNS if c not in frame.columns]
    if missing:
        raise TickDataUnavailableError(f"{context}: tick frame missing columns {missing}")
    if frame.empty:
        return frame

    if frame["timestamp"].isna().any():
        raise TickDataUnavailableError(f"{context}: tick frame contains null timestamps")
    if (frame["bid"] <= 0).any() or (frame["ask"] <= 0).any():
        raise TickDataUnavailableError(f"{context}: tick frame contains non-positive prices")

    crossed = frame["ask"] < frame["bid"]
    if crossed.any():
        raise TickDataUnavailableError(
            f"{context}: {int(crossed.sum())} ticks have ask < bid (crossed book)"
        )
    return frame.sort_values("timestamp").reset_index(drop=True)
