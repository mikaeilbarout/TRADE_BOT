"""S&P 500 filter for the Donchian M15 bot (added 2026-09-30).

Skip a gold trade when the S&P 500 moved >= threshold % IN THE SAME direction as the trade over
the last `lookback_hours` (168 h = 7 calendar days = 5 trading days): gold then trades like a risk
asset and its breakouts fail more often. Research and numbers:
research_20260929_tight_stop/spx_filter.py (last 30% of 4 years +73.3R vs +64.4R, drawdown
14.8R vs 24.1R; 6-month ticks +$1600 vs +$1156).

Pure functions only -- no MetaTrader5 import -- so the live bot and the tests share one
implementation. All times are the broker SERVER clock (what MT5 stamps on bars), naive.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

H1 = timedelta(hours=1)
M15 = timedelta(minutes=15)


def spx_move_pct(bar_open_times, closes, decision: datetime, lookback_hours: float = 168.0):
    """% change of the S&P 500 from `lookback_hours` before the last H1 bar that had CLOSED at
    `decision` (bar open + 1 h <= decision) to that bar's close.

    bar_open_times: ascending H1 bar open times (server clock); closes: their closes.
    Returns None when there is not enough history or the data is unusable.
    """
    times = np.asarray(bar_open_times, dtype="datetime64[ns]")
    closes = np.asarray(closes, dtype=float)
    if len(times) == 0 or len(times) != len(closes):
        return None
    last = int(np.searchsorted(times, np.datetime64(decision - H1, "ns"), side="right")) - 1
    if last < 0:
        return None
    past = int(np.searchsorted(times, times[last] - np.timedelta64(int(lookback_hours * 3600), "s"), side="right")) - 1
    if past < 0 or past >= last:
        return None
    a, b = closes[past], closes[last]
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0:
        return None
    return float((b / a - 1.0) * 100.0)


def allows_entry(move_pct, side: str, threshold_pct: float) -> bool:
    """True = the trade may go ahead. Unknown move (None) never blocks (fail-open)."""
    if move_pct is None:
        return True
    direction = 1.0 if side == "long" else -1.0
    return move_pct * direction < threshold_pct


def decision_time(signal_bar_open: datetime) -> datetime:
    """The bot acts on the last CLOSED M15 bar, i.e. at that bar's close."""
    return signal_bar_open + M15
