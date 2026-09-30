"""S&P 500 filter: unit tests + equivalence with the backtest implementation on real data."""
import os
import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mt5.spx_filter import allows_entry, decision_time, spx_move_pct  # noqa: E402

T0 = datetime(2026, 9, 21, 0, 0)


def hourly(n, start=T0, step=1.0):
    times = [start + timedelta(hours=i) for i in range(n)]
    closes = [100.0 + step * i for i in range(n)]
    return times, closes


def test_uses_only_closed_bars():
    times, closes = hourly(200)
    dec = times[180] + timedelta(minutes=30)             # bar 180 still forming -> last closed is 179
    got = spx_move_pct(times, closes, dec, lookback_hours=168)
    assert got == pytest.approx((closes[179] / closes[179 - 168] - 1) * 100)


def test_bar_closing_exactly_at_decision_counts():
    times, closes = hourly(200)
    dec = times[180] + timedelta(hours=1)                # bar 180 closes exactly now
    got = spx_move_pct(times, closes, dec, lookback_hours=168)
    assert got == pytest.approx((closes[180] / closes[12] - 1) * 100)


def test_lookback_over_a_gap_takes_the_last_bar_at_or_before():
    times, closes = hourly(100)
    times = times[:50] + [t + timedelta(hours=60) for t in times[50:]]   # a 60 h gap (weekend)
    dec = times[-1] + timedelta(hours=1)
    target = times[-1] - timedelta(hours=100)
    past = max(i for i, t in enumerate(times) if t <= target)
    got = spx_move_pct(times, closes, dec, lookback_hours=100)
    assert got == pytest.approx((closes[-1] / closes[past] - 1) * 100)


def test_not_enough_history_returns_none():
    times, closes = hourly(50)
    assert spx_move_pct(times, closes, times[-1] + timedelta(hours=1), lookback_hours=168) is None
    assert spx_move_pct([], [], T0) is None
    assert spx_move_pct(times, closes, times[0]) is None      # nothing closed yet


def test_bad_values_return_none():
    times, closes = hourly(200)
    closes[31] = float("nan")
    dec = times[199] + timedelta(hours=1)                     # past bar = 199 - 168 = 31
    assert spx_move_pct(times, closes, dec) is None
    closes[31] = 0.0
    assert spx_move_pct(times, closes, dec) is None


@pytest.mark.parametrize("move,side,allowed", [
    (0.95, "long", False), (0.95, "short", True),
    (-0.95, "short", False), (-0.95, "long", True),
    (0.9, "long", False),          # exactly the threshold blocks (backtest used `< threshold` to allow)
    (0.8999, "long", True),
    (None, "long", True), (None, "short", True),
])
def test_allows_entry(move, side, allowed):
    assert allows_entry(move, side, 0.9) is allowed


def test_decision_time_is_signal_bar_close():
    assert decision_time(datetime(2026, 9, 30, 3, 45)) == datetime(2026, 9, 30, 4, 0)


CACHE = os.path.join(os.path.dirname(ROOT), "trade_dataset_20260929", "intermarket_h1.parquet")


@pytest.mark.skipif(not os.path.exists(CACHE), reason="intermarket cache not built")
def test_same_answer_as_the_backtest_on_real_data():
    """The live function must give exactly the backtest's value (research spx_filter.py) and
    therefore the same allow/block decision, for every M15 bar over the whole history."""
    stub = types.ModuleType("MetaTrader5")
    for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
        setattr(stub, k, 1)
    sys.modules.setdefault("MetaTrader5", stub)
    sys.path.insert(0, os.path.join(ROOT, "research_20260929_tight_stop"))
    import spx_filter as research
    spx = pd.read_parquet(CACHE).SPX500.dropna()
    rng = np.random.default_rng(1)
    start, end = spx.index[0], spx.index[-1]
    minutes = rng.integers(0, int((end - start).total_seconds() // 900), 3000)
    signal_bars = [(start + pd.Timedelta(minutes=15 * int(m))).to_pydatetime() for m in minutes]
    for ts in signal_bars:
        expected = research.spx_5d_pct(ts)
        got = spx_move_pct(spx.index, spx.values, decision_time(ts), lookback_hours=168)
        if np.isnan(expected):
            assert got is None
        else:
            assert got == pytest.approx(expected, rel=1e-12)
        for side in ("long", "short"):
            assert allows_entry(got, side, research.THRESHOLD) == research.spx_filter(ts, side)
