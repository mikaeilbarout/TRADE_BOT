"""Regression: simulate_donchian must not see an H4 bar before it has closed."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from strategy.donchian import simulate_donchian


class Risk:
    risk_per_trade_pct = 0.2
    max_daily_loss_pct = 100.0


def frames(future_h4_close):
    """Flat M15 market with one breakout bar at 10:30; the H4 bar 08:00-12:00 that CONTAINS
    it closes at `future_h4_close`. The last H4 bar CLOSED at 10:45 (04:00-08:00) is bearish."""
    ts = pd.date_range("2026-01-05 00:00", "2026-01-05 11:45", freq="15min")  # ends before 12:00
    close = np.full(len(ts), 100.0)
    k = ts.get_loc(pd.Timestamp("2026-01-05 10:30"))
    close[k] = 105.0                                                      # breaks the 10-bar high
    close[k + 1:] = 95.0                                                  # a long from 10:30 is stopped out
    low = pd.DataFrame({"ts": ts, "open": close, "high": close + .1, "low": close - .1, "close": close})
    h4_ts = pd.date_range("2025-12-01 00:00", "2026-01-05 08:00", freq="4h")
    h4_close = np.full(len(h4_ts), 100.0); h4_close[-2] = 90.0            # 04:00 bar: well below the EMA
    h4_close[-1] = future_h4_close                                        # 08:00 bar: closes at 12:00
    high = pd.DataFrame({"ts": h4_ts, "open": h4_close, "high": h4_close, "low": h4_close, "close": h4_close})
    return low, high


class NoLookAhead(unittest.TestCase):
    def run_engine(self, future):
        low, high = frames(future)
        trades, _ = simulate_donchian(low, high, n_period=10, atr_period=3, ema_trend_period=5,
            atr_stop_multiplier=3.0, reward_risk_ratio=3.0, time_stop_minutes=10080, risk_cfg=Risk())
        return trades

    def test_future_h4_close_cannot_create_a_long(self):
        # Before the fix, a bullish FUTURE close of the containing H4 bar produced a long at 10:30.
        trades = self.run_engine(future=130.0)
        self.assertFalse((trades.get("side", pd.Series(dtype=str)) == "long").any())

    def test_result_independent_of_unclosed_h4_bar(self):
        a, b = self.run_engine(future=130.0), self.run_engine(future=60.0)
        self.assertEqual(len(a), len(b))


if __name__ == "__main__":
    unittest.main()
