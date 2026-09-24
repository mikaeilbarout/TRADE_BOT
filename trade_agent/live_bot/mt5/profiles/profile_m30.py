"""
XAUUSD M30 strategy profile
Entry: 30-minute timeframe | Trend filter: 4-hour timeframe
Algorithm: Donchian Channel Breakout (see strategy/donchian.py)

Update 2026-09-05 (switched from EMA+RSI+MACD to Donchian Breakout, spread-
aware from the start): see profile_m1.py's docstring for why spread had to
be modeled directly in the optimization objective, not just checked
afterward, after the M1 tight-stop pick failed with real costs.

  Algorithm            train_calmar  test_calmar
  Donchian (spread-aware)     15.16         9.95
  RSI+MACD (previous)          8.37         6.95

Correction 2026-09-06 (methodology bug found and fixed -- see profile_m1.py
for the full story): the 35.38 number above was computed WITHOUT the same
daily loss guard (max_daily_loss_pct=3%) that mt5/live_bot_mt5.py actually
enforces live. Re-verified with the canonical engine
(strategy/donchian.py::simulate_donchian, guard + spread both included):
514 trades, Calmar 23.12, max_dd -13.7%.

Update 2026-09-06 (trend-strength filter added): a loss-pattern study found
win-vs-loss trades differed most consistently (across the other 4 profiles)
on distance from the trend EMA at entry -- M30 was the ONE exception in
that raw win/loss comparison (losses had slightly larger trend distance,
opposite of the other profiles), so this filter was not expected to help
here. Tested anyway rather than skip it based on a correlation alone: swept
min_trend_strength_pct 0-1.0% walk-forward, and 0.3% still scored best
(train 12.17, test 12.86) -- it helped despite the raw pattern pointing the
other way, a reminder that an aggregate correlation doesn't always predict
what a threshold filter does. Full dataset: 484 trades, Calmar 30.73,
max_dd -13.9%.

Correction 2026-09-06 (time_stop bug found and fixed): simulate_donchian's
time_stop check used to approximate elapsed time as bar-count times the
dataset-wide median bar interval, silently compressing any weekend/holiday
data gap to a single median-sized step -- wrong versus the live bot's real
wall-clock `datetime.utcnow() - open_time`. Fixed to use real elapsed time
between timestamps. M30's long time_stop (14 days) means most of its
trades are long enough to span at least one weekend, so this is the other
most materially affected profile: 484 trades -> 491, Calmar 30.73 -> 41.19,
max_dd -13.9% -> -10.3% (walk-forward: train 16.42, test 12.70, score
12.70 -- still solidly positive).
"""

from dataclasses import replace
import MetaTrader5 as mt5
from mt5.config_mt5 import RISK as BASE_RISK

ALGORITHM = "donchian"

ENTRY_TIMEFRAME = mt5.TIMEFRAME_M30
TREND_TIMEFRAME = mt5.TIMEFRAME_H4
MAGIC_NUMBER = 991030

N_PERIOD = 55
ATR_PERIOD = 14
EMA_TREND_PERIOD = 50
MIN_TREND_STRENGTH_PCT = 0.3
COOLDOWN_LOSSES_TO_TRIGGER = 4  # see profile_m15.py's docstring for the mechanism, added 2026-09-08
COOLDOWN_HOURS = 72.0           # re-tuned 2026-09-08 after a data refresh -- was 48h, see docstring below

RISK = replace(
    BASE_RISK,
    reward_risk_ratio=3.0,
    atr_stop_multiplier=3.0,
    time_stop_minutes=20160,  # 14 days
    breakeven_at_r=2.0,  # see config_mt5.py::RiskConfig.breakeven_at_r's docstring
)

# Update 2026-09-17 (breakeven-at-2R added): the same bar-based backtest that
# helped M15 (+$1659) made THIS profile's result worse (418 baseline trades
# net +$9,967 -> 426 trades net +$9,346, -$622) -- enabled here anyway per
# explicit user decision after being shown this number, not because the
# backtest recommends it for M30. Revisit/disable if live results confirm
# the backtest's direction.

# Backtest result (2026-09-06, canonical engine -- day-loss-guard + trend-strength filter + time_stop fix
# + FULL Razor cost model: spread + commission + overnight swap (see strategy/donchian.py docstring), ~1824 days / 5 years):
# 491 trades, full-dataset Calmar 26.71, max_dd -12.1%. Walk-forward score 10.10 (down from 41.19/-10.3%
# spread-only -- this profile's 14-day time_stop means most trades pay several nights of swap).
#
# Update 2026-09-08 (cooldown after a losing streak, originally 4 losses -> 48h): matches this
# profile's own long time_stop (14 days). Walk-forward at the time: score 10.10 -> 12.53.
#
# Update 2026-09-08 (re-tuned after a data refresh, 4 losses -> 72h): data/xauusd_*.csv refreshed
# from 2026-09-04 to 2026-09-08 (see profile_h1.py's docstring for why this matters); re-swept.
# 48h was still positive on fresh data (score 11.99, above the no-cooldown baseline of 9.86) but no
# longer the best -- 72h scored 13.93. Given this profile's already-long holding times (14-day
# time_stop), a longer cooldown fits the same pattern.
#
# Update 2026-09-08 (verified with REAL TICK-BY-TICK data, atr_mult/rr left unchanged): M5 turned out
# completely broken under tick verification, prompting a check of M1/M30/H1 too. Tested current
# params (atr_mult=3.0, rr=3.0) plus 4 other bar-based top candidates trade-by-trade against real
# ticks (mt5.copy_ticks_range, tick-window 2023-04-17 onward). Current params tick-verified score:
# 6.64 -- the BEST of all 5 candidates tested (next best was 5.36 at atr_mult=3.5/rr=1.5; two
# candidates that looked good on bars, atr_mult=2.5/rr=4.0 and 2.5/rr=3.5, scored only 2.58 and 0.79
# tick-verified). No change made -- current live parameters are confirmed, not just assumed.
# Run: python backtest_verify.py --profile m30
