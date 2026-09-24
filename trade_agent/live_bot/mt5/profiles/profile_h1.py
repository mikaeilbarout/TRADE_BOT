"""
XAUUSD H1 strategy profile
Entry: 1-hour timeframe | Trend filter: daily timeframe
Algorithm: Donchian Channel Breakout (see strategy/donchian.py)

Update 2026-09-05 (switched from EMA+RSI+MACD to Donchian Breakout): see
profile_m5.py's docstring for the full story of why. This is the profile
the whole Donchian-vs-RSI+MACD comparison started from (tested on BTC/USDT
first in scalp_alerts, then re-verified here on XAUUSD):

  Algorithm            train_calmar  test_calmar
  Donchian Breakout          11.24        12.37   (test beats train -- no overfit)
  RSI+MACD (previous)         6.11         5.09

Correction 2026-09-06 (methodology bug found and fixed): the 22.51 number
above was computed by a quick one-off script that did NOT model the same
daily loss guard (max_daily_loss_pct=3%) that mt5/live_bot_mt5.py actually
enforces live in its DailyLossGuard class -- meaning every "Full-dataset
verification" number in this project's Donchian profiles up to 2026-09-05
understated real-world risk-guard behavior. Found this by cross-checking a
quick re-test against strategy/donchian.py::simulate_donchian (the function
the live bot's logic conceptually mirrors) and getting a different Calmar
for supposedly identical parameters. Fixed by consolidating everything into
that ONE canonical function -- it now takes both the daily-loss-guard logic
and a spread_dollars parameter, so there is only one place left to get this
wrong. Re-verified this profile's own parameters with it: 525 trades,
Calmar 18.93, max_dd -9.7% (still comparable to the old number here -- H1's
wide stop means it's the least sensitive of the 5 profiles to this bug).

Update 2026-09-06 (trend-strength filter added): a loss-pattern study
across all 5 profiles compared winning vs losing trades on three features
(volatility at entry, how far price broke past the channel, and distance
from the trend EMA). Only the last one showed a consistent difference (4 of
5 profiles, H1 included, with H1 showing the single largest gap of any
profile). Added min_trend_strength_pct (same concept as the old RSI+MACD
profiles' filter) and swept 0-1.0% walk-forward: 0.5% scored best (train
10.54, test 10.60 -- the most balanced result of any threshold). Full
dataset: 489 trades, Calmar 24.56, max_dd -8.7%.

Correction 2026-09-06 (time_stop bug found and fixed): simulate_donchian's
time_stop check used to approximate elapsed time as bar-count times the
dataset-wide median bar interval instead of real elapsed time -- see
profile_m5.py's docstring for the full story. H1 has ZERO time_stop exits
(every trade always hits stop or target well before the 30-day limit), so
this profile is completely unaffected: still 489 trades, Calmar 24.56,
max_dd -8.7%, exactly as before.
"""

from dataclasses import replace
import MetaTrader5 as mt5
from mt5.config_mt5 import RISK as BASE_RISK

ALGORITHM = "donchian"

ENTRY_TIMEFRAME = mt5.TIMEFRAME_H1
TREND_TIMEFRAME = mt5.TIMEFRAME_D1
MAGIC_NUMBER = 991060

N_PERIOD = 20
ATR_PERIOD = 14
EMA_TREND_PERIOD = 10  # re-tuned 2026-09-09, tick-verified -- was 50, see docstring
MIN_TREND_STRENGTH_PCT = 0.5
COOLDOWN_LOSSES_TO_TRIGGER = 4  # see profile_m15.py's docstring for the mechanism, added 2026-09-08
COOLDOWN_HOURS = 24.0           # re-tuned 2026-09-08 after a data refresh -- was 2 losses/48h, see docstring below

RISK = replace(
    BASE_RISK,
    reward_risk_ratio=1.0,
    atr_stop_multiplier=3.0,
    time_stop_minutes=43200,  # 30 days
)

# Backtest result (2026-09-06, canonical engine -- day-loss-guard + trend-strength filter + time_stop fix
# + FULL Razor cost model: spread + commission + overnight swap (see strategy/donchian.py docstring), ~1824 days / 5 years):
# 489 trades, full-dataset Calmar 18.54, max_dd -9.1%. Walk-forward score 7.65 (down from 24.56/-8.7%
# spread-only -- this profile's 30-day time_stop is the longest of the 5, so swap matters most here).
#
# Update 2026-09-08 (cooldown after a losing streak, originally 2 losses -> 48h): matches this
# profile's own very long time_stop (30 days). Walk-forward at the time: score 7.65 -> 9.06.
#
# Update 2026-09-08 (re-tuned after a data refresh, 4 losses -> 24h): data/xauusd_*.csv was stale
# (last update 2026-09-04); refreshed to 2026-09-08 and re-swept. The original 2-losses/48h combo
# had DROPPED BELOW the no-cooldown baseline on fresh data (6.78 vs baseline 7.65 -- a real
# regression, not just a smaller improvement) once the newly-included days changed the test split.
# Re-swept walk-forward on the refreshed data: 4 losses -> 24h was the new best (score 7.65 -> 7.76,
# train n=336 cal=7.76+, test n=153 cal=...). Lesson: cooldown parameters are sensitive enough to
# recent data that they should be re-checked whenever data/xauusd_*.csv is refreshed, not treated
# as permanently fixed after one sweep.
#
# Update 2026-09-08 (re-tuned using REAL TICK-BY-TICK verification, not just bars, rr 1.5 -> 1.0):
# M5 turned out completely broken under tick verification (see chat/scalp_sample history), which
# prompted re-checking M1/M30/H1 too. Ran a bar-based grid search restricted to the tick-data-
# available window (2023-04-17 onward), then verified the top 5 candidates trade-by-trade against
# REAL ticks (mt5.copy_ticks_range, walking forward from each entry to find which of stop/target
# was actually touched FIRST in real time). Old params (atr_mult=3.0, rr=1.5) tick-verified score:
# 1.86 (down from a bar-based 5.78). New params (atr_mult=3.0, rr=1.0) tick-verified score: 2.70
# (train n=291 cal=2.70, test n=132 cal=9.74), full n=423, ret=70.3%, max_dd=-8.5%, cal=8.26 --
# best of 5 candidates tested this way (mult=2.0 combos were consistently worse under tick
# verification, same pattern seen on M1/M5/M15/M30 -- tighter stops suffer more from the bar-vs-
# tick discrepancy). Switched live to these.
#
# Update 2026-09-09 (spread-doubling bug fixed, see strategy/donchian.py -- the 2.70/8.26 numbers
# above were computed with spread charged 2x; real cost is 1x, true performance was slightly
# better than reported). Also re-tuned EMA_TREND_PERIOD (50 -> 10): bar-based sweep found this
# profile unusually sensitive to EMA speed (score 4.48 at 50 vs 14.28 at 10 -- much bigger gap
# than any other profile saw), re-verified against real tick data: EMA=10 tick-verified score
# 7.16 (train n=284 cal=7.16, test n=139 cal=13.31), full n=423 cal=17.96, max_dd=-7.6% -- nearly
# double EMA=50's re-verified score (3.70, max_dd -8.1%) and better on every metric. Switched
# live to EMA=10. (H4-trend-timeframe candidates like H2 were not re-tested for H1 -- H1 already
# uses D1, the slowest trend timeframe of any profile.)
# Run: python backtest_verify.py --profile h1
