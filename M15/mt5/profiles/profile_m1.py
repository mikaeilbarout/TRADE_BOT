"""
XAUUSD M1 strategy profile
Entry: 1-minute timeframe | Trend filter: 15-minute timeframe
Algorithm: Donchian Channel Breakout (see strategy/donchian.py)

Update 2026-09-05 (switched from EMA+RSI+MACD to Donchian Breakout, but NOT
on the first attempt): the initial no-cost walk-forward pick (n=10,
atr_mult=1.5, rr=2.5) looked spectacular -- 2693% return, Calmar 83 over
just 100 days -- but that pick's median holding time was 9 MINUTES, meaning
an extremely tight ATR-based stop. Checked the real economics: at that
tight a stop, position size for a $10 risk budget requires buying ~$18,600
of gold (1859x the risk amount!), and even a modest $0.30/oz round-trip
spread costs ~$1.23 per trade -- reran with spread actually modeled and
the tight-stop pick LOST money (-34.3% return, Calmar -0.51). This exactly
mirrors the crypto fee-vs-notional problem found earlier the same day (see
the now-deleted scalp_crypto project's history) -- a tight-stop strategy can
look incredible gross and be unusable net of real transaction costs.

Fix: re-ran the grid search with spread cost included DIRECTLY in the
walk-forward objective (not just checked afterward), searching wider
ATR-stop-multipliers (up to 12x). Landed on n_period=55, atr_mult=3.0,
rr=2.0, time_stop=0.5 days.

Correction 2026-09-06 (methodology bug found and fixed): the numbers above
(Calmar 19.31) were computed with a script that didn't model the SAME daily
loss guard (max_daily_loss_pct=3%) that mt5/live_bot_mt5.py actually
enforces live. Consolidated everything into ONE canonical engine
(strategy/donchian.py::simulate_donchian, which now includes both the
guard and spread) -- re-verified Calmar with these same parameters: 778
trades, Calmar 7.11, max_dd -16.5%. Lower than the earlier (wrong) number,
but this is the number that actually matches live behavior; see
profile_h1.py's docstring for the full story and why scalp-bot's RSI+MACD
comparison (Calmar 2.63 for M1, done correctly from the start) still loses.

Update 2026-09-06 (trend-strength filter added): a loss-pattern study
across all 5 profiles found win-vs-loss trades differed most consistently
(4 of 5 profiles, M1 included) on how far price was from the trend EMA at
entry -- trades taken with the trend already well-established won more
often than ones taken right at a marginal crossing. Added
min_trend_strength_pct (same idea as the old RSI+MACD profiles' filter),
swept threshold values 0-1.0% walk-forward: 1.0% scored best (train 5.65,
test 5.55 -- balanced, no big gap). Full-dataset result: 150 trades (down
from 778 -- a much smaller, more selective sample), Calmar 13.78, max_dd
-6.4% (down from -16.5%).

Correction 2026-09-06 (time_stop bug found and fixed): simulate_donchian's
time_stop check used to approximate elapsed time as bar-count times the
dataset-wide median bar interval, which silently compresses any
weekend/holiday data gap to a single median-sized step -- wrong versus the
live bot's real wall-clock `datetime.utcnow() - open_time`. Fixed to use
real elapsed time between timestamps. M1 has very few time_stop exits (2),
so the effect here is small: Calmar 13.78 -> 13.95, max_dd -6.4% -> -6.3%.

Update 2026-09-08 (cooldown after a losing streak, COOLDOWN_LOSSES_TO_TRIGGER
+ COOLDOWN_HOURS): same mechanism added to profile_m15.py, see its docstring
for the full rationale/design. Swept for M1 too; the strongest full-range
result was 2 losses -> 48h (score 5.32 -> 8.10), but per explicit request
this profile's cooldown was constrained to 15-30 minutes (M1 trades far
more frequently than the other profiles -- a 48h pause would be a much
bigger relative interruption here). Within that constrained range, 2
losses -> 18min was the best (score 5.32 -> 5.46) -- a small but real
improvement, not the strongest one available if the range were unconstrained.
"""

from dataclasses import replace
import MetaTrader5 as mt5
from mt5.config_mt5 import RISK as BASE_RISK

ALGORITHM = "donchian"

ENTRY_TIMEFRAME = mt5.TIMEFRAME_M1
TREND_TIMEFRAME = mt5.TIMEFRAME_M15
MAGIC_NUMBER = 991001

N_PERIOD = 55
ATR_PERIOD = 14
EMA_TREND_PERIOD = 40  # re-tuned 2026-09-09, tick-verified -- was 50, see docstring
MIN_TREND_STRENGTH_PCT = 1.0
COOLDOWN_LOSSES_TO_TRIGGER = 2  # see docstring, 2026-09-08 -- constrained to 15-30min per request
COOLDOWN_HOURS = 0.3            # 18 minutes

RISK = replace(
    BASE_RISK,
    reward_risk_ratio=1.5,
    atr_stop_multiplier=4.0,
    time_stop_minutes=720,  # 0.5 days
)

# Backtest result (2026-09-08, canonical engine -- cooldown added (2 losses -> 18min), see docstring above):
# walk-forward score 5.46 (train cal=6.84 n=103, test cal=5.46 n=45). Small improvement over the
# no-cooldown baseline (5.32); a longer cooldown scored higher (8.10 at 48h) but was out of the
# requested 15-30min range for this profile.
#
# Update 2026-09-08 (re-tuned using REAL TICK-BY-TICK verification, not just bars): M5's parameters
# turned out completely broken under tick verification (see chat/scalp_sample history), which
# prompted re-checking M1/M30/H1 too, since all three use tight-ish stops like M5 did. Ran a bar-based
# grid search restricted to the tick-data-available window (2023-04-17 onward -- this broker's tick
# history doesn't go back further), then verified the top 5 candidates trade-by-trade against REAL
# ticks (mt5.copy_ticks_range, walking forward from each entry to find which of stop/target was
# actually touched FIRST in real time, not just "which bar's high/low range included it").
# Old params (atr_mult=3.0, rr=2.0) tick-verified score: 2.23 (down from a bar-based 8.41 -- a real,
# not just Cosmetic, overstatement). New params (atr_mult=4.0, rr=1.5) tick-verified score: 7.67
# (train n=97 cal=7.67, test n=31 cal=8.59), full n=128, ret=61.5%, max_dd=-4.8%, cal=12.73 --
# clearly the best of 5 candidates tested this way. Switched live to these.
#
# Update 2026-09-09 (spread-doubling bug fixed, see strategy/donchian.py -- the 7.67/12.73
# numbers above were computed with spread charged 2x; real cost is 1x, so true performance
# was slightly better than reported). Also re-tuned EMA_TREND_PERIOD (50 -> 40): bar-based
# sweep found 40 scored highest (9.82 vs 50's 8.97), re-verified against real tick data:
# EMA=40 tick-verified score 9.30 (train n=82 cal=9.30, test n=25 cal=9.97), full n=107
# cal=16.60, max_dd=-3.7% -- beats EMA=50's re-verified 8.71 (max_dd -4.7%) on every metric.
# Switched live to EMA=40.
# Run: python backtest_verify.py --profile m1
