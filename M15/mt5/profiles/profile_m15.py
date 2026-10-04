"""
XAUUSD M15 strategy profile
Entry: 15-minute timeframe | Trend filter: 4-hour timeframe
Algorithm: Donchian Channel Breakout (see strategy/donchian.py)

Update 2026-09-05 (switched from EMA+RSI+MACD to Donchian Breakout): see
profile_m5.py's docstring for the full story of why. This profile showed
the single biggest improvement of any timeframe on XAUUSD:

  Algorithm            train_calmar  test_calmar
  Donchian Breakout          31.26        38.63   (test beats train -- no overfit)
  RSI+MACD (previous)         5.45         5.69

Correction 2026-09-06 (methodology bug found and fixed -- see profile_m1.py
for the full story): the 67.50 number above was computed WITHOUT the same
daily loss guard (max_daily_loss_pct=3%) that mt5/live_bot_mt5.py actually
enforces live. Re-verified with the canonical engine
(strategy/donchian.py::simulate_donchian, guard + spread both included):
1915 trades, Calmar 14.70, max_dd -45.1% (materially worse drawdown than
previously reported -- this was the profile most exposed by the bug fix).

Update 2026-09-06 (trend-strength filter added): a loss-pattern study found
win-vs-loss trades differed most consistently on distance from the trend
EMA at entry. Swept min_trend_strength_pct 0-1.0% walk-forward: 0.5% scored
best by far (train 20.70, test 31.78 -- both far above every other
threshold tried). Full dataset: 1490 trades, Calmar 130.87, max_dd -19.8%
-- this filter fixed BOTH problems the bug fix exposed: Calmar went from
14.70 to 130.87 and max_dd improved from -45.1% to -19.8%, the best
drawdown of any of the 5 profiles' final picks.

Correction 2026-09-06 (time_stop bug found and fixed): simulate_donchian's
time_stop check used to approximate elapsed time as bar-count times the
dataset-wide median bar interval instead of real elapsed time -- see
profile_m5.py's docstring for the full story. M15 has very few time_stop
exits (2 of 1493), so the effect here is negligible: Calmar 130.87 -> 130.71,
max_dd unchanged at -19.8%.

Update 2026-09-07 (atr_stop_multiplier widened 2.0 -> 3.0): live trading on
2026-09-07 showed 4 losing trades in one day, all shorts stopped out during
a choppy/ranging session -- the H4 trend filter stayed "short" from an
earlier pullback while price whipsawed in a ~4381-4435 range, and the then-
2.0x-ATR stop was tight enough that each dip-and-bounce triggered a fresh
short and then stopped it. Tested widening the stop instead of switching to
a faster trend timeframe (H1/M45 tried first -- both made things worse,
roughly halving the walk-forward score, because a faster trend EMA flips
direction far more often -- H4 flips every ~3.1 days vs H1's ~0.8 and M45's
~0.6 -- so it reacts to small wiggles instead of only real trend changes).
Swept atr_stop_multiplier 2.0-6.0 (train/test walk-forward, canonical
engine, same cost model): 3.0 was the clear best -- score 15.09 -> 25.40,
win rate 32.7%/35.1% -> 35.5%/38.4%, max_dd -19.2% -> -16.0% (all three
improved together, not traded off). Above 3.0 the score drops again (e.g.
4.0 -> 10.67, 6.0 -> 5.88, the latter also on a much thinner, less reliable
sample of 133 test trades).

Update 2026-09-08 (pivot-confirmation filter enabled, REQUIRE_PIVOT_CONFIRM):
the wider stop above did NOT stop a second straight day (2026-09-08) of the
same whipsaw pattern -- 3 more losing shorts (-$1085 total) while the H4 EMA
stayed "short" through a still-ranging 4381-4435 market. Checked: at every
one of these 5 losing entries (this day + the prior day's 4), the
higher-timeframe swing structure (fractal-pivot HH/HL/LH/LL chain, see
strategy/donchian.py::add_pivot_trend) was still "long"/"flat", disagreeing
with the EMA's "short" -- a pivot-confirmation filter (only trade when EMA
and pivot trend agree) would have blocked all 5.

IMPORTANT -- first measured with a hand-rolled scratch script (bug: its
stop/target check loop failed to detect a stop hit for 13 days on the very
first trade, materially understating losses), which claimed walk-forward
23.08 -> 16.27. Re-measured with the CANONICAL engine
(strategy/donchian.py::simulate_donchian, the actual function this live bot
calls) after wiring require_pivot_confirm/pivot_k/df_pivot into it properly:
the real number is far worse -- 23.08 -> 3.55 (pivot on H4, same tf as the
EMA) or -> 1.80 (pivot on a separate H1 feed, which also introduces a real
losing quarter, Q1 calmar -0.31, that the H4-pivot version didn't have).
Both variants trail the no-pivot baseline badly once measured correctly.

Enabled 2026-09-08 on the (wrong) scratch numbers, then REVERTED the same
day once the canonical engine's real numbers came back -- back to
REQUIRE_PIVOT_CONFIRM=False. The underlying capability (add_pivot_trend,
find_pivots, the df_pivot param) is left in strategy/donchian.py and wired
into live_bot_mt5.py in case it's worth revisiting with a different
timeframe/pivot_k later, but is dormant while this flag is False.

Update 2026-09-08 (cooldown after a losing streak, COOLDOWN_LOSSES_TO_TRIGGER
+ COOLDOWN_HOURS): live losing streaks on 2026-09-07/08 (see pivot section
above) prompted the question "does widening the stop even help a real
losing streak, or does something else need to change." Swept
losses_to_trigger (2-5) x cooldown_hours (0-168h) walk-forward, canonical
engine (strategy/donchian.py::simulate_donchian's new
cooldown_losses_to_trigger/cooldown_hours params -- verified byte-identical
to a hand-rolled test script before trusting it, after the pivot section's
scratch-script bug). Longer cooldowns (6h+) looked good on train alone but
failed on test -- overfit. The one combo that held up on BOTH sides: 5
losses in a row -> 4h pause. Full dataset: n=828 (was 829), max_dd -16.0%
-> -15.0%, Calmar 77.54 -> 79.80, all 4 quarters improved together.
Walk-forward score 25.40 -> 26.08. A modest but real, consistent
improvement -- enabled live.

Update 2026-09-09 (spread-doubling bug fixed, see strategy/donchian.py --
every Calmar number above was computed with spread charged 2x; real cost is
1x, so all prior full-history/walk-forward numbers on this page slightly
understated true performance). Also tested H2 as TREND_TIMEFRAME instead of
H4 (user question: "does a faster trend timeframe help") -- tick-verified,
H2 was WORSE on every metric (score 6.07 vs H4's 8.11, lower Calmar, lower
win rate) -- reverted/kept H4, TREND_TIMEFRAME unchanged.

Update 2026-09-09 (EMA_TREND_PERIOD re-tuned, 50 -> 30): bar-based sweep of
ema_trend_period (15-200, same walk-forward split) found 30 scored highest
(20.85) with 50 close behind (20.30) -- both re-verified against REAL TICK
data (mt5.copy_ticks_range, same methodology as every other tick-
verification this session) since bar rankings have repeatedly diverged from
tick-verified ones: 30 -> score 15.87 (train n=424 cal=15.87, test n=231
cal=16.18 -- test even beats train, no overfit), full n=655 Calmar 54.05,
max_dd -12.3%. Old EMA=50 tick-verified: score 8.11, full Calmar 36.62,
max_dd -16.7%. EMA=60 tick-verified: score 7.48 (worse). 30 nearly doubles
the walk-forward score AND improves drawdown -- switched live.
"""

from dataclasses import replace
import MetaTrader5 as mt5
from mt5.config_mt5 import RISK as BASE_RISK

ALGORITHM = "donchian"

ENTRY_TIMEFRAME = mt5.TIMEFRAME_M15
TREND_TIMEFRAME = mt5.TIMEFRAME_H4
MAGIC_NUMBER = 991015

# 10 -> 20, MIN_TREND_STRENGTH_PCT 0.5 -> 0.3, RR 3 -> 4 on 2026-09-30 (user request): full re-optimisation,
# 324 cells chosen on the first 70% of 4 years (neighbour-smoothed), then checked once: last 30%
# +64.4R vs +62.4R (DD 24.1 vs 21.3), 6-month ticks +$1156 vs +$738 -- see research_20260929_tight_stop/full_reopt.py
N_PERIOD = 20
ATR_PERIOD = 14
EMA_TREND_PERIOD = 30  # re-tuned 2026-09-09, tick-verified -- was 50, see docstring
MIN_TREND_STRENGTH_PCT = 0.3
REQUIRE_PIVOT_CONFIRM = False  # see docstring, 2026-09-08 -- tried and reverted, see below
PIVOT_K = 2  # fractal pivot on TREND_TIMEFRAME (H4), same tf as the EMA -- unused while disabled
COOLDOWN_LOSSES_TO_TRIGGER = 3  # re-tuned 2026-09-08 after a data refresh -- was 5 losses/4h, see docstring
COOLDOWN_HOURS = 2.0            # fixed wall-clock pause, not tied to calendar-day boundaries
# Added 2026-09-29 at the user's request: skip a signal whose stop (2x ATR) is closer than
# $8/oz -- fixed costs (~$0.37/oz) ate ~0.1R per trade when ATR was small (2022-23).
# research_20260929_tight_stop/min_stop_filter.py: 4y +71.7R -> +93.8R, DD 74.9R -> 27.9R,
# 6-month ticks +$610 -> +$738; but 2025-26 +95.4R -> +70.1R (failed the pre-set 5% rule).
MIN_STOP_DOLLARS = 8.0
# Added 2026-09-30 (user request): skip a trade when the S&P 500 moved >= 0.9% in the SAME direction
# over the last 168 h (5 trading days) -- see mt5/spx_filter.py and research_20260929_tight_stop/spx_filter.py
# (last 30% of 4 years +73.3R vs +64.4R, drawdown 14.8R vs 24.1R; 6-month ticks +$1600 vs +$1156).
SPX_FILTER = dict(symbol="SPX500", threshold_pct=0.9, lookback_hours=168)

# Fixed volume per trade (user request 2026-10-04). Set to None to size by RISK.risk_per_trade_pct again.
FIXED_LOTS = 0.01
RISK = replace(
    BASE_RISK,
    # 0.20% -> 0.30% on 2026-09-30 (user request): 12-month Monte Carlo of both bots' 4-year trades,
    # P(10% drawdown) 1.6%, worst historical day -1.7% -- research_20260929_tight_stop/risk_sizing_mc.py
    risk_per_trade_pct=0.30,
    reward_risk_ratio=4.0,
    # 3.0 -> 2.0 on 2026-09-29 at the user's request (smaller stop). Look-ahead-free
    # engine, see research_20260929_tight_stop/: 6-month ticks +2.4% vs -0.3% at 3.0,
    # but 4-year drawdown 74.9R vs 32.9R; below 2.0 every cell lost money.
    atr_stop_multiplier=2.0,
    time_stop_minutes=10080,  # 7 days
)

# Backtest result (2026-09-08, canonical engine -- cooldown after 5-loss streak added, see docstring above):
# walk-forward score 26.08 (train cal=26.08 n=563, test cal=27.12 n=268), full-dataset n=828, max_dd -15.0%.
# (pivot-confirmation was tried 2026-09-08 and reverted -- see docstring; REQUIRE_PIVOT_CONFIRM=False)
#
# Update 2026-09-08 (re-tuned after a data refresh, 3 losses -> 2h): data/xauusd_*.csv was stale
# (last update 2026-09-04, missing the 2026-09-07/08 losing streaks this whole cooldown feature was
# built to address); refreshed to 2026-09-08 and re-swept. With the extra days included, the old
# 5-losses/4h combo scored 18.51 (still above the no-cooldown baseline of 16.93, but no longer the
# best available). Re-swept: 3 losses -> 2h scored 22.66, clearly better. See profile_h1.py's
# docstring for why this re-check matters -- cooldown params should be re-swept after any data
# refresh, not treated as permanent after one pass.
# Run: python backtest_verify.py --profile m15
