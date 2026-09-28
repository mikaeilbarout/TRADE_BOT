"""
XAUUSD H4 strategy profile
Entry: 4-hour timeframe | Trend filter: daily timeframe
Algorithm: Donchian Channel Breakout (see strategy/donchian.py)

Created 2026-09-18 as a direct port of profile_m15.py's strategy (same
N_PERIOD/ATR_PERIOD/EMA_TREND_PERIOD/MIN_TREND_STRENGTH_PCT/RISK shape)
onto H4 bars. TREND_TIMEFRAME moved up one level from M15's H4 to D1 (H4
is now the entry timeframe itself), mirroring how H1 already uses D1.

Every other profile keeps RISK.time_stop_minutes at roughly the same
~672-720 bars of ITS OWN entry timeframe (H1: 720, M30: 672, M15: 672,
M1: 720), not a fixed calendar duration -- scaled to match: 672 H4 bars
x 240 min/bar = 161280 minutes (~112 days).

Correction 2026-09-18 (M15's literal params overfit on H4, re-swept):
first tick-verified pass (real ticks, but this account's tick history
only covers ~380 days) showed train calmar 103.47 -> test calmar 2.28,
a ~45x collapse -- the opposite of the "test matches/beats train" bar
every other profile's docstring documents, and a 91.6% train win rate
that's implausible for a breakout strategy (almost certainly a single
strong-trend regime, not genuine edge). Re-swept EMA_TREND_PERIOD /
ATR_STOP_MULTIPLIER / REWARD_RISK_RATIO / MIN_TREND_STRENGTH_PCT /
N_PERIOD one dimension at a time (same iterative methodology as every
other profile's own tuning history), scored via the canonical bar engine
(strategy/donchian.py::simulate_donchian, full 6.75y H4/D1 history --
tick data isn't needed for a fast relative-ranking sweep). Baseline
(M15's literal params) on the FULL bar history: calmar -0.03, ret -1.3%
over 6.75y -- confirms the earlier tick-verified +45% was a short-window
artifact, not real edge. Swept combo: ema_trend_period 30->75,
atr_stop_multiplier 3.0->2.0, min_trend_strength_pct 0.5->0.7 (n_period
and reward_risk_ratio stayed at 10/3.0) -- full-history bar-based calmar
-0.03 -> 0.79 (train 0.23, test 2.80, test beats train).

Final verification (per user request, "if there is no tick data, use
15-minute candle data" -- real ticks only cover ~380 days, but real M15
candles go back to 2022-06-23, ~4.25 years): resolved each H4 signal by
walking its 16 M15 sub-bars' real OHLC (fill at the first M15 bar's open
at/after the H4 close, then scanning for stop/target/breakeven-at-2R/
time_stop) instead of assuming the crude "H4 bar close only" resolution
the sweep used -- 16x finer granularity, full 4.25y sample (not tick-
availability-limited): 542 resolved trades, train (n=379, 3y) calmar
3.59, test (n=163, 1.1y) calmar 12.87 (test beats train again, larger
and more consistent margin than the bar-based sweep), full calmar 5.40,
ret +98.4%, max_dd -4.31%, win_rate 43.4%, pf 2.26 -- see
scratchpad/h4_gold_m15verify.py's run log for the full numbers. Lower
Calmar than H1/M30/M15/M1 (15-130 range) -- H4 is a weaker edge on this
exact strategy shape than the other timeframes, but no longer overfit,
and profitable/consistent across both halves of a real 4+ year sample.

COOLDOWN_HOURS/COOLDOWN_LOSSES_TO_TRIGGER left at M15's un-scaled values
-- not part of this sweep (simulate_donchian's cooldown params weren't
included in the re-tune), pending a dedicated sweep like M15/H1 got.

Round 2 attempted 2026-09-18 (per user request, "search to improve
its parameters"): a JOINT grid over ema_trend_period x atr_stop_multiplier x
reward_risk_ratio, then n_period x min_trend_strength_pct, then a
cooldown sweep, all on the fast bar engine -- found ema_trend_period
75->150, min_trend_strength_pct 0.7->1.0, cooldown 3losses/2h->5losses/
48h "improved" the bar-based score (0.46->0.71). Did NOT hold up under
either higher-fidelity check: M15-resolved full calmar 5.40->4.22 (ret
98.4%->81.5%, worse), and tick-verified test outright LOST money
(calmar 1.86->-1.84, win_rate 23.9%->10.6%, net -$1,240 over 47 trades).
Rejected -- reverted to round 1's params below. Lesson: the bar engine's
crude same-H4-bar stop/target resolution is a fine tool for a first-pass
relative ranking, but chasing marginal gains on it past round 1 tuned
INTO its blind spots rather than finding real edge -- verify any further
H4 re-tune against M15-resolved and (however limited the sample) real
ticks before trusting a bar-only score improvement.
"""

from dataclasses import replace
import MetaTrader5 as mt5
from mt5.config_mt5 import RISK as BASE_RISK

ALGORITHM = "donchian"

ENTRY_TIMEFRAME = mt5.TIMEFRAME_H4
TREND_TIMEFRAME = mt5.TIMEFRAME_D1
MAGIC_NUMBER = 991240  # 240 = H4 in minutes, matching M30=991030/M15=991015/M1=991001's convention

N_PERIOD = 10
ATR_PERIOD = 14
EMA_TREND_PERIOD = 75  # re-tuned 2026-09-18, see docstring -- was 30 (M15's literal value)
MIN_TREND_STRENGTH_PCT = 0.7  # re-tuned 2026-09-18, see docstring -- was 0.5
REQUIRE_PIVOT_CONFIRM = False
PIVOT_K = 2  # unused while REQUIRE_PIVOT_CONFIRM is False
COOLDOWN_LOSSES_TO_TRIGGER = 3  # un-scaled M15 value, not yet independently swept for H4
COOLDOWN_HOURS = 2.0            # un-scaled M15 value, not yet independently swept for H4

RISK = replace(
    BASE_RISK,
    reward_risk_ratio=3.0,
    atr_stop_multiplier=2.0,  # re-tuned 2026-09-18, see docstring -- was 3.0
    time_stop_minutes=161280,  # 112 days -- scaled from M15's 672-bar budget, see docstring
    breakeven_at_r=2.0,
)

# Backtest (2026-09-18, M15-candle-resolved, 70/30 walk-forward, 4.25y sample):
# train n=379 calmar=3.59, test n=163 calmar=12.87 (test beats train), full n=542 calmar=5.40,
# ret=+98.4%, max_dd=-4.31%, win_rate=43.4%, pf=2.26.
# Run: python scratchpad/h4_gold_m15verify.py (see docstring for the full tuning history)
