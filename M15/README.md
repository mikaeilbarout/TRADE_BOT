# scalp_sample

> **2026-09-24 — backtest numbers below are INFLATED.** `strategy/donchian.py::simulate_donchian`
> matched each M15 bar to the H4 bar still *forming* (its future close) — a look-ahead the live bot
> never has. Fixed (test: `tests/test_donchian_engine.py`). With the fix, FundedNext data and costs
> (incl. its swap: long −107, short −47 points/lot/night), the M15 profile makes about +61R over
> 2022-06→2026-09 (0.2% risk ≈ +12%), almost all of it in 2025–2026; 2022–2024 ≈ flat. The Calmar
> figures in the tables below were produced with the look-ahead and should not be relied on.
> See `research_20260924/`.

XAUUSD (gold) live trading bot on MetaTrader5 (Pepperstone demo, separate
account from `scalp-bot`). Donchian Channel Breakout across 5 timeframe
profiles, each walk-forward validated (70% train / 30% held-out test,
scored by `min(train_calmar, test_calmar)` — never the full-dataset number
alone, which overfits).

## Relationship to scalp-bot

`scalp-bot` is the **untouched baseline** (original EMA+RSI+MACD
mean-reversion strategy) — never modified, kept for live comparison.
`scalp_sample` is where every walk-forward-validated update lands. The two
run on separate MT5 demo accounts side by side so their live results can be
compared head-to-head on identical, real-time market data.

## History: why Donchian, not RSI+MACD

Started as a copy of scalp-bot's EMA+RSI+MACD strategy. After being asked
why every project kept reusing RSI+MACD instead of testing genuinely
different algorithms against 5 years of data, Donchian Channel Breakout
(pure trend-following: long on a close above the N-period high with
higher-timeframe trend agreement, short on the equivalent breakdown — see
`strategy/donchian.py`) was compared against Bollinger mean-reversion and
the original RSI+MACD, and won clearly on both BTC/USDT and XAUUSD. All 5
profiles below were switched over and recalibrated from scratch.

`strategy/signals.py` (the original RSI+MACD logic) is kept and still
imported by `mt5/live_bot_mt5.py` as a fallback code path — a profile can
still set `ALGORITHM = "rsi_macd"` — but no current profile uses it.

## Live profiles (mt5/profiles/)

| Profile | Entry / Trend TF | n_period | ema_trend | min_str% | RR | ATR mult | time stop | Magic | Full Calmar | Full maxdd |
|---|---|---|---|---|---|---|---|---|---|---|
| M1  | M1 / M15 | 55 | 50  | 1.0 | 2.0 | 3.0 | 12h | 991001 | 13.28  | -6.4%  |
| M5  | M5 / H1  | 55 | 200 | 0.3 | 3.0 | 3.0 | 12h | 991005 | 27.32  | -17.3% |
| M15 | M15 / H4 | 10 | 50  | 0.5 | 3.0 | 2.0 | 7d  | 991015 | 86.43  | -21.2% |
| M30 | M30 / H4 | 55 | 50  | 0.3 | 3.0 | 3.0 | 14d | 991030 | 26.71  | -12.1% |
| H1  | H1 / D1  | 20 | 50  | 0.5 | 1.5 | 3.0 | 30d | 991060 | 18.54  | -9.1%  |

Run `python backtest_verify.py --profile <name>` any time `data/xauusd_*.csv`
is refreshed, to reproduce every number above.

Each profile file's docstring has the full calibration history, including
three consolidated-into-one-canonical-function bug fixes/additions, all in
`strategy/donchian.py::simulate_donchian`:

1. **2026-09-06**: an early grid-search script didn't model the live bot's
   `DailyLossGuard`, giving inflated numbers.
2. **2026-09-06**: the time_stop check approximated elapsed time as
   bar-count × median bar interval instead of real elapsed time,
   understating how long a trade had been open whenever it spanned a
   weekend/holiday gap — materially changed M5 and M30, negligible elsewhere.
3. **2026-09-06**: added the full real cost model for this Razor account
   (confirmed live via `mt5.symbol_info`) -- Razor commission ($7/lot
   round-trip, confirmed from Pepperstone's pricing page) and overnight swap
   (currently -$79.73/lot/night long, +$29.62/lot/night short -- a live,
   drifting rate, not permanent). Swap turned out to be the single biggest
   previously-missing cost: it alone cut M15's Calmar from 130.71 to ~109 and
   M30's from 41.19 to ~28 (the two longest-average-holding-time profiles
   after H1) before the Razor commission was added on top too. The table
   above is the fully-costed, final number for each profile.

## Structure

- `mt5/config_mt5.py` — MT5 connection + base risk/strategy config
- `mt5/profiles/profile_{m1,m5,m15,m30,h1}.py` — per-timeframe Donchian parameters + calibration history
- `mt5/live_bot_mt5.py` — the live bot (`--profile` selects which one runs; supports both `donchian` and `rsi_macd` algorithms per-profile)
- `mt5/notifier.py` — Telegram trade notifications
- `mt5/export_history.py` — pulls historical XAUUSD data from MT5 into `data/*.csv`
- `mt5/track_performance.py`, `mt5/weekly_report.py` — live performance reporting utilities
- `mt5/run_with_watchdog.bat` — auto-restart wrapper for 24/7 operation
- `strategy/donchian.py` — the canonical Donchian engine (`simulate_donchian` — matches live bot behavior exactly, including the daily-loss guard)
- `strategy/signals.py`, `strategy/risk.py` — the original RSI+MACD logic and position sizing, kept as the `rsi_macd` fallback path
- `data/xauusd_*.csv` — historical XAUUSD data used for calibration

## Running live

Needs its own `.env` with a **separate** demo account's `MT5_LOGIN` /
`MT5_PASSWORD` / `MT5_SERVER` (never reuse scalp-bot's account). Magic
numbers (991xxx) keep orders distinct from scalp-bot (990xxx) and
scalp_silver (992xxx) even if ever pointed at the same account.

```
pip install -r requirements.txt
python mt5/live_bot_mt5.py --profile m15
```

Run once per profile (separate terminal/process) to run more than one
timeframe at a time. For auto-restart: `mt5\run_with_watchdog.bat m15`
(or m1 / m5 / m30 / h1). See `mt5/SETUP_MT5.md` for full MT5 terminal setup.

## Re-verifying the numbers

There's no single `backtest_verify.py` here (unlike scalp_silver/scalp_alerts)
— run the canonical engine directly, e.g.:

```python
from strategy.donchian import simulate_donchian
from mt5.profiles import profile_m15 as p
# ... load data/xauusd_m15.csv + data/xauusd_h4.csv, call simulate_donchian(...)
```
