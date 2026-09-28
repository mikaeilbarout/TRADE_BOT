# Review report — 2026-09-23

The code in the folder was reviewed and fixed. The current version of the project is SP2L on XAUUSD; the pattern_strategy script is a separate research tool. Old result files do not count as results of the fixed version.

## Fixes

- One shared pattern-detection engine for live trading and backtesting; aligned RR=3, EMA=20, gap distance=2 and maximum opposite move=2.
- Removed the unexecutable entry at the low/high of an already closed bar: the signal happens at bar close and the backtest enters at the next bar's open, with costs. The live target is calculated from the real entry price.
- Bid/Ask sides, slippage, gaps through the stop, stop hits in the entry bar, time exits and end-of-data settlement are all modelled. A stop already crossed by the exit price blocks the entry.
- Order intent is recorded before sending; unknown responses and partial fills are kept; duplicate sends after a crash are prevented; retries on definitive rejections are limited; state is reconciled with the broker's deal history.
- Atomic state saving, a proper run lock, and separate state for accounts and for demo/live mode. Corrupt state is never silently cleared; an account change while running stops ordering.
- Pending patterns are kept between runs; patterns formed while a position is open are dropped; hypothetical past trades are not rebuilt at start-up.
- Checks on all open positions and orders for the symbol, price and broker data validity, margin and minimum stop distance. Volume is calculated after rounding the stop, with room for commission and price deviation.
- Time exits for the bot's own positions and positions without a stop; an unknown close response is not blindly repeated.
- OHLC and time validation, duplicate timestamps rejected, unfinished bar removed, time-zone-aware timestamps supported, and the incorrect H1 join on unsorted data fixed.
- Fixed a negative index in the three-bar pattern, zero gap, invalid filter direction/mode, and detection of the four-bar pattern from the first allowed index. Removed old validation claims whose documentation no longer exists.

## Behaviour notes

The EMA is now an exponential average with a limited window of 10× the period, so the full-history and live-window results are identical; it is therefore not exactly the old recursive EMA. After installation or a settings change, the program starts from the latest closed bar and waits for a fresh pattern.

The default dry-run mode only records signals and requests; it is not a full position simulator. An unknown order response deliberately blocks new entries until its execution is confirmed from the broker history; if automatic confirmation is not possible, a manual check of the broker state is required. Do not delete the state file to clear this block.

## Verification

44 automated tests passed against a fully fake MetaTrader5 API; no test connects to an account. Coverage includes a crash after sending, save errors, partial fills, unknown responses, account changes, risk calculation, state recovery, invalid data, executable entry, time exits, stop gaps and no use of future bars.

The backtest ran on the 100,000 available M15 bars; the 70/30 split is at 2025-06-17 13:45. The settings were not optimised on this split.

| Part | Trades | Win rate | Profit factor | Total R | Max drawdown R |
|---|---:|---:|---:|---:|---:|
| Train | 122 | 27.87% | 1.122 | 10.745 | 13.402 |
| Test | 56 | 32.14% | 1.271 | 11.311 | 8.704 |

These results are a historical diagnostic; the history of how the parameters were chosen cannot be verified, so the test part should not be treated as a fresh independent test. The execution model is OHLC, not a tick backtest. Historical spread, 2-point slippage and a commission of 7 per lot are included; swap, all dynamic limits and real broker latency are not modelled. A full audit of the raw tick files was not done. Fixing bugs does not guarantee profitability or the absence of other bugs.

## Files and running

- original_source.zip and original_manifest.json: backup of the sources before the fixes, with their hashes.
- tests.txt: test output.
- backtest_output.txt: backtest output.
- ../../data/slp2_reviewed_20260923/: trades and report.json of the current version.

From the project root:

```powershell
python -m unittest discover -s tests -v
python scripts/sp2l_m15_backtest.py
python SLP2.py --help
```

No live bot was run and no order was sent to the broker during this review.
