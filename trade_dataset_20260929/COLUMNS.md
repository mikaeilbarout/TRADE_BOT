# Donchian trade dataset (live settings)

**Files**
- `donchian_dataset.csv` — for Excel or an AI model; all columns numeric, 4 decimals.
- `donchian_dataset.parquet` / `donchian_dataset_full.parquet` — same data with exact dtypes (not in git: `*.parquet` is ignored; rebuild with `build_dataset.py`).
- `schema.json` — name, dtype, unit and description of every column, plus the columns removed as chance and the cells filled with the column mode.
- `build_dataset.py` — rebuilds everything (needs the local M15/tick data and read-only MT5 for the related markets).

**Bot:** Donchian M15, 20-bar channel, only with the H4 EMA30 trend (price at least 0.3% away), stop 2 x ATR14 with an 8 USD minimum, take profit 4R, risk 0.2% per trade in R terms.
**599 trades**, one continuous history, never two positions at once:
- 2022-07 to 2026-03-20: look-ahead-free M15 bar engine
- from 2026-03-23: real-tick simulation (`is_tick_sim` = 1)

**Rule:** every column is known **before** the order is sent; only the last three (`outcome`, `win`, `R`) are the result.

## Columns
| Column | Type | Meaning |
|---|---|---|
| `trade_id` | int | trade number, time order |
| `hour_utc` | int | hour of the decision, UTC |
| `weekday` | int | 0 = Monday .. 4 = Friday (6 = Sunday-evening open) |
| `session` | int | 0 Asia 00-06, 1 London 07-11, 2 London/NY overlap 12-15, 3 New York 16-20, 4 late 21-23 (UTC) |
| `is_long` | 0/1 | 1 = buy, 0 = sell |
| `entry_price` | float | planned entry (close of the signal bar) |
| `m15_atr14_usd` | float | M15 ATR(14) in USD; stop = 2x this (min 8), target = 4x the stop |
| `m15_atr_vs_5d_avg` | float | M15 ATR / its 5-day average (above 1 = more volatile than usual) |
| `d1_atr14_usd` | float | daily ATR(14), previous completed day |
| `d1_adx14` | float | daily ADX(14), previous completed day (trend strength) |
| `d1_trend_with_trade` | 0/1 | 1 = trade in the direction of the daily EMA50 |
| `day1_change_pct` .. `day7_change_pct` | float | gold % change on each of the last 7 completed trading days (market direction, + = up) |
| `month_change_pct` | float | gold % change over the last 30 calendar days |
| `h4c1_*` .. `h4c5_*` | float | the last 5 closed H4 candles (1 = most recent): `change_pct` close vs open, `range_pct` size, `close_position` 0 = closed at the low, 1 = at the high |
| `h4_trend_age_bars` | int | H4 bars since price last crossed the H4 EMA30 (6 = 1 day) |
| `h4_ema30_distance_pct` | float | distance from the H4 EMA30, + = in the trade direction |
| `h4_ema30_slope_1d_pct` | float | H4 EMA30 change over the last day, + = in the trade direction |
| `channel_width_atr` | float | 20-bar channel width in ATR |
| `breakout_size_atr` | float | how far the signal bar closed beyond the channel, in ATR |
| `signal_bar_range_atr` | float | signal bar size in ATR |
| `signal_bar_close_position` | float | where the signal bar closed in its range, 1 = at the extreme in the trade direction |
| `move_1h_atr` / `move_4h_atr` / `move_24h_atr` | float | price move over the last 1/4/24 h in ATR, + = in the trade direction |
| `signal_bar_volume_vs_20` | float | tick volume of the signal bar / average of the 20 bars before |
| `volume_4h_vs_5d` | float | tick volume of the last 4 h / 5-day average |
| `spread_vs_5d` | float | signal-bar spread / 5-day average |
| `eurusd_4h_pct` / `eurusd_24h_pct` / `eurusd_5d_pct` | float | EURUSD change (+ = weaker dollar) |
| `usdjpy_24h_pct` | float | USDJPY change (+ = stronger dollar) |
| `silver_24h_pct` | float | silver 24 h change |
| `silver_minus_gold_24h_pct` | float | silver minus gold 24 h change |
| `spx500_24h_pct` / `spx500_5d_pct` | float | S&P 500 change (history from 2022-10) |
| `prior_loss_streak` | int | consecutive losses right before this trade |
| `prev_trade_R` | float | result of the previous trade in R |
| `prev_trade_same_side` | 0/1 | 1 = previous trade had the same direction |
| `hours_since_prev_exit` | float | hours since the previous trade closed |
| `is_tick_sim` | 0/1 | 1 = simulated on real ticks, 0 = on M15 bars |
| **`outcome`** | int | **result:** 1 take profit, -1 stop loss, 0 7-day time stop |
| **`win`** | 0/1 | **result:** 1 = profitable after costs |
| **`R`** | float | **result:** profit in units of risk after costs (a full win is about +3.9R) |

Directional columns (distance, slope, move, breakout) are positive in the trade's direction, so buys and sells are comparable;
`day*`, `month`, `h4c*`, EURUSD/USDJPY/silver/S&P columns are raw market direction.
Empty cells were filled with the column mode (listed in `schema.json` under `filled_with_mode`).
Analyses of this data (logistic regression, decision tree, random forest, XGBoost-style boosting) found no
column set that predicts wins on newer trades; see the `*.json` results next to the scripts.
