# TRADE_BOT

Two live XAUUSD (gold) bots for MetaTrader 5 and the AI service that reviews their signals.

| Folder | What it is | Run |
|---|---|---|
| `SLP2/` | **SLP2** — SP2L 4-candle spike/gap pattern, entry on the first pullback, M15, RR 5 | `python SLP2.py` (dry-run) / `python SLP2.py --live` |
| `M15/` | **Donchian** — 10-bar channel breakout with an H4 EMA30 trend filter, M15, RR 3, ATR stop | `mt5\run_with_watchdog.bat m15` |
| `trade_agent/` | **AI review service** (FastAPI, Docker) — approves/rejects bot signals | `docker compose up -d --build` |
| `combined_tick_backtest.py` | Both bots together on 6 months of ticks | — |
| `restart_live_bots.bat` | Stop and restart both live bots | — |

Both bots detect the broker server clock (UTC offset) automatically, ask the AI service before every
entry (fail-open: an unreachable service or its own data failure does not block a trade) and send
Telegram notifications.

## Setup
1. Copy each `.env.example` to `.env` (in `SLP2/`, `M15/`, `trade_agent/`) and fill in your own values.
   Real `.env` files are git-ignored — never commit them.
2. `pip install -r SLP2/requirements.txt -r M15/requirements.txt`
3. Market data files (`*.parquet`) for the SLP2 backtests are not included; export them from MT5.

## Tests
- `cd SLP2 && python -m unittest discover -s tests`
- `cd M15 && python -m unittest discover -s tests`
- `cd trade_agent && python -m pytest`

Trading involves risk; backtest results are not a guarantee of future performance.
