# Tick backtest of both bots together — 2026-09-24

Period: 2026-03-23 to 2026-09-22 (real ticks), account 25,561 USD, 0.2% risk per bot, commission 7 USD/lot. Swap and the AI review are not modelled.
SLP2: 22 trades, 6 wins, +469 USD (+1.84%), PF 1.51, drawdown 409 USD.
Donchian: 113 trades, 31 wins, −84 USD (−0.33%), PF 0.98, drawdown 645 USD.
Combined: +385 USD (+1.5%), PF 1.07, max drawdown 791 USD (3.1%).

Key finding: the Donchian backtest engine (strategy/donchian.py::simulate_donchian) matched the H4 trend with merge_asof on the H4 bar's "open" time
and so saw that bar's future close (look-ahead). The live bot only sees closed bars.
With this look-ahead the tick simulator gives +28R (matching the project's own tick_verified list at +26R); without it, about zero.
Over 4 years with the project's own engine: as-is +95.9%, fixed +14.9% (starting from 10,000 USD). The figures in the Donchian bot's README are overstated.
Files: all_trades.csv, donchian_WITH_lookahead_diagnostic.csv, report.json. No bot was changed.
