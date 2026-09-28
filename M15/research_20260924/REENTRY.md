# Donchian same-bar re-entry test — 2026-09-24

Simulation on the FundedNext 5-minute bar path (rules registered in advance; primary hypothesis: "after an exit, do not enter again until the next bar").
Decision period (2025-05 to 2026-03; separate from the period where the problem was seen):
  current behaviour +41.2R (202 trades, PF 1.27, drawdown 27.6R) | next bar +36.2R (drawdown 26.7R) | fresh signal +23.1R.
  → The adoption rule failed; the current behaviour was kept.
Tick period (2026-03 to 2026-09, calibration only): current −2.7R (tick simulator −1.7R, same 113 trades and 31 wins), next bar +2.5R.
Takeaway: the re-entries themselves lose money, but blocking them does not improve the total, because the bot usually enters one bar later at a similar price.
