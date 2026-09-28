# SP2L optimisation result — 2026-09-23

98 unique combinations were tested with the fixed seed 23092026. Selection used only the first 70% of the history. The ranking score was total R divided by maximum drawdown R (floored at 5R), with a 0.5 penalty for each losing time third; at least 10 trades per third were required. Account risk was not changed.

Search space: RR 1, 1.5, 2, 2.5, 3, 4; gap 0.5, 1, 1.5, 2; max stop distance 6, 10, 15; spike factor 1.25, 1.5, 2; EMA 20, 40, 60, 100 or off; allowed opposite moves 1, 2, 3. This is a limited random search, not an exhaustive one, and not proof of the best possible setting.

Best training setting: RR=4, EMA=60, max_stop=15, max_opposite=1, gap=2, spike_size=1.5. The trend filter and EMA stayed on.

| Metric | Current setting, evaluation | Train pick, evaluation |
|---|---:|---:|
| Trades | 56 | 69 |
| Win rate | 32.14% | 23.19% |
| Total R | 11.311 | 6.348 |
| Profit factor | 1.271 | 1.112 |
| Max drawdown R | 8.704 | 11.036 |

In training, the pick made 33.835R with profit factor 1.364 and a 12.032R drawdown; all three training parts were positive. However, the result did not carry over to the evaluation part. The adoption rule was saved before the search ran: at least 40 evaluation trades, profit factor above 1, more profit than the current setting, and drawdown at most 1.5× the current setting. The profit condition failed, so the active settings were not changed and no alternative was chosen from the evaluation results.

With 5-point slippage and a commission of 10 per lot, the pick made 5.827R in evaluation with profit factor 1.102. Changing costs can also change the trade path, so the cost test is not just subtracting a fixed amount from the same trades.

The evaluation boundary is 2025-06-17 13:45. The final part had already been looked at earlier in the conversation, so it is not a completely untouched test. Execution is bar-based with historical spread; swap, real rounded volume and all broker limits are not modelled. The best training result does not guarantee future performance.

protocol.json holds all combinations and the pre-registered score; report.json has the ranking, statistics for every combination and the cost comparison. baseline_test_trades.csv and selected_test_trades.csv hold the compared trades. Reproducible script: scripts/optimize_sp2l.py. No real order was sent.
