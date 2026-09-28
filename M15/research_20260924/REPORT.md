# Donchian backtest engine fix and re-evaluation — 2026-09-24

Bug: simulate_donchian read the H4 trend from the bar still forming (its future close). Fixed; the test tests/test_donchian_engine.py
fails on the old engine and passes on the new one.
FundedNext data (2022-06 to 2026-09) with real costs: spread 0.30, commission 7 USD/lot, FundedNext swap (long −107, short −47 points/lot/night).

Current settings: total +61.5R (PF 1.11, drawdown 32.9R ≈ 6.6% at 0.2% risk). By year: 2022 −3.6, 2023 +1.5, 2024 +2.9, 2025 +32.1, 2026 +28.7.
First 70% (to 2025-06): only +8.9R with a 32.9R drawdown; 14% of the 243 combinations were profitable. Last 30%: +53.9R (PF 1.28).
Selected (trend strength 0.3, 4 ATR stop): last 30% +47.7R < +53.9R → rejected; current settings kept.
The fixed engine agrees with ticks (86 of 86 shared trades give the same result). The remaining difference comes from the live bot
re-entering on the same bar after a stop (extra tick trades −10.9R over 6 months).
