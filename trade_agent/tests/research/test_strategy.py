from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Side
from research.backtest.indicators import (
    atr,
    compute_indicator_frame,
    ema,
    percentile_rank,
    rolling_high,
    rsi,
)
from research.strategy.seventy_thirty import (
    SeventyThirtyStrategy,
    StrategyParams,
    parameter_grid,
)

UTC = timezone.utc


def synthetic_bars(closes: list[float], start: datetime | None = None) -> pd.DataFrame:
    """Bar FIXTURES with a controlled shape, used to verify that the strategy
    rules fire where they should. These are not market data and are never
    used to produce reported backtest results."""
    start = start or datetime(2024, 6, 3, 7, 0, tzinfo=UTC)
    n = len(closes)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [start + timedelta(minutes=15 * i) for i in range(n)], utc=True
            ),
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "tick_count": [80] * n,
            "bid_close": [c - 0.1 for c in closes],
            "ask_close": [c + 0.1 for c in closes],
            "spread_mean": [0.2] * n,
            "spread_max": [0.4] * n,
            "is_partial": [False] * n,
        }
    )


# --- indicator causality ---------------------------------------------------


def test_indicators_are_causal():
    """The critical property: appending a future bar must not change any
    earlier indicator value. A non-causal indicator silently injects
    look-ahead bias into every trade."""
    closes = [2400 + i * 0.5 for i in range(300)]
    first = compute_indicator_frame(synthetic_bars(closes), ema_fast=50, ema_slow=200)
    extended = compute_indicator_frame(
        synthetic_bars(closes + [2600.0, 2700.0]), ema_fast=50, ema_slow=200
    )

    for column in ["ema_fast", "ema_slow", "rsi", "atr", "breakout_high", "atr_percentile"]:
        pd.testing.assert_series_equal(
            first[column], extended[column].iloc[: len(first)], check_names=False
        )


def test_rolling_high_excludes_the_current_bar():
    """Otherwise a bar always 'breaks out' above its own high."""
    highs = pd.Series([10.0, 11.0, 12.0, 9.0, 8.0])
    excluded = rolling_high(highs, period=2, exclude_current=True)
    # Bar 3's window is bars 1-2, so its own high (9.0) is not in the window.
    assert excluded.iloc[3] == 12.0
    # Bar 4's window is bars 2-3 -> max 12.0, again excluding bar 4 itself.
    assert excluded.iloc[4] == 12.0
    # Shifting costs one extra warm-up bar, which is the intended trade-off.
    assert pd.isna(excluded.iloc[1])


def test_ema_and_rsi_basic_behavior():
    rising = pd.Series([float(i) for i in range(1, 60)])
    assert rsi(rising).iloc[-1] == pytest.approx(100.0)
    assert ema(rising, 10).iloc[-1] < rising.iloc[-1]


def test_atr_is_positive_and_tracks_range():
    frame = synthetic_bars([2400 + (i % 7) for i in range(100)])
    values = atr(frame["high"], frame["low"], frame["close"], 14).dropna()
    assert (values > 0).all()


def test_percentile_rank_is_trailing_only():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    ranks = percentile_rank(series, window=3)
    assert pd.isna(ranks.iloc[1])
    assert ranks.iloc[-1] == pytest.approx(1.0)  # 100 exceeds both prior values


# --- strategy rules --------------------------------------------------------


def _uptrend_then_breakout(n: int = 320, seed: int = 7) -> list[float]:
    """A long uptrend with realistic bar-to-bar noise (so ATR actually
    varies and its trailing percentile is meaningful), a quiet
    consolidation, then a decisive break upward."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.2, n)
    trend = [2400 + i * 0.6 + float(noise[i]) for i in range(n)]
    consolidation = [trend[-1] + float(np.sin(i / 2)) * 0.8 for i in range(30)]
    breakout = [consolidation[-1] + 2.5 * (i + 1) for i in range(6)]
    return trend + consolidation + breakout


def _rule_test_params(**overrides) -> StrategyParams:
    """Params for testing entry RULES: the volatility band is opened up so
    each rule can be tested in isolation. The band itself is covered by its
    own tests below."""
    defaults = dict(
        ema_fast=20,
        ema_slow=100,
        breakout_lookback=12,
        volatility_window=50,
        atr_percentile_min=0.0,
        atr_percentile_max=1.0,
    )
    defaults.update(overrides)
    return StrategyParams(**defaults)


def test_strategy_emits_a_long_signal_on_an_uptrend_breakout():
    strategy = SeventyThirtyStrategy(_rule_test_params())
    prepared = strategy.prepare(synthetic_bars(_uptrend_then_breakout()))
    signals = strategy.generate(prepared)

    assert signals, "expected at least one breakout signal"
    first = signals[0]
    assert first.side == Side.BUY
    assert first.stop_loss < first.entry < first.take_profit
    assert first.risk_reward_ratio == pytest.approx(2.0, abs=0.01)  # 3.0/1.5 ATR multiples


def test_signal_records_entry_reason_and_market_conditions():
    strategy = SeventyThirtyStrategy(_rule_test_params())
    signals = strategy.generate(strategy.prepare(synthetic_bars(_uptrend_then_breakout())))
    signal = signals[0]

    assert "regime" in signal.entry_reason and "broke" in signal.entry_reason
    conditions = signal.market_conditions
    for key in ["session", "trend", "rsi", "atr", "ema_50", "ema_200", "atr_percentile"]:
        assert key in conditions, key
    assert conditions["trend"] == "UPTREND"


def test_strategy_never_trades_against_its_own_trend_filter():
    strategy = SeventyThirtyStrategy(_rule_test_params())
    prepared = strategy.prepare(synthetic_bars(_uptrend_then_breakout()))
    for signal in strategy.generate(prepared):
        trend = prepared["trend"].iat[signal.bar_index]
        expected = Side.BUY if trend == "UPTREND" else Side.SELL
        assert signal.side == expected


def test_session_filter_excludes_disallowed_hours():
    strategy = SeventyThirtyStrategy(_rule_test_params(allowed_sessions=("NEW_YORK",)))
    prepared = strategy.prepare(synthetic_bars(_uptrend_then_breakout()))
    for signal in strategy.generate(prepared):
        assert signal.market_conditions["session"] == "NEW_YORK"


def test_cooldown_enforces_spacing_between_signals():
    strategy = SeventyThirtyStrategy(_rule_test_params(cooldown_bars=25))
    signals = strategy.generate(strategy.prepare(synthetic_bars(_uptrend_then_breakout())))
    indices = [s.bar_index for s in signals]
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    assert all(gap >= 25 for gap in gaps)


def test_warmup_rows_produce_no_signals():
    """Bars before the indicators are defined must never trade."""
    strategy = SeventyThirtyStrategy(
        _rule_test_params(ema_fast=50, ema_slow=200, volatility_window=200)
    )
    prepared = strategy.prepare(synthetic_bars(_uptrend_then_breakout()))
    for signal in strategy.generate(prepared):
        assert signal.bar_index >= 200


def test_out_of_sample_warmup_bars_are_not_traded():
    """With `is_test` present (out-of-sample mode), only test-period bars may
    produce signals even though warm-up bars are supplied for indicators."""
    strategy = SeventyThirtyStrategy(_rule_test_params())
    prepared = strategy.prepare(synthetic_bars(_uptrend_then_breakout()))
    boundary = len(prepared) - 20
    prepared["is_test"] = [i >= boundary for i in range(len(prepared))]

    for signal in strategy.generate(prepared):
        assert signal.bar_index >= boundary


def test_parameter_grid_is_small_and_filters_bad_combinations():
    grid = parameter_grid()
    assert 0 < len(grid) < 500, "grid must stay small to limit overfitting"
    for params in grid:
        assert params.ema_fast < params.ema_slow
        assert params.implied_rr >= 1.5


def test_params_are_serializable_for_sealing():
    params = StrategyParams()
    payload = params.to_dict()
    assert isinstance(payload["allowed_sessions"], list)
    assert StrategyParams(**payload).to_dict() == payload


# --- volatility band (tested on its own, since rule tests open it up) ------


def test_narrowing_the_volatility_band_yields_a_strict_subset_of_signals():
    """Fixture-independent property: tightening the band can only remove
    signals, never add or move them."""
    bars_frame = synthetic_bars(_uptrend_then_breakout())
    permissive = SeventyThirtyStrategy(_rule_test_params())
    narrow = SeventyThirtyStrategy(
        _rule_test_params(atr_percentile_min=0.60, atr_percentile_max=0.95)
    )

    all_signals = permissive.generate(permissive.prepare(bars_frame))
    narrowed = narrow.generate(narrow.prepare(bars_frame))

    assert all_signals, "fixture should produce signals with an open band"
    assert len(narrowed) < len(all_signals)
    assert {s.bar_index for s in narrowed} < {s.bar_index for s in all_signals}


def test_every_signal_respects_the_volatility_band():
    params = _rule_test_params(atr_percentile_min=0.30, atr_percentile_max=0.95)
    strategy = SeventyThirtyStrategy(params)
    prepared = strategy.prepare(synthetic_bars(_uptrend_then_breakout()))
    for signal in strategy.generate(prepared):
        pct = prepared["atr_percentile"].iat[signal.bar_index]
        assert params.atr_percentile_min <= pct <= params.atr_percentile_max
