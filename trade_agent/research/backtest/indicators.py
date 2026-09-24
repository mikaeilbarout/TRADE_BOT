from __future__ import annotations

import numpy as np
import pandas as pd

"""Vectorized indicators for backtesting.

Every function here is causal: the value at bar i uses only bars <= i. That
is the single most important property in this file -- a non-causal indicator
(e.g. a centered rolling window) silently injects look-ahead bias into every
trade, and the resulting backtest looks excellent and means nothing.
"""


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # All-gain windows have no loss to divide by: RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, out.fillna(50.0))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


def rolling_high(high: pd.Series, period: int, exclude_current: bool = True) -> pd.Series:
    """Highest high of the last `period` bars.

    `exclude_current=True` shifts the window back one bar so a breakout
    comparison against the CURRENT bar's close is not comparing the close to
    a high that the same bar produced -- which would make every breakout
    trivially true.
    """
    window = high.shift(1) if exclude_current else high
    return window.rolling(period, min_periods=period).max()


def rolling_low(low: pd.Series, period: int, exclude_current: bool = True) -> pd.Series:
    window = low.shift(1) if exclude_current else low
    return window.rolling(period, min_periods=period).min()


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Causal percentile rank of the latest value within a trailing window,
    used to judge whether current volatility is unusually low or high
    without referencing the full-sample distribution (which would leak)."""
    return series.rolling(window, min_periods=window).apply(
        lambda values: (values[:-1] < values[-1]).mean(), raw=True
    )


def swing_structure(high: pd.Series, low: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Simple causal market-structure read: are we making higher highs and
    higher lows (uptrend), lower highs and lower lows (downtrend), or
    neither (range)?"""
    prior_high = rolling_high(high, lookback)
    prior_low = rolling_low(low, lookback)
    higher_high = high > prior_high
    lower_low = low < prior_low

    structure = pd.Series("RANGE", index=high.index, dtype="object")
    structure[higher_high & ~lower_low] = "HIGHER_HIGH"
    structure[lower_low & ~higher_high] = "LOWER_LOW"
    structure[higher_high & lower_low] = "OUTSIDE"
    return pd.DataFrame(
        {"structure": structure, "prior_high": prior_high, "prior_low": prior_low}
    )


def session_of(timestamps: pd.Series) -> pd.Series:
    """Trading session by UTC hour. Gold's character differs markedly
    between the Asian range and the London/NY trend hours, so the strategy
    can filter on it."""
    hour = timestamps.dt.hour
    session = pd.Series("ASIA", index=timestamps.index, dtype="object")
    session[(hour >= 7) & (hour < 12)] = "LONDON"
    session[(hour >= 12) & (hour < 16)] = "OVERLAP"
    session[(hour >= 16) & (hour < 21)] = "NEW_YORK"
    return session


def compute_indicator_frame(
    candles: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    rsi_period: int = 14,
    atr_period: int = 14,
    breakout_lookback: int = 20,
    structure_lookback: int = 20,
    volatility_window: int = 200,
) -> pd.DataFrame:
    """Attach every indicator the strategy and the technical agent need.

    Returned frame is aligned 1:1 with `candles`; warm-up rows contain NaN
    and are skipped by the executor rather than being filled in.
    """
    frame = candles.copy()
    close, high, low = frame["close"], frame["high"], frame["low"]

    frame["ema_fast"] = ema(close, ema_fast)
    frame["ema_slow"] = ema(close, ema_slow)
    frame["ema_50"] = ema(close, 50)
    frame["ema_200"] = ema(close, 200)
    frame["rsi"] = rsi(close, rsi_period)
    frame["atr"] = atr(high, low, close, atr_period)
    frame[["macd", "macd_signal", "macd_hist"]] = macd(close)
    frame["breakout_high"] = rolling_high(high, breakout_lookback)
    frame["breakout_low"] = rolling_low(low, breakout_lookback)
    frame[["structure", "prior_high", "prior_low"]] = swing_structure(
        high, low, structure_lookback
    )
    frame["atr_percentile"] = percentile_rank(frame["atr"], volatility_window)
    frame["session"] = session_of(frame["timestamp"])
    frame["trend"] = np.where(
        frame["ema_fast"] > frame["ema_slow"],
        "UPTREND",
        np.where(frame["ema_fast"] < frame["ema_slow"], "DOWNTREND", "RANGE"),
    )
    return frame
