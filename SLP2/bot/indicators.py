"""Pure indicator functions -- identical formulas to the ones used throughout
the backtest scripts, kept standalone here so the live bot has no dependency
on the research/scripts folder.
"""

import pandas as pd
import numpy as np


def smooth(series, period=14, window=128):
    """Finite exponentially weighted kernel; identical for batch and live tails.

    Version 2 deliberately bounds indicator memory, preventing a different EWM
    seed on every live poll. Initial missing values propagate through warmup.
    """
    weights = (1 - 1 / period) ** np.arange(window)
    if len(series) < window:
        return pd.Series(np.nan, index=series.index)
    values = np.convolve(series.to_numpy(dtype=float), weights, mode="valid") / weights.sum()
    return pd.Series(np.r_[np.full(window - 1, np.nan), values], index=series.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = smooth(gain, period)
    avg_loss = smooth(loss, period)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return smooth(tr, period)


def wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = smooth(tr, period)
    plus_di = 100 * smooth(pd.Series(plus_dm, index=df.index), period) / atr_
    minus_di = 100 * smooth(pd.Series(minus_dm, index=df.index), period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return smooth(dx.fillna(0).where(atr_.notna()), period)


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)


def kaufman_er_rolling(close: pd.Series, window: int) -> pd.Series:
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net / path


def bollinger_bands(close: pd.Series, period: int, n_std: float):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma - n_std * std, sma, sma + n_std * std
