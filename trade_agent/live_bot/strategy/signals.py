"""
Strategy logic: trend filter (EMA) + entry trigger (RSI + MACD) + volatility filter (ATR).
This module only produces signals; it never sends orders.
"""

import pandas as pd
import numpy as np


def add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(cfg.rsi_period).mean()
    avg_loss = loss.rolling(cfg.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema_f = df["close"].ewm(span=cfg.macd_fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=cfg.macd_slow, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["macd_signal"] = df["macd"].ewm(span=cfg.macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(cfg.atr_period).mean()
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    return df


def trend_direction(higher_tf_row, min_strength_pct: float = 0.0) -> str:
    """
    Trend direction from the higher timeframe row: 'long', 'short', or 'flat'.

    min_strength_pct (optional, default 0 = old behavior): requires the EMA
    spread to be at least this percent of price before trusting a crossover.
    Right at a crossover the EMA trend filter is laggiest and most likely to
    still disagree with where price is actually already heading (see
    mt5/profiles/profile_m5.py for the real trade sequence that motivated
    this), so treating a barely-flipped EMA pair as "flat" avoids entries at
    exactly the worst moment.
    """
    fast = higher_tf_row["ema_fast"]
    slow = higher_tf_row["ema_slow"]
    if min_strength_pct > 0:
        close = higher_tf_row["close"]
        if close and not pd.isna(close) and not pd.isna(fast) and not pd.isna(slow):
            spread_pct = abs(fast - slow) / close * 100
            if spread_pct < min_strength_pct:
                return "flat"
    if fast > slow:
        return "long"
    elif fast < slow:
        return "short"
    return "flat"


def volatility_ok(row, cfg) -> bool:
    if pd.isna(row["atr_pct"]):
        return False
    return cfg.min_atr_pct <= row["atr_pct"] <= cfg.max_atr_pct


def entry_signal(row, prev_row, trend: str, cfg) -> str | None:
    """
    Entry signal, only in the direction of the higher-timeframe trend.
    Returns 'long', 'short', or None.

    cfg.require_macd_confirmation (default True) controls whether an MACD
    histogram cross is also required, or whether an RSI extreme alone is enough.

    cfg.require_rsi_turning (default False): requires RSI to already be
    reversing (rising for a long, falling for a short) rather than just past
    the oversold/overbought threshold. Without this, entries can fire while
    RSI (and price) is still falling, catching a pullback before it's done --
    backtesting found this materially reduces drawdown for profiles with
    wide RSI thresholds far from the midline (e.g. M1's 35/65), but hurts
    profiles with thresholds near the midline (e.g. 48/52), where RSI
    naturally oscillates too much for "already turning" to mean much.
    """
    if not volatility_ok(row, cfg):
        return None

    require_macd = getattr(cfg, "require_macd_confirmation", True)
    require_rsi_turning = getattr(cfg, "require_rsi_turning", False)
    macd_cross_up = prev_row["macd_hist"] <= 0 and row["macd_hist"] > 0
    macd_cross_down = prev_row["macd_hist"] >= 0 and row["macd_hist"] < 0

    if trend == "long" and row["rsi"] <= cfg.rsi_oversold:
        rsi_ok = not require_rsi_turning or row["rsi"] > prev_row["rsi"]
        if rsi_ok and (not require_macd or macd_cross_up):
            return "long"
    if trend == "short" and row["rsi"] >= cfg.rsi_overbought:
        rsi_ok = not require_rsi_turning or row["rsi"] < prev_row["rsi"]
        if rsi_ok and (not require_macd or macd_cross_down):
            return "short"
    return None
