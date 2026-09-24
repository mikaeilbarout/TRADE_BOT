"""Pure UTC/price validation shared by research and live snapshots."""
import numpy as np
import pandas as pd


def utc_naive(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing timestamp")
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def validate_bars(frame, *, sort=False):
    frame = frame.copy()
    required = ["bar_time", "open", "high", "low", "close"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing OHLC/time columns")
    frame["bar_time"] = pd.to_datetime(frame.bar_time, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    if frame.bar_time.isna().any() or frame.bar_time.duplicated().any():
        raise ValueError("Missing/duplicate bar timestamps")
    if sort:
        frame = frame.sort_values("bar_time", kind="stable")
    if not frame.bar_time.is_monotonic_increasing:
        raise ValueError("Bars must be chronological")
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid OHLC prices")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
        (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC range")
    return frame.reset_index(drop=True)
