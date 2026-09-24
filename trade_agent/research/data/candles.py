from __future__ import annotations

import pandas as pd

from research.data.sources.base import TICK_COLUMNS, TickDataUnavailableError

CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tick_count",
    "bid_close",
    "ask_close",
    "spread_mean",
    "spread_max",
    "is_partial",
]


def build_candles(
    ticks: pd.DataFrame, timeframe_minutes: int = 15, price: str = "mid"
) -> pd.DataFrame:
    """Aggregate raw ticks into exact fixed-width OHLC bars.

    Design decisions that matter for backtest accuracy:

    * Bars are labelled by their OPENING time and are left-closed /
      right-open, so the 14:30 bar contains ticks in [14:30, 14:45). A
      strategy acting on the 14:30 bar's close therefore acts at 14:45.
    * OHLC is built from the MID price by default, while `bid_close` and
      `ask_close` are kept per bar so the executor can fill buys at the ask
      and sells at the bid using the real recorded spread rather than an
      assumed one.
    * `volume` is summed tick volume and `tick_count` is the number of ticks;
      for FX/CFD tick data, tick_count is usually the more meaningful
      activity measure, so both are preserved.
    * Empty periods produce NO bar at all. Gaps (weekends, holidays, halts)
      are left as genuine gaps rather than forward-filled, because inventing
      flat bars would both distort indicators and fabricate tradeable prices.
    """
    missing = [c for c in TICK_COLUMNS if c not in ticks.columns]
    if missing:
        raise TickDataUnavailableError(f"tick frame missing columns {missing}")
    if ticks.empty:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    frame = ticks.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        raise TickDataUnavailableError("tick timestamps must be datetime64")
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")

    frame = frame.sort_values("timestamp")
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    frame["spread"] = frame["ask"] - frame["bid"]

    if price == "mid":
        price_series = "mid"
    elif price in {"bid", "ask"}:
        price_series = price
    else:
        raise ValueError(f"unsupported price basis: {price!r}")

    rule = f"{timeframe_minutes}min"
    grouped = frame.set_index("timestamp").resample(
        rule, label="left", closed="left", origin="epoch"
    )

    candles = grouped.agg(
        open=(price_series, "first"),
        high=(price_series, "max"),
        low=(price_series, "min"),
        close=(price_series, "last"),
        volume=("bid_volume", "sum"),
        tick_count=(price_series, "count"),
        bid_close=("bid", "last"),
        ask_close=("ask", "last"),
        spread_mean=("spread", "mean"),
        spread_max=("spread", "max"),
    )

    # Periods with no ticks resample to NaN rows -- drop them rather than
    # inventing prices for a closed market.
    candles = candles[candles["tick_count"] > 0].copy()
    candles["is_partial"] = candles["tick_count"] < 2
    candles = candles.reset_index()

    _validate_candles(candles)
    return candles[CANDLE_COLUMNS]


def _validate_candles(candles: pd.DataFrame) -> None:
    if candles.empty:
        return
    bad_high = candles["high"] < candles[["open", "close"]].max(axis=1)
    bad_low = candles["low"] > candles[["open", "close"]].min(axis=1)
    if bad_high.any() or bad_low.any():
        raise TickDataUnavailableError(
            "aggregation produced impossible candles "
            f"({int(bad_high.sum())} high violations, {int(bad_low.sum())} low violations)"
        )
    if candles["timestamp"].duplicated().any():
        raise TickDataUnavailableError("aggregation produced duplicate bar timestamps")
    if not candles["timestamp"].is_monotonic_increasing:
        raise TickDataUnavailableError("aggregated bars are not chronologically ordered")


def candle_coverage(candles: pd.DataFrame, timeframe_minutes: int = 15) -> dict:
    """Summarize how complete the bar series is, so a thin period is visible
    rather than silently producing a confident-looking backtest."""
    if candles.empty:
        return {"bars": 0, "first": None, "last": None, "expected_bars": 0, "coverage_pct": 0.0}

    first = candles["timestamp"].iloc[0]
    last = candles["timestamp"].iloc[-1]
    span_minutes = (last - first).total_seconds() / 60.0
    expected = int(span_minutes // timeframe_minutes) + 1
    return {
        "bars": int(len(candles)),
        "first": first,
        "last": last,
        "expected_bars": expected,
        # Well under 100% is normal for FX/CFDs: weekends and the daily break
        # are real gaps, not missing data.
        "coverage_pct": round(len(candles) / expected * 100, 2) if expected else 0.0,
        "median_tick_count": float(candles["tick_count"].median()),
        "median_spread": float(candles["spread_mean"].median()),
    }
