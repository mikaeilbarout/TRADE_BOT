"""Alternative research signal detector, separate from the live SP2L engine.

A confirmed fractal sets direction; three progressive candles and a positive
fair-value gap define an occurrence. Returned close prices are references,
not executable market fills. Historical tuning claims are not verified here.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot.indicators import atr as _atr
from bot.bars import validate_bars

M15_PATH = ROOT / "data" / "XAUUSD_M15_5y.parquet"
H1_PATH = ROOT / "data" / "XAUUSD_H1_full.parquet"
# Retained research settings; these are not the live SP2L parameters.
FRACTAL_LEGS = 3
MAX_BARS_FROM_TURN = 10
H1_FRACTAL_LEGS = 2
USE_H1_FILTER = False
H1_MODE = "counter"
PIP = 0.10
MIN_GAP = 1.50  # legacy constant, unused by the ATR-based detector
MIN_GAP_ATR_MULT = 0.4
RR = 4.0
BAD_HOURS = set()
EMA_TREND_PERIOD = 200
USE_EMA_FILTER = False


def load_m15():
    df = validate_bars(pd.read_parquet(M15_PATH), sort=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = df[df.bar_time + pd.Timedelta(minutes=15) <= now].reset_index(drop=True)
    df["ema_trend"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    df["atr"] = _atr(df, 14)
    return df


def load_h1():
    frame = validate_bars(pd.read_parquet(H1_PATH), sort=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return frame[frame.bar_time + pd.Timedelta(hours=1) <= now].reset_index(drop=True)


def attach_h1_trend(m15_df, h1_df, legs=H1_FRACTAL_LEGS):
    """Computes the same fractal-swing trend on H1 bars, then merges it onto
    the M15 frame causally: an M15 decision at time T only ever sees an H1
    trend value from an H1 bar that had FULLY CLOSED (bar_time + 1h) at or
    before T."""
    h1 = validate_bars(h1_df, sort=True)
    sh, sl = find_swings(h1, legs=legs)
    h1["h1_trend"] = trend_state(h1, sh, sl, legs=legs)
    h1["available_at"] = h1["bar_time"] + pd.Timedelta(hours=1)

    m15 = validate_bars(m15_df.assign(_original_order=np.arange(len(m15_df))), sort=True)
    m15["decision_time"] = m15["bar_time"] + pd.Timedelta(minutes=15)  # when the M15 bar actually closes
    merged = pd.merge_asof(m15, h1[["available_at", "h1_trend"]],
                           left_on="decision_time", right_on="available_at", direction="backward",
                           tolerance=pd.Timedelta(hours=1))
    merged = merged.sort_values("_original_order").drop(columns=["_original_order", "available_at"])
    merged.index = m15_df.index
    return merged


def find_swings(df, legs=FRACTAL_LEGS):
    if not isinstance(legs, int) or legs < 1:
        raise ValueError("legs must be a positive integer")
    validate_bars(df)
    high, low = df["high"].values, df["low"].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(legs, n - legs):
        window_h = high[i - legs:i + legs + 1]
        window_l = low[i - legs:i + legs + 1]
        if high[i] == window_h.max() and np.argmax(window_h) == legs:
            swing_high[i] = True
        if low[i] == window_l.min() and np.argmin(window_l) == legs:
            swing_low[i] = True
    return swing_high, swing_low


def trend_state(df, swing_high, swing_low, legs=FRACTAL_LEGS):
    """Returns an array: +1 = look for bullish setups (most recent confirmed
    fractal was a swing LOW -- the 'corner' itself marks the turn), -1 = look
    for bearish setups (most recent confirmed fractal was a swing HIGH),
    0 = no swing point confirmed yet. No structure-break/level-close
    confirmation is used -- the fractal point itself is the trend change."""
    n = len(df)
    trend = np.zeros(n, dtype=int)
    cur_trend = 0
    for i in range(n):
        confirm_idx = i - legs
        if confirm_idx >= 0:
            if swing_low[confirm_idx]:
                cur_trend = 1
            elif swing_high[confirm_idx]:
                cur_trend = -1
        trend[i] = cur_trend
    return trend


def trend_origin(df, swing_high, swing_low, legs=FRACTAL_LEGS):
    """Companion to trend_state: for each bar, the bar INDEX of the fractal
    swing point (the actual turning point candle) that set the trend value
    active at that bar. -1 if no swing point confirmed yet."""
    n = len(df)
    origin = np.full(n, -1, dtype=int)
    cur_origin = -1
    for i in range(n):
        confirm_idx = i - legs
        if confirm_idx >= 0 and (swing_low[confirm_idx] or swing_high[confirm_idx]):
            cur_origin = confirm_idx
        origin[i] = cur_origin
    return origin


def three_candle_pattern(df, i, direction, min_gap_mult=MIN_GAP_ATR_MULT, min_fvg=0.0, require_fvg=True):
    """Checks bars [i-2, i-1, i] for the strict progressive pattern + FVG
    created by the middle bar (i-1). The low/high progression gap is required
    to be at least min_gap_mult * ATR(14) at candle 1 -- adaptive to whatever
    volatility regime is active, instead of a fixed dollar amount.
    require_fvg=False disables the imbalance requirement."""
    if i < 2 or i >= len(df) or direction not in (-1, 1):
        return False
    if not all(np.isfinite(v) and v >= 0 for v in (min_gap_mult, min_fvg)):
        raise ValueError("Invalid pattern gap thresholds")
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if not np.isfinite(c1.atr) or c1.atr <= 0:
        return False
    min_gap = min_gap_mult * c1.atr
    if direction == 1:
        bullish = c1.close > c1.open and c2.close > c2.open and c3.close > c3.open
        progressive = (c2.low >= c1.low + min_gap and c2.open > c1.open and c2.close > c1.close and
                       c3.low >= c2.low + min_gap and c3.open > c2.open and c3.close > c2.close)
        if not require_fvg:
            return bullish and progressive
        fvg_size = c3.low - c1.high
        return bullish and progressive and fvg_size > 0 and fvg_size >= min_fvg
    else:
        bearish = c1.close < c1.open and c2.close < c2.open and c3.close < c3.open
        progressive = (c2.high <= c1.high - min_gap and c2.open < c1.open and c2.close < c1.close and
                       c3.high <= c2.high - min_gap and c3.open < c2.open and c3.close < c2.close)
        if not require_fvg:
            return bearish and progressive
        fvg_size = c1.low - c3.high
        return bearish and progressive and fvg_size > 0 and fvg_size >= min_fvg


def generate_signals(df, min_gap_mult=MIN_GAP_ATR_MULT, min_fvg=0.0, trend=None, bad_hours=BAD_HOURS,
                     use_ema_filter=USE_EMA_FILTER, use_h1_filter=USE_H1_FILTER, h1_mode=H1_MODE,
                     require_fvg=True, max_bars_from_turn=MAX_BARS_FROM_TURN, trend_origin_arr=None):
    """h1_mode: 'aligned' = only trade WITH the H1 fractal trend, 'counter' =
    only trade AGAINST it.
    max_bars_from_turn: candle 1 of the 3-candle pattern (bar i-2) must start
    within this many bars of the fractal swing point (the turning-point
    candle itself) -- requires a FRESH reversal, not a pattern deep inside an
    already-running move. Set to None to disable."""
    df = validate_bars(df)
    if h1_mode not in ("aligned", "counter"):
        raise ValueError("h1_mode must be aligned or counter")
    if trend is not None and (len(trend) != len(df) or not np.isin(trend, [-1, 0, 1]).all()):
        raise ValueError("Invalid trend array")
    if trend is None:
        swing_high, swing_low = find_swings(df)
        trend = trend_state(df, swing_high, swing_low)
        if max_bars_from_turn is not None and trend_origin_arr is None:
            trend_origin_arr = trend_origin(df, swing_high, swing_low)
    if max_bars_from_turn is not None and trend_origin_arr is None:
        raise ValueError("max_bars_from_turn requires trend_origin_arr when trend is precomputed externally")
    if use_h1_filter and "h1_trend" not in df.columns:
        raise ValueError("use_h1_filter=True requires df from attach_h1_trend() first")
    signals = []
    for i in range(2, len(df)):
        d = trend[i]
        if d == 0:
            continue
        if max_bars_from_turn is not None:
            origin = trend_origin_arr[i]
            if origin < 0 or not 0 <= (i - 2) - origin <= max_bars_from_turn:
                continue
        bar_time = df["bar_time"].iloc[i]
        if bad_hours and (bar_time + pd.Timedelta(minutes=15)).hour in bad_hours:
            continue
        if use_ema_filter:
            close = df["close"].iloc[i]
            ema = df["ema_trend"].iloc[i]
            aligned = (close > ema) if d == 1 else (close < ema)
            if not aligned:
                continue
        if use_h1_filter:
            h1t = df["h1_trend"].iloc[i]
            if not np.isfinite(h1t) or h1t == 0:
                continue
            if h1_mode == "aligned" and h1t != d:
                continue
            if h1_mode == "counter" and h1t != -d:
                continue
        if three_candle_pattern(df, i, d, min_gap_mult, min_fvg, require_fvg):
            c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            if d == 1:
                stop = min(c1.low, c2.low, c3.low)
            else:
                stop = max(c1.high, c2.high, c3.high)
            signals.append(dict(idx=i, bar_time=bar_time, direction=d,
                                entry=c3.close, stop=stop))
    return pd.DataFrame(signals, columns=["idx", "bar_time", "direction", "entry", "stop"])


if __name__ == "__main__":
    df = load_m15()
    print("M15 bars:", len(df))
    sig = generate_signals(df)
    print("raw pattern occurrences:", len(sig))
    print(sig.head(20).to_string(index=False))
    sig.to_csv(ROOT / "data" / "pattern_signals_reviewed.csv", index=False)
