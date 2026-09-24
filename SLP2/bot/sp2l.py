"""Causal SP2L state transitions; no broker calls or simulated fills here."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from bot.bars import validate_bars


@dataclass(frozen=True)
class Parameters:
    spike_size: float = 1.5
    gap: float = 2.0
    max_stop: float = 10.0
    rr: float = 5.0  # 3.0 until 2026-09-23; train-selected, see data/sp2l_rr_20260923/REPORT_FA.md
    ema_filter: bool = True
    ema_period: int = 20
    trend_filter: bool = True
    max_opposite: int = 2
    max_hold_minutes: int = 7500
    max_entry_delay_minutes: int = 5

    def __post_init__(self):
        for value in (self.spike_size, self.max_stop, self.rr, self.max_hold_minutes, self.max_entry_delay_minutes):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Strategy distances/durations must be positive")
        if not math.isfinite(self.gap) or self.gap < 0:
            raise ValueError("gap must be finite and nonnegative")
        if not isinstance(self.ema_period, int) or self.ema_period < 1:
            raise ValueError("ema_period must be a positive integer")
        if not isinstance(self.max_opposite, int) or self.max_opposite < 0:
            raise ValueError("max_opposite must be a nonnegative integer")


DEFAULTS = Parameters()


def add_indicators(df, ema_period=DEFAULTS.ema_period):
    """Bound EMA history so batch and rolling live windows have identical values."""
    if not isinstance(ema_period, int) or ema_period < 1:
        raise ValueError("Invalid EMA period")
    df = validate_bars(df)
    window = ema_period * 10
    weights = (1 - 2 / (ema_period + 1)) ** np.arange(window)
    df["EMA"] = np.nan
    if len(df) >= window:
        values = np.convolve(df.close.to_numpy(), weights, mode="valid") / weights.sum()
        df.loc[window-1:, "EMA"] = values
    return df


def detect_buy_setup(o, h, l, c, i, p_gap_price, spike_size=DEFAULTS.spike_size):
    if i < 3 or i >= len(c):
        return False
    a, b, cc = i-3, i-2, i-1
    return bool(l[i] < l[cc] and c[cc] > c[b] and o[cc] > o[b] and c[b] > c[a] and o[b] > o[a]
        and c[cc] > o[cc] and c[b] > o[b] and c[a] > o[a] and l[cc] > h[a] + p_gap_price
        and c[b]-o[b] > spike_size*(c[cc]-o[cc]) and c[b]-o[b] > spike_size*(c[a]-o[a])
        and c[b]-o[b] > spike_size*(c[i]-o[i]))


def detect_sell_setup(o, h, l, c, i, p_gap_price, spike_size=DEFAULTS.spike_size):
    if i < 3 or i >= len(c):
        return False
    a, b, cc = i-3, i-2, i-1
    return bool(h[i] > h[cc] and c[cc] < c[b] and o[cc] < o[b] and c[b] < c[a] and o[b] < o[a]
        and c[cc] < o[cc] and c[b] < o[b] and c[a] < o[a] and h[cc] < l[a] - p_gap_price
        and o[b]-c[b] > spike_size*(o[cc]-c[cc]) and o[b]-c[b] > spike_size*(o[a]-c[a])
        and o[b]-c[b] > spike_size*(o[i]-c[i]))


class PatternEngine:
    """An unresolved setup may be replaced by a new setup on the same closed bar.

    Persist pending between polls; do not recreate setups during actual positions.
    Array access keeps historical scans linear instead of repeated full replays.
    """
    def __init__(self, frame, params=DEFAULTS):
        self.params = params
        self.frame = add_indicators(frame, params.ema_period)
        self.o, self.h, self.l, self.c = (self.frame[col].to_numpy() for col in ("open","high","low","close"))
        self.ema = self.frame.EMA.to_numpy()
        self.times = self.frame.bar_time.to_numpy()

    def advance(self, pending, i):
        if i < 3:
            return pending, None
        p = self.params
        if pending is not None:
            pending = dict(pending)
            d = pending["direction"]
            favourable = self.h[i] > self.h[i-1] if d == 1 else self.l[i] < self.l[i-1]
            pending["opposite"] = 0 if favourable else pending["opposite"] + 1
            if p.trend_filter and pending["opposite"] > p.max_opposite:
                pending = None  # once violated, the old full-history filter can never pass
            elif (self.l[i] < self.l[i-1] if d == 1 else self.h[i] > self.h[i-1]):
                extreme = self.l[i] if d == 1 else self.h[i]
                distance = d * (extreme - pending["stop"])
                if not 0 < distance <= p.max_stop:
                    pending = None
                elif not p.ema_filter or (np.isfinite(self.ema[i]) and d*(self.c[i]-self.ema[i]) > 0):
                    # Trigger is known only at close; never promise a fill at the extreme.
                    trigger = dict(direction=d, stop=pending["stop"], ref_price=float(self.c[i]),
                        bar_time=str(pd.Timestamp(self.times[i])),
                        decision_time=str(pd.Timestamp(self.times[i])+pd.Timedelta(minutes=15)),
                        setup_time=pending["setup_time"])
                    return None, trigger
        if detect_buy_setup(self.o,self.h,self.l,self.c,i,p.gap,p.spike_size):
            pending = dict(direction=1,stop=float(self.l[i-3]),opposite=0,setup_time=str(pd.Timestamp(self.times[i])))
        elif detect_sell_setup(self.o,self.h,self.l,self.c,i,p.gap,p.spike_size):
            pending = dict(direction=-1,stop=float(self.h[i-3]),opposite=0,setup_time=str(pd.Timestamp(self.times[i])))
        return pending, None


def replay(frame, params=DEFAULTS):
    """Research-only flat scan, not a reconstruction of actual account exposure."""
    engine = PatternEngine(frame, params)
    pending, trigger = None, None
    for i in range(len(engine.frame)):
        pending, trigger = engine.advance(pending, i)
    return pending, trigger
