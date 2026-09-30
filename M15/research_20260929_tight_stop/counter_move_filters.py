"""Filters against the live case of 2026-09-30: a SHORT taken while gold was bouncing hard.

The bot's trend = price >= 0.5% on one side of the H4 EMA30. After a 3.9% one-day drop the
EMA lags: price had risen for 7 H4 candles (4115 -> 4182) and was still 0.97% below the
EMA, so the trend still read "short", and a 13-cent dip below a quiet Asian range became
a sell. Four logical ways to say "don't trade against a counter-move", each with ONE fixed
setting chosen before running (no parameter search):

  F1 h1_trend      price (last closed H1) on the trade side of the H1 EMA50
  F2 h4_momentum   last closed H4 close on the trade side of the H4 close 24h (6 bars) earlier
  F3 day_against   yesterday's daily move not more than 1% AGAINST the trade
  F4 ema_receding  price not moving back toward the H4 EMA30: |distance| now >= |distance| 6 H4 bars ago

Live settings otherwise (2x ATR stop, min $8, RR 3). Checked on: the two halves of the
pre-tick data (2022-06..2026-03) and the 6-month real-tick window.
Adoption rule (fixed before running): a filter is adopted only if it improves total R on
half A AND half B AND the tick net USD, and no drawdown is more than 1.2x the baseline's.
With 4 filters tested, one passing by luck is possible, so a pass would still be reported
with that caveat.
"""
import sys, os, json, types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); COMB = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research_20260924"))
m = types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
    setattr(m, k, 1)
sys.modules["MetaTrader5"] = m
import numpy as np
import pandas as pd
import reevaluate_donchian as rd

BASE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5, atr_stop_multiplier=2.0, reward_risk_ratio=3.0, min_stop_dollars=8.0)
TICK_START = pd.Timestamp("2026-03-23")
BAR = pd.Timedelta(minutes=15)


def context(m15):
    """per M15 bar (by open time): what was known when that bar closed"""
    ts = m15.ts
    decision = pd.DataFrame({"known": ts + BAR, "ts": ts})
    h1 = m15.set_index("ts")[["close"]].resample("1h").last().dropna()
    h1["ema50"] = h1.close.ewm(span=50, adjust=False).mean()
    h1 = h1.assign(known=h1.index + pd.Timedelta(hours=1)).reset_index(drop=True)
    h4 = m15.set_index("ts")[["close"]].resample("4h").last().dropna()
    h4["ema30"] = h4.close.ewm(span=30, adjust=False).mean()
    h4["dist"] = (h4.close - h4.ema30) / h4.close * 100
    h4["close_6ago"] = h4.close.shift(6); h4["dist_6ago"] = h4.dist.shift(6)
    h4 = h4.assign(known=h4.index + pd.Timedelta(hours=4)).reset_index(drop=True)
    d1 = m15.set_index("ts")[["close"]].resample("1D").last().dropna()
    d1["chg"] = d1.close.pct_change() * 100
    d1 = d1.assign(known=d1.index + pd.Timedelta(days=1)).reset_index(drop=True)
    c = pd.merge_asof(decision, h1.rename(columns={"close": "h1_close"}), on="known")
    c = pd.merge_asof(c, h4.rename(columns={"close": "h4_close"}), on="known")
    c = pd.merge_asof(c, d1.rename(columns={"close": "d1_close"})[["known", "chg"]], on="known")
    return c.set_index("ts")


def make_filters(c):
    def f(allow):
        def fn(ts, side):
            d = 1 if side == "long" else -1
            try:
                r = c.loc[ts]
            except KeyError:
                return True
            v = allow(r, d)
            return True if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)
        return fn
    return {
        "F1_h1_trend": f(lambda r, d: np.nan if np.isnan(r.ema50) else d * (r.h1_close - r.ema50) > 0),
        "F2_h4_momentum": f(lambda r, d: np.nan if np.isnan(r.close_6ago) else d * (r.h4_close - r.close_6ago) > 0),
        "F3_day_against": f(lambda r, d: np.nan if np.isnan(r.chg) else d * r.chg >= -1.0),
        "F4_ema_receding": f(lambda r, d: np.nan if np.isnan(r.dist_6ago) else abs(r.dist) >= abs(r.dist_6ago)),
    }


def summ(t):
    return rd.summary(t)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    pre = low[low.ts < TICK_START]; mid = pre.ts.iloc[len(pre) // 2]
    c = context(low)
    filters = make_filters(c)

    sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 2.0, 3.0, 8.0
    ctx_tick = context(frame.rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]])
    tick_filters = make_filters(ctx_tick)

    out = {}
    for name, fn in [("baseline", None)] + list(filters.items()):
        t = rd.run(low, dict(BASE, entry_filter=fn)); e = pd.to_datetime(t.entry_time)
        A, B, full = summ(t[e < mid]), summ(t[(e >= mid) & (e < TICK_START)]), summ(t)
        ctb.D_ENTRY_FILTER = None if fn is None else tick_filters[name]
        tk = ctb.stats(ctb.donchian(frame, start, end))
        out[name] = dict(A=A, B=B, full=full, tick=tk,
                         years={int(k): round(float(x), 1) for k, x in t.R.groupby(pd.to_datetime(t.exit_time).dt.year).sum().items()})
        print(f"{name:16} | A n={A['n']:3d} R={A['R']:6.1f} DD={A['DD']:5.1f} | B n={B['n']:3d} win={B['win']:.2f} R={B['R']:6.1f} DD={B['DD']:5.1f} | "
              f"4y R={full['R']:6.1f} DD={full['DD']:5.1f} | TICK n={tk['trades']} win={tk['win_rate']} ${tk['net_usd']} DD {tk['max_dd_pct']}% | {out[name]['years']}", flush=True)
    b = out["baseline"]
    for name in filters:
        o = out[name]
        o["adopt"] = bool(o["A"]["R"] > b["A"]["R"] and o["B"]["R"] > b["B"]["R"] and o["tick"]["net_usd"] > b["tick"]["net_usd"]
                          and o["A"]["DD"] <= 1.2 * b["A"]["DD"] and o["B"]["DD"] <= 1.2 * b["B"]["DD"] and o["tick"]["max_dd_usd"] <= 1.2 * b["tick"]["max_dd_usd"])
        print(f"ADOPT {name}: {o['adopt']}")
    json.dump(dict(half_split=str(mid), results=out), open(os.path.join(HERE, "counter_move_filters.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
