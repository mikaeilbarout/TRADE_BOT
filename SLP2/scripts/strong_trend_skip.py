"""SLP2: no trades when the market trend is strong (user request 2026-09-30).

Two definitions of "strong trend", fixed before running (no search):
  T1  yesterday's daily ADX(14) > 35
  T2  gold moved more than 10% (either way) over the last 20 trading days
Live SLP2 settings otherwise (RR 5). Checked on the 4-year bar backtest split 70/30 (the same
split as the SLP2 review) and on the 6-month real-tick backtest.
Adopted only if, vs no filter, it improves the LAST 30% (R, with drawdown <= 1.2x) AND the tick
result (net R). Two variants -> one passing by luck is possible.
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate, DEFAULTS
from scripts.sp2l_tick_backtest import run_ticks, TICKS
import pyarrow.parquet as pq


def daily_context(frame):
    d1 = frame.set_index("bar_time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    tr = pd.concat([d1.high - d1.low, (d1.high - d1.close.shift()).abs(), (d1.low - d1.close.shift()).abs()], axis=1).max(axis=1)
    up, dn = d1.high.diff(), -d1.low.diff()
    pdm, ndm = up.where((up > dn) & (up > 0), 0.), dn.where((dn > up) & (dn > 0), 0.)
    a = tr.ewm(alpha=1 / 14).mean(); pdi = 100 * pdm.ewm(alpha=1 / 14).mean() / a; ndi = 100 * ndm.ewm(alpha=1 / 14).mean() / a
    d1["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean()
    d1["move20"] = d1.close.pct_change(20) * 100
    return d1.shift(1)                                     # only yesterday's completed day is known


def make(kind):
    def factory(frame):
        ctx = daily_context(frame)
        days = pd.to_datetime(frame.bar_time).dt.normalize()
        x = ctx.reindex(days)
        adx, mv = x.adx.to_numpy(), x.move20.to_numpy()

        def keep(i, direction):
            if kind == "T1":
                return not (adx[i] > 35)
            if kind == "T2":
                return not (abs(mv[i]) > 10)
            return True
        return keep
    return factory


def summary(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, win=0, R=0., DD=0.)
    c = np.cumsum(r)
    return dict(n=len(r), win=round(float((r > 0).mean()), 3), R=round(float(c[-1]), 1),
                DD=round(float(np.max(np.maximum.accumulate(np.maximum(c, 0)) - c)), 1))


def main():
    frame = load_m15()
    split = int(len(frame) * .7); boundary = frame.bar_time.iloc[split]
    train = frame.iloc[:split].reset_index(drop=True)
    warm = max(250, DEFAULTS.ema_period * 10 + 4)
    test = frame.iloc[max(0, split - warm):].reset_index(drop=True)
    meta = pq.ParquetFile(TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - pd.Timedelta(minutes=15)
    tick_frame = frame[frame.bar_time < end + pd.Timedelta(minutes=15)].reset_index(drop=True)
    res = {}
    for kind, name in (("NONE", "live (no filter)"), ("T1", "T1 skip if daily ADX > 35"), ("T2", "T2 skip if 20-day move > 10%")):
        ef = None if kind == "NONE" else make(kind)
        tr = simulate(train, entry_filter=ef); te = simulate(test, evaluation_start=boundary, entry_filter=ef)
        tk = run_ticks(tick_frame, start, end, entry_filter=ef)
        res[name] = dict(first70=summary(tr.r_multiple), last30=summary(te.r_multiple), ticks=summary(tk.r_multiple))
        a, b, k = res[name]["first70"], res[name]["last30"], res[name]["ticks"]
        print(f"{name:30} | first70 n={a['n']} win={a['win']} R={a['R']:+6.1f} DD={a['DD']} | last30 n={b['n']} win={b['win']} R={b['R']:+6.1f} DD={b['DD']} | "
              f"ticks n={k['n']} win={k['win']} R={k['R']:+6.1f} DD={k['DD']}", flush=True)
    base = res["live (no filter)"]
    for name in list(res)[1:]:
        r = res[name]
        r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["ticks"]["R"] > base["ticks"]["R"])
        print(f"ADOPT {name}: {r['adopt']}")
    out = os.path.join(ROOT, "data", "slp2_strong_trend_20260930"); os.makedirs(out, exist_ok=True)
    json.dump(dict(split=str(boundary), results=res), open(os.path.join(out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
