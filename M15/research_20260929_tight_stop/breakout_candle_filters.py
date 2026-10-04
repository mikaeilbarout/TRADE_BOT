"""Three breakout-candle filters borrowed from the gold M1 bot, tested on the live Donchian setup.

Pre-registered BEFORE running (one fixed setting each, no search). Live = N20, EMA30, 0.3%, 2 ATR (min $8), RR 4,
plus the S&P 5-day filter. The signal bar = the M15 bar that closed beyond the 20-bar channel (ts = its open time).
  G1 activity   tick_volume(signal bar) / median(tick_volume of the 12 bars before it)  >= 1.0
  G2 closeloc   close within the outer 30% of the bar range on the trade side: long (close-low)/range >= 0.7, short (high-close)/range >= 0.7
  G3 extension  side-signed move over the last 15 bars / median bar range of the prior 12 bars  < P67 of that value on the
                trades of the FIRST 70% of the data (same rule as the S&P filter)
Adoption rule: improves total R on half A AND half B of the pre-tick data AND tick net USD, DD <= 1.2x baseline in each,
and a permutation test (remove the same number of random baseline trades) gives p < 0.10 on the 4-year R. 3 filters
tested, so a pass is still reported with the multiple-testing caveat.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import counter_move_filters as cm
import spx_filter as sf
rd = cm.rd
NEW = sf.NEW
BAR = pd.Timedelta(minutes=15)

def features(df):
    d = df.rename(columns={"bar_time": "ts"}).set_index("ts")[["open", "high", "low", "close", "tick_volume"]].copy()
    rng = (d.high - d.low).replace(0, np.nan)
    d["act"] = d.tick_volume / d.tick_volume.shift().rolling(12).median()
    d["loc_long"] = (d.close - d.low) / rng
    d["loc_short"] = (d.high - d.close) / rng
    scale = rng.shift().rolling(12).median()
    d["ext_long"] = (d.close - d.close.shift(15)) / scale
    d["ext_short"] = (d.close.shift(15) - d.close) / scale
    return d

def make(f, thr_ext):
    def wrap(test):
        def fn(ts, side):
            if not sf.spx_filter(ts, side):
                return False
            try:
                r = f.loc[ts]
            except KeyError:
                return True
            v = test(r, side)
            return True if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)
        return fn
    return {
        "G1_activity": wrap(lambda r, s: np.nan if np.isnan(r.act) else r.act >= 1.0),
        "G2_closeloc": wrap(lambda r, s: (np.nan if np.isnan(r.loc_long) else (r.loc_long if s == "long" else r.loc_short) >= 0.7)),
        "G3_extension": wrap(lambda r, s: (np.nan if np.isnan(r.ext_long) else (r.ext_long if s == "long" else r.ext_short) < thr_ext)),
    }

def main():
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb, pyarrow.parquet as pq
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    full_df = pd.read_parquet(rd.DATA); f = features(full_df)
    pre = low[low.ts < cm.TICK_START]; mid = pre.ts.iloc[len(pre) // 2]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    # G3 threshold: P67 of side-signed extension over the first-70% baseline trades (fixed before filtering)
    base_all = rd.run(low, dict(NEW, entry_filter=sf.spx_filter)); e0 = pd.to_datetime(base_all.entry_time)
    first = base_all[e0 < boundary]
    sig_ts = pd.to_datetime(first.entry_time) - BAR
    vals = [(f.ext_long if s == "long" else f.ext_short).get(t, np.nan) for t, s in zip(sig_ts, first.side)] if "side" in first else []
    thr = float(np.nanpercentile(vals, 67)) if len(vals) else float("nan")
    print("trade columns:", base_all.columns.tolist()[:12], "| G3 P67 threshold:", round(thr, 3), flush=True)
    filters = make(f, thr)

    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 20, 30, 0.3, 2.0, 4.0, 8.0
    out = {}; rng = np.random.default_rng(7)
    for name, fn in [("baseline(live)", sf.spx_filter)] + list(filters.items()):
        t = rd.run(low, dict(NEW, entry_filter=fn)); e = pd.to_datetime(t.entry_time)
        A, B, F = rd.summary(t[e < mid]), rd.summary(t[(e >= mid) & (e < cm.TICK_START)]), rd.summary(t)
        ctb.D_ENTRY_FILTER = fn; tk = ctb.stats(ctb.donchian(frame, start, end))
        out[name] = dict(A=A, B=B, full=F, tick=tk, trades=t[["entry_time", "R"]].assign(entry_time=lambda x: x.entry_time.astype(str)).to_dict("list"))
        print(f"{name:15} | A n={A['n']:3d} R={A['R']:6.1f} DD={A['DD']:5.1f} | B n={B['n']:3d} R={B['R']:6.1f} DD={B['DD']:5.1f} | 4y n={F['n']} win={F['win']:.2f} R={F['R']:6.1f} DD={F['DD']:5.1f} | TICK n={tk['trades']} ${tk['net_usd']} DD {tk['max_dd_pct']}%", flush=True)
    b = out["baseline(live)"]; bR = np.array(b["trades"]["R"])
    for name in filters:
        o = out[name]; k = len(bR) - o["full"]["n"]
        perm = [rng.permutation(bR)[: len(bR) - k].sum() for _ in range(5000)] if k > 0 else [bR.sum()]
        o["perm_p"] = float(np.mean(np.array(perm) >= o["full"]["R"]))
        o["adopt"] = bool(o["A"]["R"] > b["A"]["R"] and o["B"]["R"] > b["B"]["R"] and o["tick"]["net_usd"] > b["tick"]["net_usd"]
                          and o["A"]["DD"] <= 1.2 * b["A"]["DD"] and o["B"]["DD"] <= 1.2 * b["B"]["DD"]
                          and o["tick"]["max_dd_usd"] <= 1.2 * b["tick"]["max_dd_usd"] and o["perm_p"] < 0.10)
        print(f"{name}: removed {k} trades | permutation p={o['perm_p']:.3f} | ADOPT {o['adopt']}", flush=True)
    for v in out.values(): v.pop("trades", None)
    json.dump(dict(g3_threshold=thr, half_split=str(mid), results=out), open(os.path.join(HERE, "breakout_candle_filters.json"), "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
