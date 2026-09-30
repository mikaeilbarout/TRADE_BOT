"""Stricter rules for SHORT trades only (user request 2026-09-30).

Why it might help: gold rose ~2.4x from 2022 to 2026; shorts were the weaker side, and May 2026
(23 of 25 trades short in a range) was the worst month.
Live bot = N20, EMA30, trend 0.3%, 2 ATR (min $8), RR 4, S&P filter, risk 0.3%. Longs never change.
Four short-only variants, each with ONE setting fixed before running (no search):
  S1 stronger trend: a short needs price >= 0.6% below the H4 EMA30 (longs keep 0.3%)
  S2 bigger breakdown: a short needs a close below the 40-bar low (longs keep 20)
  S3 daily confirmation: a short also needs yesterday's close below the daily EMA50
  S4 smaller size: shorts at half risk (0.15%) -- same trades, results scaled
Adopted only if, vs the live bot, it improves the LAST 30% of the 4 years (R, DD <= 1.2x) AND the
6-month real ticks (net USD). Four variants -> one passing by luck is possible.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0)
BAR = pd.Timedelta(minutes=15)


def context(m15):
    """per M15 bar (open time): H4 EMA30 distance (last closed H4), 40-bar low (before the bar), daily EMA50 side (yesterday)"""
    s = m15.set_index("ts")
    h4 = s.close.resample("4h").last().dropna().to_frame("close")
    h4["dist"] = (h4.close - h4.close.ewm(span=30, adjust=False).mean()) / h4.close * 100
    h4 = h4.assign(known=h4.index + pd.Timedelta(hours=4)).reset_index(drop=True)
    d1 = s.close.resample("1D").last().dropna().to_frame("close")
    d1["below_ema50"] = d1.close < d1.close.ewm(span=50, adjust=False).mean()
    d1 = d1.assign(known=d1.index + pd.Timedelta(days=1)).reset_index(drop=True)
    c = pd.DataFrame({"ts": m15.ts, "known": m15.ts + BAR, "close": m15.close.values,
                      "low40": m15.low.rolling(40).min().shift(1).values})
    c = pd.merge_asof(c.sort_values("known"), h4[["known", "dist"]], on="known")
    c = pd.merge_asof(c, d1[["known", "below_ema50"]], on="known")
    return c.set_index("ts")


def make(kind, ctx):
    def f(ts, side):
        if not spx.spx_filter(ts, side):
            return False
        if side == "long":
            return True
        try:
            r = ctx.loc[ts]
        except KeyError:
            return True
        if kind == "S1":
            return bool(np.isnan(r.dist) or r.dist <= -0.6)
        if kind == "S2":
            return bool(np.isnan(r.low40) or r.close < r.low40)
        if kind == "S3":
            return bool(pd.isna(r.below_ema50) or r.below_ema50)
        return True
    return f


def run_all(low, test, boundary, frame, ctb, start, end, kind):
    cf, ct, ck = context(low), context(test), context(frame.rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]])
    full = rd.run(low, dict(LIVE, entry_filter=make(kind, cf)))
    last30 = rd.run(test, dict(LIVE, entry_filter=make(kind, ct)), start=boundary)
    ctb.D_ENTRY_FILTER = make(kind, ck)
    tk = ctb.donchian(frame, start, end)
    if kind == "S4":                                   # half size on shorts, same trades
        for t in (full, last30):
            t.loc[t.side == "short", "R"] *= 0.5
        tk.loc[tk.direction == "short", "usd"] *= 0.5
    return full, last30, tk


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    test = low.iloc[split - 6000:].reset_index(drop=True)
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb, pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.RISK_PCT = 0.003
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 20, 30, 0.3, 2.0, 4.0, 8.0
    res = {}
    for kind, name in (("LIVE", "live"), ("S1", "S1 short needs 0.6% trend"), ("S2", "S2 short needs 40-bar low"),
                       ("S3", "S3 short needs daily EMA50 down"), ("S4", "S4 shorts at half risk")):
        full, last30, tk = run_all(low, test, boundary, frame, ctb, start, end, kind)
        e = pd.to_datetime(full.entry_time)
        side = {sd: dict(n=int((full.side == sd).sum()), R=round(float(full.R[full.side == sd].sum()), 1),
                         win=round(float((full.R[full.side == sd] > 0).mean()), 3)) for sd in ("long", "short")}
        tside = {sd: dict(n=int((tk.direction == sd).sum()), usd=round(float(tk.usd[tk.direction == sd].sum()), 2)) for sd in ("long", "short")}
        res[name] = dict(first70=rd.summary(full[e < boundary]), last30=rd.summary(last30), full=rd.summary(full),
                         tick=ctb.stats(tk), by_side_4y=side, by_side_tick=tside)
        a, b, fu, k = res[name]["first70"], res[name]["last30"], res[name]["full"], res[name]["tick"]
        print(f"{name:32} | first70 R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} R={b['R']:6.1f} DD={b['DD']:5.1f} | 4y R={fu['R']:6.1f} DD={fu['DD']:5.1f} | "
              f"TICK n={k['trades']} win={k['win_rate']} ${k['net_usd']} DD {k['max_dd_pct']}% | 4y long {side['long']} short {side['short']} | tick {tside}", flush=True)
    base = res["live"]
    for name in list(res)[1:]:
        r = res[name]
        r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["tick"]["net_usd"] > base["tick"]["net_usd"])
        print(f"ADOPT {name}: {r['adopt']}")
    json.dump(dict(split=str(boundary), results=res), open(os.path.join(HERE, "short_strict.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
