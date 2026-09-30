"""Johansen cointegration: gold vs the US-dollar index and vs silver (user request 2026-09-30).

Part 1 -- is there a long-run equilibrium at all?  Johansen trace test (statsmodels coint_johansen,
constant term, 1 lagged difference) on daily log closes, for [gold, dollar], [gold, silver] and
[gold, dollar, silver]: over the whole period and in rolling 250-trading-day windows (step 20 days).
Engle-Granger (statsmodels coint) as a second opinion.

Part 2 -- a "distance from equilibrium" filter, only from PAST data: each day the cointegrating vector
is re-estimated on the previous 250 days (gold coefficient = 1), and z = how many standard deviations
the spread sits from its mean at yesterday's close. Gold "rich" (z >= +2) blocks a LONG, gold "cheap"
(z <= -2) blocks a SHORT. Threshold 2 fixed before running. Two variants: J1 gold-dollar, J2 gold-silver.
Baseline = the live bot (N20, EMA30, 0.3%, 2 ATR min $8, RR 4, S&P filter). Adopted only if it improves
the LAST 30% of the 4 years (R, DD <= 1.2x) AND the 6-month real ticks (net USD).
"""
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint
import dollar_index_filter as dx
cm, spx, rd = dx.cm, dx.spx, dx.rd

WINDOW, STEP, Z_LIMIT = 250, 20, 2.0
BAR = pd.Timedelta(minutes=15)


def daily_series():
    gold = pd.read_parquet(rd.DATA).set_index("bar_time").close.resample("1D").last().dropna()
    usd = dx.USD_LOG.resample("1D").last().dropna() / 100          # log dollar index
    im = pd.read_parquet(os.path.join(cm.COMB, "trade_dataset_20260929", "intermarket_h1.parquet"))
    silver = im.XAGUSD.dropna().resample("1D").last().dropna()
    df = pd.concat({"gold": np.log(gold), "usd": usd, "silver": np.log(silver)}, axis=1).dropna()
    return df


def trace_result(x):
    """(rank at 95%, trace stat r=0, 95% critical r=0, normalised first eigenvector)"""
    j = coint_johansen(x, det_order=0, k_ar_diff=1)
    rank = int(sum(j.lr1[i] > j.cvt[i, 1] for i in range(x.shape[1]) if all(j.lr1[k] > j.cvt[k, 1] for k in range(i + 1))))
    vec = j.evec[:, 0] / j.evec[0, 0]
    return rank, float(j.lr1[0]), float(j.cvt[0, 1]), vec


def part1(df):
    out = {}
    for name, cols in (("gold-dollar", ["gold", "usd"]), ("gold-silver", ["gold", "silver"]), ("gold-dollar-silver", ["gold", "usd", "silver"])):
        x = df[cols].to_numpy()
        rank, stat, crit, vec = trace_result(x)
        eg_p = float(coint(df[cols[0]], df[cols[1:]]).__getitem__(1)) if len(cols) >= 2 else None
        ranks = []
        for s in range(0, len(x) - WINDOW + 1, STEP):
            ranks.append(trace_result(x[s:s + WINDOW])[0])
        share = float(np.mean([r >= 1 for r in ranks]))
        out[name] = dict(days=len(x), rank_full=rank, trace_stat=round(stat, 2), crit95=round(crit, 2), vector=[round(float(v), 3) for v in vec],
                         engle_granger_p=round(eg_p, 4), windows=len(ranks), share_windows_cointegrated=round(share, 3))
        print(f"{name:19} | full {len(x)} days: rank {rank} (trace {stat:.2f} vs 95% {crit:.2f}), vector {np.round(vec, 3)}, "
              f"Engle-Granger p={eg_p:.3f} | rolling {WINDOW}d windows: cointegrated in {share:.0%} of {len(ranks)}", flush=True)
    return out


def rolling_z(df, cols):
    """z of the spread at each day's close, estimated only on the WINDOW days up to and including it"""
    x = df[cols].to_numpy(); z = np.full(len(x), np.nan)
    for i in range(WINDOW - 1, len(x)):
        w = x[i - WINDOW + 1:i + 1]
        vec = coint_johansen(w, det_order=0, k_ar_diff=1).evec[:, 0]
        vec = vec / vec[0]
        spread = w @ vec
        sd = spread.std()
        z[i] = (spread[-1] - spread.mean()) / sd if sd > 0 else np.nan
    return pd.Series(z, index=df.index)


def make_filter(z):
    days = z.index

    def f(ts, side):
        dec = pd.Timestamp(ts) + BAR
        k = days.searchsorted(dec.normalize(), side="left") - 1          # yesterday's completed day
        if k < 0 or np.isnan(z.iloc[k]):
            return True
        return not ((side == "long" and z.iloc[k] >= Z_LIMIT) or (side == "short" and z.iloc[k] <= -Z_LIMIT))
    return f


def part2(df):
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    test = low.iloc[split - 6000:].reset_index(drop=True)
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 20, 30, 0.3, 2.0, 4.0, 8.0
    zs = {"J1 gold-dollar": rolling_z(df, ["gold", "usd"]), "J2 gold-silver": rolling_z(df, ["gold", "silver"])}
    for name, z in zs.items():
        v = z.dropna()
        print(f"{name}: z from {v.index[0].date()}, |z|>=2 on {(v.abs() >= 2).mean():.0%} of days", flush=True)
    variants = {"live (S&P filter)": spx.spx_filter}
    for name, z in zs.items():
        variants[f"live + {name}"] = dx.both(spx.spx_filter, make_filter(z))
    res = {}
    for name, filt in variants.items():
        full = rd.run(low, dict(dx.LIVE, entry_filter=filt)); e = pd.to_datetime(full.entry_time)
        last30 = rd.run(test, dict(dx.LIVE, entry_filter=filt), start=boundary)
        ctb.D_ENTRY_FILTER = filt
        tk = ctb.stats(ctb.donchian(frame, start, end))
        res[name] = dict(first70=rd.summary(full[e < boundary]), last30=rd.summary(last30), full=rd.summary(full), tick=tk)
        a, b, f = res[name]["first70"], res[name]["last30"], res[name]["full"]
        print(f"{name:26} | first70 n={a['n']} R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} win={b['win']:.2f} R={b['R']:6.1f} DD={b['DD']:5.1f} | "
              f"4y R={f['R']:6.1f} DD={f['DD']:5.1f} | TICK n={tk['trades']} win={tk['win_rate']} ${tk['net_usd']} DD {tk['max_dd_pct']}%", flush=True)
    base = res["live (S&P filter)"]
    for name in list(variants)[1:]:
        r = res[name]
        r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["tick"]["net_usd"] > base["tick"]["net_usd"])
        print(f"ADOPT {name}: {r['adopt']}")
    return res


def main():
    df = daily_series()
    print(f"daily data {df.index[0].date()} .. {df.index[-1].date()} ({len(df)} days)\n")
    rep = dict(part1=part1(df))
    print()
    rep["part2"] = part2(df)
    json.dump(rep, open(os.path.join(HERE, "johansen.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
