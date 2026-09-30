"""Filter from new_info_test.py: skip a gold trade when the S&P 500 moved >= 0.9% IN THE SAME
direction over the last 5 trading days (gold behaving like a risk asset, not a safe haven).

The column passed the stability test (p 0.035, same direction in all 4 time blocks) but was 1 of
19 tested, so it must prove itself here. Threshold 0.9% = the 67th percentile of that value on the
FIRST 70% of the trades (fixed before running). Live settings: N20, EMA30, 0.3%, 2 ATR (min $8), RR 4.
Adopted only if, vs no filter, it improves the LAST 30% of the 4 years (R, with DD <= 1.2x) AND the
6-month real ticks (net USD). S&P history starts 2022-10; before that the filter lets everything through.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm
rd = cm.rd

NEW = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0)
THRESHOLD = 0.9
SPX = pd.read_parquet(os.path.join(cm.COMB, "trade_dataset_20260929", "intermarket_h1.parquet")).SPX500.dropna()
BAR = pd.Timedelta(minutes=15)


def spx_5d_pct(signal_bar_ts):
    dec = pd.Timestamp(signal_bar_ts) + BAR
    last = SPX.index.searchsorted(dec - pd.Timedelta(hours=1), side="right") - 1
    if last < 0:
        return np.nan
    past = SPX.index.searchsorted(SPX.index[last] - pd.Timedelta(hours=24 * 7), side="right") - 1
    if past < 0 or past >= last:
        return np.nan
    return (SPX.values[last] / SPX.values[past] - 1) * 100


def spx_filter(ts, side):
    v = spx_5d_pct(ts)
    if np.isnan(v):
        return True
    return v * (1 if side == "long" else -1) < THRESHOLD


def main():
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
    out = {}
    for name, filt in (("no filter (live)", None), ("S&P 5d filter", spx_filter)):
        full = rd.run(low, dict(NEW, entry_filter=filt)); e = pd.to_datetime(full.entry_time)
        last30 = rd.run(test, dict(NEW, entry_filter=filt), start=boundary)
        ctb.D_ENTRY_FILTER = filt
        tk = ctb.stats(ctb.donchian(frame, start, end))
        out[name] = dict(first70=rd.summary(full[e < boundary]), last30=rd.summary(last30), full=rd.summary(full), tick=tk,
                         years={int(k): round(float(v), 1) for k, v in full.R.groupby(pd.to_datetime(full.exit_time).dt.year).sum().items()})
        a, b, f = out[name]["first70"], out[name]["last30"], out[name]["full"]
        print(f"{name:18} | first70 n={a['n']} win={a['win']:.2f} R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} win={b['win']:.2f} R={b['R']:6.1f} DD={b['DD']:5.1f} | "
              f"4y R={f['R']:6.1f} DD={f['DD']:5.1f} | TICK n={tk['trades']} win={tk['win_rate']} ${tk['net_usd']} DD {tk['max_dd_pct']}% | {out[name]['years']}", flush=True)
    base, fil = out["no filter (live)"], out["S&P 5d filter"]
    adopt = bool(fil["last30"]["R"] > base["last30"]["R"] and fil["last30"]["DD"] <= 1.2 * base["last30"]["DD"]
                 and fil["tick"]["net_usd"] > base["tick"]["net_usd"])
    print(f"ADOPT: {adopt}")
    json.dump(dict(threshold=THRESHOLD, split=str(boundary), results=out, adopt=adopt), open(os.path.join(HERE, "spx_filter.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
