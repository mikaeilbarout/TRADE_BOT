"""Minimum stop distance (USD/oz) for the new live Donchian setup (N20, EMA30, 0.3%, 2 ATR, RR 4).

Protocol (fixed before running): grid 0..20; choose on the FIRST 70% of the 4 years
(R/max(5,DD) minus 0.5 per losing third, averaged with the +-1 neighbours); then one look at
the last 30% and the 6-month ticks. The pick replaces $8 only if it beats $8 on the last 30%
(R, with DD <= 1.2x) AND on ticks (net USD).
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm
rd = cm.rd

NEW = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0)
GRID = [0, 4, 6, 8, 10, 12, 15, 20]
LIVE_MIN = 8


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    train = low.iloc[:split].reset_index(drop=True); test = low.iloc[split - 6000:].reset_index(drop=True)
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR = 20, 30, 0.3, 2.0, 4.0
    rows = {}
    for x in GRID:
        tr = rd.run(train, dict(NEW, min_stop_dollars=float(x)))
        te = rd.run(test, dict(NEW, min_stop_dollars=float(x)), start=boundary)
        full = rd.run(low, dict(NEW, min_stop_dollars=float(x)))
        ctb.D_MIN_STOP = float(x)
        rows[x] = dict(raw=rd.score(train, tr), first70=rd.summary(tr), last30=rd.summary(te), full=rd.summary(full),
                       years={int(k): round(float(v), 1) for k, v in full.R.groupby(pd.to_datetime(full.exit_time).dt.year).sum().items()},
                       tick=ctb.stats(ctb.donchian(frame, start, end)))
    raw = [rows[x]["raw"] if rows[x]["raw"] is not None else -1. for x in GRID]
    smooth = [float(np.mean(raw[max(0, i - 1):i + 2])) for i in range(len(GRID))]
    for x, sm in zip(GRID, smooth):
        r = rows[x]; a, b, f, k = r["first70"], r["last30"], r["full"], r["tick"]
        print(f"min ${x:>2} | first70 n={a['n']:3d} R={a['R']:6.1f} DD={a['DD']:5.1f} sm={sm:5.2f} | last30 n={b['n']:3d} R={b['R']:6.1f} DD={b['DD']:5.1f} | "
              f"4y R={f['R']:6.1f} DD={f['DD']:5.1f} | TICK n={k['trades']} ${k['net_usd']} DD {k['max_dd_pct']}% | {r['years']}", flush=True)
    pick = GRID[int(np.argmax(smooth))]
    p, l = rows[pick], rows[LIVE_MIN]
    adopt = bool(pick != LIVE_MIN and p["last30"]["R"] > l["last30"]["R"] and p["last30"]["DD"] <= 1.2 * l["last30"]["DD"]
                 and p["tick"]["net_usd"] > l["tick"]["net_usd"])
    print(f"PICK (first 70% only): ${pick} | replaces $8: {adopt}")
    json.dump(dict(split=str(boundary), rows={str(k): v for k, v in rows.items()}, smooth=dict(zip(map(str, GRID), smooth)), pick=pick, adopt=adopt),
              open(os.path.join(HERE, "min_stop_reopt.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
