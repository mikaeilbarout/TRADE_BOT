"""Stop-loss size (x ATR14) for the CURRENT live Donchian bot (user request 2026-09-30).

Everything else as live: N20, EMA30, trend 0.3%, RR 4, min stop $8, S&P filter, risk 0.3%.
Protocol (fixed before running): grid 1.0..4.0 x ATR; choose on the FIRST 70% of the 4 years only
(R/max(5,DD) minus 0.5 per losing third, averaged with the +-1 neighbours); then one look at the
last 30% and the 6-month real ticks. The pick replaces 2.0 only if it beats 2.0 on the last 30%
(R, with DD <= 1.2x) AND on ticks (net USD).
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd

GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, reward_risk_ratio=4.0, min_stop_dollars=8.0, entry_filter=spx.spx_filter)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    train = low.iloc[:split].reset_index(drop=True); test = low.iloc[split - 6000:].reset_index(drop=True)
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb, pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.RISK_PCT = 0.003
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_RR, ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 20, 30, 0.3, 4.0, 8.0, spx.spx_filter
    atr = frame.set_index("bar_time").atr if "atr" in frame else None
    rows = {}
    for m in GRID:
        p = dict(LIVE, atr_stop_multiplier=m)
        tr, te, full = rd.run(train, p), rd.run(test, p, start=boundary), rd.run(low, p)
        ctb.D_ATR_MULT = m
        tk = ctb.donchian(frame, start, end)
        rows[m] = dict(raw=rd.score(train, tr), first70=rd.summary(tr), last30=rd.summary(te), full=rd.summary(full), tick=ctb.stats(tk),
                       years={int(k): round(float(v), 1) for k, v in full.R.groupby(pd.to_datetime(full.exit_time).dt.year).sum().items()})
    raw = [rows[m]["raw"] if rows[m]["raw"] is not None else -1. for m in GRID]
    smooth = [float(np.mean(raw[max(0, i - 1):i + 2])) for i in range(len(GRID))]
    recent_atr = float(frame.atr.tail(96 * 20).median()) if "atr" in frame else float("nan")
    print(f"median M15 ATR over the last 20 days: ${recent_atr:.2f}\n")
    for m, sm in zip(GRID, smooth):
        r = rows[m]; a, b, f, k = r["first70"], r["last30"], r["full"], r["tick"]
        print(f"stop {m:4.2f} ATR (~${max(8, m * recent_atr):5.1f} now) | first70 n={a['n']} win={a['win']:.2f} R={a['R']:6.1f} DD={a['DD']:5.1f} sm={sm:5.2f} | "
              f"last30 n={b['n']} win={b['win']:.2f} R={b['R']:6.1f} DD={b['DD']:5.1f} | 4y R={f['R']:6.1f} DD={f['DD']:5.1f} | "
              f"TICK n={k['trades']} win={k['win_rate']} ${k['net_usd']} DD {k['max_dd_pct']}% | {r['years']}", flush=True)
    pick = GRID[int(np.argmax(smooth))]
    p, l = rows[pick], rows[2.0]
    adopt = bool(pick != 2.0 and p["last30"]["R"] > l["last30"]["R"] and p["last30"]["DD"] <= 1.2 * l["last30"]["DD"] and p["tick"]["net_usd"] > l["tick"]["net_usd"])
    print(f"PICK (first 70% only): {pick} ATR | replaces 2.0: {adopt}")
    json.dump(dict(split=str(boundary), recent_atr=recent_atr, rows={str(k): v for k, v in rows.items()}, smooth=dict(zip(map(str, GRID), smooth)),
                   pick=pick, adopt=adopt), open(os.path.join(HERE, "stop_reopt_live.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
