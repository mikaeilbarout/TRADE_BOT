"""F2 (no entry against the last 24h H4 move) + reward/risk re-optimised on top of it.

Protocol (fixed before running): RR grid 1.5..6; the RR is chosen on half A of the pre-tick
data only (2022-06..2024-05, neighbour-smoothed R/max(5,DD)); then half B, the full 4 years
and the 6-month ticks are reported for every RR, next to the live setup (no F2, RR 3).
F2+RR is adopted only if the chosen RR beats the live setup on half A, half B AND ticks.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm          # stub MT5, paths, context(), make_filters()
rd = cm.rd

RRS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    pre = low[low.ts < cm.TICK_START]; mid = pre.ts.iloc[len(pre) // 2]
    f2 = cm.make_filters(cm.context(low))["F2_h4_momentum"]
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    f2_tick = cm.make_filters(cm.context(frame.rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]))["F2_h4_momentum"]
    ctb.D_ATR_MULT, ctb.D_MIN_STOP = 2.0, 8.0

    def evaluate(rr, filt, filt_tick):
        t = rd.run(low, dict(cm.BASE, reward_risk_ratio=rr, entry_filter=filt)); e = pd.to_datetime(t.entry_time)
        ctb.D_RR, ctb.D_ENTRY_FILTER = rr, filt_tick
        return dict(A=rd.summary(t[e < mid]), B=rd.summary(t[(e >= mid) & (e < cm.TICK_START)]), full=rd.summary(t),
                    tick=ctb.stats(ctb.donchian(frame, start, end)))

    live = evaluate(3.0, None, None)
    rows = {rr: evaluate(rr, f2, f2_tick) for rr in RRS}
    score = [rows[rr]["A"]["R"] / max(5., rows[rr]["A"]["DD"]) for rr in RRS]
    smooth = [float(np.mean(score[max(0, i - 1):i + 2])) for i in range(len(RRS))]
    pick = RRS[int(np.argmax(smooth))]

    def line(name, r):
        A, B, F, T = r["A"], r["B"], r["full"], r["tick"]
        return (f"{name:14} | A n={A['n']:3d} R={A['R']:6.1f} DD={A['DD']:5.1f} | B n={B['n']:3d} win={B['win']:.2f} R={B['R']:6.1f} DD={B['DD']:5.1f} | "
                f"4y R={F['R']:6.1f} DD={F['DD']:5.1f} | TICK n={T['trades']} win={T['win_rate']} ${T['net_usd']} DD {T['max_dd_pct']}%")
    print(line("LIVE (RR 3)", live))
    for rr, sm in zip(RRS, smooth):
        print(line(f"F2 + RR {rr}", rows[rr]) + f" | smoothed A score {sm:.2f}")
    p = rows[pick]
    adopt = bool(p["A"]["R"] > live["A"]["R"] and p["B"]["R"] > live["B"]["R"] and p["tick"]["net_usd"] > live["tick"]["net_usd"])
    print(f"PICK (half A only): RR {pick} | beats live on A, B and ticks: {adopt}")
    json.dump(dict(half_split=str(mid), live=live, f2={str(k): v for k, v in rows.items()}, smooth=dict(zip(map(str, RRS), smooth)),
                   pick=pick, adopt=adopt), open(os.path.join(HERE, "f2_rr.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
