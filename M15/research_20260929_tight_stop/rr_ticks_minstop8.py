"""Donchian M15 live settings (2x ATR stop, min stop $8): reward/risk on 6 months of ticks.

Only ~190 trades, so each RR is also shown per half of the window and smoothed with its
neighbours; a pick counts as robust only if it is positive in BOTH halves.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); COMB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2")); sys.path.insert(0, os.path.join(COMB, "M15"))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import combined_tick_backtest as ctb

RRS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]


def main():
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    mid = start + (end - start) / 2
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_ATR_MULT, ctb.D_MIN_STOP = 2.0, 8.0
    res = {}
    for rr in RRS:
        ctb.D_RR = rr
        t = ctb.donchian(frame, start, end)
        s = ctb.stats(t)
        h1, h2 = t[t.entry_time < mid].usd.sum(), t[t.entry_time >= mid].usd.sum()
        res[rr] = dict(s, first_half_usd=round(float(h1), 2), second_half_usd=round(float(h2), 2))
        print(f"RR {rr:<3} | n={s['trades']:3d} win={s['win_rate']:.3f} net=${s['net_usd']:8.2f} ({s['net_pct']:+.2f}%) PF={s['profit_factor']:.2f} "
              f"DD={s['max_dd_pct']:.2f}% | halves ${h1:8.2f} / ${h2:8.2f}", flush=True)
    net = [res[rr]["net_usd"] for rr in RRS]
    smooth = [float(np.mean(net[max(0, i - 1):i + 2])) for i in range(len(RRS))]
    for rr, sm in zip(RRS, smooth):
        res[rr]["smoothed_net_usd"] = round(sm, 2)
    best = RRS[int(np.argmax(smooth))]
    robust = res[best]["first_half_usd"] > 0 and res[best]["second_half_usd"] > 0
    print("smoothed:", dict(zip(RRS, [round(x) for x in smooth])))
    print(f"BEST (smoothed) RR {best} | positive in both halves: {robust} | window {start} .. {end}, split {mid}")
    json.dump(dict(window=[str(start), str(end)], split=str(mid), best=best, robust=robust, results={str(k): v for k, v in res.items()}),
              open(os.path.join(HERE, "rr_ticks_minstop8.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
