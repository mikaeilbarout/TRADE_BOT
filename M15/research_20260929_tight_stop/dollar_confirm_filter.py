"""Idea: a gold breakout the dollar does NOT confirm is more likely fake.

D3: skip a gold LONG when, over the 4 h before the signal, at least 5 of the 7 dollar pairs moved in
    the DOLLAR's favour (gold breaking up while the dollar strengthens); skip a SHORT when at least 5
    of 7 moved against the dollar.
D4: the same over the last 1 h (the breakout itself).
Both settings fixed before running (no search). Baseline = the live bot (N20, EMA30, 0.3%, 2 ATR
min $8, RR 4, S&P filter). Adopted only if it improves the LAST 30% of 4 years (R, DD <= 1.2x) AND
the 6-month real ticks (net USD). Two variants tested -> one passing by luck is possible.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import dollar_index_filter as dx
cm, spx, rd = dx.cm, dx.spx, dx.rd
N = len(dx.PAIRS)


def make_confirm(hours, need=5):
    def f(ts, side):
        _, ups = dx.usd_state(ts, hours)
        if np.isnan(ups):
            return True
        against = ups if side == "long" else N - ups     # pairs moving the way that hurts this gold trade
        return against < need
    return f


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
    variants = {"live (S&P filter)": spx.spx_filter,
                "live + D3 dollar 4h (>=5/7)": dx.both(spx.spx_filter, make_confirm(4)),
                "live + D4 dollar 1h (>=5/7)": dx.both(spx.spx_filter, make_confirm(1))}
    res = {}
    for name, filt in variants.items():
        full = rd.run(low, dict(dx.LIVE, entry_filter=filt)); e = pd.to_datetime(full.entry_time)
        last30 = rd.run(test, dict(dx.LIVE, entry_filter=filt), start=boundary)
        ctb.D_ENTRY_FILTER = filt
        tk = ctb.stats(ctb.donchian(frame, start, end))
        res[name] = dict(first70=rd.summary(full[e < boundary]), last30=rd.summary(last30), full=rd.summary(full), tick=tk,
                         years={int(k): round(float(v), 1) for k, v in full.R.groupby(pd.to_datetime(full.exit_time).dt.year).sum().items()})
        a, b, f = res[name]["first70"], res[name]["last30"], res[name]["full"]
        print(f"{name:30} | first70 n={a['n']} win={a['win']:.2f} R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} win={b['win']:.2f} R={b['R']:6.1f} DD={b['DD']:5.1f} | "
              f"4y R={f['R']:6.1f} DD={f['DD']:5.1f} | TICK n={tk['trades']} win={tk['win_rate']} ${tk['net_usd']} DD {tk['max_dd_pct']}% | {res[name]['years']}", flush=True)
    base = res["live (S&P filter)"]
    for name in list(variants)[1:]:
        r = res[name]
        r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["tick"]["net_usd"] > base["tick"]["net_usd"])
        print(f"ADOPT {name}: {r['adopt']}")
    json.dump(dict(split=str(boundary), results=res), open(os.path.join(HERE, "dollar_confirm_filter.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
