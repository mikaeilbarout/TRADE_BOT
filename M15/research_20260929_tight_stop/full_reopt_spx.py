"""Re-optimise all Donchian parameters WITH the S&P 500 filter always on (user request 2026-09-30).

Grid: channel N {10,20,40} x trend EMA {30,50,100} x trend strength {0.3,0.5,1.0}% x stop {1.5,2,3} ATR
x RR {2,3,4,5} x S&P threshold {0.6,0.9,1.2}% (min stop $8 always) = 972 runs.
Protocol (fixed before running, same as full_reopt.py): choose on the FIRST 70% of the 4 years only --
score R/max(5,DD) minus 0.5 per losing third, averaged with the +-1 grid neighbours -- then one look at
the last 30% and the 6-month real ticks. The pick replaces the live bot (N20, EMA30, 0.3%, 2 ATR, RR 4,
S&P 0.9%) only if on the last 30% it has more R with DD <= 1.2x live, AND more net USD on ticks.
No fallback to a runner-up.
"""
import sys, os, json, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm
import spx_filter as spx
rd = cm.rd

GRID = dict(n_period=[10, 20, 40], ema_trend_period=[30, 50, 100], min_trend_strength_pct=[0.3, 0.5, 1.0],
            atr_stop_multiplier=[1.5, 2.0, 3.0], reward_risk_ratio=[2.0, 3.0, 4.0, 5.0], spx_threshold=[0.6, 0.9, 1.2])
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, spx_threshold=0.9)
KEYS = list(GRID)


def spx_filter_at(threshold):
    def f(ts, side):
        v = spx.spx_5d_pct(ts)
        return True if np.isnan(v) else v * (1 if side == "long" else -1) < threshold
    return f


FILTERS = {t: spx_filter_at(t) for t in GRID["spx_threshold"]}


def params(p):
    q = {k: v for k, v in p.items() if k != "spx_threshold"}
    return dict(q, min_stop_dollars=8.0, entry_filter=FILTERS[p["spx_threshold"]])


def main():
    t0 = time.time()
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    train = low.iloc[:split].reset_index(drop=True)
    test = low.iloc[split - 6000:].reset_index(drop=True)
    cells = {}
    for i, combo in enumerate(itertools.product(*GRID.values())):
        t = rd.run(train, params(dict(zip(KEYS, combo))))
        cells[combo] = dict(raw=rd.score(train, t) if len(t) else None, train=rd.summary(t))
        if (i + 1) % 100 == 0:
            print(f"{i + 1} cells, {time.time() - t0:.0f}s", flush=True)
    for combo, c in cells.items():
        idx = [GRID[k].index(v) for k, v in zip(KEYS, combo)]; group = [c]
        for j, k in enumerate(KEYS):
            for step in (-1, 1):
                m = idx[j] + step
                if 0 <= m < len(GRID[k]):
                    nb = list(combo); nb[j] = GRID[k][m]
                    group.append(cells[tuple(nb)])
        c["smoothed"] = float(np.mean([g["raw"] if g["raw"] is not None else -1. for g in group])) if c["raw"] is not None else None
    ranked = sorted(((k, c) for k, c in cells.items() if c["smoothed"] is not None), key=lambda kc: kc[1]["smoothed"], reverse=True)

    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)

    def evaluate(p):
        full = rd.run(low, params(p)); e = pd.to_datetime(full.entry_time)
        t30 = rd.run(test, params(p), start=boundary)
        ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR = p["n_period"], p["ema_trend_period"], p["min_trend_strength_pct"], p["atr_stop_multiplier"], p["reward_risk_ratio"]
        ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 8.0, FILTERS[p["spx_threshold"]]
        return dict(last30=rd.summary(t30), full=rd.summary(full), tick=ctb.stats(ctb.donchian(frame, start, end)),
                    years={int(k): round(float(v), 1) for k, v in full.R.groupby(pd.to_datetime(full.exit_time).dt.year).sum().items()})

    live_key = tuple(LIVE[k] for k in KEYS)
    res = {}
    for label, key in [("live", live_key)] + [(f"rank {i + 1}", k) for i, (k, _) in enumerate(ranked[:3])]:
        p = dict(zip(KEYS, key)); c = cells[key]; ev = evaluate(p)
        res[label] = dict(params=p, first70=c["train"], smoothed=c["smoothed"], **ev)
        T, S, K, F = c["train"], ev["last30"], ev["tick"], ev["full"]
        print(f"{label:7} {p} | first70 R={T['R']:6.1f} DD={T['DD']:5.1f} sm={c['smoothed']:.2f} | last30 n={S['n']} win={S['win']:.2f} R={S['R']:6.1f} DD={S['DD']:5.1f} | "
              f"4y R={F['R']:6.1f} DD={F['DD']:5.1f} | TICK n={K['trades']} win={K['win_rate']} ${K['net_usd']} DD {K['max_dd_pct']}% | {ev['years']}", flush=True)
    live, pick = res["live"], res["rank 1"]
    promote = bool(pick["params"] != LIVE and pick["last30"]["R"] > live["last30"]["R"] and pick["last30"]["DD"] <= 1.2 * live["last30"]["DD"]
                   and pick["tick"]["net_usd"] > live["tick"]["net_usd"])
    share = float(np.mean([c["train"]["R"] > 0 for c in cells.values()]))
    live_rank = next(i for i, (k, _) in enumerate(ranked) if k == live_key) + 1
    print(f"PROMOTE rank 1: {promote} | live setup ranks {live_rank} of {len(ranked)} on the first 70% | {share:.0%} of cells profitable there | {time.time() - t0:.0f}s")
    json.dump(dict(split=str(boundary), results=res, promote=promote, live_rank=live_rank, share_profitable_first70=share,
                   top20=[dict(params=dict(zip(KEYS, k)), smoothed=c["smoothed"], train=c["train"]) for k, c in ranked[:20]]),
              open(os.path.join(HERE, "full_reopt_spx.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
