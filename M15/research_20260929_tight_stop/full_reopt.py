"""Re-optimise all Donchian parameters, with and without F2 (no entry against the last 24h H4 move).

Grid: channel N {10,20,40} x trend EMA {30,50,100} x trend strength {0.3,0.5,1.0}% x
stop {1.5,2,3} ATR x RR {2,3,4,5}, min stop $8 always, each with F2 off and on (648 runs).
Protocol (fixed before running, same as research_20260924): choose on the FIRST 70% of the
4-year data only -- score R/max(5,DD) minus 0.5 per losing third, averaged with the +-1 grid
neighbours; then one look at the last 30% and at the 6-month real ticks.
The pick replaces the live setup (N10, EMA30, 0.5%, 2 ATR, RR 3, no F2) only if on the last
30% it has more R and DD <= 1.2x live, AND more net USD on ticks. No fallback to a runner-up.
"""
import sys, os, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm
rd = cm.rd

GRID = dict(n_period=[10, 20, 40], ema_trend_period=[30, 50, 100], min_trend_strength_pct=[0.3, 0.5, 1.0],
            atr_stop_multiplier=[1.5, 2.0, 3.0], reward_risk_ratio=[2.0, 3.0, 4.0, 5.0])
LIVE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=0.5, atr_stop_multiplier=2.0, reward_risk_ratio=3.0)
KEYS = list(GRID)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    train = low.iloc[:split].reset_index(drop=True)
    test = low.iloc[split - 6000:].reset_index(drop=True)
    f2_train = cm.make_filters(cm.context(train))["F2_h4_momentum"]
    f2_test = cm.make_filters(cm.context(test))["F2_h4_momentum"]
    cells = {}
    for f2 in (False, True):
        for combo in itertools.product(*GRID.values()):
            p = dict(zip(KEYS, combo), min_stop_dollars=8.0, entry_filter=f2_train if f2 else None)
            t = rd.run(train, p)
            cells[(f2,) + combo] = dict(raw=rd.score(train, t) if len(t) else None, train=rd.summary(t))
        print(f"F2={f2}: {sum(1 for k in cells if k[0] == f2)} cells done", flush=True)
    for key, c in cells.items():
        f2, combo = key[0], key[1:]
        idx = [GRID[k].index(v) for k, v in zip(KEYS, combo)]; group = [c]
        for j, k in enumerate(KEYS):
            for step in (-1, 1):
                mm = idx[j] + step
                if 0 <= mm < len(GRID[k]):
                    nb = list(combo); nb[j] = GRID[k][mm]
                    group.append(cells[(f2,) + tuple(nb)])
        c["smoothed"] = float(np.mean([g["raw"] if g["raw"] is not None else -1. for g in group])) if c["raw"] is not None else None
    ranked = sorted(((k, c) for k, c in cells.items() if c["smoothed"] is not None), key=lambda kc: kc[1]["smoothed"], reverse=True)
    best_any = ranked[0]
    best_f2 = next(kc for kc in ranked if kc[0][0])
    best_nof2 = next(kc for kc in ranked if not kc[0][0])

    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    f2_tick = cm.make_filters(cm.context(frame.rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]))["F2_h4_momentum"]

    def evaluate(f2, p):
        t = rd.run(test, dict(p, min_stop_dollars=8.0, entry_filter=f2_test if f2 else None), start=boundary)
        ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR = p["n_period"], p["ema_trend_period"], p["min_trend_strength_pct"], p["atr_stop_multiplier"], p["reward_risk_ratio"]
        ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 8.0, (f2_tick if f2 else None)
        return dict(test30=rd.summary(t), tick=ctb.stats(ctb.donchian(frame, start, end)))

    res = {}
    for label, (key, c) in (("live", ((False,) + tuple(LIVE[k] for k in KEYS), cells[(False,) + tuple(LIVE[k] for k in KEYS)])),
                            ("best_overall", best_any), ("best_with_F2", best_f2), ("best_without_F2", best_nof2)):
        p = dict(zip(KEYS, key[1:])); ev = evaluate(key[0], p)
        res[label] = dict(F2=key[0], params=p, train70=c["train"], smoothed=c["smoothed"], **ev)
        T, S, K = c["train"], ev["test30"], ev["tick"]
        print(f"{label:16} F2={str(key[0]):5} {p} | first70 R={T['R']:6.1f} DD={T['DD']:5.1f} sm={c['smoothed']:.2f} | "
              f"last30 n={S['n']} win={S['win']:.2f} R={S['R']:6.1f} DD={S['DD']:5.1f} | TICK n={K['trades']} win={K['win_rate']} ${K['net_usd']} DD {K['max_dd_pct']}%", flush=True)
    live, pick = res["live"], res["best_overall"]
    promote = bool(pick["params"] != LIVE or pick["F2"]) and pick["test30"]["R"] > live["test30"]["R"] and \
        pick["test30"]["DD"] <= 1.2 * live["test30"]["DD"] and pick["tick"]["net_usd"] > live["tick"]["net_usd"]
    share = np.mean([c["train"]["R"] > 0 for c in cells.values()])
    print(f"PROMOTE best_overall: {bool(promote)} | share of all 648 cells profitable on the first 70%: {share:.0%}")
    json.dump(dict(split=str(boundary), results=res, promote=bool(promote), share_profitable_first70=float(share),
                   top10=[dict(F2=k[0], params=dict(zip(KEYS, k[1:])), smoothed=c["smoothed"], train=c["train"]) for k, c in ranked[:10]]),
              open(os.path.join(HERE, "full_reopt.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
