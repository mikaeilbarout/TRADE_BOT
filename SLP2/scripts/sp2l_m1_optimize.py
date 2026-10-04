"""Full parameter optimisation of SP2L on 1-minute bars (6 months of real-tick M1, 2026-03-23 .. 09-22). User request 2026-10-01.

Grid (18,000 cells): spike {1.25,1.5,2,3} x gap {0,.3,.5,1,2} x max stop {2,3,5,8,12} x RR {1,1.5,2,3,5,8} x EMA {off,20,60,150,300}
x opposite bars {1,2,4} x hours {all, 10-21 server clock (the lower-spread hours)}. NET = real spread per bar + 2 pts slippage + $7/lot.
Protocol (fixed before running):
  train = first 70% of the bars, test = last 30% (with a warm-up before it, evaluation from the boundary).
  train score = net R / max(5, max drawdown R) - 0.5 per losing third of the trades; needs >= 60 trades, else the cell is invalid.
  smoothed score = mean over the cell and its +-1 grid neighbours (invalid neighbours count -1).
  the cell with the best smoothed score is the ONE pick; it is then judged once on the test.
Adopted only if on the test: net R > 0, profit factor >= 1.15, >= 40 trades, net R > 0 with 5x slippage, AND the train ranking
predicts the test (rank correlation of net R per trade, train vs test, over cells with >= 30 trades in both, > 0.2).
"""
import sys, os, json, itertools, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

GRID = dict(spike=[1.25, 1.5, 2., 3.], gap=[0., .3, .5, 1., 2.], ms=[2., 3., 5., 8., 12.], rr=[1., 1.5, 2., 3., 5., 8.],
            ema=[0, 20, 60, 150, 300], opp=[1, 2, 4], hours=[None, (10, 21)])
KEYS = list(GRID)
WARM = 3500
_F = None


def init():
    global _F, _SPLIT
    from scripts.sp2l_m15_backtest import simulate  # noqa
    f = pd.read_parquet(os.path.join(ROOT, "data", "XAUUSD_M1_6m.parquet")); f = f[f.tick_volume > 0].reset_index(drop=True)
    _SPLIT = int(len(f) * .7); _F = f


def kwargs(c, slip=2):
    spike, gap, ms, rr, ema, opp, hours = c
    return dict(rr=rr, p_gap_price=gap, max_sl_dist=ms, spike_size=spike, use_ema_filter=ema > 0, ema_period=ema or 20, max_opposite_moves=opp,
                use_trend_filter=True, bar_minutes=1, session_hours=hours, slippage_points=slip)


def stats(r):
    r = np.asarray(r, float)
    if len(r) == 0: return dict(n=0, R=0., PF=0., win=0., dd=0., thirds=[0., 0., 0.])
    c = np.cumsum(r); loss = -r[r < 0].sum(); th = [float(x.sum()) for x in np.array_split(r, 3)]
    return dict(n=len(r), R=float(c[-1]), PF=float(r[r > 0].sum() / loss) if loss > 0 else 9.99, win=float((r > 0).mean()),
                dd=float(np.max(np.maximum.accumulate(np.maximum(c, 0)) - c)), thirds=th)


def run_cell(idx):
    from scripts.sp2l_m15_backtest import simulate
    c = tuple(GRID[k][i] for k, i in zip(KEYS, idx))
    tr = simulate(_F.iloc[:_SPLIT].reset_index(drop=True), **kwargs(c))
    te_frame = _F.iloc[max(0, _SPLIT - WARM):].reset_index(drop=True)
    te = simulate(te_frame, evaluation_start=_F.bar_time.iloc[_SPLIT], **kwargs(c))
    return idx, stats(tr.r_multiple), stats(te.r_multiple)


def score(s):
    if s["n"] < 60: return None
    return s["R"] / max(5., s["dd"]) - 0.5 * sum(x < 0 for x in s["thirds"])


def main():
    t0 = time.time()
    cells = list(itertools.product(*[range(len(GRID[k])) for k in KEYS]))
    print(f"{len(cells)} cells, 8 workers", flush=True)
    res = {}
    with ProcessPoolExecutor(8, initializer=init) as ex:
        for n, (idx, a, b) in enumerate(ex.map(run_cell, cells, chunksize=40)):
            res[idx] = (a, b)
            if (n + 1) % 2000 == 0: print(f"  {n + 1} done, {time.time() - t0:.0f}s", flush=True)
    raw = {i: score(v[0]) for i, v in res.items()}
    sm = {}
    for idx, s in raw.items():
        if s is None: continue
        g = [s]
        for j, k in enumerate(KEYS):
            for step in (-1, 1):
                m = idx[j] + step
                if 0 <= m < len(GRID[k]):
                    nb = list(idx); nb[j] = m; v = raw[tuple(nb)]
                    g.append(-1. if v is None else v)
        sm[idx] = float(np.mean(g))
    ranked = sorted(sm, key=sm.get, reverse=True)
    f = pd.read_parquet(os.path.join(ROOT, "data", "XAUUSD_M1_6m.parquet")); f = f[f.tick_volume > 0].reset_index(drop=True)
    split = int(len(f) * .7); boundary = f.bar_time.iloc[split]
    print(f"\ntrain {f.bar_time.iloc[0]} .. {boundary} | test {boundary} .. {f.bar_time.iloc[-1]}")
    tr_pos = np.mean([v[0]["R"] > 0 for v in res.values() if v[0]["n"] >= 60]); te_pos = np.mean([v[1]["R"] > 0 for v in res.values() if v[1]["n"] >= 30])
    print(f"cells with >= 60 train trades: {sum(v[0]['n'] >= 60 for v in res.values())} | profitable on train: {tr_pos:.1%} | profitable on test (>=30 trades): {te_pos:.1%}")
    both = [(v[0]["R"] / v[0]["n"], v[1]["R"] / v[1]["n"]) for v in res.values() if v[0]["n"] >= 30 and v[1]["n"] >= 30]
    a, b = np.array(both).T
    rho = float(pd.Series(a).rank().corr(pd.Series(b).rank()))
    top = ranked[:200]; top_te = [res[i][1] for i in top if res[i][1]["n"] >= 30]
    print(f"rank correlation train vs test (net R per trade, {len(both)} cells): {rho:+.3f}")
    print(f"top-200 smoothed train cells: profitable on test {np.mean([s['R'] > 0 for s in top_te]):.1%}, mean test R per trade {np.mean([s['R'] / s['n'] for s in top_te]):+.3f}R")
    pick = ranked[0]
    from scripts.sp2l_m15_backtest import simulate
    c = tuple(GRID[k][i] for k, i in zip(KEYS, pick)); init()
    te_frame = f.iloc[max(0, split - WARM):].reset_index(drop=True)
    s5 = stats(simulate(te_frame, evaluation_start=boundary, **kwargs(c, slip=10)).r_multiple)
    a_, b_ = res[pick]
    print("\nPICK (best smoothed train score):", dict(zip(KEYS, c)))
    print(f"  train: n={a_['n']} win={a_['win']:.2f} R={a_['R']:+.1f} PF={a_['PF']:.2f} DD={a_['dd']:.1f} smoothed={sm[pick]:.2f}")
    print(f"  TEST : n={b_['n']} win={b_['win']:.2f} R={b_['R']:+.1f} PF={b_['PF']:.2f} DD={b_['dd']:.1f} | 5x slippage R={s5['R']:+.1f}")
    adopt = bool(b_["R"] > 0 and b_["PF"] >= 1.15 and b_["n"] >= 40 and s5["R"] > 0 and rho > 0.2)
    print(f"ADOPT: {adopt}")
    for i in ranked[1:6]:
        c2 = dict(zip(KEYS, (GRID[k][j] for k, j in zip(KEYS, i)))); a2, b2 = res[i]
        print(f"  #{ranked.index(i) + 1} {c2} | train R={a2['R']:+.1f} n={a2['n']} | test R={b2['R']:+.1f} n={b2['n']}")
    out = os.path.join(ROOT, "data", "sp2l_m1_20261001"); os.makedirs(out, exist_ok=True)
    json.dump(dict(rank_corr=rho, train_profitable=tr_pos, test_profitable=te_pos, pick=dict(zip(KEYS, c)), train=a_, test=b_, slip5x=s5, adopt=adopt,
                   top20=[dict(params=dict(zip(KEYS, (GRID[k][j] for k, j in zip(KEYS, i)))), smoothed=sm[i], train=res[i][0], test=res[i][1]) for i in ranked[:20]]),
              open(os.path.join(out, "optimize_report.json"), "w"), indent=1, default=str)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
