"""SP2L on M1 with a spread cap: skip a signal when the signal bar's average tick spread is above USD 0.35
(one threshold, fixed before running; same 16 settings and the same 'worth building' rule as sp2l_m1_backtest.py)."""
import sys, os, json, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scripts.sp2l_m15_backtest import simulate
CAP = 0.35
frame = pd.read_parquet(os.path.join(ROOT, "data", "XAUUSD_M1_6m.parquet")); frame = frame[frame.tick_volume > 0].reset_index(drop=True)
mid = frame.bar_time.iloc[len(frame) // 2]
print(f"share of M1 bars with spread <= ${CAP}: {(frame.avg_spread_price <= CAP).mean():.0%} | by server hour (median spread):")
print(frame.groupby(frame.bar_time.dt.hour).avg_spread_price.median().round(2).to_dict())


def factory(fr):
    sp = fr.avg_spread_price.to_numpy()
    return lambda i, d: bool(sp[i] <= CAP)


def summ(t):
    if len(t) == 0: return dict(n=0, win=0, R=0., PF=0.)
    r = t.r_multiple.to_numpy(); loss = -r[r < 0].sum()
    return dict(n=len(r), win=round(float((r > 0).mean()), 3), R=round(float(r.sum()), 1), PF=round(float(r[r > 0].sum() / loss), 2) if loss > 0 else 9.99)


res = {}
print(f"\n{'gap':>4} {'maxstop':>7} {'ema':>4} {'rr':>3} | NET n win R PF | half1 R half2 R | 5x slip R | worth")
for (gap, ms), ema, rr in itertools.product([(2, 10), (1, 5), (0.5, 3), (0.3, 2)], [20, 300], [3, 5]):
    kw = dict(rr=float(rr), p_gap_price=gap, max_sl_dist=float(ms), ema_period=ema, bar_minutes=1, entry_filter=factory)
    n = simulate(frame, **kw); s5 = simulate(frame, slippage_points=10, **kw)
    N, A, B, S = summ(n), summ(n[n.entry_time < mid]), summ(n[n.entry_time >= mid]), summ(s5)
    worth = bool(N["R"] > 0 and N["PF"] >= 1.15 and A["R"] > 0 and B["R"] > 0 and S["R"] > 0)
    res[f"gap{gap}_ms{ms}_ema{ema}_rr{rr}"] = dict(net=N, half1=A, half2=B, slip5x=S, worth=worth)
    print(f"{gap:4} {ms:7} {ema:4} {rr:3} | {N['n']:4d} {N['win']:.2f} {N['R']:6.1f} {N['PF']:4.2f} | {A['R']:6.1f} {B['R']:6.1f} | {S['R']:6.1f} | {worth}", flush=True)
print(f"\nNET R > 0: {sum(v['net']['R'] > 0 for v in res.values())} of 16 | passing the rule: {sum(v['worth'] for v in res.values())}")
json.dump(res, open(os.path.join(ROOT, "data", "sp2l_m1_20261001", "spread_cap_report.json"), "w"), indent=1)
