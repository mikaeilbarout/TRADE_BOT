"""SP2L on 1-minute bars with RR = 1 (user request 2026-10-01). Same 6 months of real-tick M1 bars, same settings and the same
'worth building' rule as sp2l_m1_backtest.py: (gap, max stop) {(2,10),(1,5),(0.5,3),(0.3,2)} x EMA {20,300}; NET = real spread per
bar + 2 pts slippage + $7/lot; also gross (no costs) and 5x slippage; halves = first / last 3 months."""
import sys, os, json, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scripts.sp2l_m15_backtest import simulate
frame = pd.read_parquet(os.path.join(ROOT, "data", "XAUUSD_M1_6m.parquet")); frame = frame[frame.tick_volume > 0].reset_index(drop=True)
mid = frame.bar_time.iloc[len(frame) // 2]


def summ(t):
    if len(t) == 0: return dict(n=0, win=0, R=0., PF=0., avg_stop=0., exits={})
    r = t.r_multiple.to_numpy(); loss = -r[r < 0].sum()
    return dict(n=len(r), win=round(float((r > 0).mean()), 3), R=round(float(r.sum()), 1), PF=round(float(r[r > 0].sum() / loss), 2) if loss > 0 else 9.99,
                avg_stop=round(float(t.stop_dist.mean()), 2), exits=t.exit_reason.value_counts().to_dict())


res = {}
print(f"RR = 1 | {len(frame)} M1 bars | breakeven win rate before costs = 50%\n")
print(f"{'gap':>4} {'maxstop':>7} {'ema':>4} | GROSS n  win    R  PF | NET n  win     R   PF  avgstop | half1 R  half2 R | 5x slip R | worth")
for (gap, ms), ema in itertools.product([(2, 10), (1, 5), (0.5, 3), (0.3, 2)], [20, 300]):
    kw = dict(rr=1.0, p_gap_price=gap, max_sl_dist=float(ms), ema_period=ema, bar_minutes=1)
    g, n, s5 = simulate(frame, apply_costs=False, **kw), simulate(frame, **kw), simulate(frame, slippage_points=10, **kw)
    G, N, A, B, S = summ(g), summ(n), summ(n[n.entry_time < mid]), summ(n[n.entry_time >= mid]), summ(s5)
    worth = bool(N["R"] > 0 and N["PF"] >= 1.15 and A["R"] > 0 and B["R"] > 0 and S["R"] > 0)
    res[f"gap{gap}_ms{ms}_ema{ema}"] = dict(gross=G, net=N, half1=A, half2=B, slip5x=S, worth=worth)
    print(f"{gap:4} {ms:7} {ema:4} | {G['n']:5d} {G['win']:.2f} {G['R']:6.1f} {G['PF']:4.2f} | {N['n']:5d} {N['win']:.2f} {N['R']:7.1f} {N['PF']:4.2f} {N['avg_stop']:5.2f} | "
          f"{A['R']:6.1f} {B['R']:6.1f} | {S['R']:7.1f} | {worth}", flush=True)
print(f"\nNET R > 0: {sum(v['net']['R'] > 0 for v in res.values())} of {len(res)} | passing the rule: {sum(v['worth'] for v in res.values())}")
print("cost per trade in R (gross R - net R) / trades:", {k: round((v['gross']['R'] - v['net']['R']) / max(v['net']['n'], 1), 2) for k, v in res.items()})
json.dump(res, open(os.path.join(ROOT, "data", "sp2l_m1_20261001", "rr1_report.json"), "w"), indent=1)
