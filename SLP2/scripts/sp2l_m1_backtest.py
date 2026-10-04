"""SP2L on 1-minute bars, 6 months built from the real ticks (2026-03-23 .. 2026-09-22).

Question: does the SP2L pattern have an edge on M1 once real costs are paid? Two views per setting:
  GROSS = no costs (does the pattern itself predict anything on M1?)   NET = real spread per bar + 2 pts slippage + $7/lot
Grid fixed before running (no other search): (gap, max stop) in USD  {(2,10) = the M15 live values, (1,5), (0.5,3), (0.3,2)}
x EMA period {20, 300} x RR {3, 5}; spike 1.5, max_opposite 2, max hold 500 bars, as in the live M15 bot.
'Worth building' (fixed before running): NET total R > 0, profit factor >= 1.15, positive in BOTH halves (3 months each), and
still positive with 5x the slippage (10 points). Six months is short and 16 settings are tried, so the share of settings
that are positive is reported too.
"""
import sys, os, json, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scripts.sp2l_m15_backtest import simulate

frame = pd.read_parquet(os.path.join(ROOT, "data", "XAUUSD_M1_6m.parquet"))
frame = frame[frame.tick_volume > 0].reset_index(drop=True)
mid = frame.bar_time.iloc[len(frame) // 2]


def summ(t):
    if len(t) == 0: return dict(n=0, win=0, R=0., PF=0., avg_stop=0.)
    r = t.r_multiple.to_numpy(); loss = -r[r < 0].sum()
    return dict(n=len(r), win=round(float((r > 0).mean()), 3), R=round(float(r.sum()), 1), PF=round(float(r[r > 0].sum() / loss), 2) if loss > 0 else 9.99,
                avg_stop=round(float(t.stop_dist.mean()), 2))


res = {}
print(f"{len(frame)} M1 bars, {frame.bar_time.iloc[0]} .. {frame.bar_time.iloc[-1]} | median tick spread ${frame.avg_spread_price.median():.2f}, mean ${frame.avg_spread_price.mean():.2f}\n")
print(f"{'gap':>4} {'maxstop':>7} {'ema':>4} {'rr':>3} | GROSS n  win   R   PF | NET n  win    R    PF  avgstop | half1 R  half2 R | 5x slip R | worth")
for (gap, ms), ema, rr in itertools.product([(2, 10), (1, 5), (0.5, 3), (0.3, 2)], [20, 300], [3, 5]):
    kw = dict(rr=float(rr), p_gap_price=gap, max_sl_dist=float(ms), ema_period=ema, bar_minutes=1)
    g = simulate(frame, apply_costs=False, **kw)
    n = simulate(frame, **kw)
    h1, h2 = n[n.entry_time < mid], n[n.entry_time >= mid]
    s5 = simulate(frame, slippage_points=10, **kw)
    G, N, A, B, S = summ(g), summ(n), summ(h1), summ(h2), summ(s5)
    worth = bool(N["R"] > 0 and N["PF"] >= 1.15 and A["R"] > 0 and B["R"] > 0 and S["R"] > 0)
    res[f"gap{gap}_ms{ms}_ema{ema}_rr{rr}"] = dict(gross=G, net=N, half1=A, half2=B, slip5x=S, worth=worth)
    print(f"{gap:4} {ms:7} {ema:4} {rr:3} | {G['n']:5d} {G['win']:.2f} {G['R']:6.1f} {G['PF']:4.2f} | {N['n']:5d} {N['win']:.2f} {N['R']:6.1f} {N['PF']:4.2f} {N['avg_stop']:5.2f} | "
          f"{A['R']:6.1f} {B['R']:6.1f} | {S['R']:7.1f} | {worth}", flush=True)
pos = sum(1 for v in res.values() if v["net"]["R"] > 0)
print(f"\nsettings with NET total R > 0: {pos} of {len(res)} | passing the 'worth building' rule: {sum(v['worth'] for v in res.values())}")
out = os.path.join(ROOT, "data", "sp2l_m1_20261001"); os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1)
