"""Two opposite positions per signal (user idea 2026-10-01): leg A in the signal's direction with 1% risk, leg B against it with 0.5% risk,
both RR 2.5, both with the strategy's own stop distance s (A: stop -s / target +2.5s; B mirrored). Hedging account, legs independent.
Signals: live Donchian (N20, EMA30, 0.3%, 2 ATR min $8, S&P filter) and live SLP2, 4 years of M15 bars (stop first if a bar touches both).
Costs per leg: Donchian 0.30 spread + 0.07 commission per oz; SLP2 as in its backtest. 7-day time stop. P&L in % of equity, no compounding."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
from strategy.donchian import add_donchian_indicators
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate

low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
H, L, C, T = low.high.to_numpy(), low.low.to_numpy(), low.close.to_numpy(), low.ts.to_numpy()
BAR = pd.Timedelta(minutes=15); RR = 2.5; COST = 0.37; MAXB = 672


def leg(i0, d, entry, s):
    """R of one leg opened at bar index i0 (first bar that can hit stop/target)."""
    sl, tp = entry - d * s, entry + d * RR * s
    j1 = min(len(H), i0 + MAXB)
    hs = (L[i0:j1] <= sl) if d == 1 else (H[i0:j1] >= sl)
    ht = (H[i0:j1] >= tp) if d == 1 else (L[i0:j1] <= tp)
    a = int(np.argmax(hs)) if hs.any() else 10**9
    b = int(np.argmax(ht)) if ht.any() else 10**9
    if a == 10**9 and b == 10**9: return d * (C[j1 - 1] - entry) / s - COST / s
    return (-1.0 if a <= b else RR) - COST / s


def run(name, sig):
    rows = []
    for t0, d, entry, s in sig:
        i0 = int(np.searchsorted(T, np.datetime64(t0)))
        if i0 >= len(H) - 2 or s <= 0: continue
        rows.append((t0, leg(i0, d, entry, s), leg(i0, -d, entry, s)))
    r = pd.DataFrame(rows, columns=["t", "A", "B"]); r["hedge"] = 1.0 * r.A + 0.5 * r.B; r["A1"] = 1.0 * r.A; r["A05"] = 0.5 * r.A; r["B05"] = 0.5 * r.B
    years = (r.t.max() - r.t.min()).days / 365.25
    print(f"\n=== {name}: {len(r)} signals over {years:.1f} years (P&L in % of equity)")
    print(f"{'':32} total%  per-trade  win%  maxDD%  worst  ret/DD")
    for lab, col in (("A alone, 1% risk", "A1"), ("A alone, 0.5% risk", "A05"), ("B alone (against), 0.5%", "B05"), ("HEDGE PAIR (A 1% + B 0.5%)", "hedge")):
        x = r[col].to_numpy(); c = np.cumsum(x); dd = float(np.max(np.maximum.accumulate(np.maximum(c, 0)) - c))
        print(f"  {lab:30} {c[-1]:7.1f} {x.mean():8.3f} {np.mean(x > 0):6.1%} {dd:6.1f} {x.min():6.2f} {c[-1] / dd if dd else 0:7.2f}")
    both_lose = ((r.A < 0) & (r.B < 0)).mean(); print(f"  both legs lose together in {both_lose:.0%} of signals (single-leg loss 1.0% -> pair loss up to 1.5%)")
    print("  by year (total %): ", {int(k): (round(float(v1), 1), round(float(v2), 1)) for (k, v1), v2 in zip(r.groupby(r.t.dt.year).A1.sum().items(), r.groupby(r.t.dt.year).hedge.sum().values)}, "(A alone, hedge pair)")


LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0, entry_filter=spx.spx_filter)
tr = rd.run(low, LIVE); atr = add_donchian_indicators(low, 20, 14).set_index("ts").atr
sig = [(pd.Timestamp(r.entry_time) + BAR, 1 if r.side == "long" else -1, r.entry_price, 2 * atr.loc[pd.Timestamp(r.entry_time)]) for r in tr.itertuples()]
run("Donchian signals", sig)
sl = simulate(load_m15())
sig2 = [(pd.Timestamp(r.entry_time), 1 if r.direction == "long" else -1, r.entry_price, r.stop_dist) for r in sl.itertuples()]
run("SLP2 signals", sig2)
