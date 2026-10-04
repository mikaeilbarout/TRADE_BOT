"""Pause after consecutive losses, re-checked on the live setup (N20, EMA30, 0.3%, 2 ATR, RR 4, min $8, S&P).
Live = 3 losses -> 2 h pause (tuned in September on the old setup). Grid fixed before running:
none / 2 losses 6 h / 3 losses 2 h (live) / 3 losses 12 h / 4 losses 24 h. Adopted only if it beats the live pause on the
LAST 30% of the 4 years (R, DD <= 1.2x) AND on the 6-month real ticks (net USD)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
from strategy.donchian import simulate_donchian
GRID = [("3 losses -> 2 h (live)", 3, 2.0), ("2 losses -> 2 h", 2, 2.0), ("2 losses -> 4 h", 2, 4.0), ("2 losses -> 6 h", 2, 6.0), ("2 losses -> 9 h", 2, 9.0), ("2 losses -> 12 h", 2, 12.0), ("2 losses -> 24 h", 2, 24.0), ("1 loss -> 2 h", 1, 2.0)]
P = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0, entry_filter=spx.spx_filter)


def run(low, k, h, start=None):
    lo, hi = rd.frames(low)
    t, _ = simulate_donchian(lo, hi, atr_period=14, time_stop_minutes=10080, risk_cfg=rd.PROFILE.RISK, starting_equity=10000.,
                             cooldown_losses_to_trigger=k, cooldown_hours=h, **P, **rd.COSTS)
    t["R"] = t.pnl / ((t.equity_after - t.pnl) * rd.PROFILE.RISK.risk_per_trade_pct / 100)
    if start is not None: t = t[pd.to_datetime(t.entry_time) >= start]
    return t.reset_index(drop=True)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]; test = low.iloc[split - 6000:].reset_index(drop=True)
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb, pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.RISK_PCT = 0.003
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 20, 30, 0.3, 2.0, 4.0, 8.0, spx.spx_filter
    res = {}
    for name, k, h in GRID:
        full = run(low, k, h); e = pd.to_datetime(full.entry_time)
        t30 = run(test, k, h, start=boundary)
        ctb.D_COOLDOWN_LOSSES, ctb.D_COOLDOWN = (k if k else 10**6), pd.Timedelta(hours=h)
        tk = ctb.stats(ctb.donchian(frame, start, end))
        res[name] = dict(first70=rd.summary(full[e < boundary]), last30=rd.summary(t30), full=rd.summary(full), tick=tk)
        a, b, f, q = res[name]["first70"], res[name]["last30"], res[name]["full"], tk
        print(f"{name:24} | first70 n={a['n']} R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} R={b['R']:6.1f} DD={b['DD']:5.1f} | 4y n={f['n']} R={f['R']:6.1f} DD={f['DD']:5.1f} | TICK n={q['trades']} ${q['net_usd']} DD {q['max_dd_pct']}%", flush=True)
    base = res["3 losses -> 2 h (live)"]
    for name in res:
        if name == "3 losses -> 2 h (live)": continue
        r = res[name]; r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["tick"]["net_usd"] > base["tick"]["net_usd"])
        print(f"ADOPT {name}: {r['adopt']}")
    json.dump(res, open(os.path.join(HERE, "cooldown_neighbours.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
