"""Donchian M15: smaller ATR stops (user request 2026-09-29). Look-ahead-free engine.

Grid: atr_stop_multiplier x reward_risk_ratio, everything else as live (N=10, EMA30,
strength 0.5%, 0.2% risk, 3-loss/2h cooldown, 7-day time stop, FundedNext costs).
Reported: full history, first 70% and last 30%, then the 6-month tick check for the
same cells (tick path: combined_tick_backtest.donchian with the stop/RR patched).
"""
import sys, os, json, types, importlib, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); COMB = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
mt5 = types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
    setattr(mt5, k, 1)
sys.modules.setdefault("MetaTrader5", mt5)
import pandas as pd
sys.path.insert(0, os.path.join(ROOT, "research_20260924"))
import reevaluate_donchian as rd

MULTS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
RRS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
BASE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    out = {}
    for m, rr in itertools.product(MULTS, RRS):
        p = dict(BASE, atr_stop_multiplier=m, reward_risk_ratio=rr)
        t = rd.run(low, p)
        e = pd.to_datetime(t.entry_time)
        full, first, last = rd.summary(t), rd.summary(t[e < boundary]), rd.summary(t[e >= boundary])
        out[f"{m}x{rr}"] = dict(full=full, first70=first, last30=last)
        print(f"ATR x{m:<4} RR {rr:<3} | full n={full['n']:4d} win={full['win'] or 0:.2f} R={full['R']:7.1f} PF={full['PF'] or 0:.2f} DD={full['DD']:5.1f}"
              f" | 70% R={first['R']:6.1f} | 30% R={last['R']:6.1f} PF={last['PF'] or 0:.2f}", flush=True)
    json.dump(dict(split=str(boundary), grid=out), open(os.path.join(HERE, "bar_grid.json"), "w"), indent=1, default=str)

    # tick check (6 months) for the smaller stops at the live RR and at each stop's best full-history RR
    sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ticks = {}
    for m in MULTS:
        best_rr = max(RRS, key=lambda rr: out[f"{m}x{rr}"]["full"]["R"])
        for rr in sorted({3.0, best_rr}):
            ctb.D_ATR_MULT, ctb.D_RR = m, rr
            s = ctb.stats(ctb.donchian(frame, start, end))
            ticks[f"{m}x{rr}"] = s
            print(f"TICK ATR x{m:<4} RR {rr:<3} | {s}", flush=True)
    json.dump(dict(window=[str(start), str(end)], ticks=ticks), open(os.path.join(HERE, "tick_check.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
