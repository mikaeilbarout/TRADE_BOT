"""Three ideas from loss_anatomy.py, each tested ALONE on the live settings
(2x ATR stop, min stop $8, RR 3):
  A. trend age   -- no entry when the H4 EMA side is older than N H4 bars
  B. session     -- no entry during some data-clock (broker server, UTC+2/+3) hours
  C. breakeven   -- stop to the entry price once the trade reaches +X R

The ideas were found on the 6-month tick window, so that window is in-sample for them.
Protocol (fixed before running): data before the tick window (2022-06 .. 2026-03-22) is
split in half; each filter's parameter is chosen on half A only (best total R /
max(5, DD), averaged with its grid neighbours), then checked on half B and on ticks.
A filter is adopted only if, vs no filter, it improves R on half B AND the tick net
USD, with half-B drawdown no worse than 1.2x the baseline's.
"""
import sys, os, json, types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); COMB = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research_20260924"))
m = types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
    setattr(m, k, 1)
sys.modules["MetaTrader5"] = m
import numpy as np
import pandas as pd
import reevaluate_donchian as rd

BASE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5, atr_stop_multiplier=2.0,
            reward_risk_ratio=3.0, min_stop_dollars=8.0)
TICK_START = pd.Timestamp("2026-03-23")
FILTERS = {
    "A_trend_age": ("max_trend_age", [0, 6, 9, 12, 18, 24, 36]),
    "B_session": ("blocked_entry_hours", [(), (13, 14, 15, 16), (12, 13, 14, 15, 16), (13, 14, 15, 16, 17), (14, 15, 16)]),
    "C_breakeven": ("breakeven_at_r", [0.0, 1.0, 1.25, 1.5, 2.0]),
}
TICK_ATTR = {"max_trend_age": "D_MAX_TREND_AGE", "blocked_entry_hours": "D_BLOCKED_HOURS", "breakeven_at_r": "D_BREAKEVEN_R"}


def score(s):
    return s["R"] / max(5., s["DD"])


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    pre = low[low.ts < TICK_START]
    mid = pre.ts.iloc[len(pre) // 2]
    print(f"half A: {pre.ts.iloc[0]} .. {mid} | half B: {mid} .. {TICK_START} | ticks after", flush=True)

    sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 2.0, 3.0, 8.0

    def tick(param, value):
        for attr, off in (("D_MAX_TREND_AGE", 0), ("D_BLOCKED_HOURS", ()), ("D_BREAKEVEN_R", 0.0)):
            setattr(ctb, attr, off)
        setattr(ctb, TICK_ATTR[param], value)
        return ctb.stats(ctb.donchian(frame, start, end))

    report = {}
    for name, (param, grid) in FILTERS.items():
        rows = []
        for v in grid:
            t = rd.run(low, dict(BASE, **{param: v}))
            e = pd.to_datetime(t.entry_time)
            rows.append(dict(value=v, A=rd.summary(t[e < mid]), B=rd.summary(t[(e >= mid) & (e < TICK_START)]),
                             full=rd.summary(t), years={int(k): round(float(x), 1) for k, x in t.R.groupby(pd.to_datetime(t.exit_time).dt.year).sum().items()}))
        raw = [score(r["A"]) for r in rows]
        # neighbours only among the filter-on values (index 0 is "off")
        smooth = [raw[0]] + [float(np.mean(raw[max(1, i - 1):i + 2])) for i in range(1, len(rows))]
        pick_i = 1 + int(np.argmax(smooth[1:]))
        base, pick = rows[0], rows[pick_i]
        tb, tp = tick(param, grid[0]), tick(param, pick["value"])
        adopt = bool(pick["B"]["R"] > base["B"]["R"] and tp["net_usd"] > tb["net_usd"] and pick["B"]["DD"] <= 1.2 * base["B"]["DD"])
        print(f"\n=== {name} ({param})", flush=True)
        for r, sc in zip(rows, smooth):
            print(f"  {str(r['value']):22} | A n={r['A']['n']:3d} R={r['A']['R']:6.1f} DD={r['A']['DD']:5.1f} sm={sc:5.2f} | "
                  f"B n={r['B']['n']:3d} win={r['B']['win']:.2f} R={r['B']['R']:6.1f} DD={r['B']['DD']:5.1f} | 4y R={r['full']['R']:6.1f} DD={r['full']['DD']:5.1f} | {r['years']}", flush=True)
        print(f"  PICK (half A): {pick['value']} | ticks: off n={tb['trades']} ${tb['net_usd']} DD {tb['max_dd_pct']}% -> "
              f"pick n={tp['trades']} ${tp['net_usd']} DD {tp['max_dd_pct']}% | ADOPT: {adopt}", flush=True)
        report[name] = dict(param=param, rows=rows, smooth=smooth, pick=pick["value"], tick_off=tb, tick_pick=tp, adopt=adopt)
    json.dump(dict(half_split=str(mid), tick_window=[str(start), str(end)], results=report),
              open(os.path.join(HERE, "three_filters.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
