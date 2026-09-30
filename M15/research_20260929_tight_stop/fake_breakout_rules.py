"""Two rules against fake breakouts, each tested ALONE on the live settings
(2x ATR stop, min stop $8, RR 3):
  1. early exit     -- within the first N M15 bars after entry, a close back through the
                       breakout level closes the trade at market (N = 1..4)
  2. confirm entry  -- enter one bar later, only if that bar also closes beyond the level

Same protocol as three_filters.py (fixed before running): the pre-tick data
(2022-06 .. 2026-03-22) is split in half; N is chosen on half A only (neighbour-smoothed
R / max(5, DD)); a rule is adopted only if, vs no rule, it improves R on half B AND the
6-month tick net USD, with half-B drawdown no worse than 1.2x the baseline's.
The fake-breakout pattern itself was found on 2022-2026 bars (fake_breakouts.py), so
half B is not fully untouched; the tick run is the stricter check.
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
RULES = {
    "early_exit": ("early_exit_bars", "D_EARLY_EXIT_BARS", [0, 1, 2, 3, 4]),
    "confirm_entry": ("confirm_entry", "D_CONFIRM_ENTRY", [False, True]),
}


def score(s):
    return s["R"] / max(5., s["DD"])


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    pre = low[low.ts < TICK_START]; mid = pre.ts.iloc[len(pre) // 2]
    sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 2.0, 3.0, 8.0

    def tick(attr, value):
        ctb.D_EARLY_EXIT_BARS, ctb.D_CONFIRM_ENTRY = 0, False
        setattr(ctb, attr, value)
        t = ctb.donchian(frame, start, end)
        return ctb.stats(t), t

    report = {}
    for name, (param, attr, grid) in RULES.items():
        rows = []
        for v in grid:
            t = rd.run(low, dict(BASE, **{param: v})); e = pd.to_datetime(t.entry_time)
            rows.append(dict(value=v, A=rd.summary(t[e < mid]), B=rd.summary(t[(e >= mid) & (e < TICK_START)]), full=rd.summary(t),
                             years={int(k): round(float(x), 1) for k, x in t.R.groupby(pd.to_datetime(t.exit_time).dt.year).sum().items()}))
        raw = [score(r["A"]) for r in rows]
        smooth = [raw[0]] + [float(np.mean(raw[max(1, i - 1):i + 2])) for i in range(1, len(rows))]
        pick_i = 1 + int(np.argmax(smooth[1:])); base, pick = rows[0], rows[pick_i]
        ticks = {}
        for v in grid:
            s, tt = tick(attr, v)
            ticks[str(v)] = dict(s, exits=tt.reason.value_counts().to_dict())
        tb, tp = ticks[str(grid[0])], ticks[str(pick["value"])]
        adopt = bool(pick["B"]["R"] > base["B"]["R"] and tp["net_usd"] > tb["net_usd"] and pick["B"]["DD"] <= 1.2 * base["B"]["DD"])
        print(f"\n=== {name}", flush=True)
        for r, sc in zip(rows, smooth):
            tk = ticks[str(r["value"])]
            print(f"  {str(r['value']):6} | A n={r['A']['n']:3d} R={r['A']['R']:6.1f} DD={r['A']['DD']:5.1f} sm={sc:5.2f} | "
                  f"B n={r['B']['n']:3d} win={r['B']['win']:.2f} R={r['B']['R']:6.1f} DD={r['B']['DD']:5.1f} | 4y R={r['full']['R']:6.1f} DD={r['full']['DD']:5.1f} | "
                  f"TICK n={tk['trades']} win={tk['win_rate']} ${tk['net_usd']} DD {tk['max_dd_pct']}% exits {tk['exits']} | {r['years']}", flush=True)
        print(f"  PICK (half A): {pick['value']} | ADOPT: {adopt}", flush=True)
        report[name] = dict(rows=rows, smooth=smooth, pick=pick["value"], ticks=ticks, adopt=adopt)
    json.dump(dict(half_split=str(mid), results=report), open(os.path.join(HERE, "fake_breakout_rules.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
