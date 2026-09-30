"""Donchian M15, 2x ATR stop, RR 3: skip signals whose stop is smaller than X dollars.

Primary hypothesis (user, 2026-09-29): X = 8. Other values are shown for context only.
Adoption rule (fixed before running): X=8 is adopted if, versus no filter, the 4-year
total R is higher, the 4-year max drawdown is not higher, and neither 2025-26 bars nor
the 6-month tick result drop by more than 5%.
"""
import sys, os, json, types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); COMB = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research_20260924"))
m = types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
    setattr(m, k, 1)
sys.modules["MetaTrader5"] = m
import pandas as pd
import reevaluate_donchian as rd

XS = [0, 4, 6, 8, 10, 12, 15]
BASE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5, atr_stop_multiplier=2.0, reward_risk_ratio=3.0)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    res = {}
    for x in XS:
        t = rd.run(low, dict(BASE, min_stop_dollars=float(x)))
        y = pd.to_datetime(t.exit_time).dt.year
        per = {p: rd.summary(t[msk]) for p, msk in (("2022-23", y <= 2023), ("2024", y == 2024), ("2025-26", y >= 2025))}
        res[x] = dict(full=rd.summary(t), **per, years={int(k): round(float(v), 1) for k, v in t.R.groupby(y).sum().items()})
        f = res[x]["full"]
        print(f"min stop ${x:>2} | 4y n={f['n']:4d} win={f['win']:.2f} R={f['R']:6.1f} PF={f['PF']:.2f} DD={f['DD']:5.1f} | "
              f"2022-23 n={per['2022-23']['n']:3d} R={per['2022-23']['R']:6.1f} | 2024 R={per['2024']['R']:5.1f} | "
              f"2025-26 n={per['2025-26']['n']:3d} R={per['2025-26']['R']:6.1f} | by year {res[x]['years']}", flush=True)

    sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_ATR_MULT, ctb.D_RR = 2.0, 3.0
    ticks = {}
    for x in (0, 8, 12):
        ctb.D_MIN_STOP = float(x)
        ticks[x] = ctb.stats(ctb.donchian(frame, start, end)); s = ticks[x]
        print(f"TICK min stop ${x:>2} | n={s['trades']} win={s['win_rate']} net={s['net_usd']} ({s['net_pct']}%) PF={s['profit_factor']} DD={s['max_dd_pct']}%", flush=True)

    a, b = res[8], res[0]
    adopt = bool(a["full"]["R"] > b["full"]["R"] and a["full"]["DD"] <= b["full"]["DD"]
                 and a["2025-26"]["R"] >= .95 * b["2025-26"]["R"] and ticks[8]["net_usd"] >= ticks[0]["net_usd"] - .05 * abs(ticks[0]["net_usd"]))
    print("ADOPT min stop $8:", adopt)
    json.dump(dict(bars={str(k): v for k, v in res.items()}, ticks={str(k): v for k, v in ticks.items()}, adopt_8=adopt),
              open(os.path.join(HERE, "min_stop_filter.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
