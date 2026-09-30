"""Donchian M15 with the 2.0x ATR stop (live since 2026-09-29): which reward/risk ratio?

Protocol (fixed before running): choose on the FIRST 70% of the 4-year FundedNext
history only, by total_R / max(5, DD_R) averaged with the two neighbouring RR values;
then report that pick against the current RR=3 on the last 30% and on 6 months of real
ticks. The pick replaces RR=3 only if it beats RR=3 on BOTH the last 30% (R) and ticks
(net USD) without a worse last-30% drawdown than 1.5x RR=3's.
"""
import sys, os, json, types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); COMB = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
mt5 = types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
    setattr(mt5, k, 1)
sys.modules.setdefault("MetaTrader5", mt5)
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(ROOT, "research_20260924"))
import reevaluate_donchian as rd

RRS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
BASE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5, atr_stop_multiplier=2.0)


def score(s):
    return s["R"] / max(5., s["DD"])


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    rows = {}
    for rr in RRS:
        t = rd.run(low, dict(BASE, reward_risk_ratio=rr)); e = pd.to_datetime(t.entry_time)
        rows[rr] = dict(full=rd.summary(t), first70=rd.summary(t[e < boundary]), last30=rd.summary(t[e >= boundary]))
    raw = [score(rows[rr]["first70"]) for rr in RRS]
    smooth = [float(np.mean(raw[max(0, i - 1):i + 2])) for i in range(len(RRS))]
    pick = RRS[int(np.argmax(smooth))]
    for i, rr in enumerate(RRS):
        f, a, b = rows[rr]["full"], rows[rr]["first70"], rows[rr]["last30"]
        print(f"RR {rr:<3} | 70%: n={a['n']:4d} win={a['win']:.2f} R={a['R']:6.1f} DD={a['DD']:5.1f} score={raw[i]:5.2f} smooth={smooth[i]:5.2f}"
              f" | 30%: n={b['n']:3d} win={b['win']:.2f} R={b['R']:6.1f} PF={b['PF']:.2f} DD={b['DD']:5.1f} | full R={f['R']:6.1f} DD={f['DD']:5.1f}", flush=True)
    print("PICK (first 70% only):", pick, flush=True)

    sys.path.insert(0, COMB); sys.path.insert(0, os.path.join(COMB, "SLP2"))
    import combined_tick_backtest as ctb
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ticks = {}
    ctb.D_ATR_MULT = 2.0
    for rr in RRS:
        ctb.D_RR = rr
        ticks[rr] = ctb.stats(ctb.donchian(frame, start, end))
        s = ticks[rr]
        print(f"TICK RR {rr:<3} | n={s['trades']} win={s['win_rate']} net={s['net_usd']} ({s['net_pct']}%) PF={s['profit_factor']} DD={s['max_dd_pct']}%", flush=True)
    p, c = rows[pick]["last30"], rows[3.0]["last30"]
    adopt = bool(pick != 3.0 and p["R"] > c["R"] and ticks[pick]["net_usd"] > ticks[3.0]["net_usd"] and p["DD"] <= 1.5 * c["DD"])
    print("ADOPT pick over RR=3:", adopt)
    json.dump(dict(split=str(boundary), pick=pick, adopt=adopt, bars={str(k): v for k, v in rows.items()},
                   smooth=dict(zip(map(str, RRS), smooth)), ticks={str(k): v for k, v in ticks.items()},
                   tick_window=[str(start), str(end)]), open(os.path.join(HERE, "rr_at_2atr.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
