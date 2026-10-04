"""Both live bots on the 6-month real ticks with the CURRENT live settings (2026-10-04):
SLP2: RR5, risk 0.3%, spread rule 3x.  Donchian: N20, EMA30, 0.3%, 2 ATR (min $8), RR4, S&P 5d filter, risk 0.3%.
Fixed sizing equity = current balance (no compounding), commission $7/lot, swap and AI review NOT modelled."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd, pyarrow.parquet as pq
import counter_move_filters as cm, spx_filter as sf
sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
import combined_tick_backtest as ctb
from scripts.sp2l_tick_backtest import run_ticks
ctb.EQUITY, ctb.RISK_PCT = 24787.64, 0.003
ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 20, 30, 0.3, 2.0, 4.0, 8.0
ctb.D_ENTRY_FILTER = sf.spx_filter
meta = pq.ParquetFile(ctb.TICKS)
start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
slp2 = ctb.slp2_dollars(run_ticks(frame, start, end, min_stop_spread_mult=3.0)); don = ctb.donchian(frame, start, end)
both = pd.concat([slp2, don], ignore_index=True).sort_values("exit_time")
both.to_csv(os.path.join(HERE, "both_bots_6m_trades.csv"), index=False)
m = both.assign(month=both.exit_time.dt.strftime("%Y-%m")).pivot_table(index="month", columns="bot", values="usd", aggfunc="sum", fill_value=0); m["total"] = m.sum(axis=1)
rep = dict(window=[str(start), str(end)], equity=ctb.EQUITY, SLP2=ctb.stats(slp2), Donchian=ctb.stats(don), combined=ctb.stats(both), monthly_usd=m.round(2).to_dict("index"),
           by_side={b: {s: dict(n=int(len(g)), win=round(float((g.usd > 0).mean()), 3), usd=round(float(g.usd.sum()), 2)) for s, g in d.groupby("direction")} for b, d in (("SLP2", slp2), ("Donchian", don))},
           by_reason={b: d.groupby("reason").usd.agg(["count", "sum"]).round(2).to_dict("index") for b, d in (("SLP2", slp2), ("Donchian", don))})
json.dump(rep, open(os.path.join(HERE, "both_bots_6m.json"), "w"), indent=1, default=str); print(json.dumps(rep, indent=1, default=str))
