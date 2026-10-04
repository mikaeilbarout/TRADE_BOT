"""Daily loss protection: no NEW entries for the rest of the UTC day once the day's realised loss reaches the cap.
Both live bots together at 0.3% risk each (Donchian N20/EMA30/0.3%/2ATR min$8/RR4 + S&P; SLP2 RR5), on the 4-year bar
backtests and on the 6-month real-tick backtests. Day = UTC date (as the bot's guard). Realised P&L only (the bot's
guard also sees floating P&L and manual trades, so it is stricter than this). A skipped trade is simply removed.
Caps tested: none, 1, 1.5, 2, 2.5, 3, 3.5 % of the day's start equity.
Rule fixed before running: a cap is 'better' only if return/max-drawdown improves vs no cap on BOTH the 4-year bars AND
the 6-month ticks. Also reported: worst day, trades skipped, days the cap fired."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate
from scripts.sp2l_tick_backtest import run_ticks
import combined_tick_backtest as ctb, pyarrow.parquet as pq
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0, entry_filter=spx.spx_filter)
RISK = 0.003; BAR = pd.Timedelta(minutes=15); CAPS = [None, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def to_utc(ts):
    s = pd.Series(pd.to_datetime(ts))
    return s.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None).fillna(s - pd.Timedelta(hours=3)).to_numpy()


def apply_cap(t, cap):
    """t: entry, exit, R (time-sorted by entry, UTC). Returns the kept mask."""
    keep = np.ones(len(t), bool)
    if cap is None: return keep
    e = pd.to_datetime(t.entry).to_numpy(); x = pd.to_datetime(t.exit).to_numpy(); R = t.R.to_numpy()
    day = pd.to_datetime(t.entry).dt.date.to_numpy(); xday = pd.to_datetime(t.exit).dt.date.to_numpy()
    for i in range(len(t)):
        realised = sum(R[j] for j in range(i) if keep[j] and x[j] <= e[i] and xday[j] == day[i])
        if realised * RISK * 100 <= -cap: keep[i] = False
    return keep


def metrics(t, keep, years):
    k = t[keep].sort_values("exit"); r = k.R.to_numpy() * RISK
    eq = np.cumprod(1 + r); dd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(r) else 0.
    ret = float(eq[-1] ** (1 / years) - 1) if len(r) else 0.
    daily = k.groupby(pd.to_datetime(k.exit).dt.date).R.sum() * RISK * 100
    return dict(trades=int(len(k)), R=round(float(k.R.sum()), 1), yearly_return=round(ret * 100, 2), max_dd=round(dd * 100, 2),
                ret_over_dd=round(ret / dd, 2) if dd else None, worst_day=round(float(daily.min()), 2))


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    d = rd.run(low, LIVE); s = simulate(load_m15())
    bars = pd.concat([pd.DataFrame(dict(entry=to_utc(pd.to_datetime(d.entry_time) + BAR), exit=to_utc(pd.to_datetime(d.exit_time) + BAR), R=d.R.to_numpy(), bot="Donchian")),
                      pd.DataFrame(dict(entry=to_utc(pd.to_datetime(s.entry_time)), exit=to_utc(pd.to_datetime(s.exit_time) + BAR), R=s.r_multiple.to_numpy(), bot="SLP2"))]
                     ).sort_values("entry").reset_index(drop=True)
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    f = ctb.load_m15(); f = f[f.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.RISK_PCT = RISK
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 20, 30, 0.3, 2.0, 4.0, 8.0, spx.spx_filter
    dt_, st_ = ctb.donchian(f, start, end), ctb.slp2_dollars(run_ticks(f, start, end))
    tk = pd.concat([pd.DataFrame(dict(entry=to_utc(dt_.entry_time), exit=to_utc(dt_.exit_time), R=(dt_.usd / (ctb.EQUITY * RISK)).to_numpy(), bot="Donchian")),
                    pd.DataFrame(dict(entry=to_utc(st_.entry_time), exit=to_utc(st_.exit_time), R=(st_.usd / (ctb.EQUITY * RISK)).to_numpy(), bot="SLP2"))]
                   ).sort_values("entry").reset_index(drop=True)
    yb = (pd.to_datetime(bars.exit).max() - pd.to_datetime(bars.entry).min()).days / 365.25
    yt = (pd.to_datetime(tk.exit).max() - pd.to_datetime(tk.entry).min()).days / 365.25
    res = {}
    print(f"bars: {len(bars)} trades over {yb:.2f} y | ticks: {len(tk)} trades over {yt:.2f} y\n")
    print("cap   | 4-YEAR BARS: trades  R   yearly  maxDD  ret/DD  worst day  skipped days-fired | 6-MONTH TICKS: trades  R  maxDD ret/DD worst day skipped")
    for cap in CAPS:
        kb, kt = apply_cap(bars, cap), apply_cap(tk, cap)
        mb, mt = metrics(bars, kb, yb), metrics(tk, kt, yt)
        fired = int(len({pd.Timestamp(v).date() for v in bars.entry[~kb]})) if cap else 0
        res[str(cap)] = dict(bars=mb, ticks=mt, skipped_bars=int((~kb).sum()), skipped_ticks=int((~kt).sum()), days_fired_bars=fired)
        print(f"{'none' if cap is None else f'{cap:.1f}%':5} | {mb['trades']:4d} {mb['R']:7.1f} {mb['yearly_return']:6.2f}% {mb['max_dd']:5.2f}% {mb['ret_over_dd']:5.2f} {mb['worst_day']:+6.2f}%  {int((~kb).sum()):3d}  {fired:3d}   | "
              f"{mt['trades']:4d} {mt['R']:6.1f} {mt['max_dd']:5.2f}% {mt['ret_over_dd']:5.2f} {mt['worst_day']:+6.2f}%  {int((~kt).sum()):3d}", flush=True)
    b = res["None"]
    for cap in CAPS[1:]:
        r = res[str(cap)]; r["better"] = bool(r["bars"]["ret_over_dd"] > b["bars"]["ret_over_dd"] and r["ticks"]["ret_over_dd"] > b["ticks"]["ret_over_dd"])
        print(f"cap {cap}%: better than no cap on BOTH = {r['better']}")
    json.dump(res, open(os.path.join(HERE, "daily_cap.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
