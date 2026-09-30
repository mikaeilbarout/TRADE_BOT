"""Risk 0.2% -> 0.1% after 3 losses in a row, back to 0.2% after the first win (user request 2026-09-30).

Sizing does not change which trades happen (the bot's 3-loss / 2 h pause is unchanged), so each trade's
result is scaled: x0.5 while the losing streak before it is >= 3. Applied to the live bot's trades
(N20, EMA30, 0.3%, 2 ATR min $8, RR 4, S&P filter) on 4 years of bars and on 6 months of real ticks.
Variants after 2 and 4 losses are shown for context only. Tick lots are rounded to 0.01 in reality,
so the tick dollar figures are approximate (a few cents per trade).
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0,
            min_stop_dollars=8.0, entry_filter=spx.spx_filter)


def apply(res, after):
    out, streak = [], 0
    for r in res:
        m = 0.5 if after and streak >= after else 1.0
        out.append(r * m)
        streak = streak + 1 if r <= 0 else 0
    return np.array(out)


def stats(x, unit):
    c = np.cumsum(x); dd = float(np.max(np.maximum.accumulate(np.maximum(c, 0)) - c))
    worst = 0.; run = 0.
    for v in x:
        run = run + v if v <= 0 else 0.; worst = min(worst, run)
    return dict(total=round(float(c[-1]), 2), max_dd=round(dd, 2), worst_losing_run=round(worst, 2), unit=unit)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    bars = rd.run(low, LIVE).sort_values("exit_time").R.to_numpy()
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb, pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    f = ctb.load_m15(); f = f[f.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 20, 30, 0.3, 2.0, 4.0, 8.0, spx.spx_filter
    ticks = ctb.donchian(f, start, end).sort_values("exit_time").usd.to_numpy()
    out = {}
    for label, after in (("no change (live)", 0), ("half risk after 3 losses (tested rule)", 3), ("after 2 losses (context)", 2), ("after 4 losses (context)", 4)):
        b, t = stats(apply(bars, after), "R"), stats(apply(ticks, after), "USD")
        out[label] = dict(bars_4y=b, ticks_6m=t)
        print(f"{label:40} | 4y bars: total {b['total']:+7.1f}R  max DD {b['max_dd']:5.1f}R  worst losing run {b['worst_losing_run']:6.1f}R | "
              f"6m ticks: total ${t['total']:+8.2f}  max DD ${t['max_dd']:7.2f}  worst losing run ${t['worst_losing_run']:8.2f}", flush=True)
    for name, arr in (("4y bars", bars), ("6m ticks", ticks)):
        streak, n_half, wins_half = 0, 0, 0
        for r in arr:
            if streak >= 3:
                n_half += 1; wins_half += int(r > 0)
            streak = streak + 1 if r <= 0 else 0
        print(f"{name}: {len(arr)} trades, {n_half} taken at half risk (after >=3 losses), of which {wins_half} won "
              f"({wins_half / max(n_half, 1):.0%}; overall win rate {np.mean(arr > 0):.0%})")
        out[f"{name}_half_risk_trades"] = dict(trades=n_half, wins=wins_half)
    json.dump(out, open(os.path.join(HERE, "risk_after_losses.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
