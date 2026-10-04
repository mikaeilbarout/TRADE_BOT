"""Scale out: close HALF the position at +L R, leave the rest to the bot's own stop/target (user's manual early
closes worked in the live sample; breakeven stops failed on ticks, this is a different mechanism).
Two fixed variants (no search): P1 half at 1.5R, P2 half at 2R. Live Donchian (N20, EMA30, 0.3%, 2 ATR, RR 4,
min $8, S&P filter). Path: M15 highs/lows for 4 years (same-bar stop+level = stop first, conservative) and M5
highs/lows for the 6-month tick trades. Costs: commission on the closed half is paid at close (same total lots),
so no change is modelled. R after the change: stop after level = +0.5L - 0.5 ; target after level = 0.5L + 0.5*R_old.
Adopted only if it improves the LAST 30% of the 4 years AND the 6-month tick net USD, with DD <= 1.2x.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
from strategy.donchian import add_donchian_indicators
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0, min_stop_dollars=8.0, entry_filter=spx.spx_filter)


def reached(path, entry, stop, d, L, t_from, t_to):
    seg = path.loc[t_from:t_to]
    if len(seg) == 0: return False
    lvl = entry + d * L * stop
    return bool((seg.high >= lvl).any()) if d == 1 else bool((seg.low <= lvl).any())


def adjust(trades, path, atr, L, first_bar_delta):
    out = []
    for _, x in trades.iterrows():
        d = 1 if x.side == "long" else -1
        stop = 2 * atr.loc[x.signal_ts] if x.signal_ts in atr.index else np.nan
        Rold = x.R
        if np.isnan(stop): out.append(Rold); continue
        hit = reached(path, x.entry_price, stop, d, L, x.start, x.end)
        if x.outcome == "stop":
            # the level must have been reached BEFORE the stop: only count bars strictly before the exit bar
            hit = reached(path, x.entry_price, stop, d, L, x.start, x.end - first_bar_delta)
            out.append(0.5 * L - 0.5 if hit else Rold)
        elif x.outcome == "target":
            out.append(0.5 * L + 0.5 * Rold)
        else:
            out.append(0.5 * L + 0.5 * Rold if hit else Rold)
    return np.array(out)


def summ(r):
    r = np.asarray(r, float); c = np.cumsum(r)
    return dict(n=len(r), win=round(float((r > 0).mean()), 3), R=round(float(c[-1]), 1), DD=round(float(np.max(np.maximum.accumulate(np.maximum(c, 0)) - c)), 1))


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    atr = add_donchian_indicators(low, 20, 14).set_index("ts").atr
    path15 = low.set_index("ts")
    tr = rd.run(low, LIVE).sort_values("exit_time").reset_index(drop=True)
    tr["signal_ts"] = pd.to_datetime(tr.entry_time); tr["start"] = tr.signal_ts + pd.Timedelta(minutes=15); tr["end"] = pd.to_datetime(tr.exit_time)
    split = int(len(low) * .7); boundary = low.ts.iloc[split]
    sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
    import combined_tick_backtest as ctb, pyarrow.parquet as pq
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.RISK_PCT = 0.003
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP, ctb.D_ENTRY_FILTER = 20, 30, 0.3, 2.0, 4.0, 8.0, spx.spx_filter
    tk = ctb.donchian(frame, start, end).sort_values("exit_time").reset_index(drop=True)
    m5 = pd.read_parquet(os.path.join(cm.COMB, "SLP2", "data", "XAUUSD_M5_full_history.parquet")).set_index("bar_time")
    tk["side"] = tk.direction; tk["entry_price"] = tk.entry
    tk["outcome"] = tk.reason.replace({"target": "target", "stop": "stop"}); tk["R"] = tk.usd / (ctb.EQUITY * 0.003)
    tk["signal_ts"] = pd.to_datetime(tk.entry_time).dt.floor("15min") - pd.Timedelta(minutes=15)
    tk["start"] = pd.to_datetime(tk.entry_time); tk["end"] = pd.to_datetime(tk.exit_time)
    res = {}
    for name, L in (("live (no scaling)", None), ("P1 half at 1.5R", 1.5), ("P2 half at 2R", 2.0)):
        if L is None:
            rb, rt = tr.R.to_numpy(), tk.R.to_numpy()
        else:
            rb = adjust(tr, path15, atr, L, pd.Timedelta(minutes=15)); rt = adjust(tk, m5, atr, L, pd.Timedelta(minutes=5))
        e = pd.to_datetime(tr.entry_time)
        res[name] = dict(first70=summ(rb[(e < boundary).to_numpy()]), last30=summ(rb[(e >= boundary).to_numpy()]), full=summ(rb),
                         tick=summ(rt), tick_usd=round(float(rt.sum() * ctb.EQUITY * 0.003), 0))
        a, b, f, k = res[name]["first70"], res[name]["last30"], res[name]["full"], res[name]["tick"]
        print(f"{name:20} | first70 R={a['R']:6.1f} DD={a['DD']:5.1f} | last30 n={b['n']} win={b['win']} R={b['R']:6.1f} DD={b['DD']:5.1f} | 4y R={f['R']:6.1f} DD={f['DD']:5.1f} | "
              f"TICK n={k['n']} win={k['win']} R={k['R']:+.1f} DD={k['DD']} ~${res[name]['tick_usd']:+.0f}", flush=True)
    base = res["live (no scaling)"]
    for name in list(res)[1:]:
        r = res[name]
        r["adopt"] = bool(r["last30"]["R"] > base["last30"]["R"] and r["last30"]["DD"] <= 1.2 * base["last30"]["DD"] and r["tick_usd"] > base["tick_usd"])
        print(f"ADOPT {name}: {r['adopt']}")
    json.dump(res, open(os.path.join(HERE, "partial_tp.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
