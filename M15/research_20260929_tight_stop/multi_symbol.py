"""Diversification test: the live Donchian logic (N20, EMA30, trend 0.3%, 2 ATR, RR 4) on other markets,
WITHOUT re-optimising anything (a parameter search per symbol would just fit noise).
Costs per symbol from the broker: median spread + 7 USD/lot commission per unit; swap not modelled (caveat).
Minimum stop = 15x the round-trip cost per unit (gold's 8 USD is ~20x its 0.37 cost). No S&P filter (gold-specific).
Criteria for "worth adding" (fixed before running): positive R in BOTH halves of its own history, profit factor >= 1.1
in both halves, daily-R correlation with the gold Donchian < 0.3, and the pair's combined return/drawdown better than
gold alone at the same total risk."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import counter_move_filters as cm
rd = cm.rd
P = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0)
DIR = os.path.join(HERE, "multi_m15"); INFO = json.load(open(os.path.join(DIR, "info.json")))


def run_symbol(sym):
    df = pd.read_parquet(os.path.join(DIR, f"{sym}.parquet")).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    i = INFO[sym]; spread = max(i["median_spread_points"], 3.0 if i["digits"] >= 3 and i["contract"] >= 100000 else 0.0) * i["point"]; comm = 7.0 / i["contract"]   # FX floor: 0.3 pip
    costs = dict(spread_dollars=spread, commission_dollars=comm, swap_long=0., swap_short=0.)
    t = rd.run(df, dict(P, min_stop_dollars=15 * (spread + comm)), costs=costs)
    return df, t, spread + comm


def halves(t):
    e = pd.to_datetime(t.entry_time); mid = e.iloc[len(e) // 2]
    return rd.summary(t[e < mid]), rd.summary(t[e >= mid])


def main():
    gold_df, gold_t, _ = run_symbol("XAUUSD")
    gday = gold_t.groupby(pd.to_datetime(gold_t.exit_time).dt.date).R.sum()
    rows = {}
    for sym in INFO:
        df, t, cost = run_symbol(sym)
        if len(t) < 40:
            print(f"{sym}: only {len(t)} trades"); continue
        a, b = halves(t); f = rd.summary(t)
        day = t.groupby(pd.to_datetime(t.exit_time).dt.date).R.sum()
        idx = pd.date_range(min(day.index.min(), gday.index.min()), max(day.index.max(), gday.index.max()), freq="D").date
        corr = float(np.corrcoef(day.reindex(idx, fill_value=0.), gday.reindex(idx, fill_value=0.))[0, 1])
        years = (pd.to_datetime(t.exit_time).max() - pd.to_datetime(t.entry_time).min()).days / 365.25
        ok = bool(a["R"] > 0 and b["R"] > 0 and (a["PF"] or 0) >= 1.1 and (b["PF"] or 0) >= 1.1)
        rows[sym] = dict(trades=f["n"], per_year=round(f["n"] / years), win=round(f["win"], 3), R=round(f["R"], 1), DD=round(f["DD"], 1), PF=round(f["PF"] or 0, 2),
                         half1=a, half2=b, corr_with_gold=round(corr, 3), cost_per_unit=round(cost, 5), both_halves_ok=ok,
                         years={int(k): round(float(v), 1) for k, v in t.R.groupby(pd.to_datetime(t.exit_time).dt.year).sum().items()})
        print(f"{sym:7} n={f['n']:4d} ({rows[sym]['per_year']}/yr) win={f['win']:.2f} R={f['R']:7.1f} DD={f['DD']:5.1f} PF={f['PF'] or 0:.2f} | half1 R={a['R']:6.1f} PF={a['PF'] or 0:.2f} | "
              f"half2 R={b['R']:6.1f} PF={b['PF'] or 0:.2f} | corr gold {corr:+.2f} | ok={ok} | {rows[sym]['years']}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "multi_symbol.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
