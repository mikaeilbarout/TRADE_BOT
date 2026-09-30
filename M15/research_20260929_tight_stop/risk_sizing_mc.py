"""How much risk per trade before the account limits come close? (user request 2026-09-30)

Trades: both live bots over the 4 years of bars -- Donchian (N20, EMA30, 0.3%, 2 ATR min $8, RR 4,
S&P filter) and SLP2 (live defaults, RR 5) -- each result in R (1R = the risk per trade).
Assumed limits (FundedNext usual): 5% daily loss, 10% overall loss; both measured here from the
highest equity so far (stricter than a limit fixed at the starting balance).
1. Historical: max drawdown and worst day of the real 4-year sequence at each risk level.
2. Monte Carlo: 20,000 random 12-month sequences drawn from the real trades (with replacement,
   ~136 trades a year as in the backtest) -> probability of touching -5% / -10%.
   Trades are drawn independently, so clustered losing streaks are if anything under-represented.
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import counter_move_filters as cm, spx_filter as spx
rd = cm.rd
sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate as slp2_simulate

RISKS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
LIVE = dict(n_period=20, ema_trend_period=30, min_trend_strength_pct=0.3, atr_stop_multiplier=2.0, reward_risk_ratio=4.0,
            min_stop_dollars=8.0, entry_filter=spx.spx_filter)


def main():
    low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
    d = rd.run(low, LIVE)
    d = pd.DataFrame(dict(exit=pd.to_datetime(d.exit_time), R=d.R, bot="Donchian"))
    s = slp2_simulate(load_m15())
    s = pd.DataFrame(dict(exit=pd.to_datetime(s.exit_time), R=s.r_multiple, bot="SLP2"))
    t = pd.concat([d, s]).sort_values("exit").reset_index(drop=True)
    years = (t.exit.max() - t.exit.min()).days / 365.25
    per_year = int(round(len(t) / years))
    print(f"{len(t)} trades ({len(d)} Donchian, {len(s)} SLP2) over {years:.2f} years -> {per_year}/year, mean {t.R.mean():+.3f}R\n")
    R = t.R.to_numpy()
    day_R = t.groupby(t.exit.dt.date).R.sum()
    rng = np.random.default_rng(30092026)
    sims = rng.choice(R, size=(20000, per_year), replace=True)
    out = {}
    print("risk/trade | history: max DD  worst day  yearly return | 12-month Monte Carlo: P(DD>=5%)  P(DD>=10%)  median DD  95% DD  median return")
    for r in RISKS:
        f = r / 100
        eq = np.cumprod(1 + f * R)                       # compounding, like the bots (risk = % of current equity)
        dd_hist = float(np.max(1 - eq / np.maximum.accumulate(eq)))
        worst_day = float(day_R.min() * r)
        yearly = float(eq[-1] ** (1 / years) - 1)
        e = np.cumprod(1 + f * sims, axis=1)
        dds = np.max(1 - e / np.maximum.accumulate(e, axis=1), axis=1)
        ret = e[:, -1] - 1
        out[r] = dict(hist_max_dd=dd_hist, worst_day_pct=worst_day, yearly_return=yearly, p_dd5=float(np.mean(dds >= .05)),
                      p_dd10=float(np.mean(dds >= .10)), median_dd=float(np.median(dds)), p95_dd=float(np.percentile(dds, 95)),
                      median_return=float(np.median(ret)))
        o = out[r]
        print(f"   {r:4.2f}%   |      {o['hist_max_dd']:6.1%}    {o['worst_day_pct']:+6.2f}%     {o['yearly_return']:+6.1%}     |"
              f"      {o['p_dd5']:6.1%}      {o['p_dd10']:6.1%}      {o['median_dd']:6.1%}   {o['p95_dd']:6.1%}     {o['median_return']:+7.1%}")
    json.dump(dict(trades=len(t), per_year=per_year, results=out), open(os.path.join(HERE, "risk_sizing_mc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
