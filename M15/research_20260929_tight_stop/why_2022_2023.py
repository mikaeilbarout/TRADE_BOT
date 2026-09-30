"""Why did Donchian M15 with a 2x ATR stop (RR 3) lose in 2022-2023 and win in 2025-2026?"""
import sys, os, types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research_20260924"))
m = types.ModuleType("MetaTrader5")
for k in ("TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30", "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1"):
    setattr(m, k, 1)
sys.modules["MetaTrader5"] = m
import numpy as np
import pandas as pd
import reevaluate_donchian as rd
from strategy.donchian import add_donchian_indicators, add_trend_indicator

COST = .30 + .07  # spread + commission, $/oz per round trip
low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
ind = add_donchian_indicators(low, 10, 14).set_index("ts")
bars = low.set_index("ts")


def trades(mult):
    t = rd.run(low, dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5, atr_stop_multiplier=mult, reward_risk_ratio=3.))
    t["entry_time"] = pd.to_datetime(t.entry_time); t["exit_time"] = pd.to_datetime(t.exit_time)
    t["y"] = t.exit_time.dt.year
    t["period"] = np.where(t.y <= 2023, "2022-23", np.where(t.y >= 2025, "2025-26", "2024"))
    t["d"] = np.where(t.side == "long", 1, -1)
    t["stop_usd"] = (t.entry_price - (t.exit_price.where(t.outcome == "stop"))).abs()
    return t


t2 = trades(2.0)
# stop distance from the entry bar's ATR (the engine sizes the stop off the signal bar's ATR)
atr = ind.atr.reindex(t2.entry_time, method="ffill").values
t2["stop_dist"] = atr * 2.0
t2["cost_R"] = COST / t2.stop_dist
t2["hold_h"] = (t2.exit_time - t2.entry_time).dt.total_seconds() / 3600


# after a 2x stop-out: would the same entry have reached a 3x-ATR target... and would a 3x stop have survived?
def after_stop(r):
    seg = bars.loc[r.entry_time:r.entry_time + pd.Timedelta(days=7)]
    d, e, a = r.d, r.entry_price, r.stop_dist / 2
    fav = (seg.high - e) if d == 1 else (e - seg.low)
    adv = (e - seg.low) if d == 1 else (seg.high - e)
    hit_tgt2 = fav >= 6 * a     # the 2x trade's own target (3R of 2 ATR)
    hit_stop3 = adv >= 3 * a
    first_tgt = hit_tgt2.idxmax() if hit_tgt2.any() else None
    first_stop3 = hit_stop3.idxmax() if hit_stop3.any() else None
    later_target = first_tgt is not None and (first_stop3 is None or first_tgt < first_stop3)
    return pd.Series(dict(later_reached_target=later_target, survives_3x=first_stop3 is None or (first_tgt is not None and first_tgt < first_stop3)))


stops = t2[t2.outcome == "stop"].copy()
stops[["later_reached_target", "survives_3x"]] = stops.apply(after_stop, axis=1)

# market regime per year: price range, trend efficiency, H4 trend flips, D1 ADX
h4 = low.set_index("ts").resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
h4 = add_trend_indicator(h4, 30)
h4["dir"] = np.sign(h4.close - h4.ema_trend)
d1 = low.set_index("ts").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
tr = pd.concat([d1.high - d1.low, (d1.high - d1.close.shift()).abs(), (d1.low - d1.close.shift()).abs()], axis=1).max(axis=1)
up = d1.high.diff(); dn = -d1.low.diff()
pdm = up.where((up > dn) & (up > 0), 0.); ndm = dn.where((dn > up) & (dn > 0), 0.)
atr14 = tr.ewm(alpha=1 / 14).mean(); pdi = 100 * pdm.ewm(alpha=1 / 14).mean() / atr14; ndi = 100 * ndm.ewm(alpha=1 / 14).mean() / atr14
d1["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean()

print("=== market per year")
for y, g in d1.groupby(d1.index.year):
    eff = abs(g.close.iloc[-1] - g.close.iloc[0]) / g.close.diff().abs().sum()
    hy = h4[h4.index.year == y]
    flips = int((hy.dir.diff().abs() > 0).sum())
    print(f"{y}: price {g.close.iloc[0]:7.1f} -> {g.close.iloc[-1]:7.1f} ({(g.close.iloc[-1]/g.close.iloc[0]-1)*100:+5.1f}%), "
          f"trend efficiency {eff:.3f}, mean D1 ADX {g.adx.mean():4.1f}, days ADX>25 {(g.adx > 25).mean()*100:3.0f}%, H4 EMA30 side flips {flips}")

print("\n=== trades per period (2x ATR, RR 3)")
for p, g in t2.groupby("period"):
    print(f"{p}: n={len(g)} win={(g.R > 0).mean():.2f} R={g.R.sum():6.1f} | avg stop ${g.stop_dist.mean():5.2f} "
          f"({(g.stop_dist / g.entry_price).mean()*100:.2f}% of price) | cost per trade {g.cost_R.mean():.3f}R -> total costs {g.cost_R.sum():5.1f}R | "
          f"median hold {g.hold_h.median():4.1f}h | exits {g.outcome.value_counts().to_dict()}")
    for s, gs in g.groupby("side"):
        print(f"      {s:5}: n={len(gs):3d} win={(gs.R > 0).mean():.2f} R={gs.R.sum():6.1f}")

print("\n=== stopped-out 2x trades: did price later go to the target anyway?")
for p, g in stops.groupby(t2.loc[stops.index, "period"]):
    print(f"{p}: stops={len(g)} | later reached the 2x target (within 7 days) {g.later_reached_target.mean()*100:4.1f}% | "
          f"would have survived a 3x stop and hit that target {g.survives_3x.mean()*100:4.1f}% | median time to stop {g.hold_h.median() if 'hold_h' in g else t2.loc[g.index].hold_h.median():.1f}h")

print("\n=== loss streaks and quick re-entries")
for p, g in t2.groupby("period"):
    g = g.sort_values("entry_time")
    gap = (g.entry_time - g.exit_time.shift()).dt.total_seconds() / 60
    quick = (gap <= 30) & (g.outcome.shift() == "stop") & (g.side == g.side.shift())
    print(f"{p}: re-entries in the same direction within 30 min of a stop: {int(quick.sum())} ({quick.mean()*100:.0f}% of trades), their R {g.R[quick].sum():6.1f}")

print("\n=== 3x ATR for comparison")
t3 = trades(3.0)
for p, g in t3.groupby("period"):
    print(f"{p}: n={len(g)} win={(g.R > 0).mean():.2f} R={g.R.sum():6.1f}")
