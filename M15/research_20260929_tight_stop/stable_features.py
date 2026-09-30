"""Which signal-time features separate Donchian winners from losers IN EVERY YEAR?

Trades: look-ahead-free engine, 2x ATR stop, RR 3, 4.25 years (FundedNext M15).
Run twice: without the $8 minimum stop (1,259 trades -- enough per year to test) and
with it (the live setting, 670 trades) as a check.
For each feature: split into terciles (cut points from all trades), then per calendar
year compare the average R of the best and worst tercile. A feature is STABLE only if
the same tercile is better in every year with >= 20 trades per tercile, and the pooled
gap has a bootstrap 95% interval that excludes zero. Descriptive only.
"""
import sys, os, types, json
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

rng = np.random.default_rng(29092026)
low = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"})[["ts", "open", "high", "low", "close"]]
ind = add_donchian_indicators(low, 10, 14)
ind["atr_ratio"] = ind.atr / ind.atr.rolling(96 * 5).mean()           # ATR vs its ~5-day average
ind["body_pos"] = (ind.close - ind.low) / (ind.high - ind.low)         # where the signal bar closed in its range
ind["bar_range_atr"] = (ind.high - ind.low) / ind.atr
ind["mom_4h"] = ind.close.diff(16) / ind.atr                          # move over the last 4 hours, in ATR
ind = ind.set_index("ts")

h4 = low.set_index("ts").resample("4h", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
h4 = add_trend_indicator(h4, 30)
side = np.sign(h4.close - h4.ema_trend)
h4["trend_age"] = side.groupby((side != side.shift()).cumsum()).cumcount() + 1
h4["ema_dist_pct"] = (h4.close - h4.ema_trend) / h4.close * 100
h4["ema_slope"] = h4.ema_trend.pct_change(6) * 100                    # EMA change over 1 day, %
h4 = h4.assign(known=h4.index + pd.Timedelta(hours=4)).reset_index(drop=True).sort_values("known")

d1 = low.set_index("ts").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
tr = pd.concat([d1.high - d1.low, (d1.high - d1.close.shift()).abs(), (d1.low - d1.close.shift()).abs()], axis=1).max(axis=1)
up, dn = d1.high.diff(), -d1.low.diff()
pdm, ndm = up.where((up > dn) & (up > 0), 0.), dn.where((dn > up) & (dn > 0), 0.)
a14 = tr.ewm(alpha=1 / 14).mean(); pdi = 100 * pdm.ewm(alpha=1 / 14).mean() / a14; ndi = 100 * ndm.ewm(alpha=1 / 14).mean() / a14
d1["adx"] = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean()
d1["ema50"] = d1.close.ewm(span=50, adjust=False).mean()
d1 = d1.shift(1)                                                        # yesterday's completed day only
d1 = d1.assign(day=d1.index.normalize())


def features(t):
    t = t.copy()
    t["entry_time"] = pd.to_datetime(t.entry_time); t["exit_time"] = pd.to_datetime(t.exit_time)
    t = t.sort_values("entry_time").reset_index(drop=True)
    d = np.where(t.side == "long", 1, -1)
    s = ind.loc[t.entry_time]                                           # the signal bar (entry at its close)
    t["breakout_atr"] = d * (s.close.values - np.where(d == 1, s.donchian_high.values, s.donchian_low.values)) / s.atr.values
    t["channel_atr"] = (s.donchian_high.values - s.donchian_low.values) / s.atr.values
    t["atr_ratio"] = s.atr_ratio.values
    t["close_in_bar_dir"] = np.where(d == 1, s.body_pos.values, 1 - s.body_pos.values)
    t["bar_range_atr"] = s.bar_range_atr.values
    t["mom_4h_dir"] = d * s.mom_4h.values
    k = pd.merge_asof(pd.DataFrame({"known": t.entry_time + pd.Timedelta(minutes=15)}), h4, on="known", direction="backward")
    t["h4_trend_age"] = k.trend_age.values
    t["h4_ema_dist_pct"] = d * k.ema_dist_pct.values
    t["h4_ema_slope_dir"] = d * k.ema_slope.values
    dd = d1.reindex(t.entry_time.dt.normalize())
    t["d1_adx"] = dd.adx.values
    t["with_d1_ema50"] = (np.sign(dd.close.values - dd.ema50.values) == d).astype(int)
    t["hour"] = (t.entry_time + pd.Timedelta(minutes=15)).dt.hour     # broker server clock
    t["weekday"] = t.entry_time.dt.weekday
    t["is_long"] = (d == 1).astype(int)
    streak, out = 0, []
    for r in t.R:
        out.append(streak); streak = streak + 1 if r <= 0 else 0
    t["prior_loss_streak"] = out
    t["hours_since_last_exit"] = (t.entry_time - t.exit_time.shift()).dt.total_seconds() / 3600
    t["year"] = t.exit_time.dt.year
    return t


NUMERIC = ["breakout_atr", "channel_atr", "atr_ratio", "close_in_bar_dir", "bar_range_atr", "mom_4h_dir", "h4_trend_age",
           "h4_ema_dist_pct", "h4_ema_slope_dir", "d1_adx", "hour", "prior_loss_streak", "hours_since_last_exit"]
BINARY = ["with_d1_ema50", "is_long"]


def groups(t, f):
    if f in BINARY:
        return t[f].map({0: "no", 1: "yes"}), ["no", "yes"]
    if f == "prior_loss_streak":
        return pd.cut(t[f], [-1, 0, 2, 99], labels=["0", "1-2", "3+"]), ["0", "1-2", "3+"]
    q = t[f].quantile([1 / 3, 2 / 3]).values
    return pd.cut(t[f], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"]), ["low", "mid", "high"]


def analyse(t, label):
    print(f"\n######## {label}: {len(t)} trades, avg R {t.R.mean():+.3f}")
    out = {}
    for f in NUMERIC + BINARY:
        g, labs = groups(t, f)
        pooled = t.groupby(g, observed=True).R.agg(["size", "mean"])
        hi, lo = pooled["mean"].idxmax(), pooled["mean"].idxmin()
        years = {}
        for y, ty in t.groupby("year"):
            gy = g[ty.index]
            m_ = ty.groupby(gy, observed=True).R.agg(["size", "mean"])
            if all(m_.reindex([hi, lo])["size"].fillna(0) >= 20):
                years[int(y)] = round(float(m_.loc[hi, "mean"] - m_.loc[lo, "mean"]), 3)
        a, b = t.R[g == hi].values, t.R[g == lo].values
        boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
        ci = np.percentile(boots, [2.5, 97.5])
        same = sum(v > 0 for v in years.values())
        stable = len(years) >= 3 and same == len(years) and ci[0] > 0
        out[f] = dict(best=str(hi), worst=str(lo), pooled={str(k): [int(v["size"]), round(float(v["mean"]), 3)] for k, v in pooled.iterrows()},
                      gap_by_year=years, ci=[round(float(x), 3) for x in ci], stable=stable)
        cut = "" if f in BINARY or f == "prior_loss_streak" else f" (cuts {t[f].quantile(1/3):.2f} / {t[f].quantile(2/3):.2f})"
        print(f"{'STABLE ' if stable else '       '}{f:22} best={hi:>4} worst={lo:>4} | pooled "
              + " ".join(f"{k}:{int(v['size'])}/{v['mean']:+.2f}" for k, v in pooled.iterrows())
              + f" | gap by year {years} ({same}/{len(years)} same sign) | 95% CI [{ci[0]:+.2f},{ci[1]:+.2f}]{cut}")
    return out


BASE = dict(n_period=10, ema_trend_period=30, min_trend_strength_pct=.5, atr_stop_multiplier=2.0, reward_risk_ratio=3.0)
res = {}
for label, extra in (("no minimum stop", {}), ("live: minimum stop $8", {"min_stop_dollars": 8.0})):
    t = features(rd.run(low, dict(BASE, **extra)))
    t.to_csv(os.path.join(HERE, f"stable_features_{'nomin' if not extra else 'min8'}.csv"), index=False)
    res[label] = analyse(t, label)
json.dump(res, open(os.path.join(HERE, "stable_features.json"), "w"), indent=1, default=str)
