"""4-year check with M15 bar tick_volume / spread (403 live-setting trades from the bar simulator) -- bigger sample, coarser data."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
import counter_move_filters as cm, spx_filter as sf
rd = cm.rd
df = pd.read_parquet(rd.DATA).rename(columns={"bar_time": "ts"}).set_index("ts")
v = df.tick_volume.astype(float); med12 = v.shift().rolling(12).median(); med48 = v.shift(5).rolling(48).median()
F = pd.DataFrame({"act_signal": v / med12, "act_prev": v.shift() / med12, "act_last4_vs_day": v.rolling(4).mean() / med48,
                  "vol_trend": v.rolling(5).apply(lambda x: np.corrcoef(np.arange(5), x)[0, 1], raw=True),
                  "spread_rel": df.avg_spread_price / df.avg_spread_price.shift().rolling(96).median(), "range_rel": (df.high - df.low) / (df.high - df.low).shift().rolling(12).median()})
low = df.reset_index()[["ts", "open", "high", "low", "close"]]
t = rd.run(low, dict(sf.NEW, entry_filter=sf.spx_filter)); t["sig"] = pd.to_datetime(t.entry_time).dt.floor("15min") - pd.Timedelta(minutes=15)
D = t.join(F, on="sig").dropna(subset=list(F.columns)); D["win"] = (D.R > 0).astype(int)
print(f"trades {len(t)} usable {len(D)} wins {D.win.sum()} losses {(1-D.win).sum()}"); res = []
for c in F.columns:
    a, b = D[D.win == 1][c], D[D.win == 0][c]; u = stats.mannwhitneyu(a, b)
    res.append(dict(feature=c, win_median=round(float(a.median()), 3), loss_median=round(float(b.median()), 3), auc=round(float(u.statistic / (len(a) * len(b))), 3), p=float(u.pvalue), spearman_R=round(float(stats.spearmanr(D[c], D.R)[0]), 3)))
R = pd.DataFrame(res).sort_values("p"); n = len(R); R["p_bh"] = np.minimum.accumulate((R.p * n / np.arange(1, n + 1))[::-1])[::-1].clip(upper=1)
print(R.round(3).to_string(index=False))
