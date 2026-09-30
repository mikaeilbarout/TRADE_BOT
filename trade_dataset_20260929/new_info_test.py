"""Do the NEW kinds of information (activity, dollar, silver, stocks) tell good from bad trades?

Trades: Donchian with the live settings since 2026-09-30 (N20, EMA30, 0.3%, 2 ATR min $8, RR 4).
Each new column is tested raw AND in the trade's direction (x * +1 for longs, x * -1 for shorts;
for EURUSD that means "dollar moving in gold's favour"), for `win`:
  AUC, permutation p (5,000 shuffles), and the AUC in each of the 4 time blocks.
A column counts as a real, usable signal only if p < 0.05 AND the direction holds in all 4 blocks.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_parquet(os.path.join(HERE, "donchian_dataset_full.parquet")).sort_values("trade_id").reset_index(drop=True)
NEW = ["signal_bar_volume_vs_20", "volume_4h_vs_5d", "spread_vs_5d", "eurusd_4h_pct", "eurusd_24h_pct", "eurusd_5d_pct",
       "usdjpy_24h_pct", "silver_24h_pct", "silver_minus_gold_24h_pct", "spx500_24h_pct", "spx500_5d_pct"]
DIRECTIONAL = ["eurusd_4h_pct", "eurusd_24h_pct", "eurusd_5d_pct", "usdjpy_24h_pct", "silver_24h_pct",
               "silver_minus_gold_24h_pct", "spx500_24h_pct", "spx500_5d_pct"]
d = np.where(t.is_long == 1, 1, -1)
cols = {c: t[c].to_numpy(float) for c in NEW}
for c in DIRECTIONAL:
    sign = -1 if c.startswith("usdjpy") else 1          # USDJPY up = dollar stronger = against gold
    cols[c + " (in trade direction)"] = t[c].to_numpy(float) * d * sign
y = t.win.to_numpy(int)
blocks = pd.qcut(t.trade_id, 4, labels=False).to_numpy()
rng = np.random.default_rng(30092026)
perm = np.array([rng.permutation(y) for _ in range(5000)])


def auc_rank(r, yy):
    pos = yy == 1; n1 = pos.sum(); n0 = len(yy) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
for name, x in cols.items():
    r = pd.Series(x).rank().to_numpy()
    a = auc_rank(r, y)
    n1 = y.sum(); n0 = len(y) - n1
    null = (perm @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
    p = (np.sum(np.abs(null - .5) >= abs(a - .5)) + 1) / 5001
    bl = [auc_rank(pd.Series(x[blocks == b]).rank().to_numpy(), y[blocks == b]) for b in range(4)]
    same = sum(np.sign(v - .5) == np.sign(a - .5) for v in bl)
    q = pd.qcut(pd.Series(x).rank(method="first"), 3, labels=["low", "mid", "high"])
    terc = t.groupby(q, observed=True).R.mean().round(3).to_dict()
    rows.append(dict(column=name, auc=round(a, 3), p=round(p, 4), blocks=[round(v, 3) for v in bl], same_dir_blocks=int(same),
                     usable=bool(p < .05 and same == 4), mean_R_by_tercile=terc))
res = pd.DataFrame(rows).sort_values("p")
pd.set_option("display.width", 220)
print(f"{len(t)} trades, win rate {y.mean():.3f}, total R {t.R.sum():+.1f}\n")
print(res.to_string(index=False))
print(f"\nusable (p<0.05 and same direction in all 4 blocks): {list(res[res.usable].column)}")
print(f"with {len(cols)} columns tested, about {0.05 * len(cols):.1f} would pass p<0.05 by chance alone")
json.dump(res.to_dict("records"), open(os.path.join(HERE, "new_info_test.json"), "w"), indent=1, default=str)
