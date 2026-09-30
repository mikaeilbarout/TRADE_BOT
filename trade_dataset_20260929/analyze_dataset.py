"""Exploratory analysis of donchian_dataset.parquet: which pre-entry columns relate to the
result, how stable that is over time, and whether a simple model trained on the older
trades predicts the newer ones. numpy/pandas only. Output: printed report + analysis.json.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_parquet(os.path.join(HERE, "donchian_dataset.parquet")).sort_values("trade_id").reset_index(drop=True)
LABELS = ["outcome", "win", "R"]
NOT_FEATURES = {"trade_id", "entry_price", "is_tick_sim"} | set(LABELS)   # entry_price is just a clock (gold price level)
FEATS = [c for c in t.columns if c not in NOT_FEATURES]
t["block"] = pd.qcut(t.trade_id, 4, labels=["Q1 oldest", "Q2", "Q3", "Q4 newest"])
rep = {}


def auc(y, s):
    r = pd.Series(s).rank().values; pos = np.asarray(y) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


print("=" * 100)
print(f"ROWS {len(t)} | win rate {t.win.mean():.3f} | outcomes {t.outcome.value_counts().sort_index().to_dict()} "
      f"| R mean {t.R.mean():+.3f} median {t.R.median():+.3f} total {t.R.sum():+.1f}")
print("R of a win: mean %.2f | R of a loss: mean %.2f | break-even win rate at these sizes: %.1f%%" % (
    t.R[t.win == 1].mean(), t.R[t.win == 0].mean(), 100 * -t.R[t.win == 0].mean() / (t.R[t.win == 1].mean() - t.R[t.win == 0].mean())))
for col in ("block", "is_tick_sim", "is_long", "session", "weekday"):
    g = t.groupby(col, observed=True).agg(n=("R", "size"), win=("win", "mean"), R_mean=("R", "mean"), R_total=("R", "sum")).round(3)
    print(f"\n--- by {col}\n{g.to_string()}")
    rep[f"by_{col}"] = g.reset_index().astype({col: str}).to_dict("records")

print("\n" + "=" * 100)
print("FEATURES vs RESULT  (rho = rank correlation with R; AUC for win, 0.5 = no information;")
print("  stable = same direction of the AUC in all 4 time blocks; terciles = mean R low/mid/high)")
rows = []
for f in FEATS:
    x = t[f]
    blocks = [auc(g.win, g[f]) for _, g in t.groupby("block", observed=True)]
    signs = [np.sign(b - .5) for b in blocks]
    if x.nunique() > 6:
        q = pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"])
    else:
        q = x
    terc = t.groupby(q, observed=True).R.mean().round(3).to_dict()
    rows.append(dict(feature=f, rho=spearman(x, t.R), auc=auc(t.win, x), blocks=[round(b, 3) for b in blocks],
                     stable=bool(len(set(signs)) == 1 and 0 not in signs), terciles=terc))
fr = pd.DataFrame(rows)
fr["strength"] = (fr.auc - .5).abs()
fr = fr.sort_values("strength", ascending=False)
for _, r in fr.iterrows():
    print(f"  {r.feature:26} rho {r.rho:+.3f} | AUC {r.auc:.3f} | by block {r.blocks} {'STABLE' if r.stable else ''} | mean R {r.terciles}")
rep["features"] = fr.drop(columns="strength").to_dict("records")

print("\n" + "=" * 100)
print("REDUNDANT PAIRS (|rank correlation| >= 0.7)")
c = t[FEATS].rank().corr()
pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(FEATS) for b in FEATS[i + 1:] if abs(c.loc[a, b]) >= .7]
for a, b, v in sorted(pairs, key=lambda p: -abs(p[2])):
    print(f"  {a} ~ {b}: {v:+.2f}")
rep["redundant_pairs"] = [dict(a=a, b=b, rho=round(float(v), 3)) for a, b, v in pairs]

print("\n" + "=" * 100)
print("MODEL: logistic regression for `win`, trained on the oldest 70% of trades, tested on the newest 30%")
split = int(len(t) * .7)
X = t[FEATS].astype(float)
mu, sd = X.iloc[:split].mean(), X.iloc[:split].std().replace(0, 1)
Z = np.c_[np.ones(len(X)), ((X - mu) / sd).values]
y = t.win.values
w = np.zeros(Z.shape[1])
for _ in range(4000):
    p = 1 / (1 + np.exp(-Z[:split] @ w))
    w -= .1 * (Z[:split].T @ (p - y[:split]) / split + .05 * np.r_[0, w[1:]])
score = 1 / (1 + np.exp(-Z @ w))
a_tr, a_te = auc(y[:split], score[:split]), auc(y[split:], score[split:])
print(f"  AUC train {a_tr:.3f} | AUC test {a_te:.3f}")
te = t.iloc[split:].assign(score=score[split:])
res = []
for cut in (.1, .2, .3):
    thr = te.score.quantile(cut)
    keep = te[te.score > thr]
    res.append(dict(skip_lowest=cut, trades=len(keep), win=round(keep.win.mean(), 3), R_total=round(keep.R.sum(), 1)))
    print(f"  skip the {int(cut*100)}% lowest-scored test trades -> {len(keep)} trades, win {keep.win.mean():.3f}, "
          f"total R {keep.R.sum():+.1f} (all test trades: {len(te)}, win {te.win.mean():.3f}, total R {te.R.sum():+.1f})")
coef = pd.Series(w[1:], index=FEATS).sort_values(key=abs, ascending=False)
print("  largest weights (standardised):", coef.head(6).round(3).to_dict())
rep["model"] = dict(auc_train=a_tr, auc_test=a_te, skip=res, weights=coef.round(4).to_dict(), test_trades=len(te),
                    test_R=round(float(te.R.sum()), 1))
json.dump(rep, open(os.path.join(HERE, "analysis.json"), "w"), indent=1, default=str)
