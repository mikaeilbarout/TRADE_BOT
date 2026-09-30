"""Can "fake breakouts" (losers that never got beyond +0.25R) be told apart?

Trades: 2x ATR stop, RR 3, no minimum stop (1,259 trades, 4.25 years), features from
stable_features.py. Part 1 -- BEFORE entry: each feature's AUC for fake vs the rest, per
year; plus a logistic model trained on 2022-06..2024 and scored on 2025-26 only.
Part 2 -- AFTER entry: how fakes and the rest look in the first 1-4 M15 bars after the
entry (back inside the channel? adverse move?) -- the raw material for an early exit.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import stable_features as sf   # builds the features on import (prints its own tables)

t = pd.read_csv(os.path.join(HERE, "stable_features_nomin.csv"), parse_dates=["entry_time", "exit_time"])
bars = sf.ind            # M15 with donchian levels, indexed by bar open time
d = np.where(t.side == "long", 1, -1)
stop = sf.ind.atr.reindex(t.entry_time).values * 2.0
level = np.where(d == 1, sf.ind.donchian_high.reindex(t.entry_time).values, sf.ind.donchian_low.reindex(t.entry_time).values)

mfe, early = [], []
idx = bars.index
for i, x in t.iterrows():
    j = idx.get_loc(x.entry_time)
    seg = bars.iloc[j + 1: idx.get_indexer([x.exit_time])[0] + 1]            # bars after the signal bar's close
    fav = ((seg.high - x.entry_price) if d[i] == 1 else (x.entry_price - seg.low)).max() if len(seg) else 0.
    mfe.append(fav / stop[i])
    nxt = bars.iloc[j + 1: j + 5]
    row = {}
    for n in (1, 2, 4):
        s = nxt.iloc[:n]
        row[f"back_inside_{n}"] = int(((s.close - level[i]) * d[i] < 0).any())          # a close back through the breakout level
        row[f"adverse_{n}_R"] = float((((x.entry_price - s.low) if d[i] == 1 else (s.high - x.entry_price)).max()) / stop[i])
        row[f"close_move_{n}_R"] = float(d[i] * (s.close.iloc[-1] - x.entry_price) / stop[i])
    early.append(row)
t["mfe_R"] = mfe
t = pd.concat([t, pd.DataFrame(early)], axis=1)
t["fake"] = ((t.R < 0) & (t.mfe_R < 0.25)).astype(int)
print(f"\n\n==== {len(t)} trades | fake breakouts {t.fake.sum()} ({t.fake.mean()*100:.0f}%), their R {t.R[t.fake == 1].sum():.1f} "
      f"| other losers {((t.R < 0) & (t.fake == 0)).sum()} | winners {(t.R > 0).sum()}")


def auc(y, s):
    s = pd.Series(s).rank().values; pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


print("\n==== PART 1: before entry -- AUC for 'fake' (0.5 = no information; >0.5 = higher value -> more fakes)")
feats = sf.NUMERIC + sf.BINARY
for f in feats:
    ok = t[f].notna()
    per = {int(y): round(auc(g.fake.values, g[f].values), 2) for y, g in t[ok].groupby("year") if g.fake.sum() >= 10}
    print(f"  {f:22} all {auc(t.fake[ok].values, t[f][ok].values):.2f} | by year {per}")

# logistic model: train 2022-2024, score 2025-26
X = t[feats].fillna(t[feats].median())
mu, sd = X[t.year <= 2024].mean(), X[t.year <= 2024].std().replace(0, 1)
Z = ((X - mu) / sd).values; Z = np.c_[np.ones(len(Z)), Z]
y = t.fake.values; tr = (t.year <= 2024).values
w = np.zeros(Z.shape[1])
for _ in range(3000):                                  # plain gradient descent, L2 0.01
    p = 1 / (1 + np.exp(-Z[tr] @ w))
    w -= 0.1 * (Z[tr].T @ (p - y[tr]) / tr.sum() + 0.01 * np.r_[0, w[1:]])
score = 1 / (1 + np.exp(-Z @ w))
print(f"\n  logistic model: AUC train (2022-24) {auc(y[tr], score[tr]):.2f} | AUC test (2025-26) {auc(y[~tr], score[~tr]):.2f}")
te = t[~tr].assign(score=score[~tr])
q = te.score.quantile(.8)
top = te[te.score >= q]
print(f"  2025-26: 20% of trades the model finds most 'fake'-like -> fake rate {top.fake.mean()*100:.0f}% (all {te.fake.mean()*100:.0f}%), "
      f"their R {top.R.sum():+.1f} (skipping them would change total R {te.R.sum():+.1f} -> {te.R.sum() - top.R.sum():+.1f})")

print("\n==== PART 2: after entry -- first bars (1 bar = 15 min)")
grp = np.where(t.fake == 1, "fake", np.where(t.R > 0, "winner", "other loser"))
cols = [c for c in t.columns if c.startswith(("back_inside", "adverse", "close_move"))]
print(t.groupby(grp)[cols].mean().round(2).T.to_string())
for n in (1, 2, 4):
    c = f"back_inside_{n}"
    g = t.groupby(t[c])
    print(f"\n  closed back inside the channel within {n} bar(s): "
          + " | ".join(f"{'yes' if k else 'no'}: n={len(v)}, fake {v.fake.mean()*100:.0f}%, win {(v.R > 0).mean()*100:.0f}%, avg R {v.R.mean():+.2f}" for k, v in g))
    print("    by year (avg R if yes / no): " + ", ".join(
        f"{int(y)}: {gy[gy[c] == 1].R.mean():+.2f}/{gy[gy[c] == 0].R.mean():+.2f}" for y, gy in t.groupby("year")))
t.to_csv(os.path.join(HERE, "fake_breakouts_trades.csv"), index=False)
