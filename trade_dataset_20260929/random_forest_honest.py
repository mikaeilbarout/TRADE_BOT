"""Honest re-check of the random forest on the reduced dataset.

feature_selection.py chose the 12 columns using ALL 698 trades -- including the newest 30%
that random_forest.py then uses as its test set, so that test score is optimistic.
Here the SAME selection rule is applied to the training trades only (oldest 70%, its own
4 time blocks), the forest is trained on those columns, and scored on the untouched test.
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
full = pd.read_parquet(os.path.join(HERE, "donchian_dataset_full.parquet")).sort_values("trade_id").reset_index(drop=True)
split = int(len(full) * .7)
train, test = full.iloc[:split].reset_index(drop=True), full.iloc[split:].reset_index(drop=True)
CANDIDATES = [c for c in full.columns if c not in {"trade_id", "entry_price", "is_tick_sim", "outcome", "win", "R"}]
rng = np.random.default_rng(30092026)


def auc(yy, s):
    r = pd.Series(s).rank().values; pos = np.asarray(yy) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --- same rule as feature_selection.py, training trades only
y = train.win.values; ry = train.R.rank().values
blocks = pd.qcut(train.trade_id, 4, labels=False).values
perm_y = [rng.permutation(y) for _ in range(2000)]; perm_r = [rng.permutation(ry) for _ in range(2000)]
kept = []
for f in CANDIDATES:
    x = train[f].values; rx = pd.Series(x).rank().values
    a = auc(y, x); rho = np.corrcoef(rx, ry)[0, 1]
    p_win = (sum(abs(auc(py, x) - .5) >= abs(a - .5) for py in perm_y) + 1) / 2001
    p_R = (sum(abs(np.corrcoef(rx, pr)[0, 1]) >= abs(rho) for pr in perm_r) + 1) / 2001
    same = sum(np.sign(auc(y[blocks == b], x[blocks == b]) - .5) == np.sign(a - .5) for b in range(4))
    if min(p_win, p_R) < .05 or (same >= 3 and abs(a - .5) >= .03):
        kept.append(f)
print(f"columns kept using the TRAINING trades only ({len(kept)}): {kept}")

# --- the forest (same code as random_forest.py)
sys.argv = [sys.argv[0]]
src = open(os.path.join(HERE, "random_forest.py"), encoding="utf-8").read()
funcs = src[src.index("def best_split"):src.index("X = t[FEATS]")]
FEATS = kept; MAX_FEATURES = max(1, int(np.sqrt(len(FEATS)))); N_TREES, SEED = 300, 30092026
exec(funcs)
X = full[FEATS].to_numpy(float); yy = full.win.to_numpy(int)
Xtr, ytr, Xte, yte = X[:split], yy[:split], X[split:], yy[split:]
inner = int(split * .7)
grid = []
for depth, leaf in itertools.product([2, 4, 6, 10, 20], [5, 15, 30]):
    f = forest(Xtr[:inner], ytr[:inner], depth, leaf)
    grid.append((auc(ytr[inner:], predict(f, Xtr[inner:])), depth, leaf))
_, depth, leaf = max(grid)
f = forest(Xtr, ytr, depth, leaf)
s_te = predict(f, Xte)
a_te = auc(yte, s_te)
seeds = [auc(yte, predict(forest(Xtr, ytr, depth, leaf, seed=s), Xte)) for s in (1, 2, 3, 4, 5)]
print(f"chosen depth {depth}, min leaf {leaf} | AUC test {a_te:.3f} | other seeds {[round(v, 3) for v in seeds]}")
tt = test.assign(score=s_te)
k = tt[tt.score >= .27]
print(f"rule score >= 27%: {len(k)} trades, win {k.win.mean():.3f}, total R {k.R.sum():+.1f}  (all {len(tt)}, total R {tt.R.sum():+.1f})")
json.dump(dict(kept_on_train=kept, depth=depth, min_leaf=leaf, auc_test=a_te, other_seeds=seeds,
               rule=dict(trades=len(k), R=round(float(k.R.sum()), 1), all_R=round(float(tt.R.sum()), 1))),
          open(os.path.join(HERE, "random_forest_honest.json"), "w"), indent=1)
