"""Random forest (bagged CART trees, random feature subset per split) for `win` on
donchian_dataset.parquet -- numpy only, fixed seed.

Same time-ordered protocol as decision_tree.py: train = oldest 70% of trades, test =
newest 30%. Depth and minimum leaf size are chosen inside the training part (fit on its
oldest 70%, validation AUC on its newest 30%); the forest is then refit on the whole
training part and scored once on the test part. Output: printed report, random_forest.json.
"""
import os, json, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_parquet(os.path.join(HERE, "donchian_dataset.parquet")).sort_values("trade_id").reset_index(drop=True)
FEATS = [c for c in t.columns if c not in {"trade_id", "entry_price", "is_tick_sim", "outcome", "win", "R"}]
N_TREES, SEED, BREAK_EVEN = 300, 30092026, 0.27
MAX_FEATURES = max(1, int(np.sqrt(len(FEATS))))


def best_split(X, y, feats, min_leaf):
    n = len(y); tot = y.sum(); best = None
    for j in feats:
        order = np.argsort(X[:, j], kind="stable"); xs, ys = X[order, j], y[order]
        csum = np.cumsum(ys)[:-1]
        nl = np.arange(1, n); nr = n - nl
        pl = csum / nl; pr = (tot - csum) / nr
        g = (nl * pl * (1 - pl) + nr * pr * (1 - pr)) * 2 / n
        ok = (nl >= min_leaf) & (nr >= min_leaf) & (xs[:-1] != xs[1:])
        if not ok.any():
            continue
        g = np.where(ok, g, np.inf); i = int(np.argmin(g))
        if best is None or g[i] < best[0]:
            best = (g[i], j, (xs[i] + xs[i + 1]) / 2)
    return best


def grow(X, y, depth, min_leaf, rng, d=0):
    p = float(y.mean())
    if d >= depth or len(y) < 2 * min_leaf or p in (0., 1.):
        return (p,)
    feats = rng.choice(X.shape[1], MAX_FEATURES, replace=False)
    b = best_split(X, y, feats, min_leaf)
    if b is None or b[0] >= 2 * p * (1 - p) - 1e-12:
        return (p,)
    g, j, thr = b
    m = X[:, j] <= thr
    gain = (2 * p * (1 - p) - g) * len(y)
    return (p, j, thr, gain, grow(X[m], y[m], depth, min_leaf, rng, d + 1), grow(X[~m], y[~m], depth, min_leaf, rng, d + 1))


def predict_tree(node, X):
    out = np.empty(len(X))
    for k, x in enumerate(X):
        n = node
        while len(n) > 1:
            n = n[4] if x[n[1]] <= n[2] else n[5]
        out[k] = n[0]
    return out


def forest(X, y, depth, min_leaf, seed=SEED):
    rng = np.random.default_rng(seed); trees = []
    for _ in range(N_TREES):
        idx = rng.integers(0, len(y), len(y))
        trees.append(grow(X[idx], y[idx], depth, min_leaf, rng))
    return trees


def predict(trees, X):
    return np.mean([predict_tree(tr, X) for tr in trees], axis=0)


def importance(trees):
    imp = np.zeros(len(FEATS))
    def walk(n):
        if len(n) > 1:
            imp[n[1]] += n[3]; walk(n[4]); walk(n[5])
    for tr in trees:
        walk(tr)
    return imp / imp.sum()


def auc(y, s):
    r = pd.Series(s).rank().values; pos = np.asarray(y) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


X = t[FEATS].to_numpy(float); y = t.win.to_numpy(int)
split = int(len(t) * .7); inner = int(split * .7)
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
print(f"train: trades 1-{split} | test: trades {split + 1}-{len(t)} | {len(FEATS)} features, {N_TREES} trees, {MAX_FEATURES} features tried per split")

grid = []
for depth, leaf in itertools.product([2, 4, 6, 10, 20], [5, 15, 30]):
    f = forest(Xtr[:inner], ytr[:inner], depth, leaf)
    grid.append(dict(depth=depth, min_leaf=leaf, val_auc=auc(ytr[inner:], predict(f, Xtr[inner:]))))
g = pd.DataFrame(grid)
print("\nvalidation AUC inside the training part:")
print(g.pivot(index="depth", columns="min_leaf", values="val_auc").round(3).to_string())
best = g.sort_values("val_auc", ascending=False).iloc[0]
depth, leaf = int(best.depth), int(best.min_leaf)
print(f"-> chosen: depth {depth}, min leaf {leaf}")

f = forest(Xtr, ytr, depth, leaf)
s_tr, s_te = predict(f, Xtr), predict(f, Xte)
a_tr, a_te = auc(ytr, s_tr), auc(yte, s_te)
print(f"\nAUC train {a_tr:.3f} | AUC test {a_te:.3f}  (0.5 = no information)")

# stability of the test result to the random seed
seeds = [auc(yte, predict(forest(Xtr, ytr, depth, leaf, seed=s), Xte)) for s in (1, 2, 3, 4, 5)]
print(f"test AUC with 5 other seeds: {[round(x, 3) for x in seeds]}")

test = t.iloc[split:].assign(score=s_te)
test["quintile"] = pd.qcut(test.score.rank(method="first"), 5, labels=["1 lowest", "2", "3", "4", "5 highest"])
q = test.groupby("quintile", observed=True).agg(trades=("R", "size"), score=("score", "mean"), real_win=("win", "mean"), R_total=("R", "sum")).round(3)
print("\ntest trades by forest score (fifths):")
print(q.to_string())
rules = []
for name, keep in ((f"score >= {BREAK_EVEN:.0%} (break-even)", test.score >= BREAK_EVEN),
                   ("skip the lowest 20%", test.quintile != "1 lowest"),
                   ("skip the lowest 40%", ~test.quintile.isin(["1 lowest", "2"]))):
    k = test[keep]
    rules.append(dict(rule=name, trades=len(k), win=round(float(k.win.mean()), 3) if len(k) else None, R_total=round(float(k.R.sum()), 1)))
    print(f"  {name:32} -> {len(k):3d} trades, win {k.win.mean() if len(k) else float('nan'):.3f}, total R {k.R.sum():+.1f}"
          f"   (all {len(test)}, win {test.win.mean():.3f}, total R {test.R.sum():+.1f})")
imp = pd.Series(importance(f), index=FEATS).sort_values(ascending=False)
print("\nfeature importance (share of the Gini gain):", imp.head(10).round(3).to_dict())
json.dump(dict(features=FEATS, trees=N_TREES, max_features=MAX_FEATURES, chosen=dict(depth=depth, min_leaf=leaf), validation=grid,
               auc_train=a_tr, auc_test=a_te, auc_test_other_seeds=seeds, test_by_quintile=q.reset_index().astype({"quintile": str}).to_dict("records"),
               rules=rules, importance=imp.round(4).to_dict()),
          open(os.path.join(HERE, "random_forest.json"), "w"), indent=1, default=float)
