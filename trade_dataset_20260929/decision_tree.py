"""Decision tree (CART, Gini) for `win` on donchian_dataset.parquet -- numpy only.

Time-ordered, no shuffling: train = oldest 70% of trades, test = newest 30%.
Depth and minimum leaf size are chosen INSIDE the training part (fit on its oldest 70%,
score AUC on its newest 30%), then the tree is refit on the whole training part and
evaluated once on the test part. Output: printed tree + report, decision_tree.json.
"""
import os, json, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_parquet(os.path.join(HERE, "donchian_dataset.parquet")).sort_values("trade_id").reset_index(drop=True)
FEATS = [c for c in t.columns if c not in {"trade_id", "entry_price", "is_tick_sim", "outcome", "win", "R"}]
BREAK_EVEN = 0.27          # win rate at which this bot's average win (2.9R) and loss (-1.07R) cancel out


def gini(y):
    p = y.mean() if len(y) else 0.
    return 2 * p * (1 - p)


def fit(X, y, depth, min_leaf, d=0):
    node = dict(n=len(y), p=float(y.mean()))
    if d >= depth or len(y) < 2 * min_leaf or node["p"] in (0., 1.):
        return node
    best = None
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable"); xs, ys = X[order, j], y[order]
        csum = np.cumsum(ys); n = len(ys); tot = csum[-1]
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i - 1] == xs[i]:
                continue
            nl, nr = i, n - i
            pl, pr = csum[i - 1] / nl, (tot - csum[i - 1]) / nr
            g = (nl * 2 * pl * (1 - pl) + nr * 2 * pr * (1 - pr)) / n
            if best is None or g < best[0]:
                best = (g, j, (xs[i - 1] + xs[i]) / 2)
    if best is None or best[0] >= gini(y) - 1e-12:
        return node
    g, j, thr = best
    m = X[:, j] <= thr
    node.update(feature=j, thr=float(thr), gain=float((gini(y) - g) * len(y)),
                left=fit(X[m], y[m], depth, min_leaf, d + 1), right=fit(X[~m], y[~m], depth, min_leaf, d + 1))
    return node


def predict(node, x):
    while "feature" in node:
        node = node["left"] if x[node["feature"]] <= node["thr"] else node["right"]
    return node["p"], id(node)


def auc(y, s):
    r = pd.Series(s).rank().values; pos = np.asarray(y) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def show(node, depth=0, lines=None):
    lines = [] if lines is None else lines
    pad = "    " * depth
    if "feature" not in node:
        lines.append(f"{pad}-> leaf: {node['n']} trades, win rate {node['p']:.2f}")
        return lines
    f = FEATS[node["feature"]]
    lines.append(f"{pad}if {f} <= {node['thr']:.4g}:  ({node['n']} trades, win {node['p']:.2f})")
    show(node["left"], depth + 1, lines)
    lines.append(f"{pad}else ({f} > {node['thr']:.4g}):")
    show(node["right"], depth + 1, lines)
    return lines


def importance(node, imp):
    if "feature" in node:
        imp[FEATS[node["feature"]]] = imp.get(FEATS[node["feature"]], 0.) + node["gain"]
        importance(node["left"], imp); importance(node["right"], imp)
    return imp


X = t[FEATS].to_numpy(float); y = t.win.to_numpy(int)
split = int(len(t) * .7)
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
inner = int(split * .7)

print(f"train: trades 1-{split} (oldest 70%) | test: trades {split + 1}-{len(t)} (newest 30%) | features {len(FEATS)}")
print("\nhyper-parameters chosen inside the training part (validation AUC):")
grid = []
for depth, leaf in itertools.product([1, 2, 3, 4, 5, 6], [15, 30, 50]):
    tree = fit(Xtr[:inner], ytr[:inner], depth, leaf)
    s = [predict(tree, x)[0] for x in Xtr[inner:]]
    grid.append(dict(depth=depth, min_leaf=leaf, val_auc=auc(ytr[inner:], s)))
g = pd.DataFrame(grid)
print(g.pivot(index="depth", columns="min_leaf", values="val_auc").round(3).to_string())
best = g.sort_values("val_auc", ascending=False).iloc[0]
depth, leaf = int(best.depth), int(best.min_leaf)
print(f"-> chosen: depth {depth}, min leaf {leaf}")

tree = fit(Xtr, ytr, depth, leaf)
print("\nTREE (fitted on the whole training part):")
print("\n".join(show(tree)))
tr_s = [predict(tree, x)[0] for x in Xtr]
te_pred = [predict(tree, x) for x in Xte]
te_s = [p for p, _ in te_pred]
a_tr, a_te = auc(ytr, tr_s), auc(yte, te_s)
print(f"\nAUC train {a_tr:.3f} | AUC test {a_te:.3f}  (0.5 = no information)")

test = t.iloc[split:].assign(pred=te_s)
by_leaf = test.groupby("pred").agg(trades=("R", "size"), real_win=("win", "mean"), R_total=("R", "sum")).round(3)
print("\ntest trades by leaf (pred = win rate the leaf had in training):")
print(by_leaf.to_string())
take = test[test.pred >= BREAK_EVEN]
print(f"\nrule 'take a trade only if its leaf had win rate >= {BREAK_EVEN:.0%} in training' on the test trades:")
print(f"  all test trades: {len(test)}, win {test.win.mean():.3f}, total R {test.R.sum():+.1f}")
print(f"  taken:           {len(take)}, win {take.win.mean() if len(take) else float('nan'):.3f}, total R {take.R.sum():+.1f}")
imp = importance(tree, {})
tot = sum(imp.values()) or 1
print("\nfeature importance (share of the Gini gain):", {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])})
json.dump(dict(features=FEATS, chosen=dict(depth=depth, min_leaf=leaf), validation=grid, auc_train=a_tr, auc_test=a_te,
               tree=show(tree), test_by_leaf=by_leaf.reset_index().to_dict("records"),
               rule=dict(threshold=BREAK_EVEN, all_R=round(float(test.R.sum()), 1), taken=len(take), taken_R=round(float(take.R.sum()), 1)),
               importance={k: round(v / tot, 4) for k, v in imp.items()}),
          open(os.path.join(HERE, "decision_tree.json"), "w"), indent=1, default=float)
