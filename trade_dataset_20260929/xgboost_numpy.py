"""XGBoost-style gradient boosting for `win`, numpy only (the xgboost package is not installed).

Same algorithm as XGBoost's exact greedy tree booster with logistic loss: each round fits a
tree to the gradient g = p - y and hessian h = p(1 - p); split gain
  0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma,
leaf weight -G/(H+lambda), learning rate eta, min_child_weight on the hessian,
row subsampling and column subsampling per tree.

Time-ordered: train = oldest 70%, test = newest 30%. Hyper-parameters AND the number of
rounds are chosen inside the training part (fit on its oldest 70%, validation AUC on its
newest 30%). Run twice: the current dataset's columns (chosen with all trades -> optimistic)
and columns chosen on the training trades only (honest).
"""
import os, json, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {"trade_id", "entry_price", "is_tick_sim", "outcome", "win", "R"}
BREAK_EVEN, MAX_ROUNDS, SEED = 0.27, 300, 30092026


def auc(y, s):
    r = pd.Series(s).rank().values; pos = np.asarray(y) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build(X, g, h, cols, depth, lam, gamma, mcw):
    G, H = g.sum(), h.sum()
    leaf = (-G / (H + lam),)
    if depth == 0 or H < 2 * mcw:
        return leaf
    best = None
    for j in cols:
        o = np.argsort(X[:, j], kind="stable"); xs = X[o, j]
        gl = np.cumsum(g[o])[:-1]; hl = np.cumsum(h[o])[:-1]
        gr, hr = G - gl, H - hl
        gain = .5 * (gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - G ** 2 / (H + lam)) - gamma
        ok = (hl >= mcw) & (hr >= mcw) & (xs[:-1] != xs[1:])
        if not ok.any():
            continue
        gain = np.where(ok, gain, -np.inf); i = int(np.argmax(gain))
        if best is None or gain[i] > best[0]:
            best = (gain[i], j, (xs[i] + xs[i + 1]) / 2)
    if best is None or best[0] <= 0:
        return leaf
    _, j, thr = best
    m = X[:, j] <= thr
    return (None, j, thr, build(X[m], g[m], h[m], cols, depth - 1, lam, gamma, mcw),
            build(X[~m], g[~m], h[~m], cols, depth - 1, lam, gamma, mcw), best[0])


def tree_out(node, X):
    out = np.empty(len(X))
    for k, x in enumerate(X):
        n = node
        while len(n) > 1:
            n = n[3] if x[n[1]] <= n[2] else n[4]
        out[k] = n[0]
    return out


def boost(X, y, p, Xval=None, yval=None, seed=SEED):
    """returns trees and (if validation given) the validation AUC after every round"""
    rng = np.random.default_rng(seed)
    base = np.log(y.mean() / (1 - y.mean()))
    f = np.full(len(y), base); fv = None if Xval is None else np.full(len(Xval), base)
    trees, curve = [], []
    ncol = max(1, int(round(p["colsample"] * X.shape[1])))
    for _ in range(p["rounds"]):
        pr = 1 / (1 + np.exp(-f)); g, h = pr - y, pr * (1 - pr)
        rows = rng.random(len(y)) < p["subsample"]
        cols = rng.choice(X.shape[1], ncol, replace=False)
        tr = build(X[rows], g[rows], h[rows], cols, p["depth"], p["lambda"], p["gamma"], p["mcw"])
        trees.append(tr)
        f += p["eta"] * tree_out(tr, X)
        if Xval is not None:
            fv += p["eta"] * tree_out(tr, Xval); curve.append(auc(yval, fv))
    return (base, trees, p["eta"]), curve


def predict(model, X):
    base, trees, eta = model
    return base + eta * sum(tree_out(t, X) for t in trees)


def importance(model, feats):
    imp = dict.fromkeys(feats, 0.)
    def walk(n):
        if len(n) > 1:
            imp[feats[n[1]]] += n[5]; walk(n[3]); walk(n[4])
    for t in model[1]:
        walk(t)
    tot = sum(imp.values()) or 1
    return {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1]) if v > 0}


def select_on_train(train, cands, rng):
    y = train.win.values; ry = train.R.rank().values
    blocks = pd.qcut(train.trade_id, 4, labels=False).values
    perm_y = [rng.permutation(y) for _ in range(2000)]; perm_r = [rng.permutation(ry) for _ in range(2000)]
    kept = []
    for f in cands:
        x = train[f].values; rx = pd.Series(x).rank().values
        a = auc(y, x); rho = np.corrcoef(rx, ry)[0, 1]
        p_win = (sum(abs(auc(py, x) - .5) >= abs(a - .5) for py in perm_y) + 1) / 2001
        p_R = (sum(abs(np.corrcoef(rx, pr)[0, 1]) >= abs(rho) for pr in perm_r) + 1) / 2001
        same = sum(np.sign(auc(y[blocks == b], x[blocks == b]) - .5) == np.sign(a - .5) for b in range(4))
        if min(p_win, p_R) < .05 or (same >= 3 and abs(a - .5) >= .03):
            kept.append(f)
    return kept


def run(df, feats, label):
    df = df.sort_values("trade_id").reset_index(drop=True)
    split = int(len(df) * .7); inner = int(split * .7)
    X = df[feats].to_numpy(float); y = df.win.to_numpy(float)
    grid = []
    for depth, eta, mcw, lam in itertools.product([1, 2, 3, 4], [0.03, 0.1], [1, 5], [1.0, 10.0]):
        p = dict(depth=depth, eta=eta, mcw=mcw, **{"lambda": lam}, gamma=0.0, subsample=0.8, colsample=0.8, rounds=MAX_ROUNDS)
        _, curve = boost(X[:inner], y[:inner], p, X[inner:split], y[inner:split])
        r = int(np.argmax(curve)) + 1
        grid.append(dict(p, rounds=r, val_auc=curve[r - 1]))
    g = pd.DataFrame(grid).sort_values("val_auc", ascending=False)
    best = g.iloc[0].to_dict()
    p = {k: best[k] for k in ("depth", "eta", "mcw", "lambda", "gamma", "subsample", "colsample")}
    p["depth"] = int(p["depth"]); p["rounds"] = int(best["rounds"])
    model, _ = boost(X[:split], y[:split], p)
    s_tr, s_te = predict(model, X[:split]), predict(model, X[split:])
    a_tr, a_te = auc(y[:split], s_tr), auc(y[split:], s_te)
    seeds = [auc(y[split:], predict(boost(X[:split], y[:split], p, seed=s)[0], X[split:])) for s in (1, 2, 3, 4, 5)]
    test = df.iloc[split:].assign(score=1 / (1 + np.exp(-s_te)))
    test["fifth"] = pd.qcut(test.score.rank(method="first"), 5, labels=["1 lowest", "2", "3", "4", "5 highest"])
    q = test.groupby("fifth", observed=True).agg(trades=("R", "size"), real_win=("win", "mean"), R_total=("R", "sum")).round(3)
    k = test[test.score >= BREAK_EVEN]
    print(f"\n=== {label}: {len(feats)} columns")
    print("best validation settings:", {k_: best[k_] for k_ in ("depth", "eta", "mcw", "lambda", "rounds", "val_auc")})
    print(f"AUC train {a_tr:.3f} | AUC test {a_te:.3f} | other seeds {[round(v, 3) for v in seeds]}  (0.5 = no information)")
    print(q.to_string())
    print(f"rule score >= {BREAK_EVEN:.0%}: {len(k)} trades, win {k.win.mean() if len(k) else float('nan'):.3f}, total R {k.R.sum():+.1f}"
          f"   (all {len(test)}, win {test.win.mean():.3f}, total R {test.R.sum():+.1f})")
    imp = importance(model, feats)
    print("importance (share of split gain):", dict(list(imp.items())[:8]))
    return dict(columns=feats, settings=p, validation_top5=g.head(5).to_dict("records"), auc_train=a_tr, auc_test=a_te,
                auc_test_other_seeds=seeds, by_fifth=q.reset_index().astype({"fifth": str}).to_dict("records"),
                rule=dict(trades=len(k), R=round(float(k.R.sum()), 1), all_R=round(float(test.R.sum()), 1)), importance=imp)


main = pd.read_parquet(os.path.join(HERE, "donchian_dataset.parquet"))
full = pd.read_parquet(os.path.join(HERE, "donchian_dataset_full.parquet")).sort_values("trade_id").reset_index(drop=True)
out = {"standard": run(main, [c for c in main.columns if c not in EXCLUDE], "STANDARD (current dataset)")}
kept = select_on_train(full.iloc[:int(len(full) * .7)], [c for c in full.columns if c not in EXCLUDE], np.random.default_rng(30092026))
out["honest"] = run(full, kept, "HONEST (columns chosen on training trades only)")
json.dump(out, open(os.path.join(HERE, "xgboost_numpy.json"), "w"), indent=1, default=float)
