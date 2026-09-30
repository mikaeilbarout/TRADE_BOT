"""Logistic regression for `win`, numpy only, on the current dataset -- twice:
  1. standard: the columns of donchian_dataset.parquet (11 of them chosen with ALL trades,
     so this run is somewhat optimistic),
  2. honest: columns chosen with the selection rule on the TRAINING trades only
     (from donchian_dataset_full.parquet), test trades untouched.
Time-ordered: train = oldest 70%, test = newest 30%. The L2 penalty is chosen inside the
training part (fit on its oldest 70%, validation AUC on its newest 30%).
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {"trade_id", "entry_price", "is_tick_sim", "outcome", "win", "R"}
LAMBDAS = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
BREAK_EVEN = 0.27


def auc(y, s):
    r = pd.Series(s).rank().values; pos = np.asarray(y) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit(X, y, lam, iters=6000, lr=0.1):
    Z = np.c_[np.ones(len(X)), X]; w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + lam * np.r_[0, w[1:]])
    return w


def prob(w, X):
    return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), X] @ w))


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
    mu, sd = X[:split].mean(0), X[:split].std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    # penalty chosen inside the training part (scaling from the inner part only)
    mi, si = X[:inner].mean(0), X[:inner].std(0); si[si == 0] = 1
    val = {lam: auc(y[inner:split], prob(fit((X[:inner] - mi) / si, y[:inner], lam), (X[inner:split] - mi) / si)) for lam in LAMBDAS}
    lam = max(val, key=val.get)
    w = fit(Xs[:split], y[:split], lam)
    s_tr, s_te = prob(w, Xs[:split]), prob(w, Xs[split:])
    a_tr, a_te = auc(y[:split], s_tr), auc(y[split:], s_te)
    test = df.iloc[split:].assign(score=s_te)
    test["fifth"] = pd.qcut(test.score.rank(method="first"), 5, labels=["1 lowest", "2", "3", "4", "5 highest"])
    q = test.groupby("fifth", observed=True).agg(trades=("R", "size"), real_win=("win", "mean"), R_total=("R", "sum")).round(3)
    k = test[test.score >= BREAK_EVEN]
    print(f"\n=== {label}: {len(feats)} columns | penalty {lam} (validation AUC {', '.join(f'{l}:{v:.3f}' for l, v in val.items())})")
    print(f"AUC train {a_tr:.3f} | AUC test {a_te:.3f}  (0.5 = no information)")
    print(q.to_string())
    print(f"rule score >= {BREAK_EVEN:.0%}: {len(k)} trades, win {k.win.mean():.3f}, total R {k.R.sum():+.1f}   (all {len(test)}, win {test.win.mean():.3f}, total R {test.R.sum():+.1f})")
    coef = pd.Series(w[1:], index=feats).sort_values(key=abs, ascending=False)
    print("largest weights (per 1 std; + = more wins):", coef.head(8).round(3).to_dict())
    return dict(columns=feats, penalty=lam, validation=val, auc_train=a_tr, auc_test=a_te,
                by_fifth=q.reset_index().astype({"fifth": str}).to_dict("records"),
                rule=dict(trades=len(k), R=round(float(k.R.sum()), 1), all_R=round(float(test.R.sum()), 1)),
                weights=coef.round(4).to_dict())


main = pd.read_parquet(os.path.join(HERE, "donchian_dataset.parquet"))
full = pd.read_parquet(os.path.join(HERE, "donchian_dataset_full.parquet")).sort_values("trade_id").reset_index(drop=True)
out = {"standard": run(main, [c for c in main.columns if c not in EXCLUDE], "STANDARD (current dataset)")}
kept = select_on_train(full.iloc[:int(len(full) * .7)], [c for c in full.columns if c not in EXCLUDE], np.random.default_rng(30092026))
out["honest"] = run(full, kept, "HONEST (columns chosen on training trades only)")
json.dump(out, open(os.path.join(HERE, "logistic.json"), "w"), indent=1, default=float)
