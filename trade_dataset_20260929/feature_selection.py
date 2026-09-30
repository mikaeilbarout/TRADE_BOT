"""Which columns of donchian_dataset relate to the result at all, and which are pure chance?

Rule (fixed before running). A column is KEPT if either
  (a) a permutation test (5,000 shuffles of the result) gives p < 0.05 for its link with
      `win` (AUC) or with `R` (rank correlation), or
  (b) its link with `win` points the same way in at least 3 of the 4 time blocks
      (oldest to newest quarter of the trades) AND the pooled AUC is at least 0.03 from 0.5.
Everything else is REMOVED as indistinguishable from chance. If two kept columns carry the
same information (|rank correlation| >= 0.9), only the stronger one stays.
Not features and never removed: trade_id, is_tick_sim, outcome, win, R.
Caveat: the columns are judged on the same 698 trades they will be used with.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "donchian_dataset_full.parquet")
t = pd.read_parquet(SRC).sort_values("trade_id").reset_index(drop=True)
FIXED = ["trade_id", "is_tick_sim", "outcome", "win", "R"]
FEATS = [c for c in t.columns if c not in FIXED]
rng = np.random.default_rng(30092026)
N_PERM = 5000
blocks = pd.qcut(t.trade_id, 4, labels=False).values
y = t.win.values.astype(float); R = t.R.values
ry = pd.Series(R).rank().values


def auc_from_ranks(rank_x, yy):
    pos = yy == 1; n1 = pos.sum(); n0 = len(yy) - n1
    return (rank_x[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
perm_y = np.array([rng.permutation(y) for _ in range(N_PERM)])
perm_r = np.array([rng.permutation(ry) for _ in range(N_PERM)])
for f in FEATS:
    rx = pd.Series(t[f].values).rank().values
    a = auc_from_ranks(rx, y)
    rho = np.corrcoef(rx, ry)[0, 1]
    # permutation null: shuffle the result, keep the column
    pos_sum = perm_y @ rx; n1 = y.sum(); n0 = len(y) - n1
    null_a = (pos_sum - n1 * (n1 + 1) / 2) / (n1 * n0)
    p_win = (np.sum(np.abs(null_a - .5) >= abs(a - .5)) + 1) / (N_PERM + 1)
    rxc = rx - rx.mean()
    null_rho = (perm_r - perm_r.mean(axis=1, keepdims=True)) @ rxc / (np.sqrt(((perm_r - perm_r.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)) * np.sqrt((rxc ** 2).sum()))
    p_R = (np.sum(np.abs(null_rho) >= abs(rho)) + 1) / (N_PERM + 1)
    bl = []
    for b in range(4):
        m = blocks == b
        bl.append(auc_from_ranks(pd.Series(t[f].values[m]).rank().values, y[m]))
    main_sign = np.sign(a - .5)
    same = int(sum(np.sign(x - .5) == main_sign for x in bl)) if main_sign != 0 else 0
    keep_a = min(p_win, p_R) < 0.05
    keep_b = same >= 3 and abs(a - .5) >= 0.03
    rows.append(dict(column=f, auc_win=round(a, 3), rho_R=round(rho, 3), p_win=round(p_win, 4), p_R=round(p_R, 4),
                     block_auc=[round(x, 3) for x in bl], same_direction_blocks=same,
                     keep=bool(keep_a or keep_b), why=("p<0.05" if keep_a else "") + (" stable" if keep_b else "")))
res = pd.DataFrame(rows).sort_values(["keep", "p_win"], ascending=[False, True])

# redundancy among kept columns
kept = list(res[res.keep].column)
corr = t[kept].rank().corr().abs() if kept else pd.DataFrame()
strength = {r.column: max(abs(r.auc_win - .5), abs(r.rho_R) / 2) for r in res.itertuples()}
dropped_redundant = {}
for i, a in enumerate(kept):
    for b in kept[i + 1:]:
        if a in dropped_redundant or b in dropped_redundant:
            continue
        if corr.loc[a, b] >= .9:
            weak = a if strength[a] < strength[b] else b
            dropped_redundant[weak] = dict(duplicate_of=b if weak == a else a, rho=round(float(corr.loc[a, b]), 3))
res.loc[res.column.isin(dropped_redundant), ["keep", "why"]] = [False, "duplicate of a stronger column"]

pd.set_option("display.width", 220)
print(res.to_string(index=False))
keep_cols = list(res[res.keep].column)
remove_cols = list(res[~res.keep].column)
print(f"\nKEEP ({len(keep_cols)}): {keep_cols}")
print(f"REMOVE ({len(remove_cols)}): {remove_cols}")
json.dump(dict(rule=__doc__, permutations=N_PERM, results=res.to_dict("records"), keep=keep_cols, remove=remove_cols,
               redundant=dropped_redundant), open(os.path.join(HERE, "feature_selection.json"), "w"), indent=1, default=str)
