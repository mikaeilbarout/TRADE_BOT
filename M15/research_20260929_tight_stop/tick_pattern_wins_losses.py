"""Do the ticks BEFORE the signal differ between winning and losing Donchian trades? (6-month real ticks, live settings)
All features use only data up to the signal-bar close. Win = net USD > 0. Mann-Whitney + AUC, Benjamini-Hochberg over all features."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
import counter_move_filters as cm
import spx_filter as sf
sys.path.insert(0, cm.COMB); sys.path.insert(0, os.path.join(cm.COMB, "SLP2"))
import combined_tick_backtest as ctb, pyarrow.parquet as pq

def trades():
    meta = pq.ParquetFile(ctb.TICKS)
    start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
    end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - ctb.BAR
    frame = ctb.load_m15(); frame = frame[frame.bar_time < end + ctb.BAR].reset_index(drop=True)
    ctb.D_N, ctb.D_EMA, ctb.D_STRENGTH, ctb.D_ATR_MULT, ctb.D_RR, ctb.D_MIN_STOP = 20, 30, 0.3, 2.0, 4.0, 8.0
    ctb.D_ENTRY_FILTER = sf.spx_filter
    return ctb.donchian(frame, start, end)

def feats(m, t0, d):
    """t0 = signal-bar close (= entry bar open); d = +1 long / -1 short"""
    idx = m.index
    def win(a, b):  # minutes [t0-a, t0-b)
        return m.loc[(idx >= t0 - pd.Timedelta(minutes=a)) & (idx < t0 - pd.Timedelta(minutes=b))]
    sig, prev, last5, first10 = win(15, 0), win(30, 15), win(5, 0), win(15, 5)
    day = m.loc[(idx >= t0 - pd.Timedelta(hours=24)) & (idx < t0 - pd.Timedelta(minutes=15))]
    day15 = day.n.rolling(15).sum().iloc[14::15]; med15 = day15.median()
    med1 = win(60, 15).n.median(); spmed = day.sp_mean.median()
    if len(sig) < 10 or len(day) < 600 or not med15 or not med1: return None
    ud = sig.up.sum() - sig.down.sum(); tot = max(sig.up.sum() + sig.down.sum(), 1)
    cnts = [win(15 * (k + 1), 15 * k).n.sum() for k in range(1, 5)][::-1]
    return dict(act15=sig.n.sum() / med15, act_prev15=prev.n.sum() / med15, act60=win(60, 0).n.sum() / (4 * med15),
                burst=sig.n.max() / med1, accel5=(last5.n.sum() / 5) / max(first10.n.sum() / 10, 1),
                imbalance=d * ud / tot, spread_rel=sig.sp_mean.mean() / spmed, spread_max=sig.sp_max.max(),
                vol_trend=np.corrcoef(np.arange(5), cnts + [sig.n.sum()])[0, 1], quiet_before=prev.n.sum() / max(sig.n.sum(), 1),
                zero_minutes=int((sig.n < 30).sum()))

def main():
    m = pd.read_parquet(os.path.join(HERE, "tick_minute_stats.parquet"))
    t = trades(); t["t0"] = pd.to_datetime(t.entry_time).dt.floor("15min"); t["d"] = np.where(t.direction == "long", 1, -1)
    rows = []
    for _, r in t.iterrows():
        f = feats(m, r.t0, r.d)
        if f: rows.append(dict(entry=str(r.entry_time), side=r.direction, usd=r.usd, win=int(r.usd > 0), **f))
    D = pd.DataFrame(rows); print(f"trades {len(t)} -> usable {len(D)} | wins {D.win.sum()} losses {(1 - D.win).sum()}")
    cols = [c for c in D.columns if c not in ("entry", "side", "usd", "win")]; res = []
    for c in cols:
        X = D[[c, "win", "usd"]].dropna()
        if X[c].nunique() < 3: continue
        a, b = X[X.win == 1][c], X[X.win == 0][c]; u = stats.mannwhitneyu(a, b); auc = u.statistic / (len(a) * len(b))
        res.append(dict(feature=c, win_median=round(float(a.median()), 3), loss_median=round(float(b.median()), 3), auc=round(float(auc), 3),
                        p=float(u.pvalue), n=len(X), spearman_usd=round(float(stats.spearmanr(X[c], X.usd)[0]), 3)))
    R = pd.DataFrame(res).sort_values("p"); n = len(R)
    R["p_bh"] = np.minimum.accumulate((R.p * n / np.arange(1, n + 1))[::-1])[::-1].clip(upper=1)
    print(R.round(3).to_string(index=False))
    # consistency: same sign in two halves of time?
    h = D.entry < D.entry.sort_values().iloc[len(D) // 2]
    for c in R.feature[:3]:
        Y = D[[c, "win"]].dropna(); hh = h.loc[Y.index]
        au = lambda Z: round(stats.mannwhitneyu(Z[Z.win == 1][c], Z[Z.win == 0][c]).statistic / (Z.win.sum() * (1 - Z.win).sum()), 3)
        print(c, "AUC first half", au(Y[hh]), "| second half", au(Y[~hh]))
    D.to_csv(os.path.join(HERE, "tick_pattern_features.csv"), index=False); R.to_json(os.path.join(HERE, "tick_pattern_wins_losses.json"), orient="records", indent=1)
if __name__ == "__main__":
    main()
