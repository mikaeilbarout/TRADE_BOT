"""Does the S&P 500 filter (skip a trade when the S&P moved >= 0.9% the same way over 5 trading days) help SLP2?
Threshold fixed from the Donchian work (not searched). Split 70/30 as in the SLP2 review + 6-month real ticks.
Adopted only if it improves the LAST 30% (R, DD <= 1.2x) AND the tick result."""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scripts.pattern_strategy import load_m15
from scripts.sp2l_m15_backtest import simulate, DEFAULTS
from scripts.sp2l_tick_backtest import run_ticks, TICKS
SPX = pd.read_parquet(os.path.join(os.path.dirname(ROOT), "trade_dataset_20260929", "intermarket_h1.parquet")).SPX500.dropna()
T = 0.9


def move_at(decision):
    last = SPX.index.searchsorted(decision - pd.Timedelta(hours=1), side="right") - 1
    if last < 0: return np.nan
    past = SPX.index.searchsorted(SPX.index[last] - pd.Timedelta(hours=168), side="right") - 1
    return np.nan if past < 0 or past >= last else (SPX.values[last] / SPX.values[past] - 1) * 100


def factory(frame):
    dec = pd.to_datetime(frame.bar_time) + pd.Timedelta(minutes=15)             # signal decided at the bar's close
    mv = np.array([move_at(t) for t in dec])
    def keep(i, direction):
        v = mv[i]
        return True if np.isnan(v) else v * direction < T
    return keep


def summ(r):
    r = np.asarray(r, float)
    if len(r) == 0: return dict(n=0, win=0, R=0., DD=0.)
    c = np.cumsum(r); return dict(n=len(r), win=round(float((r > 0).mean()), 3), R=round(float(c[-1]), 1), DD=round(float(np.max(np.maximum.accumulate(np.maximum(c, 0)) - c)), 1))


frame = load_m15(); split = int(len(frame) * .7); boundary = frame.bar_time.iloc[split]
train = frame.iloc[:split].reset_index(drop=True); warm = max(250, DEFAULTS.ema_period * 10 + 4); test = frame.iloc[max(0, split - warm):].reset_index(drop=True)
meta = pq.ParquetFile(TICKS)
start = pd.Timestamp(meta.read_row_group(0, columns=["time_msc"]).column(0)[0].as_py()).ceil("15min")
end = pd.Timestamp(meta.read_row_group(meta.metadata.num_row_groups - 1, columns=["time_msc"]).column(0)[-1].as_py()).floor("15min") - pd.Timedelta(minutes=15)
tf = frame[frame.bar_time < end + pd.Timedelta(minutes=15)].reset_index(drop=True)
res = {}
for name, ef in (("SLP2 live", None), ("SLP2 + S&P filter", factory)):
    tr, te, tk = simulate(train, entry_filter=ef), simulate(test, evaluation_start=boundary, entry_filter=ef), run_ticks(tf, start, end, entry_filter=ef)
    res[name] = dict(first70=summ(tr.r_multiple), last30=summ(te.r_multiple), ticks=summ(tk.r_multiple))
    a, b, k = (res[name][x] for x in ("first70", "last30", "ticks"))
    print(f"{name:20} | first70 n={a['n']} R={a['R']:+6.1f} DD={a['DD']} | last30 n={b['n']} win={b['win']} R={b['R']:+6.1f} DD={b['DD']} | ticks n={k['n']} win={k['win']} R={k['R']:+6.1f} DD={k['DD']}", flush=True)
b, f = res["SLP2 live"], res["SLP2 + S&P filter"]
adopt = bool(f["last30"]["R"] > b["last30"]["R"] and f["last30"]["DD"] <= 1.2 * b["last30"]["DD"] and f["ticks"]["R"] > b["ticks"]["R"])
print("ADOPT:", adopt)
out = os.path.join(ROOT, "data", "slp2_strong_trend_20260930"); os.makedirs(out, exist_ok=True)
json.dump(dict(res, adopt=adopt), open(os.path.join(out, "spx_filter_report.json"), "w"), indent=1)
