"""One pass over the 6-month tick file -> per-minute tick statistics (cached in tick_minute_stats.parquet)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); COMB = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(COMB, "SLP2", "data", "XAUUSD_ticks_6m.parquet"); OUT = os.path.join(HERE, "tick_minute_stats.parquet")
acc = {}
f = pq.ParquetFile(SRC); prev_bid = None
for g in range(f.num_row_groups):
    t = f.read_row_group(g, columns=["time_msc", "bid", "ask"]).to_pandas()
    m = t.time_msc.values.astype("datetime64[m]").astype("int64"); b = t.bid.values; sp = (t.ask.values - b)
    d = np.diff(b, prepend=b[0] if prev_bid is None else prev_bid); prev_bid = b[-1]
    df = pd.DataFrame({"m": m, "up": d > 0, "down": d < 0, "sp": sp, "bid": b})
    r = df.groupby("m").agg(n=("bid", "size"), up=("up", "sum"), down=("down", "sum"), sp_mean=("sp", "mean"), sp_max=("sp", "max"), bid_last=("bid", "last"))
    for k, row in r.iterrows():
        a = acc.get(k)
        if a is None: acc[k] = [row.n, row.up, row.down, row.sp_mean * row.n, row.sp_max, row.bid_last]
        else: a[0] += row.n; a[1] += row.up; a[2] += row.down; a[3] += row.sp_mean * row.n; a[4] = max(a[4], row.sp_max); a[5] = row.bid_last
    if g % 20 == 0: print(g, f.num_row_groups, flush=True)
res = pd.DataFrame.from_dict(acc, orient="index", columns=["n", "up", "down", "sp_sum", "sp_max", "bid_last"]).sort_index()
res["sp_mean"] = res.sp_sum / res.n; res.index = pd.to_datetime(res.index.values.astype("datetime64[m]")); res.index.name = "minute"
res.drop(columns="sp_sum").to_parquet(OUT); print(res.shape, res.index.min(), res.index.max())
