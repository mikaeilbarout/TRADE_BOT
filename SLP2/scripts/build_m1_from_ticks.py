"""Build 1-minute bars (bid OHLC, tick count, mean spread) from the 6-month tick file. Broker server clock, like the other bar files."""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "XAUUSD_ticks_6m.parquet"); OUT = os.path.join(ROOT, "data", "XAUUSD_M1_6m.parquet")
pf = pq.ParquetFile(SRC); parts = []
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["time_msc", "bid", "ask"]).to_pandas()
    t = t[(t.bid > 0) & (t.ask > 0)]
    t = t.sort_values("time_msc")
    t["m"] = t.time_msc.dt.floor("min")
    a = t.groupby("m").agg(t0=("time_msc", "min"), t1=("time_msc", "max"), h=("bid", "max"), l=("bid", "min"), n=("bid", "size"), sp=("ask", "sum"), sb=("bid", "sum"))
    first = t.groupby("m").bid.first(); last = t.groupby("m").bid.last()
    a["o"], a["c"] = first, last
    parts.append(a.reset_index())
    if g % 20 == 0: print(g, flush=True)
p = pd.concat(parts)
out = p.sort_values(["m", "t0"]).groupby("m").agg(t0=("t0", "min"), t1=("t1", "max"), h=("h", "max"), l=("l", "min"), n=("n", "sum"), sp=("sp", "sum"), sb=("sb", "sum"))
o = p.sort_values(["m", "t0"]).groupby("m").o.first(); c = p.sort_values(["m", "t1"]).groupby("m").c.last()
df = pd.DataFrame({"bar_time": pd.to_datetime(out.index), "open": o.values, "high": out.h.values, "low": out.l.values, "close": c.values,
                   "tick_volume": out.n.values, "avg_spread_price": ((out.sp - out.sb) / out.n).values})
df = df.sort_values("bar_time").reset_index(drop=True)
df.to_parquet(OUT, index=False)
print(len(df), "M1 bars", df.bar_time.iloc[0], "->", df.bar_time.iloc[-1], "| median spread $%.3f" % df.avg_spread_price.median())
