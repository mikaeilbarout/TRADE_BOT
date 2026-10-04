"""Build XAUUSD M1 candles (UTC, bid OHLC + mean spread + tick count) from Dukascopy public tick files.
Resumable: one parquet per month in ../data/dukascopy_m1/. Usage: python fetch_dukascopy_m1.py [start_year] [end_ym]"""
import sys, struct, lzma, datetime as dt, urllib.request, urllib.error, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "dukascopy_m1"
URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
REC = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("av", ">f4"), ("bv", ">f4")])

def hour_ticks(t):
    url = URL.format(y=t.year, m=t.month - 1, d=t.day, h=t.hour)
    for attempt in range(10):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"failed {url}")
    if not raw: return None
    a = np.frombuffer(lzma.decompress(raw), dtype=REC)
    ts = pd.Timestamp(t) + pd.to_timedelta(a["ms"].astype("int64"), unit="ms")
    return pd.DataFrame({"t": ts, "bid": a["bid"] / 1000.0, "ask": a["ask"] / 1000.0})

def month_m1(y, m):
    start = dt.datetime(y, m, 1); end = dt.datetime(y + (m == 12), m % 12 + 1, 1)
    hours = [start + dt.timedelta(hours=i) for i in range(int((end - start).total_seconds() // 3600))]
    with ThreadPoolExecutor(8) as ex: parts = [p for p in ex.map(hour_ticks, hours) if p is not None and len(p)]
    if not parts: return None
    tk = pd.concat(parts); tk["spread"] = tk.ask - tk.bid; g = tk.groupby(tk.t.dt.floor("min"))
    c = g.bid.ohlc(); c["ticks"] = g.size(); c["spread"] = g.spread.mean()
    c.index.name = "time_utc"; return c

if __name__ == "__main__":
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
    last = sys.argv[2] if len(sys.argv) > 2 else "2026-10"
    ye, me = map(int, last.split("-"))
    for y in range(y0, ye + 1):
        for m in range(1, 13):
            if (y, m) > (ye, me): break
            if (y, m) < (2021, 10): continue
            f = OUT / f"XAUUSD_M1_{y}-{m:02d}.parquet"
            if f.exists(): continue
            c = month_m1(y, m)
            if c is not None: c.to_parquet(f)
            print(y, m, 0 if c is None else len(c), flush=True)
