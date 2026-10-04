"""Merge HistData.com MT-format XAUUSD M1 zips (2015-2026) into one UTC parquet.
HistData clock = UTC-5 in EU winter, UTC-4 during EU summer time (verified against broker M1 and weekly Sunday opens).
Usage: python build_histdata_m1.py <folder with HISTDATA_COM_MT_XAUUSD_M1*.zip>"""
import sys, glob, zipfile, os
import numpy as np, pandas as pd

def last_sunday(y, m):
    d = pd.Timestamp(y, m, 31)
    return d - pd.Timedelta(days=(d.dayofweek + 1) % 7)

def read_all(folder):
    parts = []
    for f in sorted(glob.glob(os.path.join(folder, "HISTDATA_COM_MT_XAUUSD_M1*.zip"))):
        z = zipfile.ZipFile(f); n = [x for x in z.namelist() if x.endswith(".csv")][0]
        d = pd.read_csv(z.open(n), header=None, names=["d", "t", "open", "high", "low", "close", "vol"])
        d["ts"] = pd.to_datetime(d.d + " " + d.t, format="%Y.%m.%d %H:%M"); parts.append(d[["ts", "open", "high", "low", "close"]])
    return pd.concat(parts, ignore_index=True)

def to_utc(ts):
    t = ts + pd.Timedelta(hours=5)                      # EU-winter mapping
    summer = pd.Series(False, index=ts.index)
    for y in range(ts.dt.year.min(), ts.dt.year.max() + 1):
        s = last_sunday(y, 3) + pd.Timedelta(hours=1); e = last_sunday(y, 10) + pd.Timedelta(hours=1)
        summer |= (t >= s) & (t < e)
    return t - pd.to_timedelta(summer.astype(int), unit="h")

if __name__ == "__main__":
    a = read_all(sys.argv[1]); a["time_utc"] = to_utc(a.ts)
    a = a.drop_duplicates("time_utc", keep="last").sort_values("time_utc")   # later copy matched broker better
    a["time_server"] = a.time_utc.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)
    out = a[["time_utc", "time_server", "open", "high", "low", "close"]].reset_index(drop=True)
    D = os.path.join(os.path.dirname(__file__), "..", "data", "histdata_m1"); os.makedirs(D, exist_ok=True)
    out.to_parquet(os.path.join(D, "XAUUSD_M1_2015_2026_histdata.parquet"), compression="zstd", index=False)
    print(len(out), out.time_utc.min(), out.time_utc.max())
