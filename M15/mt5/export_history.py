"""
Run this script on the same machine where MT5 is installed (it doesn't need to
be logged into a trading account beyond the terminal being open/installed).
It pulls historical price data and saves it to CSV for backtesting.

Usage:
    python mt5/export_history.py --days 365

Output: data/xauusd_m1.csv, data/xauusd_m5.csv, data/xauusd_m15.csv,
        data/xauusd_h1.csv, data/xauusd_h4.csv
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, UTC

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed. Run: pip install MetaTrader5")
    sys.exit(1)

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mt5.config_mt5 import MT5


def export(timeframe, tf_name, days, chunk_days=30):
    if not mt5.initialize(login=MT5.login, password=MT5.password, server=MT5.server):
        print(f"Connection failed: {mt5.last_error()}")
        sys.exit(1)

    now = datetime.now(UTC)
    all_frames = []
    remaining = days
    chunk_end = now

    while remaining > 0:
        span = min(chunk_days, remaining)
        chunk_start = chunk_end - timedelta(days=span)
        rates = mt5.copy_rates_range(MT5.symbol, timeframe, chunk_start, chunk_end)

        if rates is not None and len(rates) > 0:
            df_chunk = pd.DataFrame(rates)
            all_frames.append(df_chunk)
        else:
            print(f"  warning: no data returned for {chunk_start.date()} to {chunk_end.date()} "
                  f"({mt5.last_error()}) — skipped.")

        chunk_end = chunk_start
        remaining -= span

    if not all_frames:
        print(f"No data received for {tf_name}.")
        return

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time")
    df["ts"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["ts", "open", "high", "low", "close", "volume"]]

    os.makedirs("data", exist_ok=True)
    out_path = f"data/xauusd_{tf_name}.csv"
    df.to_csv(out_path, index=False)
    actual_days = (df["ts"].max() - df["ts"].min()).days
    print(f"Saved: {out_path} ({len(df)} candles | actual span: {actual_days} days, "
          f"from {df['ts'].min().date()} to {df['ts'].max().date()})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    export(mt5.TIMEFRAME_M1, "m1", args.days)
    export(mt5.TIMEFRAME_M5, "m5", args.days)
    export(mt5.TIMEFRAME_M15, "m15", args.days)
    export(mt5.TIMEFRAME_M30, "m30", args.days)
    export(mt5.TIMEFRAME_H1, "h1", args.days)
    export(mt5.TIMEFRAME_H4, "h4", args.days)
    export(mt5.TIMEFRAME_D1, "d1", args.days)

    mt5.shutdown()
    print("\nUpload the data/xauusd_*.csv files.")


if __name__ == "__main__":
    main()
