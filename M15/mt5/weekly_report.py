"""
Weekly trade report. Meant to run automatically every Sunday (via Windows
Task Scheduler -- see SETUP_MT5.md), but can also be run manually any time:

    python mt5/weekly_report.py
    python mt5/weekly_report.py --days 14   # look back further than 7 days

What it does:
1. Pulls every closed trade (across all profiles/magic numbers) from MT5
   for the lookback window.
2. Breaks it down per profile: trade count, win rate, net P&L.
3. Saves a dated snapshot to reports/weekly_report_<date>.csv (full trade
   list) and reports/weekly_report_<date>.txt (readable summary).
4. Appends one summary row per profile to data/weekly_summary_log.csv, so
   week-over-week trends build up over time in a single file.

This does NOT send the report anywhere (no email/Telegram) -- it just
writes it to disk. Open the .txt file, or ask Claude to read the latest one.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

MAGIC_NAMES = {991001: "M1", 991015: "M15", 991030: "M30", 991060: "H1"}
# Corrected 2026-09-09: these used to be the stale 990xxx magic numbers (990001/990005/
# 990015/990060/990030/990099), which none of the live profiles have used since the
# 2026-09-05 Donchian switch (see mt5/profiles/profile_*.py -- all now 991xxx). With the
# wrong numbers, this report's per-profile breakdown matched zero real trades and always
# printed empty, even though "Total closed trades" at the top was correct. M5 removed (its
# profile was deleted 2026-09-08).
REASON_MAP = {0: "client", 1: "mobile", 2: "web", 3: "expert/EA", 4: "SL", 5: "TP", 6: "SO"}

REPORTS_DIR = "reports"
SUMMARY_LOG_PATH = "data/weekly_summary_log.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="lookback window in days")
    args = parser.parse_args()

    if not mt5.initialize(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
    ):
        print(f"MT5 connection failed: {mt5.last_error()}")
        sys.exit(1)

    now = datetime.now()
    since = now - timedelta(days=args.days)
    # MT5's own clock can run a few hours ahead of this machine's, so a deal
    # closed "just now" may report a timestamp technically past `now` --
    # query a padded window and only enforce the *lower* bound (`since`),
    # never an upper one, so recent trades are never silently dropped.
    query_to = now + timedelta(hours=6)
    deals = mt5.history_deals_get(since - timedelta(hours=6), query_to)
    closed = sorted(
        [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and datetime.fromtimestamp(d.time) >= since],
        key=lambda d: d.time,
    ) if deals else []

    account = mt5.account_info()
    equity_now = account.equity if account else None
    mt5.shutdown()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    report_date = now.strftime("%Y-%m-%d")

    rows = []
    for d in closed:
        rows.append({
            "close_time": datetime.fromtimestamp(d.time).isoformat(timespec="seconds"),
            "profile": MAGIC_NAMES.get(d.magic, str(d.magic)),
            "side": "BUY" if d.type == mt5.ORDER_TYPE_BUY else "SELL",
            "close_price": d.price,
            "profit": d.profit,
            "reason": REASON_MAP.get(d.reason, "unknown"),
        })
    trades_df = pd.DataFrame(rows)
    trades_csv_path = f"{REPORTS_DIR}/weekly_report_{report_date}.csv"
    trades_df.to_csv(trades_csv_path, index=False)

    lines = []
    lines.append(f"Weekly trade report -- {report_date} (lookback: {args.days} days, since {since:%Y-%m-%d %H:%M})")
    lines.append("=" * 70)
    lines.append(f"Total closed trades: {len(trades_df)}")
    lines.append("")

    summary_rows = []
    grand_net = 0.0
    for label in ["M1", "M15", "M30", "H1"]:
        sub = trades_df[trades_df["profile"] == label] if len(trades_df) else pd.DataFrame()
        n = len(sub)
        if n == 0:
            continue
        wins = (sub["profit"] > 0).sum()
        losses = n - wins
        net = sub["profit"].sum()
        win_rate = wins / n * 100
        grand_net += net
        lines.append(f"{label}: {n} trades | {wins}W / {losses}L ({win_rate:.1f}% win rate) | net={net:+.2f}")
        summary_rows.append({
            "report_date": report_date, "lookback_days": args.days, "profile": label,
            "trades": n, "wins": int(wins), "losses": int(losses), "win_rate_pct": round(win_rate, 1),
            "net_pnl": round(net, 2), "equity_at_report": equity_now,
        })

    lines.append("-" * 70)
    lines.append(f"GRAND TOTAL net P&L: {grand_net:+.2f}")
    if equity_now is not None:
        lines.append(f"Account equity at report time: {equity_now:.2f}")
    lines.append("")
    lines.append(f"Full trade list: {trades_csv_path}")

    report_text = "\n".join(lines)
    txt_path = f"{REPORTS_DIR}/weekly_report_{report_date}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if os.path.exists(SUMMARY_LOG_PATH):
            summary_df.to_csv(SUMMARY_LOG_PATH, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(SUMMARY_LOG_PATH, index=False)

    print(report_text)
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {trades_csv_path}")
    print(f"Appended to: {SUMMARY_LOG_PATH}")


if __name__ == "__main__":
    main()
