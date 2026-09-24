"""
Compares each profile's REAL demo-account performance (since its current config
went live) against what the backtest predicted for the same profile.

This is the real out-of-sample test: the backtest parameters were chosen using
only historical data, so live results from here on are genuinely unseen data.
For the comparison to mean anything, don't change a profile's config without
also resetting its baseline in data/performance_baseline.json (see
`config_live_since`) -- otherwise you're comparing against a mix of old and
new settings.

Usage:
    python mt5/track_performance.py
Appends one row per profile to data/performance_tracking_log.csv each run,
so you can see how the comparison evolves over the coming weeks.
"""

import json
import os
import sys
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

BASELINE_PATH = "data/performance_baseline.json"
LOG_PATH = "data/performance_tracking_log.csv"


def main():
    if not os.path.exists(BASELINE_PATH):
        print(
            f"{BASELINE_PATH} does not exist yet -- nothing to compare against. "
            f"Create it with one entry per live profile (magic, config_live_since, "
            f"backtest_reference: {{trades_per_day, win_rate_pct, calmar}}) before running this."
        )
        sys.exit(1)
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baselines = json.load(f)

    if not mt5.initialize(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
    ):
        print(f"MT5 connection failed: {mt5.last_error()}")
        sys.exit(1)

    account = mt5.account_info()
    now = datetime.now()
    print(f"Account equity: {account.equity:.2f} {account.currency} | checked at {now:%Y-%m-%d %H:%M}\n")

    rows = []
    header = f"{'Profile':<8}{'Days live':<11}{'Trades':<19}{'WinRate':<19}{'Realized P&L'}"
    print(header)
    print("-" * len(header))

    for label, cfg in baselines.items():
        magic = cfg["magic"]
        since = datetime.fromisoformat(cfg["config_live_since"])
        ref = cfg["backtest_reference"]

        # query a padded window (MT5's internal clock can differ from this
        # machine's), then filter precisely in Python using the SAME
        # datetime.fromtimestamp() conversion `since`/`now` are built from
        query_from = since - pd.Timedelta(days=1)
        query_to = now + pd.Timedelta(days=1)
        deals = mt5.history_deals_get(query_from, query_to)
        closed = [
            d for d in deals
            if d.entry == mt5.DEAL_ENTRY_OUT and d.magic == magic
            and since <= datetime.fromtimestamp(d.time) <= now
        ] if deals else []

        n_trades = len(closed)
        n_wins = sum(1 for d in closed if d.profit > 0)
        win_rate = (n_wins / n_trades * 100) if n_trades else 0.0
        total_pnl = sum(d.profit for d in closed)

        days_live = max((now - since).total_seconds() / 86400, 0.01)
        expected_trades = ref["trades_per_day"] * days_live
        trades_col = f"{n_trades} (expect ~{expected_trades:.1f})"
        winrate_col = f"{win_rate:.1f}% (bt {ref['win_rate_pct']:.1f}%)"

        print(f"{label:<8}{days_live:<11.2f}{trades_col:<19}{winrate_col:<19}{total_pnl:+.2f} {account.currency}")

        rows.append({
            "checked_at": now.isoformat(timespec="seconds"),
            "profile": label,
            "days_live": round(days_live, 2),
            "actual_trades": n_trades,
            "expected_trades": round(expected_trades, 1),
            "actual_win_rate_pct": round(win_rate, 1),
            "backtest_win_rate_pct": ref["win_rate_pct"],
            "actual_realized_pnl": round(total_pnl, 2),
            "backtest_trades_per_day": ref["trades_per_day"],
            "backtest_calmar": ref["calmar"],
        })

    mt5.shutdown()

    log_df = pd.DataFrame(rows)
    if os.path.exists(LOG_PATH):
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)
    print(f"\nAppended to {LOG_PATH}")


if __name__ == "__main__":
    main()
