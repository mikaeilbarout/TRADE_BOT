"""
Reproduces the walk-forward numbers quoted in each mt5/profiles/profile_*.py,
using the canonical strategy/donchian.py::simulate_donchian engine. Run this
any time the data/xauusd_*.csv files are refreshed to confirm the
parameters still hold up. Added 2026-09-06 (this project previously had no
single verification script -- see README.md).

Usage:
    python backtest_verify.py --profile m15     (or m1 / m5 / m30 / h1)
"""

import argparse
import importlib
import pandas as pd
from strategy.donchian import simulate_donchian

XAUUSD_SPREAD = 0.30  # $/oz round-trip, observed bid-ask
# Full real cost model for this Razor account (confirmed live via mt5.symbol_info --
# see strategy/donchian.py::simulate_donchian's docstring for sourcing/caveats).
XAUUSD_COMMISSION = 7.00 / 100    # Pepperstone Razor, confirmed: $7.00/lot round-trip for XAUUSD on MT5
XAUUSD_SWAP_LONG = -79.73 / 100   # $/oz/night, live snapshot 2026-09-06 -- drifts with interest rates
XAUUSD_SWAP_SHORT = 29.62 / 100

TF_DATA_FILES = {
    "m1": ("xauusd_m1.csv", "xauusd_m15.csv"),
    "m15": ("xauusd_m15.csv", "xauusd_h4.csv"),
    "m30": ("xauusd_m30.csv", "xauusd_h4.csv"),
    "h1": ("xauusd_h1.csv", "xauusd_d1.csv"),
}


def metrics(trades, final_equity, starting=1000.0):
    if trades.empty:
        return 0, 0.0, 0.0, 0.0, 0.0
    eq = pd.concat([pd.Series([starting]), trades["equity_after"]], ignore_index=True)
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max * 100
    max_dd = dd.min()
    total_return = (final_equity - starting) / starting * 100
    calmar = total_return / abs(max_dd) if max_dd != 0 else (999 if total_return > 0 else -999)
    win_rate = (trades["pnl"] > 0).mean() * 100
    return len(trades), win_rate, total_return, max_dd, calmar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["m1", "m15", "m30", "h1"], default="m15")
    args = parser.parse_args()

    profile = importlib.import_module(f"mt5.profiles.profile_{args.profile}")
    entry_file, trend_file = TF_DATA_FILES[args.profile]

    df_e_raw = pd.read_csv(f"data/{entry_file}", parse_dates=["ts"])
    df_t_raw = pd.read_csv(f"data/{trend_file}", parse_dates=["ts"])
    df_t_raw = df_t_raw[df_t_raw["ts"] >= df_e_raw["ts"].min()].reset_index(drop=True)

    # split by BAR COUNT, not calendar time (fixed 2026-09-09 -- this used to split by
    # calendar-time span, `t0 + (t1-t0)*0.7`, a different convention than every tick-
    # verification walk-forward script this session used (`int(len(df)*0.7)`), which
    # gives a different, not directly comparable, cutoff date/score for the same "70/30").
    t0, t1 = df_e_raw["ts"].min(), df_e_raw["ts"].max()
    split_t = df_e_raw["ts"].iloc[int(len(df_e_raw) * 0.7)]
    print(f"{args.profile.upper()}: {entry_file}/{trend_file} span: {t0} -> {t1} ({(t1 - t0).days} days)")

    train_e = df_e_raw[df_e_raw["ts"] < split_t].reset_index(drop=True)
    train_t = df_t_raw[df_t_raw["ts"] < split_t].reset_index(drop=True)
    test_e = df_e_raw[df_e_raw["ts"] >= split_t].reset_index(drop=True)
    test_t = df_t_raw[df_t_raw["ts"] >= split_t].reset_index(drop=True)

    kwargs = dict(
        n_period=profile.N_PERIOD, atr_period=profile.ATR_PERIOD,
        ema_trend_period=profile.EMA_TREND_PERIOD, atr_stop_multiplier=profile.RISK.atr_stop_multiplier,
        reward_risk_ratio=profile.RISK.reward_risk_ratio, time_stop_minutes=profile.RISK.time_stop_minutes,
        risk_cfg=profile.RISK, starting_equity=1000.0, spread_dollars=XAUUSD_SPREAD,
        commission_dollars=XAUUSD_COMMISSION, swap_long=XAUUSD_SWAP_LONG, swap_short=XAUUSD_SWAP_SHORT,
        min_trend_strength_pct=getattr(profile, "MIN_TREND_STRENGTH_PCT", 0.0),
        require_pivot_confirm=getattr(profile, "REQUIRE_PIVOT_CONFIRM", False),
        pivot_k=getattr(profile, "PIVOT_K", 2),
        cooldown_losses_to_trigger=getattr(profile, "COOLDOWN_LOSSES_TO_TRIGGER", 0),
        cooldown_hours=getattr(profile, "COOLDOWN_HOURS", 0.0),
    )

    tr_trades, tr_eq = simulate_donchian(train_e, train_t, **kwargs)
    te_trades, te_eq = simulate_donchian(test_e, test_t, **kwargs)
    full_trades, full_eq = simulate_donchian(df_e_raw, df_t_raw, **kwargs)

    n_tr, wr_tr, ret_tr, dd_tr, cal_tr = metrics(tr_trades, tr_eq)
    n_te, wr_te, ret_te, dd_te, cal_te = metrics(te_trades, te_eq)
    n_f, wr_f, ret_f, dd_f, cal_f = metrics(full_trades, full_eq)

    print(f"\nTrain (70%):  n={n_tr:4d}  winrate={wr_tr:5.1f}%  return={ret_tr:7.1f}%  maxdd={dd_tr:6.1f}%  calmar={cal_tr:6.2f}")
    print(f"Test  (30%):  n={n_te:4d}  winrate={wr_te:5.1f}%  return={ret_te:7.1f}%  maxdd={dd_te:6.1f}%  calmar={cal_te:6.2f}")
    print(f"Full (100%):  n={n_f:4d}  winrate={wr_f:5.1f}%  return={ret_f:7.1f}%  maxdd={dd_f:6.1f}%  calmar={cal_f:6.2f}")
    print(f"\nWalk-forward score (min of train/test calmar): {min(cal_tr, cal_te):.2f}")


if __name__ == "__main__":
    main()
