from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from research.backtest.engine import BacktestResult

# 15-minute bars, ~24h x 5d markets: used to annualize the Sharpe ratio from
# per-trade returns. Documented rather than buried so the number can be
# challenged.
TRADING_DAYS_PER_YEAR = 252


class MonthlyPerformance(BaseModel):
    month: str  # YYYY-MM
    trades: int
    net_profit: float
    return_pct: float
    win_rate: float
    ending_balance: float


class PerformanceMetrics(BaseModel):
    # Headline
    initial_balance: float
    final_balance: float
    net_profit: float
    return_pct: float

    # Counts
    total_signals: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float

    # Quality
    profit_factor: float | None
    gross_profit: float
    gross_loss: float
    average_trade: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    expectancy_r: float
    average_planned_rr: float
    average_realized_r: float
    payoff_ratio: float | None

    # Risk
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_start: datetime | None
    max_drawdown_end: datetime | None
    longest_drawdown_days: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Behavior
    average_duration_minutes: float | None
    total_commission: float
    exit_reason_counts: dict = Field(default_factory=dict)
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0

    monthly: list[MonthlyPerformance] = Field(default_factory=list)
    equity_curve: list[dict] = Field(default_factory=list)


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _drawdown(equity: pd.DataFrame) -> tuple[float, float, datetime | None, datetime | None, float | None]:
    """Peak-to-trough drawdown on the realized equity curve."""
    if equity.empty:
        return 0.0, 0.0, None, None, None

    running_peak = equity["balance"].cummax()
    drawdown = equity["balance"] - running_peak
    trough_pos = int(drawdown.idxmin())
    max_dd = float(-drawdown.loc[trough_pos])
    if max_dd <= 0:
        return 0.0, 0.0, None, None, None

    peak_value = float(running_peak.loc[trough_pos])
    peak_rows = equity.loc[:trough_pos]
    peak_rows = peak_rows[peak_rows["balance"] >= peak_value]
    start = peak_rows["timestamp"].iloc[-1] if not peak_rows.empty else None
    end = equity["timestamp"].loc[trough_pos]

    # Recovery: first point after the trough that regains the old peak.
    after = equity.loc[trough_pos:]
    recovered = after[after["balance"] >= peak_value]
    recovery_point = recovered["timestamp"].iloc[0] if not recovered.empty else None
    longest_days = (
        (recovery_point - start).total_seconds() / 86400 if (start and recovery_point) else None
    )

    return (
        max_dd,
        (max_dd / peak_value * 100) if peak_value else 0.0,
        start.to_pydatetime() if hasattr(start, "to_pydatetime") else start,
        end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
        longest_days,
    )


def _streaks(profits: list[float]) -> tuple[int, int]:
    best = worst = current_win = current_loss = 0
    for profit in profits:
        if profit > 0:
            current_win += 1
            current_loss = 0
        elif profit < 0:
            current_loss += 1
            current_win = 0
        else:
            current_win = current_loss = 0
        best = max(best, current_win)
        worst = max(worst, current_loss)
    return best, worst


def _sharpe(returns: pd.Series, trades_per_year: float) -> float | None:
    """Annualized Sharpe from per-trade returns.

    Reported only with enough trades to be meaningful -- a Sharpe computed
    from a handful of trades is noise dressed as rigor.
    """
    if len(returns) < 20:
        return None
    std = returns.std(ddof=1)
    if not std or math.isnan(std) or std == 0:
        return None
    return float(returns.mean() / std * math.sqrt(trades_per_year))


def _sortino(returns: pd.Series, trades_per_year: float) -> float | None:
    if len(returns) < 20:
        return None
    downside = returns[returns < 0]
    if downside.empty:
        return None
    downside_std = downside.std(ddof=1)
    if not downside_std or math.isnan(downside_std) or downside_std == 0:
        return None
    return float(returns.mean() / downside_std * math.sqrt(trades_per_year))


def compute_metrics(result: BacktestResult) -> PerformanceMetrics:
    trades = result.trades
    initial = result.initial_balance
    final = result.final_balance

    equity = pd.DataFrame(result.equity_curve)
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        equity = equity.sort_values("timestamp").reset_index(drop=True)

    if not trades:
        max_dd, max_dd_pct, dd_start, dd_end, dd_days = _drawdown(equity)
        return PerformanceMetrics(
            initial_balance=initial,
            final_balance=final,
            net_profit=final - initial,
            return_pct=0.0,
            total_signals=result.signals_generated,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            profit_factor=None,
            gross_profit=0.0,
            gross_loss=0.0,
            average_trade=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            expectancy_r=0.0,
            average_planned_rr=0.0,
            average_realized_r=0.0,
            payoff_ratio=None,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_start=dd_start,
            max_drawdown_end=dd_end,
            longest_drawdown_days=dd_days,
            sharpe_ratio=None,
            sortino_ratio=None,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            average_duration_minutes=None,
            total_commission=0.0,
            equity_curve=result.equity_curve,
        )

    frame = pd.DataFrame([t.model_dump() for t in trades])
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    profits = frame["profit"]

    wins = frame[profits > 0]
    losses = frame[profits < 0]
    breakeven = frame[profits == 0]

    gross_profit = float(wins["profit"].sum())
    gross_loss = float(-losses["profit"].sum())

    # Per-trade return on the balance at entry, for risk-adjusted ratios.
    returns = frame["profit"] / frame["balance_before"].replace(0, pd.NA)
    returns = returns.dropna().astype(float)

    span_days = max(
        (frame["entry_time"].max() - frame["entry_time"].min()).total_seconds() / 86400, 1.0
    )
    trades_per_year = len(frame) / span_days * 365.0

    max_dd, max_dd_pct, dd_start, dd_end, dd_days = _drawdown(equity)
    best_streak, worst_streak = _streaks(profits.tolist())

    longs = frame[frame["direction"] == "BUY"]
    shorts = frame[frame["direction"] == "SELL"]

    monthly = _monthly(frame, initial)

    return PerformanceMetrics(
        initial_balance=initial,
        final_balance=final,
        net_profit=final - initial,
        return_pct=(final - initial) / initial * 100 if initial else 0.0,
        total_signals=result.signals_generated,
        total_trades=len(frame),
        winning_trades=len(wins),
        losing_trades=len(losses),
        breakeven_trades=len(breakeven),
        win_rate=len(wins) / len(frame) * 100,
        profit_factor=_safe_div(gross_profit, gross_loss),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        average_trade=float(profits.mean()),
        average_win=float(wins["profit"].mean()) if not wins.empty else 0.0,
        average_loss=float(losses["profit"].mean()) if not losses.empty else 0.0,
        largest_win=float(profits.max()),
        largest_loss=float(profits.min()),
        expectancy_r=float(frame["r_multiple"].mean()),
        average_planned_rr=float(frame["planned_risk_reward"].mean()),
        average_realized_r=float(frame["r_multiple"].mean()),
        payoff_ratio=_safe_div(
            float(wins["profit"].mean()) if not wins.empty else 0.0,
            abs(float(losses["profit"].mean())) if not losses.empty else 0.0,
        ),
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
        longest_drawdown_days=dd_days,
        sharpe_ratio=_sharpe(returns, trades_per_year),
        sortino_ratio=_sortino(returns, trades_per_year),
        max_consecutive_wins=best_streak,
        max_consecutive_losses=worst_streak,
        average_duration_minutes=float(frame["duration_minutes"].dropna().mean())
        if frame["duration_minutes"].notna().any()
        else None,
        total_commission=float(frame["commission"].sum()),
        exit_reason_counts=frame["exit_reason"].value_counts().to_dict(),
        long_trades=len(longs),
        short_trades=len(shorts),
        long_win_rate=float((longs["profit"] > 0).mean() * 100) if not longs.empty else 0.0,
        short_win_rate=float((shorts["profit"] > 0).mean() * 100) if not shorts.empty else 0.0,
        monthly=monthly,
        equity_curve=result.equity_curve,
    )


def _monthly(frame: pd.DataFrame, initial_balance: float) -> list[MonthlyPerformance]:
    if frame.empty:
        return []
    working = frame.copy()
    working["month"] = working["entry_time"].dt.strftime("%Y-%m")

    out: list[MonthlyPerformance] = []
    balance = initial_balance
    for month, group in working.groupby("month", sort=True):
        net = float(group["profit"].sum())
        start_balance = balance
        balance += net
        out.append(
            MonthlyPerformance(
                month=month,
                trades=len(group),
                net_profit=net,
                return_pct=(net / start_balance * 100) if start_balance else 0.0,
                win_rate=float((group["profit"] > 0).mean() * 100),
                ending_balance=balance,
            )
        )
    return out
