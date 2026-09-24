from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.models.enums import Side
from research.backtest.engine import BacktestEngine
from research.backtest.metrics import compute_metrics
from research.config import BacktestConfig, CostModel, RiskModel
from research.strategy.base import StrategySignal

UTC = timezone.utc
BASE = datetime(2024, 6, 3, 8, 0, tzinfo=UTC)


def bars(rows: list[tuple[float, float, float, float]], spread: float = 0.20) -> pd.DataFrame:
    """Build a bar frame from (open, high, low, close) tuples.

    FIXTURES for verifying execution arithmetic -- not market data.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [BASE + timedelta(minutes=15 * i) for i in range(len(rows))], utc=True
            ),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100.0] * len(rows),
            "tick_count": [50] * len(rows),
            "spread_mean": [spread] * len(rows),
            "spread_max": [spread * 1.5] * len(rows),
        }
    )


def signal(side: Side = Side.BUY, bar_index: int = 0, entry: float = 2400.0,
           stop: float = 2390.0, target: float = 2420.0) -> StrategySignal:
    return StrategySignal(
        signal_id=f"sig-{bar_index}-{side.value}",
        bar_index=bar_index,
        signal_time=BASE + timedelta(minutes=15 * bar_index),
        symbol="XAUUSD",
        side=side,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        entry_reason="fixture",
        market_conditions={"session": "LONDON"},
    )


def engine(**overrides) -> BacktestEngine:
    config = BacktestConfig(
        costs=CostModel(
            slippage_price=0.0, stop_slippage_price=0.0, commission_per_lot_per_side=0.0
        ),
        risk=RiskModel(initial_balance=100_000.0, risk_per_trade_pct=1.0),
        **overrides,
    )
    return BacktestEngine(config)


# --- execution timing ------------------------------------------------------


def test_entry_is_filled_on_the_next_bar_not_the_signal_bar():
    """The signal bar's close must never be tradeable -- that is look-ahead."""
    frame = bars([(2400, 2401, 2399, 2400), (2405, 2406, 2404, 2405), (2410, 2421, 2409, 2420)])
    trade, note = engine().simulate_signal(frame, signal(bar_index=0), 100_000.0)
    assert note == "executed"
    # Filled around bar 1's open (2405), not bar 0's close (2400).
    assert trade.entry_time == frame["timestamp"].iloc[1]
    assert trade.entry_price == pytest.approx(2405.10)  # 2405 + half the 0.20 spread


def test_signal_on_the_final_bar_cannot_be_executed():
    frame = bars([(2400, 2401, 2399, 2400)])
    trade, note = engine().simulate_signal(frame, signal(bar_index=0), 100_000.0)
    assert trade is None
    assert "no next bar" in note


# --- spread and slippage ---------------------------------------------------


def test_buy_fills_at_ask_and_sell_fills_at_bid():
    frame = bars([(2400, 2401, 2399, 2400), (2400, 2401, 2399, 2400), (2400, 2401, 2399, 2400)],
                 spread=1.00)
    buy, _ = engine().simulate_signal(frame, signal(Side.BUY), 100_000.0)
    sell, _ = engine().simulate_signal(
        frame, signal(Side.SELL, entry=2400.0, stop=2410.0, target=2380.0), 100_000.0
    )
    assert buy.entry_price == pytest.approx(2400.5)   # paid the ask
    assert sell.entry_price == pytest.approx(2399.5)  # received the bid


def test_falls_back_to_configured_spread_when_bar_has_none():
    frame = bars([(2400, 2401, 2399, 2400)] * 3)
    frame["spread_mean"] = float("nan")
    config = BacktestConfig(
        costs=CostModel(fallback_spread_price=0.80, slippage_price=0.0,
                        stop_slippage_price=0.0, commission_per_lot_per_side=0.0)
    )
    trade, _ = BacktestEngine(config).simulate_signal(frame, signal(), 100_000.0)
    assert trade.entry_price == pytest.approx(2400.4)  # half of 0.80


# --- exit resolution -------------------------------------------------------


def test_stop_is_assumed_to_fill_first_when_a_bar_hits_both_levels():
    """The pessimistic assumption. Choosing the target here is the classic
    way a backtest flatters itself."""
    frame = bars([
        (2400, 2401, 2399, 2400),
        (2400, 2401, 2399, 2400),   # entry bar
        (2400, 2425, 2385, 2420),   # range covers BOTH stop 2390 and target 2420
    ])
    trade, _ = engine().simulate_signal(frame, signal(), 100_000.0)
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.profit < 0


def test_gap_through_the_stop_fills_at_the_open_not_the_stop_level():
    frame = bars([
        (2400, 2401, 2399, 2400),
        (2400, 2401, 2399, 2400),   # entry ~2400.1
        (2370, 2375, 2365, 2372),   # gapped far below the 2390 stop
    ])
    trade, _ = engine().simulate_signal(frame, signal(), 100_000.0)
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.exit_price == pytest.approx(2370.0)  # the gap open, worse than the stop
    assert trade.r_multiple < -1.0  # a real gap loses more than 1R


def test_take_profit_fills_at_the_level():
    frame = bars([
        (2400, 2401, 2399, 2400),
        (2400, 2401, 2399, 2400),
        (2402, 2425, 2401, 2424),
    ])
    trade, _ = engine().simulate_signal(frame, signal(), 100_000.0)
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.r_multiple == pytest.approx(2.0, abs=0.05)


def test_short_trade_stop_and_target_are_mirrored():
    frame = bars([
        (2400, 2401, 2399, 2400),
        (2400, 2401, 2399, 2400),
        (2398, 2399, 2378, 2380),
    ])
    trade, _ = engine().simulate_signal(
        frame, signal(Side.SELL, entry=2400.0, stop=2410.0, target=2380.0), 100_000.0
    )
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.profit > 0


def test_unresolved_trade_is_closed_at_end_of_data():
    frame = bars([(2400, 2401, 2399, 2400)] * 4)
    trade, _ = engine().simulate_signal(frame, signal(), 100_000.0)
    assert trade.exit_reason == "END_OF_DATA"


# --- sizing ----------------------------------------------------------------


def test_position_size_targets_the_configured_risk_percentage():
    eng = engine()
    volume, risk = eng.position_size(balance=100_000.0, entry=2400.0, stop=2390.0)
    # 1% of 100k = $1000 risk; 10 points x 100oz = $1000/lot -> 1.0 lot
    assert volume == pytest.approx(1.0)
    assert risk == pytest.approx(1000.0)


def test_position_size_scales_down_for_a_wider_stop():
    eng = engine()
    narrow, _ = eng.position_size(100_000.0, 2400.0, 2390.0)
    wide, _ = eng.position_size(100_000.0, 2400.0, 2360.0)
    assert wide < narrow


def test_trade_below_minimum_volume_is_skipped_not_upsized():
    config = BacktestConfig(risk=RiskModel(initial_balance=100.0, risk_per_trade_pct=0.1))
    frame = bars([(2400, 2401, 2399, 2400)] * 3)
    trade, note = BacktestEngine(config).simulate_signal(frame, signal(), 100.0)
    assert trade is None
    assert "below instrument minimum" in note


def test_commission_is_charged_on_both_sides():
    config = BacktestConfig(
        costs=CostModel(slippage_price=0.0, stop_slippage_price=0.0,
                        commission_per_lot_per_side=5.0),
        risk=RiskModel(initial_balance=100_000.0, risk_per_trade_pct=1.0),
    )
    frame = bars([(2400, 2401, 2399, 2400), (2400, 2401, 2399, 2400), (2402, 2425, 2401, 2424)])
    trade, _ = BacktestEngine(config).simulate_signal(frame, signal(), 100_000.0)
    assert trade.commission == pytest.approx(trade.volume * 5.0 * 2)
    assert trade.profit == pytest.approx(trade.gross_profit - trade.commission)


# --- full run and concurrency ---------------------------------------------


def test_run_enforces_one_open_position_at_a_time():
    frame = bars([(2400, 2401, 2399, 2400)] * 20)
    signals = [signal(bar_index=i) for i in (0, 1, 2)]
    result = engine().run(frame, signals, run_id="t1")
    assert len(result.trades) == 1
    assert len(result.skipped) == 2
    assert all(s.skipped_by == "engine_constraint" for s in result.skipped)
    assert all("already open" in s.skip_reason for s in result.skipped)


def test_run_enforces_daily_trade_cap():
    config = BacktestConfig(
        risk=RiskModel(initial_balance=100_000.0, risk_per_trade_pct=1.0, max_trades_per_day=1),
        costs=CostModel(slippage_price=0.0, stop_slippage_price=0.0,
                        commission_per_lot_per_side=0.0),
    )
    # Each trade resolves on its entry bar so nothing blocks on concurrency.
    frame = bars([(2400, 2425, 2399, 2424)] * 12)
    signals = [signal(bar_index=i) for i in (0, 3, 6)]
    result = BacktestEngine(config).run(frame, signals, run_id="t2")
    assert len(result.trades) == 1
    assert any("daily trade cap" in s.skip_reason for s in result.skipped)


def test_equity_curve_compounds_sequentially():
    frame = bars([(2400, 2425, 2399, 2424)] * 12)
    signals = [signal(bar_index=i) for i in (0, 3, 6)]
    result = engine().run(frame, signals, run_id="t3")
    balances = [point["balance"] for point in result.equity_curve]
    assert balances[0] == 100_000.0
    assert result.final_balance == balances[-1]
    assert result.final_balance != 100_000.0


# --- metrics ---------------------------------------------------------------


def test_metrics_on_a_known_mix_of_wins_and_losses():
    win_bars = bars([(2400, 2401, 2399, 2400), (2400, 2401, 2399, 2400), (2402, 2425, 2401, 2424)])
    loss_bars = bars([(2400, 2401, 2399, 2400), (2400, 2401, 2399, 2400), (2398, 2399, 2385, 2388)])

    eng = engine()
    winner, _ = eng.simulate_signal(win_bars, signal(), 100_000.0)
    loser, _ = eng.simulate_signal(loss_bars, signal(), 100_000.0)

    from research.backtest.engine import BacktestResult

    result = BacktestResult(
        run_id="m1",
        run_kind="baseline",
        trades=[winner, loser],
        equity_curve=[
            {"timestamp": winner.entry_time, "balance": 100_000.0, "trade_id": None},
            {"timestamp": winner.exit_time, "balance": winner.balance_after, "trade_id": winner.trade_id},
            {"timestamp": loser.exit_time, "balance": winner.balance_after + loser.profit,
             "trade_id": loser.trade_id},
        ],
        initial_balance=100_000.0,
        final_balance=winner.balance_after + loser.profit,
        signals_generated=2,
    )
    metrics = compute_metrics(result)

    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == pytest.approx(50.0)
    assert metrics.profit_factor is not None
    assert metrics.largest_win == pytest.approx(winner.profit)
    assert metrics.largest_loss == pytest.approx(loser.profit)
    assert metrics.max_drawdown > 0
    assert len(metrics.monthly) == 1
    # Sharpe needs a meaningful sample; two trades must not produce one.
    assert metrics.sharpe_ratio is None


def test_metrics_handle_a_run_with_no_trades():
    from research.backtest.engine import BacktestResult

    metrics = compute_metrics(
        BacktestResult(
            run_id="empty", run_kind="baseline", trades=[],
            equity_curve=[{"timestamp": BASE, "balance": 100_000.0, "trade_id": None}],
            initial_balance=100_000.0, final_balance=100_000.0, signals_generated=7,
        )
    )
    assert metrics.total_trades == 0
    assert metrics.total_signals == 7
    assert metrics.win_rate == 0.0
    assert metrics.profit_factor is None
