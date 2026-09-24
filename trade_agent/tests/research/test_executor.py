from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.services.risk_service import RiskService
from research.ai.decision import SignalDecision, decisions_by_signal
from research.ai.gate import replay_risk_settings
from research.ai.schemas import FinalAction
from research.backtest.engine import (
    FILL_LIMIT_TOUCH,
    BacktestEngine,
    ExecutionPlan,
)
from research.backtest.executor import (
    BY_AI,
    BY_BOT_THROTTLE,
    BY_EXECUTION,
    BY_RISK,
    DecisionExecutor,
    run_with_decisions,
)
from research.backtest.indicators import compute_indicator_frame
from research.backtest.metrics import compute_metrics
from research.config import BacktestConfig
from research.strategy.base import StrategySignal
from tests.conftest import make_settings

UTC = timezone.utc
BASE = datetime(2025, 6, 3, 8, 0, tzinfo=UTC)


# --- fixtures --------------------------------------------------------------
def rising_bars(n: int = 300, step: float = 0.4) -> pd.DataFrame:
    """A steadily rising series: a BUY's target is reached, a BUY pullback
    limit below the market never is. Both are properties the tests rely on."""
    closes = [2400 + i * step for i in range(n)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [BASE + timedelta(minutes=15 * i) for i in range(n)], utc=True
            ),
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "tick_count": [80] * n,
            "bid_close": [c - 0.15 for c in closes],
            "ask_close": [c + 0.15 for c in closes],
            "spread_mean": [0.3] * n,
            "spread_max": [0.5] * n,
            "is_partial": [False] * n,
        }
    )
    return compute_indicator_frame(frame, ema_fast=20, ema_slow=100, volatility_window=50)


def dipping_bars(n: int = 300) -> pd.DataFrame:
    """Rises, then dips below the signal close, then rises again.

    Gives a BUY pullback limit something to fill against, so the MODIFY fill
    path is exercised on a series where the price genuinely comes back.
    """
    closes = []
    for i in range(n):
        value = 2400 + i * 0.4
        if 201 <= i <= 206:  # a dip a few bars after the signal at 200
            value -= 9.0
        closes.append(value)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [BASE + timedelta(minutes=15 * i) for i in range(n)], utc=True
            ),
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "tick_count": [80] * n,
            "bid_close": [c - 0.15 for c in closes],
            "ask_close": [c + 0.15 for c in closes],
            "spread_mean": [0.3] * n,
            "spread_max": [0.5] * n,
            "is_partial": [False] * n,
        }
    )
    return compute_indicator_frame(frame, ema_fast=20, ema_slow=100, volatility_window=50)


def signal(bars: pd.DataFrame, bar_index: int, signal_id: str, **overrides) -> StrategySignal:
    entry = float(bars["close"].iat[bar_index])
    payload = dict(
        signal_id=signal_id,
        bar_index=bar_index,
        signal_time=bars["timestamp"].iat[bar_index].to_pydatetime(),
        symbol="XAUUSD",
        side="BUY",
        entry=entry,
        stop_loss=entry - 8.0,
        take_profit=entry + 16.0,
        entry_reason="fixture",
        market_conditions={"session": "LONDON"},
    )
    payload.update(overrides)
    return StrategySignal(**payload)


def decision(signal_id: str, action: FinalAction, **overrides) -> SignalDecision:
    payload = dict(
        signal_id=signal_id,
        signal_time=BASE,
        action=action,
        confidence=0.8,
        reason="fixture decision",
    )
    payload.update(overrides)
    return SignalDecision(**payload)


@pytest.fixture
def executor() -> DecisionExecutor:
    config = BacktestConfig()
    return DecisionExecutor(
        config=config,
        engine=BacktestEngine(config),
        risk_service=RiskService(replay_risk_settings(make_settings())),
        counterfactual_balance=config.risk.initial_balance,
    )


# --- APPROVE ---------------------------------------------------------------
def test_approve_executes_the_original_signal(executor):
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(bars, [sig], [decision("s1", FinalAction.APPROVE)])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.signal_id == "s1"
    assert trade.ai_decision == "APPROVE"
    assert not trade.was_modified
    # Original levels, untouched, with the one-bar execution delay.
    assert trade.requested_entry == pytest.approx(sig.entry)
    assert trade.entry_time == bars["timestamp"].iat[101].to_pydatetime()
    assert executor.summary.approved_executed == 1


def test_approved_trade_carries_the_full_agent_chain(executor):
    bars = rising_bars()
    chain_decision = decision(
        "s1",
        FinalAction.APPROVE,
        technical={"decision": "PASS", "confidence": 0.9},
        news={"decision": "PASS"},
        sentiment=None,
        final={"action": "APPROVE"},
        skip_reasons={"sentiment": "no point-in-time sentiment data"},
        reason_codes=["ALL_ALIGNED"],
        deciding_rule="AGENT_DECISION",
        weighted_score=71.5,
        cost_usd=0.0042,
        agents_called=["technical", "news", "final"],
    )
    result = executor.run_with_decisions(bars, [signal(bars, 100, "s1")], [chain_decision])

    trade = result.trades[0]
    assert trade.agent_chain["technical"]["confidence"] == 0.9
    assert trade.agent_chain["skip_reasons"]["sentiment"].startswith("no point-in-time")
    assert trade.ai_reason_codes == ["ALL_ALIGNED"]
    assert trade.ai_weighted_score == 71.5
    assert trade.ai_cost_usd == pytest.approx(0.0042)


# --- REJECT -----------------------------------------------------------------
def test_reject_does_not_execute_but_is_recorded(executor):
    bars = rising_bars()
    result = executor.run_with_decisions(
        bars,
        [signal(bars, 100, "s1")],
        [decision("s1", FinalAction.REJECT, reason="fixture reason", blocking_agent="technical",
                  reason_codes=["TECHNICAL_VETO"], deciding_rule="BLOCK_VETO")],
    )

    assert result.trades == []
    assert len(result.skipped) == 1
    skipped = result.skipped[0]
    assert skipped.signal_id == "s1"
    assert skipped.skipped_by == BY_AI
    assert skipped.ai_decision == "REJECT"
    assert skipped.skip_reason == "fixture reason"
    assert skipped.blocking_agent == "technical"
    assert skipped.ai_reason_codes == ["TECHNICAL_VETO"]
    # ...and the hypothetical outcome is attached.
    assert skipped.counterfactual_available
    assert skipped.counterfactual_r is not None
    assert skipped.counterfactual_profit is not None


# --- WAIT --------------------------------------------------------------------
def test_wait_executes_the_original_signal_like_approve(executor):
    """WAIT means "not enough evidence either way" (missing data, low
    confidence), not "this is a probable loser" the way REJECT is. Per the
    user: only REJECT should ever block a trade -- WAIT trades normally, the
    label is kept purely for the audit trail."""
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(
        bars, [sig], [decision("s1", FinalAction.WAIT, reason="fixture reason")]
    )

    assert len(result.trades) == 1
    assert result.skipped == []
    trade = result.trades[0]
    assert trade.signal_id == "s1"
    assert trade.ai_decision == "WAIT"
    assert not trade.was_modified
    assert trade.requested_entry == pytest.approx(sig.entry)
    assert executor.summary.waited_by_ai == 1


def test_rejection_never_changes_the_balance(executor):
    bars = rising_bars()
    result = executor.run_with_decisions(
        bars, [signal(bars, 100, "s1")], [decision("s1", FinalAction.REJECT)]
    )
    assert result.final_balance == result.initial_balance
    # The counterfactual was a profitable trade, and it still did not move the
    # equity curve -- the whole point of scoring it on a fixed balance.
    assert result.skipped[0].counterfactual_profit > 0
    assert len(result.equity_curve) == 1


# --- MODIFY ----------------------------------------------------------------
def test_modify_fills_as_a_limit_when_price_returns(executor):
    bars = dipping_bars()
    sig = signal(bars, 200, "s1")
    pullback = sig.entry - 5.0
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                was_modified=True,
                modified_entry=pullback,
                modified_sl=pullback - 8.0,
                modified_tp=pullback + 16.0,
            )
        ],
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.was_modified
    assert trade.modified_fill_kind == FILL_LIMIT_TOUCH
    # A resting order fills at its price, not at a bar open.
    assert trade.entry_price == pytest.approx(pullback)
    assert trade.entry_slippage == 0.0
    assert trade.original_entry == pytest.approx(sig.entry)
    assert trade.stop_loss == pytest.approx(pullback - 8.0)
    assert executor.summary.modified_executed == 1


def test_modify_that_never_fills_is_recorded_not_forced(executor):
    # Rising series: a BUY limit below the market is never touched.
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    pullback = sig.entry - 6.0
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                was_modified=True,
                modified_entry=pullback,
                modified_sl=pullback - 8.0,
                modified_tp=pullback + 16.0,
            )
        ],
    )

    assert result.trades == []
    assert executor.summary.modified_unfilled == 1
    skipped = result.skipped[0]
    assert skipped.skipped_by == BY_EXECUTION
    assert "never touched" in skipped.skip_reason
    # The ORIGINAL signal is still counterfactually scored.
    assert skipped.counterfactual_available


def test_modify_at_the_signal_price_is_a_market_order(executor):
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                was_modified=True,
                modified_entry=sig.entry,          # unchanged entry
                modified_sl=sig.entry - 6.0,       # tighter stop
                modified_tp=sig.entry + 18.0,
            )
        ],
    )
    trade = result.trades[0]
    assert trade.was_modified
    assert trade.modified_fill_kind == "NEXT_BAR_OPEN"
    assert trade.entry_slippage > 0  # a market order crosses the spread


def test_modify_is_rejected_when_a_level_is_missing(executor):
    bars = rising_bars()
    result = executor.run_with_decisions(
        bars,
        [signal(bars, 100, "s1")],
        [decision("s1", FinalAction.MODIFY, modified_entry=2400.0, modified_sl=None)],
    )
    assert result.trades == []
    assert "missing entry, stop_loss or take_profit" in result.skipped[0].skip_reason
    assert result.skipped[0].skipped_by == BY_RISK


def test_modify_with_inverted_levels_is_refused(executor):
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                modified_entry=sig.entry,
                modified_sl=sig.entry + 8.0,   # stop ABOVE entry on a BUY
                modified_tp=sig.entry + 16.0,
            )
        ],
    )
    assert result.trades == []
    assert "incoherent" in result.skipped[0].skip_reason


def test_modify_that_breaks_the_rr_floor_is_rejected_by_the_risk_engine(executor):
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                modified_entry=sig.entry,
                modified_sl=sig.entry - 10.0,
                modified_tp=sig.entry + 5.0,  # RR 0.5, below the 1.5 minimum
            )
        ],
    )
    assert result.trades == []
    skipped = result.skipped[0]
    assert skipped.skipped_by == BY_RISK
    assert "execution guard" in skipped.skip_reason
    assert "RR" in skipped.skip_reason


def test_modify_cannot_flip_the_direction(executor):
    """A 'modification' that reverses the trade is a different trade.

    The schema pins the side to the original signal, so a SELL-shaped set of
    levels on a BUY signal is refused as incoherent rather than executed as a
    short.
    """
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                modified_entry=sig.entry,
                modified_sl=sig.entry + 8.0,
                modified_tp=sig.entry - 16.0,
            )
        ],
    )
    assert result.trades == []
    assert "incoherent" in result.skipped[0].skip_reason


def test_modify_disabled_rejects_instead_of_executing():
    bars = rising_bars()
    config = BacktestConfig()
    executor = DecisionExecutor(
        config=config,
        engine=BacktestEngine(config),
        risk_service=RiskService(replay_risk_settings(make_settings())),
        allow_modify=False,
    )
    sig = signal(bars, 100, "s1")
    result = executor.run_with_decisions(
        bars,
        [sig],
        [
            decision(
                "s1",
                FinalAction.MODIFY,
                modified_entry=sig.entry - 2.0,
                modified_sl=sig.entry - 10.0,
                modified_tp=sig.entry + 14.0,
            )
        ],
    )
    assert result.trades == []
    assert "modification is disabled" in result.skipped[0].skip_reason


# --- risk engine is never bypassed -----------------------------------------
def test_the_risk_engine_still_screens_an_approved_signal(executor):
    """An APPROVE cannot execute a trade the deterministic rules refuse."""
    bars = rising_bars()
    sig = signal(bars, 100, "s1")
    # R:R of 0.5 -- the strategy would never emit this, but the point is that
    # an AI approval does not get it past the gate.
    bad = sig.model_copy(update={"take_profit": sig.entry + 4.0})
    result = executor.run_with_decisions(bars, [bad], [decision("s1", FinalAction.APPROVE)])

    assert result.trades == []
    assert result.skipped[0].skipped_by == BY_RISK
    assert executor.summary.blocked_by_gate == 1


def test_missing_decision_is_excluded_not_executed(executor):
    """A signal the AI never decided on is excluded and counted.

    The undecided signal is the EARLIER one: engine constraints are evaluated
    before the decision, so a later signal would be skipped for the open
    position instead and this would not test what it claims to.
    """
    bars = rising_bars()
    result = executor.run_with_decisions(
        bars,
        [signal(bars, 100, "s1"), signal(bars, 140, "s2")],
        [decision("s2", FinalAction.APPROVE)],  # nothing for s1
    )
    assert [t.signal_id for t in result.trades] == ["s2"]
    assert executor.summary.missing_decision == 1
    excluded = [s for s in result.skipped if s.signal_id == "s1"][0]
    assert excluded.skipped_by == "no_decision"
    assert "no AI decision was recorded" in excluded.skip_reason


def test_duplicate_decisions_are_refused():
    with pytest.raises(ValueError, match="duplicate decision"):
        decisions_by_signal(
            [decision("s1", FinalAction.APPROVE), decision("s1", FinalAction.REJECT)]
        )


# --- experiment mechanics --------------------------------------------------
def test_baseline_and_ai_runs_share_the_engine_constraints(executor):
    """Same signals, all approved: Experiment B must reproduce Experiment A
    exactly. Any difference here is a difference in mechanics, not in AI."""
    bars = rising_bars()
    signals = [signal(bars, i, f"s{i}") for i in (100, 140, 180, 220)]

    baseline = executor.run_baseline(bars, signals)
    ai = executor.run_with_decisions(
        bars, signals, [decision(f"s{i}", FinalAction.APPROVE) for i in (100, 140, 180, 220)]
    )

    assert [t.signal_id for t in baseline.trades] == [t.signal_id for t in ai.trades]
    assert baseline.final_balance == pytest.approx(ai.final_balance)
    assert [t.entry_price for t in baseline.trades] == [
        pytest.approx(t.entry_price) for t in ai.trades
    ]


def test_experiment_b_produces_a_full_result_set(executor):
    bars = rising_bars()
    signals = [signal(bars, i, f"s{i}") for i in (100, 140, 180, 220, 260)]
    decisions = [
        decision("s100", FinalAction.APPROVE),
        decision("s140", FinalAction.REJECT, blocking_agent="news"),
        decision("s180", FinalAction.APPROVE),
        decision("s220", FinalAction.WAIT, blocking_agent="news"),
        decision("s260", FinalAction.APPROVE),
    ]
    result = executor.run_with_decisions(bars, signals, decisions)
    metrics = compute_metrics(result)

    assert result.trades and result.skipped
    assert len(result.equity_curve) == len(result.trades) + 1
    assert result.final_balance != result.initial_balance
    assert metrics.total_trades == len(result.trades)
    assert metrics.max_drawdown_pct >= 0
    assert metrics.monthly
    assert metrics.equity_curve


def test_functional_entry_point_returns_result_and_summary():
    bars = rising_bars()
    config = BacktestConfig()
    result, summary = run_with_decisions(
        bars,
        [signal(bars, 100, "s1"), signal(bars, 160, "s2")],
        [decision("s1", FinalAction.APPROVE), decision("s2", FinalAction.REJECT)],
        config=config,
        risk_service=RiskService(replay_risk_settings(make_settings())),
    )
    assert len(result.trades) == 1
    assert summary.rejected_by_ai == 1
    assert summary.counterfactuals_scored == 1


# --- fill semantics --------------------------------------------------------
def test_limit_entry_is_cancelled_if_the_target_is_reached_first():
    """If price runs to the target without filling, the order is cancelled.

    Assuming we could still board the move afterwards would be the flattering
    assumption, and it is the one this rule refuses.
    """
    bars = rising_bars(n=300, step=3.0)  # fast rise: target hit quickly
    config = BacktestConfig()
    engine = BacktestEngine(config)
    sig = signal(bars, 100, "s1")
    plan = ExecutionPlan(
        signal=sig,
        entry=sig.entry - 5.0,
        stop_loss=sig.entry - 13.0,
        take_profit=sig.entry + 11.0,
        fill_kind=FILL_LIMIT_TOUCH,
        limit_expiry_bars=20,
        was_modified=True,
    )
    trade, note = engine.simulate_plan(bars, plan, 100_000.0)
    assert trade is None
    assert "take-profit before the entry" in note


def test_excursions_are_recorded_on_executed_trades():
    bars = rising_bars()
    config = BacktestConfig()
    engine = BacktestEngine(config)
    trade, note = engine.simulate_signal(bars, signal(bars, 100, "s1"), 100_000.0)
    assert note == "executed"
    assert trade.max_favorable_excursion_r > 0
    assert trade.max_adverse_excursion_r >= 0


# --- the production bot's portfolio rules ----------------------------------
def bot_executor(**risk_overrides) -> DecisionExecutor:
    """An executor with the bot's throttles configured explicitly."""
    config = BacktestConfig()
    for key, value in risk_overrides.items():
        setattr(config.risk, key, value)
    return DecisionExecutor(
        config=config,
        engine=BacktestEngine(config),
        risk_service=RiskService(replay_risk_settings(make_settings())),
        counterfactual_balance=config.risk.initial_balance,
    )


def falling_bars(n: int = 400) -> pd.DataFrame:
    """A steadily falling series, so a BUY is always a loser.

    Needed to drive the loss-streak cooldown: it only arms after real losses.
    """
    closes = [2400 - i * 0.5 for i in range(n)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [BASE + timedelta(minutes=15 * i) for i in range(n)], utc=True
            ),
            "open": closes,
            "high": [c + 0.6 for c in closes],
            "low": [c - 0.6 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "tick_count": [80] * n,
            "bid_close": [c - 0.15 for c in closes],
            "ask_close": [c + 0.15 for c in closes],
            "spread_mean": [0.3] * n,
            "spread_max": [0.5] * n,
            "is_partial": [False] * n,
        }
    )
    return compute_indicator_frame(frame, ema_fast=20, ema_slow=100, volatility_window=50)


def test_time_stop_closes_a_position_the_bot_would_have_closed():
    """The bot's time stop, measured in real elapsed minutes."""
    bars = rising_bars(n=400, step=0.02)  # drifts too slowly to reach the target
    executor = bot_executor(time_stop_minutes=120, max_trades_per_day=50)
    sig = signal(bars, 50, "s1")
    # Far enough that the slow drift reaches neither level, but inside the risk
    # engine's max-stop-distance cap (3% of price) and above the 1.5 RR floor.
    wide = sig.model_copy(
        update={"stop_loss": sig.entry - 60.0, "take_profit": sig.entry + 150.0}
    )
    result = executor.run_baseline(bars, [wide])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "TIME_STOP"
    elapsed = (trade.exit_time - trade.entry_time).total_seconds() / 60
    assert elapsed >= 120


def test_a_target_reached_before_the_time_stop_wins():
    """Stop and target are resolved first; the time stop is only a fallback.

    Run with the bot's real 7-day time stop: the rising series reaches its
    target in about ten hours, so the target must be the recorded exit.
    """
    bars = rising_bars(n=400)
    executor = bot_executor(time_stop_minutes=10080, max_trades_per_day=50)
    result = executor.run_baseline(bars, [signal(bars, 50, "s1")])
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TAKE_PROFIT"


def test_time_stop_of_zero_disables_it():
    bars = rising_bars(n=400, step=0.02)
    executor = bot_executor(time_stop_minutes=0, max_trades_per_day=50)
    sig = signal(bars, 50, "s1")
    wide = sig.model_copy(
        update={"stop_loss": sig.entry - 60.0, "take_profit": sig.entry + 150.0}
    )
    result = executor.run_baseline(bars, [wide])
    assert result.trades[0].exit_reason == "END_OF_DATA"


def test_cooldown_pauses_entries_after_a_losing_streak():
    """Three consecutive losses arm a two-hour wall-clock pause."""
    bars = falling_bars(400)
    executor = bot_executor(
        cooldown_losses_to_trigger=3,
        cooldown_hours=2.0,
        max_trades_per_day=50,
        time_stop_minutes=0,
    )
    # BUYs into a falling market: every one loses.
    signals = [signal(bars, i, f"s{i}") for i in (20, 40, 60, 63, 66)]
    result = executor.run_with_decisions(bars, signals, decisions=None)

    assert executor.summary.blocked_by_cooldown >= 1
    paused = [s for s in result.skipped if s.skipped_by == BY_BOT_THROTTLE]
    assert paused
    assert "loss-streak cooldown" in paused[0].skip_reason
    # The trades that did execute were losers, which is what armed it.
    assert all(t.profit < 0 for t in result.trades)


def test_cooldown_is_not_armed_when_disabled():
    bars = falling_bars(400)
    executor = bot_executor(
        cooldown_losses_to_trigger=0, max_trades_per_day=50, time_stop_minutes=0
    )
    signals = [signal(bars, i, f"s{i}") for i in (20, 40, 60, 63, 66)]
    executor.run_with_decisions(bars, signals, decisions=None)
    assert executor.summary.blocked_by_cooldown == 0


def test_a_win_resets_the_loss_streak():
    """Two losses then a win must not arm a three-loss cooldown."""
    bars = falling_bars(400)
    executor = bot_executor(
        cooldown_losses_to_trigger=3, cooldown_hours=2.0, max_trades_per_day=50,
        time_stop_minutes=0,
    )
    losers = [signal(bars, i, f"L{i}") for i in (20, 40)]
    # A SELL into a falling market wins.
    winner = signal(bars, 60, "W60")
    entry = winner.entry
    winner = winner.model_copy(
        update={"side": "SELL", "stop_loss": entry + 8.0, "take_profit": entry - 16.0}
    )
    executor.run_with_decisions(bars, [*losers, winner], decisions=None)
    assert executor.summary.blocked_by_cooldown == 0


def test_daily_loss_guard_stops_trading_for_the_rest_of_the_day():
    bars = falling_bars(400)
    executor = bot_executor(
        max_daily_loss_pct=0.05,   # trips on the first loss
        max_trades_per_day=50,
        cooldown_losses_to_trigger=0,
        time_stop_minutes=0,
    )
    signals = [signal(bars, i, f"s{i}") for i in (20, 40, 60)]
    result = executor.run_with_decisions(bars, signals, decisions=None)

    assert executor.summary.blocked_by_daily_guard >= 1
    guarded = [
        s for s in result.skipped
        if s.skipped_by == BY_BOT_THROTTLE and "daily loss guard" in s.skip_reason
    ]
    assert guarded


def test_the_daily_guard_is_disabled_at_one_hundred_percent():
    """100 is the bot's current live setting, meaning off."""
    bars = falling_bars(400)
    executor = bot_executor(
        max_daily_loss_pct=100.0, max_trades_per_day=50,
        cooldown_losses_to_trigger=0, time_stop_minutes=0,
    )
    executor.run_with_decisions(
        bars, [signal(bars, i, f"s{i}") for i in (20, 40, 60)], decisions=None
    )
    assert executor.summary.blocked_by_daily_guard == 0


def test_throttled_signals_still_get_a_counterfactual():
    """A signal the bot's own rules declined is still scored.

    Otherwise the cost of the throttles themselves is invisible.
    """
    bars = falling_bars(400)
    executor = bot_executor(
        cooldown_losses_to_trigger=2, cooldown_hours=4.0, max_trades_per_day=50,
        time_stop_minutes=0,
    )
    signals = [signal(bars, i, f"s{i}") for i in (20, 40, 43, 46)]
    result = executor.run_with_decisions(bars, signals, decisions=None)
    throttled = [s for s in result.skipped if s.skipped_by == BY_BOT_THROTTLE]
    assert throttled
    assert all(s.counterfactual_available for s in throttled)


def test_both_arms_face_the_same_throttles():
    """The throttles are the bot's, not the AI's -- they must apply to A and B."""
    bars = falling_bars(400)
    signals = [signal(bars, i, f"s{i}") for i in (20, 40, 60, 63, 66)]

    baseline_executor = bot_executor(
        cooldown_losses_to_trigger=3, cooldown_hours=2.0, max_trades_per_day=50,
        time_stop_minutes=0,
    )
    baseline = baseline_executor.run_baseline(bars, signals)
    baseline_throttled = baseline_executor.summary.blocked_by_cooldown

    ai_executor = bot_executor(
        cooldown_losses_to_trigger=3, cooldown_hours=2.0, max_trades_per_day=50,
        time_stop_minutes=0,
    )
    ai = ai_executor.run_with_decisions(
        bars, signals, [decision(s.signal_id, FinalAction.APPROVE) for s in signals]
    )

    assert ai_executor.summary.blocked_by_cooldown == baseline_throttled
    assert [t.signal_id for t in ai.trades] == [t.signal_id for t in baseline.trades]
