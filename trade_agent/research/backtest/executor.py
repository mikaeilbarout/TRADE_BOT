from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from app.models.signal import TradeSignal
from app.models.trade import ModifiedTrade
from app.services.risk_service import AccountState, RiskService
from research.ai.decision import SignalDecision, decisions_by_signal
from research.ai.gate import snapshot_at
from research.ai.schemas import FinalAction
from research.backtest.engine import (
    FILL_LIMIT_TOUCH,
    FILL_NEXT_BAR_OPEN,
    BacktestEngine,
    BacktestResult,
    ExecutionPlan,
)
from research.backtest.trade import SkippedSignal, Trade
from research.config import BacktestConfig
from research.strategy.base import StrategySignal

"""The execution bridge: AI decisions -> actual trades.

Before this module existed, the AI layer produced `SignalDecision` objects and
the backtest engine consumed `StrategySignal` objects, and nothing joined
them -- so Experiment B (the AI-filtered equity curve) could not be produced
at all.

The design decision that matters here: ONE executor runs both experiments.
Experiment A calls it with no decisions, Experiment B with the AI decisions,
and everything else -- sizing, the deterministic gate, the execution guard,
concurrency, daily caps, fills, commission, slippage -- is the same code on
the same bars. Two separately written runners would drift, and a comparison
between two drifted runners measures the drift, not the AI.

Pipeline per signal, identical in both experiments except for step 3:

  1. Engine constraints: a position already open, or the daily cap reached.
  2. Deterministic pre-trade gate (`RiskService.pre_check`).
  3. (Experiment B only) the AI decision.
  4. Execution guard (`RiskService.final_guard`) on the exact trade about to
     be sent -- original or modified. Nothing the AI produced can skip it.
  5. Fill on `BacktestEngine.simulate_plan`.

Every signal that does not become a trade is recorded as a `SkippedSignal`
with its counterfactual outcome, so a rejection is never silently dropped.
"""

# Who stopped a signal from becoming a trade.
BY_ENGINE = "engine_constraint"
BY_RISK = "risk_engine"
BY_AI = "ai_pipeline"
BY_MISSING = "no_decision"
# An order that was accepted by every check but never filled -- a modified
# limit entry the market did not come back to. Distinct from a rejection: the
# trade was allowed, the price simply never arrived.
BY_EXECUTION = "execution"
# A throttle the production bot enforces: the loss-streak cooldown or the
# daily-loss guard. Kept separate from `engine_constraint` so the comparison can
# say how many signals the BOT's own rules declined, not just how many the
# mechanics declined.
BY_BOT_THROTTLE = "bot_throttle"


@dataclass
class ExecutionSummary:
    """Counts of what the executor did, for the run report and for asserting
    the two experiments saw the same signal universe."""

    signals_considered: int = 0
    executed: int = 0
    approved_executed: int = 0
    modified_executed: int = 0
    modified_unfilled: int = 0
    rejected_by_ai: int = 0
    waited_by_ai: int = 0
    blocked_by_gate: int = 0
    blocked_by_guard: int = 0
    blocked_by_engine: int = 0
    blocked_by_cooldown: int = 0
    blocked_by_daily_guard: int = 0
    missing_decision: int = 0
    counterfactuals_scored: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DecisionExecutor:
    """Turns signals -- and optionally AI decisions about them -- into trades.

    `risk_service` must be built from replay settings
    (`research.ai.gate.replay_risk_settings`), which neutralize only the two
    wall-clock rules that cannot apply to historical bars. Every monetary and
    structural limit stays live.
    """

    config: BacktestConfig
    engine: BacktestEngine
    risk_service: RiskService
    allow_modify: bool = True
    # Counterfactuals are sized off a FIXED balance, never the running one, so
    # that scoring a rejected trade cannot influence -- or be influenced by --
    # either experiment's equity curve.
    counterfactual_balance: float | None = None
    summary: ExecutionSummary = field(default_factory=ExecutionSummary)

    # --- public API -------------------------------------------------------
    def run_baseline(
        self,
        bars: pd.DataFrame,
        signals: list[StrategySignal],
        run_id: str | None = None,
    ) -> BacktestResult:
        """Experiment A: the strategy alone, through the same gate and guard."""
        return self.run_with_decisions(
            bars, signals, decisions=None, run_id=run_id, run_kind="baseline"
        )

    def run_with_decisions(
        self,
        bars: pd.DataFrame,
        signals: list[StrategySignal],
        decisions: list[SignalDecision] | dict[str, SignalDecision] | None,
        run_id: str | None = None,
        run_kind: str | None = None,
    ) -> BacktestResult:
        """Execute a signal set, optionally filtered by the AI layer.

        `decisions=None` runs Experiment A. Passing decisions runs Experiment
        B; a signal with no decision is skipped and counted rather than
        silently executed or silently dropped, because a partially decided
        pilot must not be compared against a fully traded baseline.
        """
        if isinstance(decisions, list):
            index = decisions_by_signal(decisions)
        else:
            index = dict(decisions) if decisions else None

        run_kind = run_kind or ("ai" if index is not None else "baseline")
        run_id = run_id or str(uuid.uuid4())[:12]
        self.summary = ExecutionSummary(signals_considered=len(signals))

        balance = self.config.risk.initial_balance
        cf_balance = self.counterfactual_balance or self.config.risk.initial_balance
        trades: list[Trade] = []
        skipped: list[SkippedSignal] = []
        equity: list[dict] = [
            {
                "timestamp": bars["timestamp"].iloc[0].to_pydatetime(),
                "balance": balance,
                "trade_id": None,
            }
        ]
        open_until: datetime | None = None
        trades_per_day: dict[date, int] = {}
        # --- the bot's own throttles, carried across the run ---------------
        consecutive_losses = 0
        cooldown_until: datetime | None = None
        current_day: date | None = None
        day_start_balance = balance
        daily_guard_tripped = False
        risk = self.config.risk

        for signal in sorted(signals, key=lambda s: s.signal_time):
            decision = index.get(signal.signal_id) if index is not None else None
            day = signal.signal_time.date()
            if day != current_day:
                # A new UTC day resets the daily guard and its reference equity,
                # matching the live bot.
                current_day = day
                day_start_balance = balance
                daily_guard_tripped = False
            taken_today = trades_per_day.get(day, 0)
            position_open = open_until is not None and signal.signal_time <= open_until

            account = AccountState(
                balance=balance,
                trades_today=taken_today,
                open_positions=1 if position_open else 0,
                market_open=True,
            )

            # --- the bot's throttles, evaluated before the trade itself ----
            # Ordered ahead of the gate because the live bot never even looks at
            # a signal while it is paused; counting these as risk rejections
            # would misattribute them.
            if daily_guard_tripped:
                self.summary.blocked_by_daily_guard += 1
                skipped.append(
                    self._with_counterfactual(
                        self._skip(
                            signal,
                            f"daily loss guard: the day's loss reached "
                            f"{risk.max_daily_loss_pct}% of opening equity, so the "
                            "bot stops taking entries for the rest of the UTC day",
                            BY_BOT_THROTTLE,
                            decision,
                        ),
                        bars, signal, cf_balance,
                    )
                )
                continue
            if cooldown_until is not None and signal.signal_time < cooldown_until:
                self.summary.blocked_by_cooldown += 1
                skipped.append(
                    self._with_counterfactual(
                        self._skip(
                            signal,
                            f"loss-streak cooldown: paused until "
                            f"{cooldown_until.isoformat()} after "
                            f"{risk.cooldown_losses_to_trigger} consecutive losses",
                            BY_BOT_THROTTLE,
                            decision,
                        ),
                        bars, signal, cf_balance,
                    )
                )
                continue

            outcome = self._evaluate(
                bars, signal, decision, index is not None, account, position_open,
                taken_today, balance,
            )
            if isinstance(outcome, SkippedSignal):
                skipped.append(self._with_counterfactual(outcome, bars, signal, cf_balance))
                continue

            trade = outcome
            if decision is not None:
                trade.ai_decision = decision.action.value
                trade.ai_confidence = decision.confidence
                trade.ai_reason = decision.reason
                trade.ai_reason_codes = list(decision.reason_codes)
                trade.ai_deciding_rule = decision.deciding_rule
                trade.ai_weighted_score = decision.weighted_score
                trade.ai_cost_usd = decision.cost_usd
                trade.agent_chain = decision.chain()

            trades.append(trade)
            balance = trade.balance_after
            open_until = trade.exit_time
            trades_per_day[day] = taken_today + 1

            # --- update the bot's throttle state from the closed trade ------
            if risk.cooldown_losses_to_trigger > 0:
                if trade.profit < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= risk.cooldown_losses_to_trigger:
                        # Wall-clock pause from the losing EXIT, not from the
                        # signal, and not tied to calendar-day boundaries.
                        cooldown_until = (trade.exit_time or signal.signal_time) + timedelta(
                            hours=risk.cooldown_hours
                        )
                        consecutive_losses = 0
                else:
                    consecutive_losses = 0
            if risk.max_daily_loss_pct < 100.0 and day_start_balance > 0:
                loss_pct = (day_start_balance - balance) / day_start_balance * 100
                if loss_pct >= risk.max_daily_loss_pct:
                    daily_guard_tripped = True
            self.summary.executed += 1
            if trade.was_modified:
                self.summary.modified_executed += 1
            else:
                self.summary.approved_executed += 1
            equity.append(
                {
                    "timestamp": trade.exit_time,
                    "balance": balance,
                    "trade_id": trade.trade_id,
                }
            )

        return BacktestResult(
            run_id=run_id,
            run_kind=run_kind,
            trades=trades,
            skipped=skipped,
            equity_curve=equity,
            initial_balance=self.config.risk.initial_balance,
            final_balance=balance,
            period_start=bars["timestamp"].iloc[0].to_pydatetime(),
            period_end=bars["timestamp"].iloc[-1].to_pydatetime(),
            signals_generated=len(signals),
        )

    # --- per-signal decision ----------------------------------------------
    def _evaluate(
        self,
        bars: pd.DataFrame,
        signal: StrategySignal,
        decision: SignalDecision | None,
        ai_run: bool,
        account: AccountState,
        position_open: bool,
        taken_today: int,
        balance: float,
    ) -> Trade | SkippedSignal:
        # --- 1. engine constraints, applied identically in both runs ------
        if position_open:
            self.summary.blocked_by_engine += 1
            return self._skip(signal, "a position was already open", BY_ENGINE, decision)
        if taken_today >= self.config.risk.max_trades_per_day:
            self.summary.blocked_by_engine += 1
            return self._skip(
                signal,
                f"daily trade cap of {self.config.risk.max_trades_per_day} reached",
                BY_ENGINE,
                decision,
            )

        market = snapshot_at(self.config, bars, signal.bar_index, signal.symbol)
        volume, _ = self.engine.position_size(balance, signal.entry, signal.stop_loss)
        if volume < self.config.instrument.min_volume:
            self.summary.blocked_by_gate += 1
            return self._skip(
                signal,
                f"position size {volume} below instrument minimum "
                f"{self.config.instrument.min_volume}",
                BY_RISK,
                decision,
            )

        original = self._trade_signal(signal, volume)

        # --- 2. deterministic pre-trade gate ------------------------------
        pre = self.risk_service.pre_check(original, account, market=market)
        if not pre.passed:
            self.summary.blocked_by_gate += 1
            return self._skip(
                signal,
                "deterministic gate: " + "; ".join(pre.violations),
                BY_RISK,
                decision,
            )

        # --- 3. the AI layer (Experiment B only) --------------------------
        plan = ExecutionPlan.from_signal(signal)
        guarded: TradeSignal | ModifiedTrade = original

        if ai_run:
            if decision is None:
                self.summary.missing_decision += 1
                return self._skip(
                    signal,
                    "no AI decision was recorded for this signal, so it is excluded "
                    "from the AI run rather than executed or dropped",
                    BY_MISSING,
                    None,
                )
            if decision.action == FinalAction.REJECT:
                self.summary.rejected_by_ai += 1
                return self._skip(signal, decision.reason, BY_AI, decision)
            if decision.action == FinalAction.WAIT:
                # WAIT means "not enough evidence to be confident either way"
                # (missing data, low confidence) -- NOT a considered "this is
                # a probable loser" the way REJECT is. Per the user: the AI's
                # job is narrowly to catch high-confidence losers and raise
                # win rate by cutting those; everything else, including WAIT,
                # trades normally. Only count it for the audit trail; fall
                # through to execute the original (unmodified) signal exactly
                # like an implicit APPROVE.
                self.summary.waited_by_ai += 1
            if decision.action == FinalAction.MODIFY:
                if not self.allow_modify:
                    self.summary.rejected_by_ai += 1
                    return self._skip(
                        signal,
                        "MODIFY decision but modification is disabled by configuration",
                        BY_AI,
                        decision,
                    )
                built = self._modified_plan(signal, decision, balance)
                if isinstance(built, str):
                    self.summary.blocked_by_guard += 1
                    return self._skip(signal, built, BY_RISK, decision)
                plan, guarded = built

        # --- 4. execution guard on the exact trade being sent -------------
        guard = self.risk_service.final_guard(
            guarded, account, market=market, original_signal=original
        )
        if not guard.passed:
            self.summary.blocked_by_guard += 1
            return self._skip(
                signal,
                "execution guard: " + "; ".join(guard.violations),
                BY_RISK,
                decision,
            )

        # --- 5. fill -------------------------------------------------------
        trade, note = self.engine.simulate_plan(bars, plan, balance)
        if trade is None:
            if plan.fill_kind == FILL_LIMIT_TOUCH:
                self.summary.modified_unfilled += 1
                return self._skip(signal, note, BY_EXECUTION, decision)
            self.summary.blocked_by_engine += 1
            return self._skip(signal, note, BY_ENGINE, decision)
        return trade

    # --- modification -----------------------------------------------------
    def _modified_plan(
        self, signal: StrategySignal, decision: SignalDecision, balance: float
    ) -> tuple[ExecutionPlan, ModifiedTrade] | str:
        """Validate an AI modification and turn it into an execution plan.

        Returns an error string instead of raising so the caller can record
        the rejection reason on the skipped signal. The direction and symbol
        are pinned to the original signal here, and re-checked by the risk
        engine afterwards: a "modification" that flips a BUY into a SELL is a
        different trade, not an adjustment.
        """
        entry, stop, target = (
            decision.modified_entry,
            decision.modified_sl,
            decision.modified_tp,
        )
        if entry is None or stop is None or target is None:
            return (
                "MODIFY decision is missing entry, stop_loss or take_profit; "
                "refusing to guess the missing level"
            )

        volume, _ = self.engine.position_size(balance, entry, stop)
        if volume < self.config.instrument.min_volume:
            return (
                f"modified position size {volume} below instrument minimum "
                f"{self.config.instrument.min_volume}"
            )

        # A modified entry away from the signal price is a resting order; one
        # at the signal price is still a market order. Deciding this from the
        # numbers rather than from a flag means the fill semantics cannot
        # disagree with the prices.
        is_limit = abs(entry - signal.entry) > 1e-9
        try:
            guarded = ModifiedTrade(
                symbol=signal.symbol,
                side=signal.side,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                volume=volume,
                order_type="LIMIT" if is_limit else "MARKET",
            )
        except ValueError as exc:
            # The schema refuses inverted levels outright (BUY with the stop
            # above entry, and so on).
            return f"modified levels are incoherent: {exc}"

        plan = ExecutionPlan(
            signal=signal,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            fill_kind=FILL_LIMIT_TOUCH if is_limit else FILL_NEXT_BAR_OPEN,
            limit_expiry_bars=self.config.risk.modified_limit_expiry_bars,
            was_modified=True,
        )
        return plan, guarded

    def _trade_signal(self, signal: StrategySignal, volume: float) -> TradeSignal:
        return TradeSignal(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=volume,
            timeframe=f"M{self.config.timeframe_minutes}",
            strategy="seventy_thirty",
            timestamp=signal.signal_time,
        )

    # --- skipping and counterfactuals -------------------------------------
    def _skip(
        self,
        signal: StrategySignal,
        reason: str,
        by: str,
        decision: SignalDecision | None,
    ) -> SkippedSignal:
        return SkippedSignal(
            signal_id=signal.signal_id,
            signal_time=signal.signal_time,
            symbol=signal.symbol,
            direction=signal.side,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            skip_reason=reason,
            skipped_by=by,
            ai_decision=decision.action.value if decision else None,
            ai_confidence=decision.confidence if decision else None,
            ai_reason_codes=list(decision.reason_codes) if decision else [],
            ai_deciding_rule=decision.deciding_rule if decision else None,
            # Fall back to the mechanical stage when no agent is answerable,
            # so every skipped signal carries an attribution.
            blocking_agent=(
                decision.blocking_agent if decision and decision.blocking_agent else by
            ),
            ai_cost_usd=decision.cost_usd if decision else 0.0,
            agent_chain=decision.chain() if decision else {},
        )

    def _with_counterfactual(
        self,
        skipped: SkippedSignal,
        bars: pd.DataFrame,
        signal: StrategySignal,
        cf_balance: float,
    ) -> SkippedSignal:
        """Score what the ORIGINAL signal would have done, had it been taken.

        Always the original levels, never a modification: the question this
        answers is "was the strategy's trade a good one", which is what makes
        "how many profitable trades did the AI reject" meaningful.

        Sized off a fixed balance and never added to any equity curve, so this
        is diagnostic only and cannot move Experiment A or B by a cent.
        """
        trade, note = self.engine.simulate_signal(bars, signal, cf_balance)
        if trade is None:
            skipped.counterfactual_available = False
            skipped.counterfactual_unavailable_reason = note
            return skipped

        self.summary.counterfactuals_scored += 1
        skipped.counterfactual_available = True
        skipped.counterfactual_entry_time = trade.entry_time
        skipped.counterfactual_entry_price = trade.entry_price
        skipped.counterfactual_exit_time = trade.exit_time
        skipped.counterfactual_exit_price = trade.exit_price
        skipped.counterfactual_exit_reason = trade.exit_reason
        skipped.counterfactual_r = trade.r_multiple
        skipped.counterfactual_profit = trade.profit
        skipped.counterfactual_is_win = trade.profit > 0
        skipped.counterfactual_mfe_r = trade.max_favorable_excursion_r
        skipped.counterfactual_mae_r = trade.max_adverse_excursion_r
        skipped.counterfactual_volume = trade.volume
        skipped.counterfactual_bars_held = trade.bars_held
        return skipped


def run_with_decisions(
    bars: pd.DataFrame,
    signals: list[StrategySignal],
    decisions: list[SignalDecision] | None,
    config: BacktestConfig,
    risk_service: RiskService,
    engine: BacktestEngine | None = None,
    allow_modify: bool = True,
    run_id: str | None = None,
    run_kind: str | None = None,
) -> tuple[BacktestResult, ExecutionSummary]:
    """Functional entry point for the execution bridge.

    Returns the result and the execution summary together, because the counts
    (how many signals the AI rejected, how many modifications never filled)
    are needed to interpret the equity curve and should not have to be
    recomputed from the trade list.
    """
    executor = DecisionExecutor(
        config=config,
        engine=engine or BacktestEngine(config),
        risk_service=risk_service,
        allow_modify=allow_modify,
    )
    result = executor.run_with_decisions(
        bars, signals, decisions, run_id=run_id, run_kind=run_kind
    )
    return result, executor.summary
