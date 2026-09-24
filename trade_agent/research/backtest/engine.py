from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from app.models.enums import Side
from research.config import BacktestConfig
from research.backtest.trade import SkippedSignal, Trade
from research.strategy.base import StrategySignal

# How an entry is assumed to fill.
FILL_NEXT_BAR_OPEN = "NEXT_BAR_OPEN"
FILL_LIMIT_TOUCH = "LIMIT_TOUCH"


@dataclass
class ExecutionPlan:
    """What is actually sent to market for one signal.

    The baseline run builds a plan straight from the signal. The AI run
    builds it from the signal for an APPROVE, or from the agent's levels for
    a MODIFY -- but both go through the same executor, so an AI trade can
    never be filled by more generous mechanics than a baseline trade.
    """

    signal: StrategySignal
    entry: float
    stop_loss: float
    take_profit: float
    fill_kind: str = FILL_NEXT_BAR_OPEN
    # LIMIT_TOUCH only: how many bars the resting order stays live before it
    # is cancelled unfilled. 0 means "use the configured default".
    limit_expiry_bars: int = 0
    # Provenance carried onto the resulting Trade.
    was_modified: bool = False

    @property
    def side(self) -> Side:
        return self.signal.side

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def target_distance(self) -> float:
        return abs(self.take_profit - self.entry)

    @classmethod
    def from_signal(cls, signal: StrategySignal) -> "ExecutionPlan":
        return cls(
            signal=signal,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            fill_kind=FILL_NEXT_BAR_OPEN,
        )


class BacktestResult(BaseModel):
    run_id: str
    run_kind: str
    trades: list[Trade] = Field(default_factory=list)
    skipped: list[SkippedSignal] = Field(default_factory=list)
    equity_curve: list[dict] = Field(default_factory=list)
    initial_balance: float = 0.0
    final_balance: float = 0.0
    period_start: datetime | None = None
    period_end: datetime | None = None
    signals_generated: int = 0


class BacktestEngine:
    """Bar-by-bar executor with explicit, conservative fill assumptions.

    The assumptions that decide whether a backtest is honest:

    * **Execution is delayed one bar.** A signal generated on the close of
      bar i is filled at the OPEN of bar i+1. A strategy can never trade at
      the price that triggered it.
    * **Fills cross the spread.** Buys fill at ask, sells at bid, using the
      spread actually recorded in the tick data for that bar (falling back to
      the configured minimum only when no spread was recorded), plus
      configured slippage.
    * **Stops are assumed to fill first.** When a bar's range contains both
      the stop and the target, the stop is taken. Without intrabar tick
      replay the true order is unknown, and assuming the favorable one is the
      single most common way backtests flatter themselves.
    * **Stops fill worse than their trigger** by `stop_slippage_price`;
      targets fill at the level (a limit order).
    * **Gaps are honored.** If a bar opens beyond the stop, the fill is the
      open, not the stop level -- so weekend gaps hurt, as they do live.
    * Only one position at a time by default; concurrent signals are recorded
      as skipped with a reason rather than silently dropped.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    # --- sizing ---------------------------------------------------------
    def position_size(self, balance: float, entry: float, stop: float) -> tuple[float, float]:
        """Volume sized so a stop-out costs exactly risk_per_trade_pct.

        Returns (volume, risk_amount). Volume is floored to the instrument's
        step, which is why the realized risk can be slightly under target.
        """
        spec = self._config.instrument
        risk_budget = balance * (self._config.risk.risk_per_trade_pct / 100.0)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0, 0.0

        raw_volume = risk_budget / (stop_distance * spec.contract_size)
        steps = int(raw_volume / spec.volume_step)
        volume = round(steps * spec.volume_step, 8)
        risk_amount = volume * stop_distance * spec.contract_size
        return volume, risk_amount

    # --- execution ------------------------------------------------------
    def _entry_fill(self, bar: pd.Series, side: Side) -> tuple[float, float]:
        """Fill price at the next bar's open, crossing the spread."""
        costs = self._config.costs
        spread = bar.get("spread_mean")
        if spread is None or pd.isna(spread) or spread <= 0:
            spread = costs.fallback_spread_price
        half = spread / 2.0
        mid_open = float(bar["open"])
        if side == Side.BUY:
            price = mid_open + half + costs.slippage_price
        else:
            price = mid_open - half - costs.slippage_price
        return price, abs(price - mid_open)

    def _resolve_exit(
        self, bar: pd.Series, side: Side, stop: float, target: float
    ) -> tuple[float, str] | None:
        """Decide whether this bar closes the position, and at what price."""
        costs = self._config.costs
        high, low, open_price = float(bar["high"]), float(bar["low"]), float(bar["open"])

        if side == Side.BUY:
            # Gap through the stop: fill at the open, not the stop level.
            if open_price <= stop:
                return open_price - costs.stop_slippage_price, "STOP_LOSS"
            if open_price >= target:
                return open_price, "TAKE_PROFIT"
            if low <= stop:
                return stop - costs.stop_slippage_price, "STOP_LOSS"
            if high >= target:
                return target, "TAKE_PROFIT"
        else:
            if open_price >= stop:
                return open_price + costs.stop_slippage_price, "STOP_LOSS"
            if open_price <= target:
                return open_price, "TAKE_PROFIT"
            if high >= stop:
                return stop + costs.stop_slippage_price, "STOP_LOSS"
            if low <= target:
                return target, "TAKE_PROFIT"
        return None

    def _pnl(self, side: Side, entry: float, exit_price: float, volume: float) -> float:
        move = (exit_price - entry) if side == Side.BUY else (entry - exit_price)
        return move * volume * self._config.instrument.contract_size

    def _limit_fill_index(
        self, bars: pd.DataFrame, plan: ExecutionPlan, start_index: int
    ) -> tuple[int | None, str]:
        """When, if ever, a resting limit entry is touched.

        Explicit and deterministic semantics for an AI-modified entry, which
        is typically a pullback price the market has not reached yet:

        * The order rests from the bar AFTER the signal (same one-bar delay as
          a market entry -- an agent's decision cannot be acted on inside the
          bar that produced the signal).
        * A BUY fills when the bar's ASK low reaches the limit; a SELL when
          its BID high does. The spread is crossed to get filled, exactly as
          for a market order, so a limit entry is not quietly cheaper.
        * A fill is booked AT THE LIMIT PRICE even when the bar gapped through
          it. Real fills on a gap are better than the limit; assuming the
          limit is the pessimistic choice.
        * Exit resolution then starts on the fill bar itself, so a bar that
          touches the entry and the stop is a loss. Without tick replay the
          within-bar order is unknown and this is the unfavorable assumption.
        * The order is cancelled if the take-profit is reached before the
          entry -- the move happened without us, and pretending we could
          still get on board afterwards is the flattering assumption.
        * It expires unfilled after `limit_expiry_bars`. An unfilled order is
          recorded as a skipped signal, never as a fill at some later price.
        """
        expiry = plan.limit_expiry_bars or self._config.risk.modified_limit_expiry_bars
        costs = self._config.costs
        last_index = min(len(bars) - 1, start_index + expiry - 1)
        if start_index > last_index:
            return None, "no bars remain for the modified limit entry to fill in"

        for offset in range(start_index, last_index + 1):
            bar = bars.iloc[offset]
            spread = bar.get("spread_mean")
            if spread is None or pd.isna(spread) or spread <= 0:
                spread = costs.fallback_spread_price
            half = float(spread) / 2.0
            high, low = float(bar["high"]), float(bar["low"])

            if plan.side == Side.BUY:
                touched = (low + half) <= plan.entry
                target_first = high >= plan.take_profit
            else:
                touched = (high - half) >= plan.entry
                target_first = low <= plan.take_profit

            if touched:
                return offset, "filled"
            if target_first:
                return None, (
                    "modified limit entry cancelled: price reached the take-profit "
                    "before the entry was touched"
                )

        return None, (
            f"modified limit entry never touched within {expiry} bars and expired"
        )

    def _excursions(
        self,
        bars: pd.DataFrame,
        side: Side,
        entry_price: float,
        entry_index: int,
        exit_index: int,
        stop_distance: float,
    ) -> tuple[float | None, float | None]:
        """Best and worst the position ever looked, in R.

        Measured on bar extremes between fill and exit inclusive. Bar highs
        and lows understate intrabar travel, so these are conservative
        magnitudes, not exact ones.
        """
        if stop_distance <= 0 or exit_index < entry_index:
            return None, None
        window = bars.iloc[entry_index : exit_index + 1]
        high = float(window["high"].max())
        low = float(window["low"].min())
        if side == Side.BUY:
            favorable, adverse = high - entry_price, entry_price - low
        else:
            favorable, adverse = entry_price - low, high - entry_price
        return (
            round(max(favorable, 0.0) / stop_distance, 4),
            round(max(adverse, 0.0) / stop_distance, 4),
        )

    def simulate_signal(
        self, bars: pd.DataFrame, signal: StrategySignal, balance: float
    ) -> tuple[Trade | None, str]:
        """Execute one signal at its own levels, filled at the next bar's open.

        Thin wrapper over `simulate_plan` so the baseline run, the AI run and
        the counterfactual scoring of rejected signals all go through exactly
        one execution path.
        """
        return self.simulate_plan(bars, ExecutionPlan.from_signal(signal), balance)

    def simulate_plan(
        self, bars: pd.DataFrame, plan: ExecutionPlan, balance: float
    ) -> tuple[Trade | None, str]:
        """Execute one plan against the bar series.

        Used for real execution AND for computing the counterfactual result of
        a signal the AI rejected -- identical mechanics either way, so
        rejected-trade analysis is apples to apples.
        """
        signal = plan.signal
        first_index = signal.bar_index + 1  # next-bar execution
        if first_index >= len(bars):
            return None, "signal on final bar: no next bar to execute at"

        if plan.fill_kind == FILL_LIMIT_TOUCH:
            fill_index, note = self._limit_fill_index(bars, plan, first_index)
            if fill_index is None:
                return None, note
            entry_index = fill_index
            entry_bar = bars.iloc[entry_index]
            # A resting order fills at its price, not at the bar's open.
            entry_price = plan.entry
            slippage = 0.0
            # The stop and target were chosen around this exact price, so they
            # are used as given rather than re-derived.
            stop, target = plan.stop_loss, plan.take_profit
            exit_scan_start = entry_index
        else:
            entry_index = first_index
            entry_bar = bars.iloc[entry_index]
            entry_price, slippage = self._entry_fill(entry_bar, plan.side)
            # Re-derive stop/target distances around the ACTUAL fill so the
            # risk stays the intended multiple of ATR rather than drifting
            # with slippage.
            stop_distance = plan.stop_distance
            target_distance = plan.target_distance
            if plan.side == Side.BUY:
                stop = entry_price - stop_distance
                target = entry_price + target_distance
            else:
                stop = entry_price + stop_distance
                target = entry_price - target_distance
            exit_scan_start = entry_index

        stop_distance = abs(entry_price - stop)
        target_distance = abs(target - entry_price)

        volume, risk_amount = self.position_size(balance, entry_price, stop)
        if volume < self._config.instrument.min_volume:
            if self._config.risk.skip_if_below_min_volume:
                return None, (
                    f"position size {volume} below instrument minimum "
                    f"{self._config.instrument.min_volume}"
                )
            volume = self._config.instrument.min_volume
            risk_amount = volume * stop_distance * self._config.instrument.contract_size

        commission = (
            volume * self._config.costs.commission_per_lot_per_side * 2
        )  # both sides

        exit_price: float | None = None
        exit_reason = "END_OF_DATA"
        exit_time: datetime | None = None
        bars_held = 0
        exit_index = len(bars) - 1
        # The bot's time stop. Measured in REAL elapsed minutes, not bar count:
        # FX data has genuine multi-day gaps where the market is closed, and
        # counting bars would compress a weekend into one step and hold a
        # position far longer than the live bot would.
        time_stop = self._config.risk.time_stop_minutes
        entry_stamp = entry_bar["timestamp"]

        for offset in range(exit_scan_start, len(bars)):
            bar = bars.iloc[offset]
            bars_held = offset - entry_index + 1
            resolved = self._resolve_exit(bar, plan.side, stop, target)
            if resolved is not None:
                exit_price, exit_reason = resolved
                exit_time = bar["timestamp"].to_pydatetime()
                exit_index = offset
                break
            if time_stop and offset > exit_scan_start:
                elapsed = (bar["timestamp"] - entry_stamp).total_seconds() / 60.0
                if elapsed >= time_stop:
                    # Stop and target are checked first, so a bar that hits one
                    # of them on the time-stop bar exits there rather than here.
                    exit_price = float(bar["close"])
                    exit_reason = "TIME_STOP"
                    exit_time = bar["timestamp"].to_pydatetime()
                    exit_index = offset
                    break

        if exit_price is None:  # still open when data ran out
            last_bar = bars.iloc[-1]
            exit_price = float(last_bar["close"])
            exit_time = last_bar["timestamp"].to_pydatetime()
            exit_reason = "END_OF_DATA"

        mfe_r, mae_r = self._excursions(
            bars, plan.side, entry_price, entry_index, exit_index, stop_distance
        )

        gross = self._pnl(plan.side, entry_price, exit_price, volume)
        net = gross - commission
        entry_time = entry_bar["timestamp"].to_pydatetime()

        trade = Trade(
            trade_id=self._trade_id(signal),
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=plan.side,
            entry_time=entry_time,
            entry_price=entry_price,
            requested_entry=plan.entry,
            entry_slippage=slippage,
            stop_loss=stop,
            take_profit=target,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            volume=volume,
            gross_profit=gross,
            commission=commission,
            profit=net,
            r_multiple=(net / risk_amount) if risk_amount else 0.0,
            planned_risk_reward=(target_distance / stop_distance) if stop_distance else 0.0,
            risk_amount=risk_amount,
            duration_minutes=(
                (exit_time - entry_time).total_seconds() / 60.0 if exit_time else None
            ),
            bars_held=bars_held,
            entry_reason=signal.entry_reason,
            market_conditions=signal.market_conditions,
            balance_before=balance,
            balance_after=balance + net,
            max_favorable_excursion_r=mfe_r,
            max_adverse_excursion_r=mae_r,
            was_modified=plan.was_modified,
            modified_fill_kind=plan.fill_kind if plan.was_modified else None,
            original_entry=signal.entry if plan.was_modified else None,
            original_stop_loss=signal.stop_loss if plan.was_modified else None,
            original_take_profit=signal.take_profit if plan.was_modified else None,
        )
        return trade, "executed"

    # --- full run -------------------------------------------------------
    def run(
        self,
        bars: pd.DataFrame,
        signals: list[StrategySignal],
        run_id: str,
        run_kind: str = "baseline",
    ) -> BacktestResult:
        """Execute every signal in order, respecting concurrency and daily
        limits, and tracking equity sequentially so position sizes compound
        exactly as they would live."""
        balance = self._config.risk.initial_balance
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
        trades_per_day: dict = {}

        for signal in sorted(signals, key=lambda s: s.signal_time):
            day = signal.signal_time.date()
            taken_today = trades_per_day.get(day, 0)

            if open_until is not None and signal.signal_time <= open_until:
                skipped.append(
                    self._skip(signal, "a position was already open", "engine_constraint")
                )
                continue
            if taken_today >= self._config.risk.max_trades_per_day:
                skipped.append(
                    self._skip(
                        signal,
                        f"daily trade cap of {self._config.risk.max_trades_per_day} reached",
                        "engine_constraint",
                    )
                )
                continue

            trade, note = self.simulate_signal(bars, signal, balance)
            if trade is None:
                skipped.append(self._skip(signal, note, "engine_constraint"))
                continue

            trades.append(trade)
            balance = trade.balance_after
            open_until = trade.exit_time
            trades_per_day[day] = taken_today + 1
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
            initial_balance=self._config.risk.initial_balance,
            final_balance=balance,
            period_start=bars["timestamp"].iloc[0].to_pydatetime(),
            period_end=bars["timestamp"].iloc[-1].to_pydatetime(),
            signals_generated=len(signals),
        )

    @staticmethod
    def _skip(signal: StrategySignal, reason: str, by: str) -> SkippedSignal:
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
        )

    @staticmethod
    def _trade_id(signal: StrategySignal) -> str:
        return hashlib.sha1(
            f"{signal.signal_id}|{signal.signal_time.isoformat()}".encode()
        ).hexdigest()[:12]
