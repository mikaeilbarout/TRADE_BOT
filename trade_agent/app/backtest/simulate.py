from __future__ import annotations

from dataclasses import dataclass

from app.models.market_data import Candle


@dataclass
class SimulatedOutcome:
    status: str  # "WIN" | "LOSS" | "NOT_FILLED" | "OPEN"
    r_multiple: float


def simulate_outcome(
    side: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    future_candles: list[Candle],
    immediate_fill: bool = True,
) -> SimulatedOutcome:
    """Deterministic bar-by-bar replay to score one trade using ONLY
    candles that occur after the decision was made -- this is evaluating
    an already-decided trade's result, not feeding information back into
    the decision, so it does not reintroduce look-ahead bias.

    `immediate_fill=True` assumes a market order fills at `entry` on the
    very next candle (matches how the original bot behaves). Set it False
    to require price to actually trade through a (modified) limit entry
    before the position is considered open -- if it never does, the trade
    is marked NOT_FILLED with r_multiple 0.

    When both the stop and target fall inside the same candle's range, the
    stop is assumed to trigger first (the standard conservative backtest
    convention, since intra-candle order is unknown).
    """
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk == 0:
        return SimulatedOutcome("NOT_FILLED", 0.0)

    is_buy = side.upper() == "BUY"
    filled = immediate_fill

    for candle in future_candles:
        if not filled:
            touched_entry = candle.low <= entry <= candle.high
            if touched_entry:
                filled = True
            continue

        if is_buy:
            hit_stop = candle.low <= stop_loss
            hit_target = candle.high >= take_profit
        else:
            hit_stop = candle.high >= stop_loss
            hit_target = candle.low <= take_profit

        if hit_stop:
            return SimulatedOutcome("LOSS", -1.0)
        if hit_target:
            return SimulatedOutcome("WIN", reward / risk)

    if not filled:
        return SimulatedOutcome("NOT_FILLED", 0.0)
    return SimulatedOutcome("OPEN", 0.0)
