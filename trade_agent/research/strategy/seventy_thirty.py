from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.models.enums import Side
from research.backtest.indicators import compute_indicator_frame
from research.strategy.base import Strategy, StrategySignal

"""SUPERSEDED: this was the placeholder strategy that stood in for the real
trading bot while it was unavailable.

The experiment now runs `research/strategy/donchian_scalp.py`, ported from the
production bot (`scalp-sample-v2`). Nothing in the experiment path imports this
module any more -- `ExperimentConfig`, the optimizer and the research CLI all
use the Donchian strategy. It is kept only because its tests still exercise the
`Strategy` interface and the indicator helpers; it can be deleted on request.
"""


class StrategyParams(BaseModel):
    """SUPERSEDED placeholder parameters -- see the module note below.

    The parameter space fitted on the development period only.

    These defaults are the STARTING POINT of the search, not a result. The
    values actually used for the out-of-sample test are whatever the
    optimizer selects on the first 70% of the data, and they are recorded in
    the strategy seal.
    """

    # Regime filter
    ema_fast: int = 50
    ema_slow: int = 200
    # Entry trigger
    breakout_lookback: int = 20
    require_close_beyond: bool = True
    # Exhaustion guard: don't buy into an already-overbought push
    rsi_period: int = 14
    rsi_long_max: float = 72.0
    rsi_short_min: float = 28.0
    # Volatility band as a trailing percentile of ATR: skip dead and chaotic regimes
    atr_period: int = 14
    atr_percentile_min: float = 0.20
    atr_percentile_max: float = 0.90
    volatility_window: int = 200
    # Exits, expressed in ATR so they adapt to regime rather than fixed points
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 3.0
    # Throttles
    cooldown_bars: int = 8
    allowed_sessions: tuple[str, ...] = ("LONDON", "OVERLAP", "NEW_YORK")

    def to_dict(self) -> dict:
        data = self.model_dump()
        data["allowed_sessions"] = list(self.allowed_sessions)
        return data

    @property
    def implied_rr(self) -> float:
        return self.atr_target_mult / self.atr_stop_mult


class SeventyThirtyStrategy(Strategy):
    """Trend-filtered volatility breakout on XAUUSD 15-minute bars.

    The name refers to the 70/30 DEVELOPMENT METHODOLOGY (parameters fitted
    on the first 70% of history, then frozen and tested on the final 30%),
    not to any 70/30 ratio inside the rules.

    Rules, in order of evaluation at each bar close:

      1. Regime: EMA(fast) vs EMA(slow) decides the only permitted
         direction. Longs in an uptrend, shorts in a downtrend, nothing in
         between -- so the strategy never fights its own trend filter.
      2. Trigger: the close must break beyond the highest high (or lowest
         low) of the previous `breakout_lookback` bars. The lookback window
         excludes the current bar, so a bar cannot break its own extreme.
      3. Exhaustion guard: RSI must not already be extended past
         `rsi_long_max` / `rsi_short_min`, which filters the late entries
         that give breakout systems their worst fills.
      4. Volatility band: ATR's trailing percentile must sit inside
         [atr_percentile_min, atr_percentile_max] -- skipping both dead
         ranges where a breakout is noise and chaotic spikes where stops are
         meaningless. The percentile is trailing, never full-sample, so it
         does not leak.
      5. Session filter: only the configured sessions trade.
      6. Cooldown: at least `cooldown_bars` bars since the last signal,
         which stops one volatile hour producing a cluster of correlated
         trades.

    Stops and targets are ATR multiples measured at the signal bar, giving a
    constant implied R:R while adapting the absolute distances to the
    prevailing volatility.
    """

    name = "seventy_thirty_trend_breakout"

    def __init__(self, params: StrategyParams | None = None, symbol: str = "XAUUSD") -> None:
        self._params = params or StrategyParams()
        self._symbol = symbol

    @property
    def params(self) -> dict:
        return self._params.to_dict()

    @property
    def typed_params(self) -> StrategyParams:
        return self._params

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        p = self._params
        return compute_indicator_frame(
            candles,
            ema_fast=p.ema_fast,
            ema_slow=p.ema_slow,
            rsi_period=p.rsi_period,
            atr_period=p.atr_period,
            breakout_lookback=p.breakout_lookback,
            volatility_window=p.volatility_window,
        )

    def generate(self, prepared: pd.DataFrame) -> list[StrategySignal]:
        p = self._params
        signals: list[StrategySignal] = []
        last_signal_index: int | None = None

        required = ["ema_fast", "ema_slow", "rsi", "atr", "breakout_high", "breakout_low"]
        usable = prepared[required].notna().all(axis=1)
        # `is_test` marks the tradeable region when running out-of-sample with
        # warm-up context; absent it, every bar is tradeable.
        tradeable = (
            prepared["is_test"] if "is_test" in prepared.columns
            else pd.Series(True, index=prepared.index)
        )

        atr_pct = prepared["atr_percentile"]
        for i in prepared.index:
            if not usable.iat[i] or not bool(tradeable.iat[i]):
                continue
            if last_signal_index is not None and (i - last_signal_index) < p.cooldown_bars:
                continue
            if prepared["session"].iat[i] not in p.allowed_sessions:
                continue

            pct = atr_pct.iat[i]
            if pd.isna(pct) or not (p.atr_percentile_min <= pct <= p.atr_percentile_max):
                continue

            close = float(prepared["close"].iat[i])
            atr_value = float(prepared["atr"].iat[i])
            if atr_value <= 0:
                continue

            trend = prepared["trend"].iat[i]
            rsi_value = float(prepared["rsi"].iat[i])
            breakout_high = float(prepared["breakout_high"].iat[i])
            breakout_low = float(prepared["breakout_low"].iat[i])

            side: Side | None = None
            if trend == "UPTREND" and close > breakout_high and rsi_value <= p.rsi_long_max:
                side = Side.BUY
            elif trend == "DOWNTREND" and close < breakout_low and rsi_value >= p.rsi_short_min:
                side = Side.SELL
            if side is None:
                continue

            if side == Side.BUY:
                stop = close - p.atr_stop_mult * atr_value
                target = close + p.atr_target_mult * atr_value
                broke = breakout_high
            else:
                stop = close + p.atr_stop_mult * atr_value
                target = close - p.atr_target_mult * atr_value
                broke = breakout_low

            signal_time = prepared["timestamp"].iat[i].to_pydatetime()
            signals.append(
                StrategySignal(
                    signal_id=self._signal_id(signal_time, side),
                    bar_index=int(i),
                    signal_time=signal_time,
                    symbol=self._symbol,
                    side=side,
                    entry=close,
                    stop_loss=stop,
                    take_profit=target,
                    entry_reason=(
                        f"{trend.lower()} regime (EMA{p.ema_fast} vs EMA{p.ema_slow}); "
                        f"close {close:.2f} broke {p.breakout_lookback}-bar "
                        f"{'high' if side == Side.BUY else 'low'} {broke:.2f}; "
                        f"RSI {rsi_value:.1f} within limit; ATR percentile {pct:.2f} in band"
                    ),
                    market_conditions=self._market_conditions(prepared, i, atr_value, rsi_value),
                )
            )
            last_signal_index = i

        return signals

    def _market_conditions(
        self, prepared: pd.DataFrame, i: int, atr_value: float, rsi_value: float
    ) -> dict:
        def value(column: str):
            if column not in prepared.columns:
                return None
            raw = prepared[column].iat[i]
            if isinstance(raw, (np.floating, float)):
                return None if pd.isna(raw) else round(float(raw), 5)
            if isinstance(raw, (np.integer, int)):
                return int(raw)
            return str(raw)

        close = float(prepared["close"].iat[i])
        return {
            "session": value("session"),
            "trend": value("trend"),
            "structure": value("structure"),
            "close": round(close, 3),
            "ema_fast": value("ema_fast"),
            "ema_slow": value("ema_slow"),
            "ema_50": value("ema_50"),
            "ema_200": value("ema_200"),
            "rsi": round(rsi_value, 2),
            "atr": round(atr_value, 4),
            "atr_pct_of_price": round(atr_value / close * 100, 4) if close else None,
            "atr_percentile": value("atr_percentile"),
            "macd_hist": value("macd_hist"),
            "breakout_high": value("breakout_high"),
            "breakout_low": value("breakout_low"),
            "spread_mean": value("spread_mean"),
            "tick_count": value("tick_count"),
        }

    @staticmethod
    def _signal_id(signal_time, side: Side) -> str:
        raw = f"{signal_time.isoformat()}|{side.value}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


def parameter_grid(
    ema_fast: tuple[int, ...] = (20, 50),
    ema_slow: tuple[int, ...] = (100, 200),
    breakout_lookback: tuple[int, ...] = (12, 20, 40),
    atr_stop_mult: tuple[float, ...] = (1.0, 1.5, 2.0),
    atr_target_mult: tuple[float, ...] = (2.0, 3.0, 4.5),
    cooldown_bars: tuple[int, ...] = (4, 8),
) -> list[StrategyParams]:
    """A deliberately SMALL grid.

    Every additional axis multiplies the number of chances to fit noise in
    the development period. This grid is sized so the winner is chosen from
    a few hundred candidates rather than hundreds of thousands, and
    combinations with a sub-1.5 implied R:R are dropped rather than tested.
    """
    candidates: list[StrategyParams] = []
    for fast, slow, lookback, stop_mult, target_mult, cooldown in product(
        ema_fast, ema_slow, breakout_lookback, atr_stop_mult, atr_target_mult, cooldown_bars
    ):
        if fast >= slow:
            continue
        if target_mult / stop_mult < 1.5:
            continue
        candidates.append(
            StrategyParams(
                ema_fast=fast,
                ema_slow=slow,
                breakout_lookback=lookback,
                atr_stop_mult=stop_mult,
                atr_target_mult=target_mult,
                cooldown_bars=cooldown,
            )
        )
    return candidates
