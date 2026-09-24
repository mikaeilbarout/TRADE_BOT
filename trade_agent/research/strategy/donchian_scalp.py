from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.models.enums import Side
from research.strategy.base import Strategy, StrategySignal

"""The real trading bot: Donchian breakout with a higher-timeframe trend filter.

Ported from `scalp-sample-v2`, the M15 profile (`mt5/profiles/profile_m15.py`
and `strategy/donchian.py`). This replaces the placeholder strategy that stood
in for it while the bot was unavailable. The rules are the bot's, not an
interpretation of them:

  1. Donchian channel over `n_period` bars, EXCLUDING the current bar
     (`.shift(1)`), so a bar cannot break its own extreme.
  2. Higher-timeframe trend: price above/below a single EMA on the trend
     timeframe, and at least `min_trend_strength_pct` away from it -- a
     marginal crossing counts as "flat" and blocks the trade.
  3. Long when the trend is long AND the close exceeds the Donchian high;
     short on the mirror condition. No RSI, no MACD.
  4. Stop at `atr_stop_mult` x ATR(14), target at `reward_risk_ratio` x the
     stop distance.

ATR here is a simple mean of True Range, not Wilder's smoothing, because that
is what the bot computes; using the more common Wilder ATR would change every
stop distance and therefore every trade.

--- ONE DELIBERATE CORRECTION TO THE BOT'S BACKTEST --------------------------

The bot's own backtest aligns the trend timeframe with

    pd.merge_asof(df_low, df_high[["ts", "trend"]], on="ts", direction="backward")

where `ts` is the bar's OPEN time (MT5's `copy_rates_range` returns the open
time, and `mt5/export_history.py` writes it unchanged). An M15 bar at 16:15
therefore matches the H4 bar stamped 16:00 -- a bar that does not close until
20:00. Its `close`, and so its EMA and trend verdict, are not knowable at 16:15.
The bot's backtest reads up to four hours into the future on every single bar.

The bot's LIVE code does not do this. `mt5/live_bot_mt5.py` fetches
`count + 1` bars and drops the forming one specifically so that `.iloc[-1]` is
"a real closed bar" (its own comment), then reads the trend from that. Live is
correct; the backtest is not.

So `trend_alignment="closed_bar"` (the default here) uses the last trend bar to
have CLOSED at or before the entry bar's close -- which is simultaneously the
leakage-free choice AND the one that matches live behaviour.

`trend_alignment="legacy_open_bar"` reproduces the bot's backtest exactly. It
exists so a differential test can prove this port is faithful to the original,
and it is look-ahead biased by construction. It must not be used for the
experiment, and `ExperimentConfig` does not offer it.
"""

# Alignment modes. Named rather than boolean so the biased one cannot be
# selected by passing `True` to something that reads like a good idea.
TREND_CLOSED_BAR = "closed_bar"
TREND_LEGACY_OPEN_BAR = "legacy_open_bar"


class DonchianParams(BaseModel):
    """The bot's M15 profile parameters, as configured live.

    Defaults are the values in `mt5/profiles/profile_m15.py` at the commit this
    was ported from, including the overrides that profile applies to the base
    `RiskConfig`. They are the bot's live settings, not a starting guess -- but
    the out-of-sample methodology still requires them to be re-selected on the
    development period and sealed before the final 30% is touched, because they
    were tuned on the whole history.
    """

    # --- entry channel ---
    n_period: int = 10
    atr_period: int = 14
    # --- higher-timeframe trend ---
    ema_trend_period: int = 30
    trend_timeframe_minutes: int = 240      # H4
    min_trend_strength_pct: float = 0.5
    # --- exits (profile overrides of the base risk config) ---
    atr_stop_mult: float = 3.0
    reward_risk_ratio: float = 3.0
    time_stop_minutes: int = 10080          # 7 days
    # --- throttles the live bot enforces ---
    cooldown_losses_to_trigger: int = 3
    cooldown_hours: float = 2.0
    # --- dormant in the live profile, kept because the bot keeps it ---
    require_pivot_confirm: bool = False
    pivot_k: int = 2

    def to_dict(self) -> dict:
        return self.model_dump()

    @property
    def implied_rr(self) -> float:
        return self.reward_risk_ratio


# --- indicators, matching the bot's implementations exactly -----------------
def add_donchian_channel(frame: pd.DataFrame, n_period: int) -> pd.DataFrame:
    """Donchian high/low over the PREVIOUS n_period bars.

    `.shift(1)` is what makes this causal: the channel a bar is tested against
    never includes that bar.
    """
    out = frame.copy()
    out["donchian_high"] = out["high"].rolling(n_period).max().shift(1)
    out["donchian_low"] = out["low"].rolling(n_period).min().shift(1)
    return out


def add_true_range_atr(frame: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    """ATR as a SIMPLE mean of True Range -- the bot's definition.

    Deliberately not Wilder's smoothing. The bot sizes every stop off this
    value, so substituting the more usual ATR would silently change every
    trade's stop, target and position size.
    """
    out = frame.copy()
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["true_range"] = true_range
    out["atr"] = true_range.rolling(atr_period).mean()
    return out


def add_trend_ema(frame: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    """Single EMA on the trend timeframe (not a fast/slow crossover)."""
    out = frame.copy()
    out["ema_trend"] = out["close"].ewm(span=ema_period, adjust=False).mean()
    return out


def trend_verdict(
    close: np.ndarray, ema: np.ndarray, min_strength_pct: float
) -> np.ndarray:
    """Vectorised trend direction: long / short / flat.

    `min_strength_pct` treats a marginal crossing as flat, which is the filter
    the bot's own notes credit with its largest single improvement.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        distance_pct = np.abs(close - ema) / close * 100
    return np.where(
        pd.isna(ema),
        "flat",
        np.where(
            (min_strength_pct > 0) & (distance_pct < min_strength_pct),
            "flat",
            np.where(close > ema, "long", "short"),
        ),
    )


def find_pivots(high: np.ndarray, low: np.ndarray, k: int = 2):
    """Fractal pivot highs/lows over a (2k+1)-bar centred window."""
    window = 2 * k + 1
    rolling_max = (
        pd.Series(high).rolling(window, center=True, min_periods=window).max().to_numpy()
    )
    rolling_min = (
        pd.Series(low).rolling(window, center=True, min_periods=window).min().to_numpy()
    )
    is_high = high == rolling_max
    is_low = low == rolling_min
    is_high[np.isnan(rolling_max)] = False
    is_low[np.isnan(rolling_min)] = False
    return is_high, is_low


def add_pivot_trend(frame: pd.DataFrame, pivot_k: int = 2) -> pd.DataFrame:
    """Swing-structure trend from a chained HH/HL/LH/LL pivot sequence.

    A pivot is only confirmed `pivot_k` bars after it forms, which is why the
    loop reads index `i - pivot_k`: it cannot be known to be a local extreme
    before then. Dormant in the live profile (`require_pivot_confirm=False`),
    ported because the bot keeps the capability.
    """
    out = frame.copy()
    high = out["high"].to_numpy()
    low = out["low"].to_numpy()
    count = len(out)
    is_high, is_low = find_pivots(high, low, pivot_k)

    chain_last_type: str | None = None
    last_high = previous_high = np.nan
    last_low = previous_low = np.nan
    trend = np.full(count, "flat", dtype=object)

    for i in range(count):
        confirmed = i - pivot_k
        if confirmed >= 0:
            if is_high[confirmed]:
                value = high[confirmed]
                if chain_last_type == "high":
                    if value > last_high:
                        last_high = value
                else:
                    previous_high, last_high = last_high, value
                    chain_last_type = "high"
            if is_low[confirmed]:
                value = low[confirmed]
                if chain_last_type == "low":
                    if value < last_low:
                        last_low = value
                else:
                    previous_low, last_low = last_low, value
                    chain_last_type = "low"
        if not (
            np.isnan(previous_high)
            or np.isnan(last_high)
            or np.isnan(previous_low)
            or np.isnan(last_low)
        ):
            higher_high = last_high > previous_high
            higher_low = last_low > previous_low
            lower_high = last_high < previous_high
            lower_low = last_low < previous_low
            if higher_high and higher_low:
                trend[i] = "long"
            elif lower_high and lower_low:
                trend[i] = "short"

    out["pivot_trend"] = trend
    return out


class DonchianScalpStrategy(Strategy):
    """The production bot's signal logic, as a research Strategy.

    Only signal GENERATION lives here. Fills, position sizing, the time stop,
    the loss-streak cooldown and the daily guard are execution concerns and are
    enforced by `research/backtest/executor.py`, so that Experiment A and
    Experiment B share one execution path.
    """

    name = "donchian_scalp_m15"

    def __init__(
        self,
        params: DonchianParams | None = None,
        symbol: str = "XAUUSD",
        trend_frame: pd.DataFrame | None = None,
        trend_alignment: str = TREND_CLOSED_BAR,
    ) -> None:
        self._params = params or DonchianParams()
        self._symbol = symbol
        self._trend_frame = trend_frame
        if trend_alignment not in (TREND_CLOSED_BAR, TREND_LEGACY_OPEN_BAR):
            raise ValueError(
                f"unknown trend_alignment {trend_alignment!r}; use "
                f"{TREND_CLOSED_BAR!r} or {TREND_LEGACY_OPEN_BAR!r}"
            )
        self._alignment = trend_alignment

    @property
    def params(self) -> dict:
        payload = self._params.to_dict()
        payload["trend_alignment"] = self._alignment
        return payload

    @property
    def typed_params(self) -> DonchianParams:
        return self._params

    def with_trend_frame(self, trend_frame: pd.DataFrame) -> "DonchianScalpStrategy":
        return DonchianScalpStrategy(
            self._params, self._symbol, trend_frame, self._alignment
        )

    # --- preparation ------------------------------------------------------
    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Attach the channel, ATR and the aligned higher-timeframe trend.

        The trend frame is derived here when not supplied, by resampling the
        entry bars -- exact for OHLC, and aligned below so only closed trend
        bars are ever read.
        """
        from research.data.candle_import import resample_candles

        p = self._params
        prepared = add_donchian_channel(candles, p.n_period)
        prepared = add_true_range_atr(prepared, p.atr_period)

        entry_minutes = self._infer_entry_minutes(candles)
        trend_source = self._trend_frame
        if trend_source is None:
            trend_source = resample_candles(
                candles[["timestamp", "open", "high", "low", "close", "volume"]],
                p.trend_timeframe_minutes,
                entry_minutes,
            )
        trend_source = add_trend_ema(trend_source, p.ema_trend_period)
        trend_source = trend_source.copy()
        trend_source["trend_ema_verdict"] = trend_verdict(
            trend_source["close"].to_numpy(),
            trend_source["ema_trend"].to_numpy(),
            p.min_trend_strength_pct,
        )

        if p.require_pivot_confirm:
            trend_source = add_pivot_trend(trend_source, p.pivot_k)
            trend_source["trend"] = np.where(
                (trend_source["trend_ema_verdict"] == "long")
                & (trend_source["pivot_trend"] == "long"),
                "long",
                np.where(
                    (trend_source["trend_ema_verdict"] == "short")
                    & (trend_source["pivot_trend"] == "short"),
                    "short",
                    "flat",
                ),
            )
        else:
            trend_source["trend"] = trend_source["trend_ema_verdict"]

        prepared = self._align_trend(prepared, trend_source, entry_minutes)
        return prepared.reset_index(drop=True)

    def _align_trend(
        self, prepared: pd.DataFrame, trend_source: pd.DataFrame, entry_minutes: int
    ) -> pd.DataFrame:
        """Attach each entry bar's trend verdict.

        In `closed_bar` mode the key is the trend bar's CLOSE time against the
        entry bar's CLOSE time, so a trend bar becomes visible only once it has
        finished -- matching the live bot, and leaking nothing.
        """
        columns = ["trend", "ema_trend", "trend_ema_verdict"]
        if "pivot_trend" in trend_source.columns:
            columns.append("pivot_trend")

        if self._alignment == TREND_LEGACY_OPEN_BAR:
            # The bot's backtest behaviour, reproduced for verification only:
            # keyed on the trend bar's OPEN time, so a forming bar's close is
            # read up to a full trend-bar early.
            left = prepared[["timestamp"]].sort_values("timestamp")
            right = trend_source[["timestamp", *columns]].sort_values("timestamp")
            merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
        else:
            entry_close = prepared["timestamp"] + pd.Timedelta(minutes=entry_minutes)
            left = pd.DataFrame(
                {"_key": entry_close, "_order": np.arange(len(prepared))}
            ).sort_values("_key")
            trend_close = (
                trend_source["bar_close_time"]
                if "bar_close_time" in trend_source.columns
                else trend_source["timestamp"]
                + pd.Timedelta(minutes=self._params.trend_timeframe_minutes)
            )
            # Keep the Series rather than `.values`: taking the numpy array
            # drops the tz and merge_asof then refuses the mismatched key dtype.
            right = trend_source[columns].copy()
            right["_key"] = pd.Series(trend_close).reset_index(drop=True)
            right = right.sort_values("_key")
            merged = (
                pd.merge_asof(left, right, on="_key", direction="backward")
                .sort_values("_order")
                .drop(columns=["_key", "_order"])
                .reset_index(drop=True)
            )

        out = prepared.copy()
        for column in columns:
            out[column] = merged[column].values
        return out

    @staticmethod
    def _infer_entry_minutes(candles: pd.DataFrame) -> int:
        """Bar length from the data's own modal spacing.

        Read from the data rather than configured, so a mislabelled timeframe
        cannot silently shift the trend alignment.
        """
        if len(candles) < 3:
            return 15
        deltas = candles["timestamp"].diff().dt.total_seconds().div(60).dropna()
        if deltas.empty:
            return 15
        return int(deltas.mode().iloc[0])

    # --- signals ----------------------------------------------------------
    def generate(self, prepared: pd.DataFrame) -> list[StrategySignal]:
        p = self._params
        signals: list[StrategySignal] = []

        required = ["donchian_high", "donchian_low", "atr", "trend"]
        usable = prepared[required].notna().all(axis=1)
        tradeable = (
            prepared["is_test"]
            if "is_test" in prepared.columns
            else pd.Series(True, index=prepared.index)
        )

        close = prepared["close"].to_numpy()
        donchian_high = prepared["donchian_high"].to_numpy()
        donchian_low = prepared["donchian_low"].to_numpy()
        atr = prepared["atr"].to_numpy()
        trend = prepared["trend"].to_numpy()
        timestamps = prepared["timestamp"]

        for i in range(len(prepared)):
            if not usable.iat[i] or not bool(tradeable.iat[i]):
                continue
            atr_value = float(atr[i])
            if atr_value <= 0:
                continue

            direction = trend[i]
            side: Side | None = None
            if direction == "long" and close[i] > donchian_high[i]:
                side = Side.BUY
            elif direction == "short" and close[i] < donchian_low[i]:
                side = Side.SELL
            if side is None:
                continue

            entry = float(close[i])
            stop_distance = atr_value * p.atr_stop_mult
            target_distance = stop_distance * p.reward_risk_ratio
            if side == Side.BUY:
                stop_loss = entry - stop_distance
                take_profit = entry + target_distance
                broke = float(donchian_high[i])
            else:
                stop_loss = entry + stop_distance
                take_profit = entry - target_distance
                broke = float(donchian_low[i])

            signal_time = timestamps.iat[i].to_pydatetime()
            signals.append(
                StrategySignal(
                    signal_id=self._signal_id(signal_time, side),
                    bar_index=int(i),
                    signal_time=signal_time,
                    symbol=self._symbol,
                    side=side,
                    entry=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_reason=(
                        f"{direction} trend on M{p.trend_timeframe_minutes} "
                        f"EMA{p.ema_trend_period} (min strength "
                        f"{p.min_trend_strength_pct}%); close {entry:.2f} broke the "
                        f"{p.n_period}-bar "
                        f"{'high' if side == Side.BUY else 'low'} {broke:.2f}; stop "
                        f"{p.atr_stop_mult}x ATR ({atr_value:.2f}), target "
                        f"{p.reward_risk_ratio}R"
                    ),
                    market_conditions=self._market_conditions(
                        prepared, i, atr_value, direction
                    ),
                )
            )

        return signals

    def _market_conditions(
        self, prepared: pd.DataFrame, i: int, atr_value: float, direction: str
    ) -> dict:
        def value(column: str):
            if column not in prepared.columns:
                return None
            raw = prepared[column].iat[i]
            if isinstance(raw, (np.floating, float)):
                return None if pd.isna(raw) else round(float(raw), 5)
            if isinstance(raw, (np.integer, int)):
                return int(raw)
            return None if raw is None else str(raw)

        close = float(prepared["close"].iat[i])
        return {
            "trend": direction,
            "trend_timeframe": f"M{self._params.trend_timeframe_minutes}",
            "trend_ema": value("ema_trend"),
            "trend_ema_distance_pct": (
                round(abs(close - float(prepared["ema_trend"].iat[i])) / close * 100, 4)
                if not pd.isna(prepared["ema_trend"].iat[i])
                else None
            ),
            "pivot_trend": value("pivot_trend"),
            "close": round(close, 3),
            "donchian_high": value("donchian_high"),
            "donchian_low": value("donchian_low"),
            "donchian_period": self._params.n_period,
            "atr": round(atr_value, 4),
            "atr_pct_of_price": round(atr_value / close * 100, 4) if close else None,
            "session": value("session"),
            "spread_mean": value("spread_mean"),
            "volume": value("volume"),
        }

    @staticmethod
    def _signal_id(signal_time, side: Side) -> str:
        raw = f"donchian|{signal_time.isoformat()}|{side.value}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


def parameter_grid(
    n_period: tuple[int, ...] = (10, 20, 40),
    ema_trend_period: tuple[int, ...] = (30, 50),
    min_trend_strength_pct: tuple[float, ...] = (0.0, 0.5),
    atr_stop_mult: tuple[float, ...] = (2.0, 3.0, 4.0),
    reward_risk_ratio: tuple[float, ...] = (1.5, 3.0),
) -> list[DonchianParams]:
    """The bot's own tuning axes, as a bounded grid.

    These are the knobs the bot's notes record sweeping -- channel length, trend
    EMA, the minimum-strength filter, the stop multiple and the reward ratio --
    with the live values inside the range rather than at its edge, so the
    development-set search can either confirm or move them. Combinations whose
    reward:risk falls below the risk engine's floor are dropped rather than
    evaluated and rejected later.
    """
    candidates: list[DonchianParams] = []
    for period, ema, strength, stop_mult, reward in product(
        n_period, ema_trend_period, min_trend_strength_pct, atr_stop_mult, reward_risk_ratio
    ):
        if reward < 1.5:
            continue
        candidates.append(
            DonchianParams(
                n_period=period,
                ema_trend_period=ema,
                min_trend_strength_pct=strength,
                atr_stop_mult=stop_mult,
                reward_risk_ratio=reward,
            )
        )
    return candidates
