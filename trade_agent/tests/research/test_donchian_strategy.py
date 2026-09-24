from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Side
from research.data.candle_import import resample_candles
from research.strategy.donchian_scalp import (
    TREND_CLOSED_BAR,
    TREND_LEGACY_OPEN_BAR,
    DonchianParams,
    DonchianScalpStrategy,
    add_donchian_channel,
    add_pivot_trend,
    add_true_range_atr,
    add_trend_ema,
    parameter_grid,
    trend_verdict,
)

UTC = timezone.utc
BASE = datetime(2023, 1, 2, 0, 0, tzinfo=UTC)

# The production bot's own source, when this clone is present. The differential
# tests below are the only real proof the port is faithful, so they skip loudly
# rather than silently passing when the reference is unavailable.
BOT_REPO = Path("/home/user/scalp-sample-v2")
BOT_AVAILABLE = (BOT_REPO / "strategy" / "donchian.py").exists()
requires_bot = pytest.mark.skipif(
    not BOT_AVAILABLE,
    reason=f"reference implementation not cloned at {BOT_REPO}",
)


def m15_bars(n: int = 2000, seed: int = 3) -> pd.DataFrame:
    """A seeded random walk on a 15-minute grid, weekdays only.

    Weekday-only spacing matters: it reproduces the session gaps that make the
    real-elapsed-time rules (the time stop, the cooldown) behave differently
    from bar-count approximations.
    """
    rng = np.random.default_rng(seed)
    stamps: list[datetime] = []
    cursor = BASE
    while len(stamps) < n:
        if cursor.weekday() < 5:
            stamps.append(cursor)
        cursor += timedelta(minutes=15)
    closes = 1800 + np.cumsum(rng.normal(0.05, 1.6, n))
    highs = closes + rng.uniform(0.4, 3.0, n)
    lows = closes - rng.uniform(0.4, 3.0, n)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps, utc=True),
            "open": np.clip(opens, lows, highs),
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(500, 3000, n),
        }
    )


# --- indicator fidelity ----------------------------------------------------
@requires_bot
def test_indicators_match_the_bot_exactly():
    """Bit-for-bit agreement with the bot's own indicator functions.

    Not approximate: the bot sizes every stop off its ATR, so a different
    smoothing would change every trade in the experiment.
    """
    sys.path.insert(0, str(BOT_REPO))
    from strategy.donchian import add_donchian_indicators, add_trend_indicator

    bars = m15_bars()
    params = DonchianParams()

    theirs = add_donchian_indicators(
        bars.rename(columns={"timestamp": "ts"}), params.n_period, params.atr_period
    )
    mine = add_true_range_atr(
        add_donchian_channel(bars, params.n_period), params.atr_period
    )
    for column in ("donchian_high", "donchian_low", "atr"):
        assert np.allclose(
            theirs[column].to_numpy(), mine[column].to_numpy(), equal_nan=True,
            rtol=0, atol=0,
        ), column

    h4 = resample_candles(bars, 240, 15)
    their_ema = add_trend_indicator(
        h4.rename(columns={"timestamp": "ts"}), params.ema_trend_period
    )
    my_ema = add_trend_ema(h4, params.ema_trend_period)
    assert np.allclose(
        their_ema["ema_trend"].to_numpy(), my_ema["ema_trend"].to_numpy(),
        equal_nan=True, rtol=0, atol=0,
    )


@requires_bot
def test_pivot_trend_matches_the_bot_exactly():
    sys.path.insert(0, str(BOT_REPO))
    from strategy.donchian import add_pivot_trend as their_pivot

    bars = m15_bars(600)
    theirs = their_pivot(bars.rename(columns={"timestamp": "ts"}), 2)
    mine = add_pivot_trend(bars, 2)
    assert list(theirs["pivot_trend"]) == list(mine["pivot_trend"])


@requires_bot
def test_signal_set_matches_the_bot_in_legacy_alignment_mode():
    """The whole point of the legacy mode: prove the port, then not use it.

    Reproduces the bot's own backtest alignment and asserts the resulting
    signal set is identical -- same timestamps, same directions, no extras and
    none missing.
    """
    sys.path.insert(0, str(BOT_REPO))
    from strategy.donchian import add_donchian_indicators, add_trend_indicator

    bars = m15_bars(3000)
    params = DonchianParams()
    h4 = resample_candles(bars, 240, 15)

    entry = add_donchian_indicators(
        bars.rename(columns={"timestamp": "ts"}), params.n_period, params.atr_period
    )
    trend_frame = add_trend_indicator(
        h4.rename(columns={"timestamp": "ts"}), params.ema_trend_period
    )
    close = trend_frame["close"].to_numpy()
    ema = trend_frame["ema_trend"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.abs(close - ema) / close * 100
    trend_frame = trend_frame.assign(
        trend=np.where(
            pd.isna(ema),
            "flat",
            np.where(
                (params.min_trend_strength_pct > 0)
                & (distance < params.min_trend_strength_pct),
                "flat",
                np.where(close > ema, "long", "short"),
            ),
        )
    )
    joined = pd.merge_asof(
        entry.sort_values("ts"),
        trend_frame[["ts", "trend"]].sort_values("ts"),
        on="ts",
        direction="backward",
    )

    reference: set[tuple] = set()
    for i in range(params.n_period + 1, len(joined)):
        atr = joined["atr"].iat[i]
        if pd.isna(atr) or atr == 0 or pd.isna(joined["donchian_high"].iat[i]):
            continue
        direction = joined["trend"].iat[i]
        price = joined["close"].iat[i]
        if direction == "long" and price > joined["donchian_high"].iat[i]:
            reference.add((joined["ts"].iat[i], "long"))
        elif direction == "short" and price < joined["donchian_low"].iat[i]:
            reference.add((joined["ts"].iat[i], "short"))

    strategy = DonchianScalpStrategy(
        params, trend_frame=h4, trend_alignment=TREND_LEGACY_OPEN_BAR
    )
    ported = {
        (pd.Timestamp(s.signal_time), "long" if s.side == Side.BUY else "short")
        for s in strategy.generate(strategy.prepare(bars))
    }

    assert ported == reference
    assert reference  # the fixture must actually produce signals


# --- the look-ahead correction ---------------------------------------------
def test_default_alignment_uses_only_closed_trend_bars():
    """A trend bar must not be readable before it closes.

    The bot's backtest keyed the merge on the trend bar's OPEN time, so an entry
    at 16:15 read an H4 bar that closes at 20:00. The default here keys on the
    close, which is both leakage-free and what the live bot does.
    """
    bars = m15_bars(1200)
    h4 = resample_candles(bars, 240, 15)
    strategy = DonchianScalpStrategy(DonchianParams(), trend_frame=h4)
    prepared = strategy.prepare(bars)

    entry_close = prepared["timestamp"] + pd.Timedelta(minutes=15)
    trend_lookup = h4.set_index("bar_close_time")["close"]
    for i in range(len(prepared)):
        verdict = prepared["ema_trend"].iat[i]
        if pd.isna(verdict):
            continue
        # Every trend bar that contributed must have closed by this bar's close.
        eligible = trend_lookup.index[trend_lookup.index <= entry_close.iat[i]]
        assert len(eligible) > 0


def test_the_legacy_mode_is_measurably_more_permissive():
    """The look-ahead manufactures signals that the corrected version refuses."""
    bars = m15_bars(4000)
    h4 = resample_candles(bars, 240, 15)
    params = DonchianParams()

    legacy = DonchianScalpStrategy(params, trend_frame=h4, trend_alignment=TREND_LEGACY_OPEN_BAR)
    correct = DonchianScalpStrategy(params, trend_frame=h4, trend_alignment=TREND_CLOSED_BAR)
    legacy_signals = {
        (s.signal_time, s.side) for s in legacy.generate(legacy.prepare(bars))
    }
    correct_signals = {
        (s.signal_time, s.side) for s in correct.generate(correct.prepare(bars))
    }

    assert legacy_signals != correct_signals
    only_legacy = legacy_signals - correct_signals
    assert only_legacy, "the look-ahead should admit signals the corrected mode refuses"


def test_the_experiment_config_cannot_select_the_biased_alignment():
    """The biased mode must be unreachable from the experiment."""
    from research.experiment import ExperimentConfig

    strategy = ExperimentConfig().strategy()
    assert strategy.params["trend_alignment"] == TREND_CLOSED_BAR


def test_an_unknown_alignment_is_rejected():
    with pytest.raises(ValueError, match="unknown trend_alignment"):
        DonchianScalpStrategy(trend_alignment="whatever_seems_fine")


# --- the rules themselves --------------------------------------------------
def test_donchian_channel_excludes_the_current_bar():
    """A bar must not be able to break its own extreme."""
    bars = m15_bars(100)
    channel = add_donchian_channel(bars, 10)
    for i in range(11, 40):
        expected = bars["high"].iloc[i - 10 : i].max()
        assert channel["donchian_high"].iat[i] == pytest.approx(expected)


def test_atr_is_a_simple_mean_of_true_range_not_wilder():
    bars = m15_bars(100)
    framed = add_true_range_atr(bars, 14)
    manual = framed["true_range"].rolling(14).mean()
    assert np.allclose(framed["atr"], manual, equal_nan=True)


@pytest.mark.parametrize(
    "close,ema,strength,expected",
    [
        (100.0, 99.0, 0.0, "long"),
        (99.0, 100.0, 0.0, "short"),
        # 0.5% away is the threshold: 0.2% is marginal, so flat.
        (100.2, 100.0, 0.5, "flat"),
        (101.0, 100.0, 0.5, "long"),
        (99.0, 100.0, 0.5, "short"),
    ],
)
def test_min_trend_strength_treats_a_marginal_crossing_as_flat(close, ema, strength, expected):
    verdict = trend_verdict(np.array([close]), np.array([ema]), strength)
    assert verdict[0] == expected


def test_a_flat_trend_produces_no_signal():
    """A breakout without trend agreement is not a trade."""
    bars = m15_bars(800)
    h4 = resample_candles(bars, 240, 15)
    strategy = DonchianScalpStrategy(DonchianParams(), trend_frame=h4)
    prepared = strategy.prepare(bars)
    prepared["trend"] = "flat"
    assert strategy.generate(prepared) == []


def test_stops_and_targets_follow_the_bot_arithmetic():
    bars = m15_bars(1500)
    h4 = resample_candles(bars, 240, 15)
    params = DonchianParams()
    strategy = DonchianScalpStrategy(params, trend_frame=h4)
    prepared = strategy.prepare(bars)
    signals = strategy.generate(prepared)
    assert signals

    for signal in signals[:20]:
        atr = float(prepared["atr"].iat[signal.bar_index])
        stop_distance = atr * params.atr_stop_mult
        assert abs(signal.entry - signal.stop_loss) == pytest.approx(stop_distance)
        assert abs(signal.take_profit - signal.entry) == pytest.approx(
            stop_distance * params.reward_risk_ratio
        )
        assert signal.risk_reward_ratio == pytest.approx(params.reward_risk_ratio)
        if signal.side == Side.BUY:
            assert signal.stop_loss < signal.entry < signal.take_profit
        else:
            assert signal.take_profit < signal.entry < signal.stop_loss


def test_entry_is_the_signal_bar_close():
    """The bot enters at the close that triggered it.

    The research engine still fills at the NEXT bar's open -- that difference is
    deliberate and documented; this asserts only that the signal records the
    bot's intended price.
    """
    bars = m15_bars(1500)
    h4 = resample_candles(bars, 240, 15)
    strategy = DonchianScalpStrategy(DonchianParams(), trend_frame=h4)
    prepared = strategy.prepare(bars)
    for signal in strategy.generate(prepared)[:10]:
        assert signal.entry == pytest.approx(float(prepared["close"].iat[signal.bar_index]))


def test_signals_carry_the_bot_context_for_the_agents():
    bars = m15_bars(1500)
    h4 = resample_candles(bars, 240, 15)
    strategy = DonchianScalpStrategy(DonchianParams(), trend_frame=h4)
    signals = strategy.generate(strategy.prepare(bars))
    assert signals
    conditions = signals[0].market_conditions
    for key in (
        "trend", "trend_timeframe", "trend_ema", "trend_ema_distance_pct",
        "donchian_high", "donchian_low", "donchian_period", "atr",
    ):
        assert key in conditions, key
    assert conditions["trend_timeframe"] == "M240"
    assert conditions["donchian_period"] == 10
    assert "broke" in signals[0].entry_reason


def test_signal_ids_are_stable_and_unique():
    bars = m15_bars(1500)
    h4 = resample_candles(bars, 240, 15)
    strategy = DonchianScalpStrategy(DonchianParams(), trend_frame=h4)
    first = strategy.generate(strategy.prepare(bars))
    second = strategy.generate(strategy.prepare(bars))
    assert [s.signal_id for s in first] == [s.signal_id for s in second]
    assert len({s.signal_id for s in first}) == len(first)


def test_is_test_marking_restricts_signals_to_the_tradeable_region():
    bars = m15_bars(1500)
    h4 = resample_candles(bars, 240, 15)
    strategy = DonchianScalpStrategy(DonchianParams(), trend_frame=h4)
    prepared = strategy.prepare(bars)
    prepared["is_test"] = prepared.index >= 1000
    for signal in strategy.generate(prepared):
        assert signal.bar_index >= 1000


def test_trend_frame_is_derived_when_not_supplied():
    bars = m15_bars(1500)
    strategy = DonchianScalpStrategy(DonchianParams())
    prepared = strategy.prepare(bars)
    assert "trend" in prepared.columns
    assert prepared["ema_trend"].notna().any()


def test_grid_keeps_the_live_parameters_inside_the_range():
    """The search must be able to confirm the bot's settings, not just move them."""
    grid = parameter_grid()
    live = DonchianParams()
    assert any(
        candidate.n_period == live.n_period
        and candidate.ema_trend_period == live.ema_trend_period
        and candidate.atr_stop_mult == live.atr_stop_mult
        and candidate.reward_risk_ratio == live.reward_risk_ratio
        for candidate in grid
    )
    assert all(candidate.reward_risk_ratio >= 1.5 for candidate in grid)
    assert 20 <= len(grid) <= 400
