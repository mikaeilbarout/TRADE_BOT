from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models.market_data import Candle
from app.services.technical_service import atr, compute_indicators, ema, higher_tf_alignment, macd, rsi


def _candles(closes: list[float]) -> list[Candle]:
    now = datetime.now(timezone.utc)
    out = []
    for i, c in enumerate(closes):
        out.append(
            Candle(
                timestamp=now - timedelta(minutes=len(closes) - i),
                open=c,
                high=c * 1.001,
                low=c * 0.999,
                close=c,
                volume=100,
            )
        )
    return out


def test_ema_converges_toward_recent_values():
    values = [100.0] * 30 + [110.0] * 30
    result = ema(values, 10)
    assert result[-1] > 105.0
    assert result[0] == 100.0


def test_rsi_all_gains_is_100():
    values = [float(i) for i in range(1, 30)]  # strictly increasing
    assert rsi(values, period=14) == 100.0


def test_rsi_insufficient_data_returns_none():
    assert rsi([1.0, 2.0, 3.0], period=14) is None


def test_macd_insufficient_data_returns_none_triplet():
    line, signal, hist = macd([1.0] * 10)
    assert line is None and signal is None and hist is None


def test_atr_positive_for_volatile_series():
    closes = [100 + (i % 5) for i in range(30)]
    candles = _candles(closes)
    value = atr(candles, period=14)
    assert value is not None
    assert value > 0


def test_compute_indicators_returns_trend_field():
    closes = [100.0 + i * 0.1 for i in range(220)]  # steady uptrend
    candles = _candles(closes)
    indicators = compute_indicators("H1", candles)
    assert indicators.trend in {"UPTREND", "DOWNTREND", "RANGE"}
    assert indicators.recent_high is not None
    assert indicators.recent_low is not None


def test_higher_tf_alignment():
    assert higher_tf_alignment("BUY", "UPTREND") is True
    assert higher_tf_alignment("SELL", "UPTREND") is False
    assert higher_tf_alignment("BUY", "DOWNTREND") is False
    assert higher_tf_alignment("SELL", "DOWNTREND") is True
    assert higher_tf_alignment("BUY", "RANGE") is False
