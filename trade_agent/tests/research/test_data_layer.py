from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from research.data.candles import build_candles, candle_coverage
from research.data.sources.base import TickDataUnavailableError, validate_tick_frame
from research.data.sources.csv_source import CsvTickSource
from research.data.sources.dukascopy import decode_bi5, hour_url
from research.data.split import DataSplit, LeakageError, StrategySeal, guard_reseal

UTC = timezone.utc


def make_ticks(rows: list[tuple[datetime, float, float]]) -> pd.DataFrame:
    """Build a tick frame from (timestamp, bid, ask) triples.

    NOTE: these are hand-written FIXTURES for testing the aggregation
    arithmetic. They are never used as market data for a backtest -- the
    backtest requires real vendor ticks and fails closed without them.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([r[0] for r in rows], utc=True),
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
            "bid_volume": [1.0] * len(rows),
            "ask_volume": [1.0] * len(rows),
        }
    )


# --- Dukascopy bi5 decoding ------------------------------------------------


def test_dukascopy_url_uses_zero_indexed_month():
    """Dukascopy months are 0-indexed: January is 00, December is 11.
    Getting this wrong silently fetches a different month's data."""
    assert hour_url("XAUUSD", datetime(2024, 1, 15, 14, tzinfo=UTC)).endswith(
        "XAUUSD/2024/00/15/14h_ticks.bi5"
    )
    assert hour_url("XAUUSD", datetime(2024, 12, 31, 23, tzinfo=UTC)).endswith(
        "XAUUSD/2024/11/31/23h_ticks.bi5"
    )
    assert hour_url("xauusd", datetime(2024, 6, 1, 0, tzinfo=UTC)).endswith(
        "XAUUSD/2024/05/01/00h_ticks.bi5"
    )


def _encode_bi5(ticks: list[tuple[int, int, int, float, float]]) -> bytes:
    packer = struct.Struct(">IIIff")
    return lzma.compress(b"".join(packer.pack(*t) for t in ticks))


def test_decode_bi5_roundtrip_applies_point_divisor():
    hour = datetime(2024, 6, 3, 10, tzinfo=UTC)
    payload = _encode_bi5(
        [
            (0, 2345_678, 2345_178, 1.5, 2.5),      # ask 2345.678, bid 2345.178
            (1_500, 2346_000, 2345_500, 1.0, 1.0),  # 1.5s into the hour
        ]
    )
    frame = decode_bi5(payload, hour, point_divisor=1000.0)

    assert len(frame) == 2
    assert frame["ask"].iloc[0] == pytest.approx(2345.678)
    assert frame["bid"].iloc[0] == pytest.approx(2345.178)
    assert frame["timestamp"].iloc[0] == pd.Timestamp(hour)
    assert frame["timestamp"].iloc[1] == pd.Timestamp(hour + timedelta(milliseconds=1500))


def test_decode_bi5_empty_payload_is_valid_closed_market():
    frame = decode_bi5(b"", datetime(2024, 6, 8, 3, tzinfo=UTC), 1000.0)
    assert frame.empty  # weekend: no ticks is a real answer, not an error


def test_decode_bi5_rejects_corrupt_payload():
    with pytest.raises(TickDataUnavailableError):
        decode_bi5(b"not-lzma-at-all", datetime(2024, 6, 3, 10, tzinfo=UTC), 1000.0)


def test_decode_bi5_rejects_truncated_records():
    truncated = lzma.compress(b"\x00" * 17)  # not a multiple of 20 bytes
    with pytest.raises(TickDataUnavailableError, match="not a multiple"):
        decode_bi5(truncated, datetime(2024, 6, 3, 10, tzinfo=UTC), 1000.0)


# --- tick validation -------------------------------------------------------


def test_validate_rejects_crossed_book():
    ticks = make_ticks([(datetime(2024, 6, 3, 10, tzinfo=UTC), 2400.0, 2399.0)])
    with pytest.raises(TickDataUnavailableError, match="crossed book"):
        validate_tick_frame(ticks, "fixture")


def test_validate_rejects_non_positive_prices():
    ticks = make_ticks([(datetime(2024, 6, 3, 10, tzinfo=UTC), 0.0, 1.0)])
    with pytest.raises(TickDataUnavailableError, match="non-positive"):
        validate_tick_frame(ticks, "fixture")


# --- tick -> 15m aggregation ----------------------------------------------


def test_candles_are_left_closed_right_open_15m_buckets():
    base = datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
    ticks = make_ticks(
        [
            (base, 2400.0, 2400.4),                        # 14:30 bar opens
            (base + timedelta(minutes=5), 2405.0, 2405.4),  # high
            (base + timedelta(minutes=10), 2395.0, 2395.4),  # low
            (base + timedelta(minutes=14, seconds=59), 2402.0, 2402.4),  # close
            (base + timedelta(minutes=15), 2410.0, 2410.4),  # 14:45 bar
        ]
    )
    candles = build_candles(ticks, timeframe_minutes=15)

    assert len(candles) == 2
    first = candles.iloc[0]
    assert first["timestamp"] == pd.Timestamp(base)
    assert first["open"] == pytest.approx(2400.2)   # mid of 2400.0/2400.4
    assert first["high"] == pytest.approx(2405.2)
    assert first["low"] == pytest.approx(2395.2)
    assert first["close"] == pytest.approx(2402.2)
    assert first["tick_count"] == 4                  # the 14:45 tick is excluded
    assert candles.iloc[1]["timestamp"] == pd.Timestamp(base + timedelta(minutes=15))


def test_candles_align_to_quarter_hour_boundaries():
    """Bars must sit on :00/:15/:30/:45 regardless of when the first tick
    arrives, otherwise every backtest bar is offset from reality."""
    ticks = make_ticks(
        [
            (datetime(2024, 6, 3, 14, 37, tzinfo=UTC), 2400.0, 2400.4),
            (datetime(2024, 6, 3, 14, 52, tzinfo=UTC), 2401.0, 2401.4),
        ]
    )
    candles = build_candles(ticks, timeframe_minutes=15)
    assert list(candles["timestamp"].dt.minute) == [30, 45]


def test_candles_preserve_real_spread_for_execution():
    base = datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
    ticks = make_ticks(
        [
            (base, 2400.0, 2400.2),               # spread 0.2
            (base + timedelta(minutes=1), 2400.0, 2400.8),  # spread 0.8
        ]
    )
    candles = build_candles(ticks, timeframe_minutes=15)
    row = candles.iloc[0]
    assert row["spread_mean"] == pytest.approx(0.5)
    assert row["spread_max"] == pytest.approx(0.8)
    assert row["bid_close"] == pytest.approx(2400.0)
    assert row["ask_close"] == pytest.approx(2400.8)


def test_empty_periods_produce_no_bars_rather_than_flat_fills():
    """A weekend gap must stay a gap: fabricated flat bars would distort
    indicators and invent tradeable prices."""
    ticks = make_ticks(
        [
            (datetime(2024, 6, 7, 20, 0, tzinfo=UTC), 2400.0, 2400.4),   # Friday
            (datetime(2024, 6, 10, 1, 0, tzinfo=UTC), 2410.0, 2410.4),   # Monday
        ]
    )
    candles = build_candles(ticks, timeframe_minutes=15)
    assert len(candles) == 2  # not ~200 flat weekend bars
    gap_hours = (candles["timestamp"].iloc[1] - candles["timestamp"].iloc[0]).total_seconds() / 3600
    assert gap_hours > 24


def test_candle_coverage_reports_thin_data():
    ticks = make_ticks(
        [(datetime(2024, 6, 3, 14, 30, tzinfo=UTC) + timedelta(minutes=15 * i), 2400.0 + i, 2400.4 + i)
         for i in range(10)]
    )
    coverage = candle_coverage(build_candles(ticks), timeframe_minutes=15)
    assert coverage["bars"] == 10
    assert coverage["coverage_pct"] == 100.0


def test_build_candles_on_empty_ticks_returns_empty_frame():
    from research.data.sources.base import empty_tick_frame

    assert build_candles(empty_tick_frame()).empty


# --- CSV source ------------------------------------------------------------


def test_csv_source_reads_user_supplied_ticks(tmp_path):
    path = tmp_path / "ticks.csv"
    path.write_text(
        "timestamp,bid,ask\n"
        "2024-06-03 14:30:00,2400.0,2400.4\n"
        "2024-06-03 14:31:00,2401.0,2401.4\n"
    )
    source = CsvTickSource(path)
    frame = source.fetch_hour("XAUUSD", datetime(2024, 6, 3, 14, tzinfo=UTC))
    assert len(frame) == 2
    assert frame["ask"].iloc[1] == pytest.approx(2401.4)


def test_csv_source_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(TickDataUnavailableError, match="tick file not found"):
        CsvTickSource(tmp_path / "nope.csv").load_all()


def test_csv_source_refuses_to_invent_a_spread(tmp_path):
    path = tmp_path / "mid.csv"
    path.write_text("timestamp,price\n2024-06-03 14:30:00,2400.0\n")
    source = CsvTickSource(path, price_column="price", assumed_spread=None)
    with pytest.raises(TickDataUnavailableError, match="Refusing to invent"):
        source.load_all()


# --- 70/30 chronological split + leakage guard ----------------------------


def _candle_series(bars: int = 1000) -> pd.DataFrame:
    base = datetime(2021, 1, 4, 0, 0, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [base + timedelta(minutes=15 * i) for i in range(bars)], utc=True
            ),
            "open": [2000.0 + i * 0.1 for i in range(bars)],
            "high": [2001.0 + i * 0.1 for i in range(bars)],
            "low": [1999.0 + i * 0.1 for i in range(bars)],
            "close": [2000.5 + i * 0.1 for i in range(bars)],
            "volume": [100.0] * bars,
            "tick_count": [50] * bars,
            "bid_close": [2000.3 + i * 0.1 for i in range(bars)],
            "ask_close": [2000.7 + i * 0.1 for i in range(bars)],
            "spread_mean": [0.4] * bars,
            "spread_max": [0.6] * bars,
            "is_partial": [False] * bars,
        }
    )


def test_split_is_chronological_and_70_30():
    split = DataSplit(_candle_series(1000), development_fraction=0.70, embargo_bars=50)
    summary = split.summary()
    assert summary["development_bars"] == 700
    assert summary["out_of_sample_bars"] == 300
    assert summary["development_fraction_actual"] == 0.70
    # The test period is strictly after the development period.
    assert split.out_of_sample_start > split.development_end
    assert summary["sealed"] is False


def test_development_data_is_freely_available():
    split = DataSplit(_candle_series(1000), embargo_bars=50)
    dev = split.development()
    assert len(dev) == 700
    assert dev["timestamp"].max() <= pd.Timestamp(split.development_end)


def test_out_of_sample_is_refused_before_the_strategy_is_sealed(tmp_path):
    split = DataSplit(
        _candle_series(1000), embargo_bars=50, seal_path=tmp_path / "seal.json"
    )
    with pytest.raises(LeakageError, match="sealed"):
        split.out_of_sample()


def test_out_of_sample_available_after_sealing(tmp_path):
    seal_path = tmp_path / "seal.json"
    split = DataSplit(_candle_series(1000), embargo_bars=50, seal_path=seal_path)
    params = {"ema_fast": 50, "atr_mult": 1.5}

    StrategySeal.create("test_strategy", params, split).save(seal_path)

    frame = split.out_of_sample(params)
    # Warm-up bars are included for indicator continuity but marked untradeable.
    assert (~frame["is_test"]).sum() == 50
    assert frame["is_test"].sum() == 300
    assert frame.loc[frame["is_test"], "timestamp"].min() == pd.Timestamp(
        split.out_of_sample_start
    )


def test_out_of_sample_rejects_parameters_that_differ_from_the_seal(tmp_path):
    seal_path = tmp_path / "seal.json"
    split = DataSplit(_candle_series(1000), embargo_bars=50, seal_path=seal_path)
    StrategySeal.create("test_strategy", {"ema_fast": 50}, split).save(seal_path)

    with pytest.raises(LeakageError, match="differ from the sealed strategy"):
        split.out_of_sample({"ema_fast": 20})  # quietly retuned


def test_reseal_is_refused_by_default(tmp_path):
    seal_path = tmp_path / "seal.json"
    split = DataSplit(_candle_series(1000), embargo_bars=50, seal_path=seal_path)
    StrategySeal.create("test_strategy", {"a": 1}, split).save(seal_path)

    with pytest.raises(LeakageError, match="already exists"):
        guard_reseal(seal_path, reason="tweak after peeking", allow=False)


def test_reseal_when_allowed_is_recorded_in_history(tmp_path):
    seal_path = tmp_path / "seal.json"
    split = DataSplit(_candle_series(1000), embargo_bars=50, seal_path=seal_path)
    StrategySeal.create("test_strategy", {"a": 1}, split).save(seal_path)

    guard_reseal(seal_path, reason="explicit re-optimization", allow=True)

    seal = StrategySeal.load(seal_path)
    assert len(seal.reseal_history) == 1
    assert seal.reseal_history[0]["reason"] == "explicit re-optimization"


def test_split_rejects_embargo_longer_than_development_period():
    with pytest.raises(ValueError, match="too short for an embargo"):
        DataSplit(_candle_series(100), development_fraction=0.70, embargo_bars=200)


def test_split_requires_ordered_candles():
    frame = _candle_series(500).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="chronologically ordered"):
        DataSplit(frame)
