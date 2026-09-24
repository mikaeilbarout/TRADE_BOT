from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.manifest import sha256_file

"""Import a supplied OHLC candle export, and derive higher timeframes from it.

This exists because a broker export of finished bars is a different input from
the tick stream `research/data/candles.py` builds bars out of, and conflating
them would misstate what the dataset actually is. Two consequences are recorded
rather than papered over:

* **The spread is not observed.** A bar export carries no bid/ask, so the
  backtest's spread is a MODEL (`CostModel.fallback_spread_price`), not a
  measurement. `bid_close`/`ask_close`/`spread_mean` are left absent instead of
  being synthesised from the close, and the manifest says the spread is modelled.
  Inventing a bid/ask from a mid price would look like data and be arithmetic.
* **Bars are already aggregated.** Whatever the vendor did to build them is
  fixed; this importer validates and relabels, it does not re-derive.

Higher timeframes are resampled from the supplied bars. That is exact for
OHLC (first/max/min/last over a contained window) and is causal as long as a
resampled bar is only used from its own close onward -- which the point-in-time
merge in the strategy layer enforces by taking the last CLOSED higher-timeframe
bar at or before each entry bar.
"""

# The canonical column set the research stack expects. Columns the source
# genuinely cannot provide are left out, never filled in.
REQUIRED_SOURCE_COLUMNS = ("open", "high", "low", "close")

# Common spellings for the timestamp column across broker exports.
TIMESTAMP_ALIASES = (
    "ts", "timestamp", "time", "datetime", "date", "open_time", "Date", "Time",
)


class CandleImportError(Exception):
    """The supplied file cannot be imported as candles, with the reason.

    Raised rather than coerced: a file whose OHLC relationships are broken, or
    whose timestamps do not parse, is not something to repair silently.
    """


@dataclass
class ImportReport:
    """What was imported, and every doubt worth recording."""

    source_path: str
    source_sha256: str
    rows_in: int
    rows_out: int
    symbol: str
    timeframe_minutes: int
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    duplicates_dropped: int = 0
    unparseable_dropped: int = 0
    source_timezone: str = "assumed UTC"
    spread_observed: bool = False
    volume_present: bool = False
    zero_volume_bars: int = 0
    session_gaps: int = 0
    largest_gap_hours: float = 0.0
    modal_spacing_minutes: float | None = None
    off_grid_bars: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["first_timestamp"] = (
            self.first_timestamp.isoformat() if self.first_timestamp else None
        )
        payload["last_timestamp"] = (
            self.last_timestamp.isoformat() if self.last_timestamp else None
        )
        return payload


def _find_timestamp_column(frame: pd.DataFrame) -> str:
    for candidate in TIMESTAMP_ALIASES:
        if candidate in frame.columns:
            return candidate
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in TIMESTAMP_ALIASES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise CandleImportError(
        f"no timestamp column found; looked for {list(TIMESTAMP_ALIASES)} among "
        f"{list(frame.columns)}"
    )


def load_candle_csv(
    path: Path,
    symbol: str = "XAUUSD",
    timeframe_minutes: int = 15,
    source_timezone: str = "UTC",
) -> tuple[pd.DataFrame, ImportReport]:
    """Load and validate a supplied candle export.

    `source_timezone` is applied to naive timestamps. It defaults to UTC and the
    assumption is recorded, because a broker export in server time silently
    shifts every bar by the server offset -- a failure that looks like a
    strategy result rather than a data error.
    """
    path = Path(path)
    if not path.exists():
        raise CandleImportError(f"candle file not found: {path}")

    frame = pd.read_csv(path)
    rows_in = len(frame)
    if frame.empty:
        raise CandleImportError(f"{path.name} contains no rows")

    timestamp_column = _find_timestamp_column(frame)
    lowered = {column: column.lower() for column in frame.columns}
    frame = frame.rename(columns=lowered)
    timestamp_column = timestamp_column.lower()

    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in frame.columns]
    if missing:
        raise CandleImportError(
            f"{path.name} is missing required OHLC column(s) {missing}; found "
            f"{sorted(frame.columns)}"
        )

    notes: list[str] = []
    stamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    unparseable = int(stamps.isna().sum())
    if unparseable:
        notes.append(f"{unparseable} row(s) dropped for unparseable timestamps")

    if stamps.dt.tz is None:
        if source_timezone.upper() == "UTC":
            stamps = stamps.dt.tz_localize("UTC")
            tz_note = "source timestamps were naive and are treated as UTC"
        else:
            stamps = stamps.dt.tz_localize(source_timezone).dt.tz_convert("UTC")
            tz_note = f"source timestamps were naive, localized as {source_timezone} then converted to UTC"
    else:
        stamps = stamps.dt.tz_convert("UTC")
        tz_note = "source timestamps carried a timezone and were converted to UTC"
    notes.append(tz_note)

    frame["timestamp"] = stamps
    frame = frame.dropna(subset=["timestamp"])

    duplicates = int(frame["timestamp"].duplicated().sum())
    if duplicates:
        # Keep the FIRST occurrence: a re-exported bar is usually the same bar,
        # and picking the later one silently prefers whichever row the vendor
        # happened to write last.
        frame = frame.drop_duplicates(subset=["timestamp"], keep="first")
        notes.append(f"{duplicates} duplicate timestamp(s) dropped, keeping the first")

    frame = frame.sort_values("timestamp").reset_index(drop=True)

    for column in REQUIRED_SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(REQUIRED_SOURCE_COLUMNS)].isna().any().any():
        raise CandleImportError(
            f"{path.name}: non-numeric value(s) in OHLC columns; refusing to guess"
        )

    # --- OHLC integrity: fatal, because a broken bar is not fixable ---------
    broken_range = frame["high"] < frame["low"]
    outside = (
        (frame["close"] > frame["high"]) | (frame["close"] < frame["low"])
        | (frame["open"] > frame["high"]) | (frame["open"] < frame["low"])
    )
    nonpositive = (frame[list(REQUIRED_SOURCE_COLUMNS)] <= 0).any(axis=1)
    for mask, what in (
        (broken_range, "high < low"),
        (outside, "open or close outside [low, high]"),
        (nonpositive, "non-positive price"),
    ):
        if mask.any():
            first = frame[mask].iloc[0]
            raise CandleImportError(
                f"{path.name}: {int(mask.sum())} bar(s) with {what} "
                f"(first at {first['timestamp']}). The export is not internally "
                "consistent; it is not repaired automatically."
            )

    # --- volume ------------------------------------------------------------
    volume_present = "volume" in frame.columns
    zero_volume = 0
    if volume_present:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        zero_volume = int((frame["volume"] <= 0).sum())
        if zero_volume:
            notes.append(
                f"{zero_volume} bar(s) have zero or negative volume; kept, because a "
                "quiet bar is real, but they are worth knowing about"
            )
    else:
        frame["volume"] = 0.0
        notes.append("source has no volume column; volume is recorded as 0, not invented")

    # --- spacing -----------------------------------------------------------
    deltas = frame["timestamp"].diff().dt.total_seconds().div(60)
    modal = float(deltas.mode().iloc[0]) if not deltas.dropna().empty else None
    session_gaps = int((deltas > timeframe_minutes * 2).sum())
    largest_gap = float(deltas.max() / 60) if not deltas.dropna().empty else 0.0
    # A bar not aligned to the timeframe grid means the export is not the
    # timeframe it claims to be.
    off_grid = int(
        (
            (frame["timestamp"].dt.minute % timeframe_minutes != 0)
            | (frame["timestamp"].dt.second != 0)
        ).sum()
    )
    if off_grid:
        notes.append(
            f"{off_grid} bar(s) are not aligned to the {timeframe_minutes}-minute grid"
        )
    if modal is not None and abs(modal - timeframe_minutes) > 1e-6:
        notes.append(
            f"the most common bar spacing is {modal:.0f} minutes, not the declared "
            f"{timeframe_minutes}"
        )

    # --- spread: absent, and that is recorded ------------------------------
    spread_observed = {"bid", "ask", "bid_close", "ask_close", "spread"} & set(frame.columns)
    if not spread_observed:
        notes.append(
            "source carries no bid/ask, so the backtest spread is a MODEL "
            "(CostModel.fallback_spread_price), not a measurement. No synthetic "
            "bid/ask is written."
        )

    canonical = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    # Columns the research stack looks for; left NA so the engine's documented
    # fallbacks apply rather than a fabricated value being used.
    canonical["tick_count"] = pd.NA
    canonical["bid_close"] = pd.NA
    canonical["ask_close"] = pd.NA
    canonical["spread_mean"] = pd.NA
    canonical["spread_max"] = pd.NA
    canonical["is_partial"] = False

    report = ImportReport(
        source_path=str(path),
        source_sha256=sha256_file(path),
        rows_in=rows_in,
        rows_out=len(canonical),
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        first_timestamp=canonical["timestamp"].iloc[0].to_pydatetime(),
        last_timestamp=canonical["timestamp"].iloc[-1].to_pydatetime(),
        duplicates_dropped=duplicates,
        unparseable_dropped=unparseable,
        source_timezone=source_timezone,
        spread_observed=bool(spread_observed),
        volume_present=volume_present,
        zero_volume_bars=zero_volume,
        session_gaps=session_gaps,
        largest_gap_hours=round(largest_gap, 2),
        modal_spacing_minutes=modal,
        off_grid_bars=off_grid,
        notes=notes,
    )
    return canonical, report


def resample_candles(
    frame: pd.DataFrame, target_minutes: int, source_minutes: int = 15
) -> pd.DataFrame:
    """Aggregate finished bars into a higher timeframe.

    Exact for OHLC: open is the window's first open, high its max, low its min,
    close its last close. Bars are labelled by the window's START and closed on
    the left, matching the convention the entry-timeframe bars already use.

    Causality is NOT established here -- it is established by the consumer,
    which may only read a higher-timeframe bar once that bar has closed. The
    strategy layer does that with a backward as-of merge on the bar's close
    time; see `research/strategy/donchian_scalp.py`.
    """
    if target_minutes % source_minutes:
        raise ValueError(
            f"target timeframe {target_minutes}m is not a whole multiple of the "
            f"source {source_minutes}m; aggregation would straddle bars"
        )
    if frame.empty:
        return frame.copy()

    work = frame.set_index("timestamp").sort_index()
    rule = f"{target_minutes}min"
    aggregated = work.resample(rule, label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    # Empty windows (weekends, holidays) produce no bar at all rather than a
    # row of NaNs -- an invented flat bar would be a trading opportunity that
    # never existed.
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated = aggregated.reset_index()
    aggregated["bar_close_time"] = aggregated["timestamp"] + pd.Timedelta(
        minutes=target_minutes
    )
    return aggregated


def write_candles(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
