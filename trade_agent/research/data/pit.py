from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from research.data.ingest.base import INADMISSIBLE_PROVENANCE, Provenance
from research.data.ingest.normalize import read_dataset

"""`get_information_available_at(timestamp)` -- the one door into historical data.

Every agent in the backtest reads through this module and nothing else. It has
one job: given a moment T, return what was genuinely knowable at T, and nothing
that became knowable afterwards.

The filter, per kind:

* **News** -- `available_at <= T`, where `available_at` is the later of the
  article's publication time and the moment our aggregator exposed it. Using the
  later of the two means an aggregator's ingest lag can never be spent as a head
  start.
* **Economic releases** -- `available_at <= T` for the value, with two
  refinements. A REVISED vintage is never returned as a value (a correction
  published in July was not knowable in June). And the SCHEDULE is separable
  from the outcome: a release known to be scheduled for 14:30 is public
  information at 13:00, so `scheduled_events` may look forward, while the
  released figure stays withheld until it printed.
* **Sentiment** -- `available_at <= T` AND the provenance must be
  POINT_IN_TIME_CAPTURE. A RETROSPECTIVE value is refused outright rather than
  filtered by time, because its timestamp is not when the value existed.

Two properties make this hard to get wrong by accident:

1. The comparison is `<=` against a single `available_at` column that the
   ingestion layer computed once. Callers cannot choose a different column.
2. Inadmissible provenance is dropped at load time, before any query runs, so a
   revised figure or a retrospectively scored sentiment cannot be returned by
   any code path in this module.
"""


class LookaheadError(Exception):
    """A query would have returned information from after the as-of moment.

    Raised rather than filtered: reaching this means an invariant is broken
    somewhere upstream, and continuing would produce a quietly wrong backtest.
    """


@dataclass
class AvailableInformation:
    """Everything knowable at one moment, with the reasons for what is absent."""

    as_of: datetime
    news: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar: pd.DataFrame = field(default_factory=pd.DataFrame)
    scheduled_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    sentiment: pd.DataFrame = field(default_factory=pd.DataFrame)
    unavailable: dict[str, str] = field(default_factory=dict)

    def has(self, kind: str) -> bool:
        frame = getattr(self, kind, None)
        return frame is not None and not frame.empty

    def summary(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "news": len(self.news),
            "calendar": len(self.calendar),
            "scheduled_events": len(self.scheduled_events),
            "sentiment": len(self.sentiment),
            "unavailable": dict(self.unavailable),
        }


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _drop_inadmissible(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows whose provenance disqualifies them, at load time."""
    if frame.empty or "provenance" not in frame.columns:
        return frame, 0
    provenance = frame["provenance"].fillna("").astype(str).str.upper()
    keep = ~provenance.isin(INADMISSIBLE_PROVENANCE)
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def _availability_column(frame: pd.DataFrame) -> pd.Series:
    """The timestamp a row became knowable.

    `timestamp` is written by the ingestion layer as the already-computed
    availability (the later of publication and discovery). A dataset that also
    carries `published_at` and `discovered_at` is re-checked here against the
    maximum of the two, so a hand-edited file cannot move availability earlier.
    """
    primary = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    candidates = [primary]
    for column in ("published_at", "discovered_at"):
        if column in frame.columns:
            candidates.append(pd.to_datetime(frame[column], utc=True, errors="coerce"))
    if len(candidates) == 1:
        return primary
    stacked = pd.concat(candidates, axis=1)
    return stacked.max(axis=1)


class PointInTimeDataset:
    """Loaded historical datasets, queryable as of any moment.

    Loads once, filters per query. The frames are kept sorted so a query is a
    binary search rather than a scan, which matters when a full out-of-sample run
    asks this question a few thousand times.
    """

    def __init__(
        self,
        news: pd.DataFrame | None = None,
        calendar: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        unavailable: dict[str, str] | None = None,
    ) -> None:
        self._unavailable = dict(unavailable or {})
        self._news, news_dropped = self._prepare(news, "news")
        self._calendar, calendar_dropped = self._prepare(calendar, "economic_calendar")
        self._sentiment, sentiment_dropped = self._prepare(sentiment, "sentiment")
        self.dropped_inadmissible = {
            "news": news_dropped,
            "economic_calendar": calendar_dropped,
            "sentiment": sentiment_dropped,
        }

    def _prepare(self, frame: pd.DataFrame | None, kind: str) -> tuple[pd.DataFrame, int]:
        if frame is None or frame.empty:
            self._unavailable.setdefault(
                kind, f"{kind} dataset is empty or was not supplied"
            )
            return pd.DataFrame(), 0
        work = frame.copy()
        work, dropped = _drop_inadmissible(work)
        if work.empty:
            self._unavailable.setdefault(
                kind,
                f"every {kind} row was dropped as inadmissible provenance "
                f"({dropped} row(s))",
            )
            return work, dropped
        work["_available_at"] = _availability_column(work)
        work = work.dropna(subset=["_available_at"])
        work = work.sort_values("_available_at").reset_index(drop=True)
        return work, dropped

    # --- the query --------------------------------------------------------
    def _slice(
        self, frame: pd.DataFrame, as_of: datetime, lookback: timedelta | None, limit: int | None
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        cutoff = pd.Timestamp(as_of)
        available = frame["_available_at"]
        mask = available <= cutoff
        if lookback is not None:
            mask &= available >= cutoff - pd.Timedelta(lookback)
        selected = frame[mask]
        if limit is not None and len(selected) > limit:
            selected = selected.tail(limit)
        result = selected.drop(columns=["_available_at"]).reset_index(drop=True)

        # Belt and braces: assert the invariant on the way out. If this ever
        # fires, the bug is upstream and the run must stop rather than continue
        # with contaminated context.
        if not selected.empty and selected["_available_at"].max() > cutoff:
            raise LookaheadError(
                f"point-in-time query for {as_of.isoformat()} selected a record "
                f"available at {selected['_available_at'].max()}"
            )
        return result

    def news_at(
        self,
        as_of: datetime,
        lookback: timedelta = timedelta(hours=24),
        limit: int | None = 20,
    ) -> pd.DataFrame:
        return self._slice(self._news, _as_utc(as_of), lookback, limit)

    def calendar_at(
        self,
        as_of: datetime,
        lookback: timedelta = timedelta(days=7),
        limit: int | None = 20,
    ) -> pd.DataFrame:
        """Released figures known at `as_of`.

        REVISED vintages were already dropped at load time, so this returns
        original releases only -- the values that were actually public.
        """
        return self._slice(self._calendar, _as_utc(as_of), lookback, limit)

    def scheduled_events_at(
        self,
        as_of: datetime,
        lookahead: timedelta = timedelta(hours=24),
        lookback: timedelta = timedelta(hours=2),
    ) -> pd.DataFrame:
        """Events SCHEDULED around `as_of`, with future values withheld.

        The one place this module deliberately looks forward, and only at the
        calendar: that a release is due at 14:30 is public knowledge beforehand,
        and a blackout rule needs it. The released figure is blanked for any
        event that has not yet printed, so the schedule can be seen without the
        outcome.
        """
        if self._calendar.empty:
            return pd.DataFrame()
        cutoff = pd.Timestamp(_as_utc(as_of))
        available = self._calendar["_available_at"]
        window = self._calendar[
            (available >= cutoff - pd.Timedelta(lookback))
            & (available <= cutoff + pd.Timedelta(lookahead))
        ].copy()
        if window.empty:
            return pd.DataFrame()

        not_yet = window["_available_at"] > cutoff
        for column in ("released_value", "actual", "previous"):
            if column in window.columns:
                window.loc[not_yet, column] = None
        window["already_released"] = ~not_yet
        window["withheld_reason"] = None
        window.loc[not_yet, "withheld_reason"] = "not yet released at this timestamp"
        window["minutes_until"] = (
            (window["_available_at"] - cutoff).dt.total_seconds() / 60.0
        ).round(1)
        return window.drop(columns=["_available_at"]).reset_index(drop=True)

    def sentiment_at(
        self,
        as_of: datetime,
        lookback: timedelta = timedelta(hours=6),
        limit: int | None = 10,
    ) -> pd.DataFrame:
        """Point-in-time sentiment only.

        RETROSPECTIVE rows were dropped at load; nothing here can return them.
        """
        return self._slice(self._sentiment, _as_utc(as_of), lookback, limit)

    def get_information_available_at(
        self,
        as_of: datetime,
        news_lookback: timedelta = timedelta(hours=24),
        calendar_lookback: timedelta = timedelta(days=7),
        sentiment_lookback: timedelta = timedelta(hours=6),
        schedule_lookahead: timedelta = timedelta(hours=24),
        news_limit: int | None = 20,
    ) -> AvailableInformation:
        """Everything knowable at `as_of`, and why anything missing is missing."""
        as_of = _as_utc(as_of)
        info = AvailableInformation(as_of=as_of, unavailable=dict(self._unavailable))
        info.news = self.news_at(as_of, news_lookback, news_limit)
        info.calendar = self.calendar_at(as_of, calendar_lookback)
        info.scheduled_events = self.scheduled_events_at(as_of, schedule_lookahead)
        info.sentiment = self.sentiment_at(as_of, sentiment_lookback)

        # A dataset that exists but has no rows in the window is a different
        # statement from one that is absent, and the agents are told which.
        for kind, frame in (
            ("news", info.news),
            ("economic_calendar", info.calendar),
            ("sentiment", info.sentiment),
        ):
            if frame.empty and kind not in info.unavailable:
                info.unavailable[kind] = (
                    f"no {kind} records in the lookback window ending "
                    f"{as_of.isoformat()} (the dataset covers this period)"
                )
        return info

    def coverage(self) -> dict:
        def span(frame: pd.DataFrame) -> dict:
            if frame.empty:
                return {"rows": 0, "first": None, "last": None}
            return {
                "rows": len(frame),
                "first": frame["_available_at"].min().isoformat(),
                "last": frame["_available_at"].max().isoformat(),
            }

        return {
            "news": span(self._news),
            "economic_calendar": span(self._calendar),
            "sentiment": span(self._sentiment),
            "unavailable": dict(self._unavailable),
            "dropped_inadmissible": dict(self.dropped_inadmissible),
        }


def load_point_in_time_dataset(
    news_path: Path | None = None,
    calendar_path: Path | None = None,
    sentiment_path: Path | None = None,
) -> PointInTimeDataset:
    """Load the datasets from disk.

    Local files only, by design: the agents must never reach the network during
    a historical run, so the only way information enters the backtest is a file
    the ingestion pipeline wrote and validation approved.
    """
    unavailable: dict[str, str] = {}

    def load(path: Path | None, kind: str) -> pd.DataFrame | None:
        if path is None:
            unavailable[kind] = f"no {kind} dataset path configured"
            return None
        if not Path(path).exists():
            unavailable[kind] = (
                f"{kind} dataset not found at {path}; run the ingestion pipeline "
                f"(python -m research_data fetch-{'news' if kind == 'news' else kind}) "
                "-- no data is invented"
            )
            return None
        return read_dataset(Path(path))

    return PointInTimeDataset(
        news=load(news_path, "news"),
        calendar=load(calendar_path, "economic_calendar"),
        sentiment=load(sentiment_path, "sentiment"),
        unavailable=unavailable,
    )


def get_information_available_at(
    dataset: PointInTimeDataset, as_of: datetime, **kwargs
) -> AvailableInformation:
    """Module-level convenience wrapper, for the name the design asks for."""
    return dataset.get_information_available_at(as_of, **kwargs)
