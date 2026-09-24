from __future__ import annotations

from datetime import datetime, timezone

from app.models.market_data import Candle, Quote
from app.models.news import NewsItem
from app.models.sentiment import SentimentItem
from app.providers.market_data.base import MarketDataProvider, MarketDataProviderError
from app.providers.news.base import NewsProvider
from app.providers.sentiment.base import SentimentProvider


class LookaheadViolationError(Exception):
    """Raised if a historical fixture would leak future data into a
    decision. This is a hard stop, not a warning -- section 28 requires
    zero look-ahead bias in backtests."""


def _as_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


class HistoricalMarketDataProvider(MarketDataProvider):
    """Serves pre-recorded OHLCV pinned to one signal's `as_of` timestamp.

    Every candle must be timestamped at or before `as_of`; violating that
    raises immediately rather than silently feeding the pipeline
    information it could not have had at decision time.
    """

    name = "historical"

    def __init__(
        self,
        as_of: datetime,
        candles_by_timeframe: dict[str, list[Candle]],
        bid: float,
        ask: float,
    ) -> None:
        self._as_of = _as_utc(as_of)
        self._candles_by_timeframe = candles_by_timeframe
        self._bid = bid
        self._ask = ask
        for tf, candles in candles_by_timeframe.items():
            for c in candles:
                if _as_utc(c.timestamp) > self._as_of:
                    raise LookaheadViolationError(
                        f"candle at {c.timestamp} on {tf} is after as_of {as_of}"
                    )

    async def get_quote(self, symbol: str) -> Quote:
        # The quote carries the historical bar time, not "now": staleness is
        # a live-trading rule and the backtest harness disables it explicitly
        # rather than having fake-fresh timestamps hide it.
        return Quote(bid=self._bid, ask=self._ask, timestamp=self._as_of)

    async def get_candles(
        self, symbol: str, timeframe: str, count: int = 200
    ) -> list[Candle]:
        candles = self._candles_by_timeframe.get(timeframe)
        if not candles:
            raise MarketDataProviderError(
                f"no historical candles recorded for timeframe {timeframe}"
            )
        return candles[-count:]


class HistoricalNewsProvider(NewsProvider):
    """Serves pre-recorded news, pinned to one signal's `as_of` timestamp.
    Any item timestamped after `as_of` is a look-ahead violation."""

    name = "historical"

    def __init__(self, as_of: datetime, items: list[NewsItem]) -> None:
        as_of_utc = _as_utc(as_of)
        for item in items:
            if _as_utc(item.timestamp) > as_of_utc:
                raise LookaheadViolationError(
                    f"news item '{item.title}' at {item.timestamp} is after as_of {as_of}"
                )
        self._items = items

    async def fetch(self, query_terms: list[str], lookback_minutes: int) -> list[NewsItem]:
        return list(self._items)


class HistoricalSentimentProvider(SentimentProvider):
    """Serves pre-recorded sentiment material, pinned to one signal's
    `as_of` timestamp, with the same hard look-ahead guard."""

    name = "historical"

    def __init__(self, as_of: datetime, items: list[SentimentItem]) -> None:
        as_of_utc = _as_utc(as_of)
        for item in items:
            if _as_utc(item.timestamp) > as_of_utc:
                raise LookaheadViolationError(
                    f"sentiment item from {item.source} at {item.timestamp} "
                    f"is after as_of {as_of}"
                )
        self._items = items

    async def fetch(
        self, symbol: str, query_terms: list[str], lookback_minutes: int
    ) -> list[SentimentItem]:
        return list(self._items)
