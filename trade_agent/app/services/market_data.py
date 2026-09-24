from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from app.config.settings import Settings
from app.services.clock import Clock, system_clock
from app.models.market_data import MarketSnapshot, TimeframeSeries
from app.providers.market_data.base import MarketDataProvider, MarketDataProviderError
from app.services.technical_service import atr, compute_indicators


class MarketDataUnavailableError(Exception):
    """Raised when a fresh, valid snapshot cannot be produced. The pipeline
    must treat this as fail-closed (REJECT/WAIT), never proceed with stale
    or partial data silently."""


def _current_session(now: datetime) -> str:
    hour = now.astimezone(timezone.utc).hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 16:
        return "OVERLAP"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "ASIA"


class MarketDataService:
    """The single shared source of normalized market data.

    All agents consume this instead of independently calling market-data
    APIs -- one fetch per signal, consistent numbers across the whole
    pipeline, and one place to enforce freshness/staleness rules.
    """

    def __init__(
        self, provider: MarketDataProvider, settings: Settings, clock: Clock = system_clock
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._clock = clock

    async def get_snapshot(
        self, symbol: str, timeframes: list[str], candles_per_timeframe: int = 220
    ) -> MarketSnapshot:
        start = time.monotonic()
        try:
            quote, *series_results = await asyncio.gather(
                self._provider.get_quote(symbol),
                *[
                    self._provider.get_candles(symbol, tf, candles_per_timeframe)
                    for tf in timeframes
                ],
            )
        except MarketDataProviderError as exc:
            raise MarketDataUnavailableError(str(exc)) from exc
        except Exception as exc:  # unexpected provider bug -> still fail closed
            raise MarketDataUnavailableError(f"unexpected provider error: {exc}") from exc

        timeframe_map = {
            tf: TimeframeSeries(timeframe=tf, candles=candles)
            for tf, candles in zip(timeframes, series_results)
        }
        if not timeframe_map:
            raise MarketDataUnavailableError("no timeframes requested")
        for tf, series in timeframe_map.items():
            if not series.candles:
                raise MarketDataUnavailableError(f"no candles returned for timeframe {tf}")

        indicators = {
            tf: compute_indicators(tf, series.candles)
            for tf, series in timeframe_map.items()
        }

        entry_tf = timeframes[-1]
        entry_candles = timeframe_map[entry_tf].candles
        entry_atr = atr(entry_candles)
        last_price = quote.mid
        volatility_pct = (entry_atr / last_price * 100) if entry_atr and last_price else None

        now = self._clock()
        quote_ts = (
            quote.timestamp
            if quote.timestamp.tzinfo
            else quote.timestamp.replace(tzinfo=timezone.utc)
        )
        # Freshness is measured against the feed's own timestamp, so a
        # frozen feed is caught even though its responses arrive instantly.
        freshness_seconds = max(0.0, (now - quote_ts).total_seconds())
        is_stale = freshness_seconds > self._settings.market_data_max_staleness_seconds

        return MarketSnapshot(
            symbol=symbol.upper(),
            bid=quote.bid,
            ask=quote.ask,
            last_price=last_price,
            spread=quote.spread,
            session=_current_session(now),
            atr=entry_atr,
            volatility_pct=volatility_pct,
            timeframes=timeframe_map,
            indicators=indicators,
            generated_at=now,
            source=self._provider.name,
            quote_timestamp=quote_ts,
            is_stale=is_stale,
            freshness_seconds=freshness_seconds,
            fetch_latency_seconds=time.monotonic() - start,
            entry_timeframe=entry_tf,
        )
