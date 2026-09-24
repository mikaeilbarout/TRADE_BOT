from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.market_data import Candle, Quote


class MarketDataProviderError(Exception):
    """Raised on any upstream failure: network error, bad symbol, stale feed,
    broker disconnect. MarketDataService treats this as 'no data available'
    and the pipeline fails closed."""


class MarketDataProvider(ABC):
    """Abstraction over wherever raw price data actually comes from (a
    broker feed, MT5, ccxt, a REST market-data API...). Swap the concrete
    implementation without touching MarketDataService or any agent.
    """

    name: str = "base"

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest tick, including the FEED's own timestamp.

        Implementations must report the venue/feed timestamp, not
        `datetime.now()` -- staleness enforcement depends on it. If your
        feed genuinely does not expose one, use the arrival time and
        document that limitation in the provider.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_candles(
        self, symbol: str, timeframe: str, count: int = 200
    ) -> list[Candle]:
        raise NotImplementedError
