from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone

from app.models.market_data import Candle, Quote
from app.providers.market_data.base import MarketDataProvider

_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

_BASE_PRICES = {
    "XAUUSD": 3650.0,
    "BTCUSD": 63000.0,
    "ETHUSD": 3100.0,
    "EURUSD": 1.09,
    "GBPUSD": 1.27,
    "USDJPY": 149.5,
}


def _seed_for(symbol: str, timeframe: str) -> int:
    digest = hashlib.sha256(f"{symbol}:{timeframe}".encode()).hexdigest()
    return int(digest[:8], 16)


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic synthetic OHLCV generator.

    This exists purely so the rest of the system (services, agents, tests,
    the API) has something real to run against before a broker/market-data
    feed is wired in. It is NOT meant for production trading decisions --
    replace it by implementing MarketDataProvider against your actual feed
    (MT5, ccxt, a REST vendor) and pointing MARKET_DATA_PROVIDER at it.
    """

    name = "mock"

    async def get_quote(self, symbol: str) -> Quote:
        base = _BASE_PRICES.get(symbol.upper(), 100.0)
        spread = base * 0.0002
        return Quote(
            bid=base - spread / 2,
            ask=base + spread / 2,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_candles(
        self, symbol: str, timeframe: str, count: int = 200
    ) -> list[Candle]:
        base = _BASE_PRICES.get(symbol.upper(), 100.0)
        minutes = _TIMEFRAME_MINUTES.get(timeframe, 5)
        seed = _seed_for(symbol, timeframe)
        rng_state = seed
        candles: list[Candle] = []
        price = base * 0.985
        now = datetime.now(timezone.utc)

        for i in range(count):
            rng_state = (1103515245 * rng_state + 12345) & 0x7FFFFFFF
            noise = ((rng_state % 2000) - 1000) / 1000.0  # in [-1, 1]
            drift = math.sin(i / 20.0) * 0.002
            pct_move = drift + noise * 0.003
            open_p = price
            close_p = open_p * (1 + pct_move)
            high_p = max(open_p, close_p) * (1 + abs(noise) * 0.0015)
            low_p = min(open_p, close_p) * (1 - abs(noise) * 0.0015)
            volume = 100 + (rng_state % 500)
            ts = now - timedelta(minutes=minutes * (count - i))
            candles.append(
                Candle(
                    timestamp=ts,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=float(volume),
                )
            )
            price = close_p

        return candles
