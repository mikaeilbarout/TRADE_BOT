from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from app.models.market_data import Candle, Quote
from app.providers.market_data.base import MarketDataProvider, MarketDataProviderError

_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_USER_AGENT = "Mozilla/5.0 (compatible; trade-agent-market-data/1.0)"

# Yahoo has no working XAUUSD spot ticker -- `XAUUSD=X` (the FX-style cross
# this used to map to) now 404s ("symbol may be delisted"), confirmed live
# 2026-09-14. GC=F (COMEX continuous gold futures) is the closest public,
# keyless proxy that actually returns data and tracks spot gold closely
# (small, usually immaterial basis). XAGUSD=X was not re-verified after this
# same failure mode was found for gold; check it the same way before relying
# on it.
_SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "XAGUSD": "XAGUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

# Yahoo does not expose a native 4-hour bucket; H4 candles are built here by
# merging four consecutive H1 candles aligned to UTC 00/04/08/12/16/20.
_INTERVAL_MAP = {
    "M1": ("1m", "7d"),
    "M5": ("5m", "60d"),
    "M15": ("15m", "60d"),
    "M30": ("30m", "60d"),
    "H1": ("60m", "730d"),
    "H4": ("60m", "730d"),
    "D1": ("1d", "5y"),
}

# Yahoo's chart API reports the last traded price, not a two-sided quote.
# The spread below is a documented model (same honesty requirement as the
# rest of this codebase's data layer), not a measurement -- there is no free,
# keyless source of real bid/ask for spot gold.
_MODELED_SPREAD_PCT = 0.0003


class YahooFinanceMarketDataProvider(MarketDataProvider):
    """Free, keyless market data from Yahoo Finance's public chart endpoint.

    This is a real, live feed (not synthetic) but it is an unofficial API
    with no uptime guarantee or bid/ask -- treat it as a way to get running
    without a paid vendor, and replace it with a broker/vendor feed before
    trading real size.
    """

    name = "yahoo"

    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self._timeout = timeout_seconds

    def _yahoo_symbol(self, symbol: str) -> str:
        return _SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}=X")

    async def _fetch_chart(self, symbol: str, interval: str, range_: str) -> dict:
        url = _CHART_ENDPOINT.format(symbol=self._yahoo_symbol(symbol))
        params = {"interval": interval, "range": range_}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    url, params=params, headers={"User-Agent": _USER_AGENT}
                )
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MarketDataProviderError(f"yahoo finance fetch failed: {exc}") from exc

        try:
            result = payload["chart"]["result"][0]
        except (KeyError, IndexError, TypeError) as exc:
            error = payload.get("chart", {}).get("error") if isinstance(payload, dict) else None
            raise MarketDataProviderError(
                f"yahoo finance returned no data for {symbol!r}: {error}"
            ) from exc
        return result

    async def get_quote(self, symbol: str) -> Quote:
        result = await self._fetch_chart(symbol, "1m", "1d")
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice")
        epoch = meta.get("regularMarketTime")
        if price is None or epoch is None:
            raise MarketDataProviderError(
                f"yahoo finance quote missing price/time for {symbol!r}"
            )
        timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
        half_spread = price * _MODELED_SPREAD_PCT / 2
        return Quote(bid=price - half_spread, ask=price + half_spread, timestamp=timestamp)

    async def get_candles(
        self, symbol: str, timeframe: str, count: int = 200
    ) -> list[Candle]:
        mapping = _INTERVAL_MAP.get(timeframe)
        if mapping is None:
            raise MarketDataProviderError(f"unsupported timeframe: {timeframe!r}")
        interval, range_ = mapping
        result = await self._fetch_chart(symbol, interval, range_)

        timestamps = result.get("timestamp") or []
        quote_block = (result.get("indicators", {}).get("quote") or [{}])[0]
        opens = quote_block.get("open") or []
        highs = quote_block.get("high") or []
        lows = quote_block.get("low") or []
        closes = quote_block.get("close") or []
        volumes = quote_block.get("volume") or []

        candles: list[Candle] = []
        for i, ts in enumerate(timestamps):
            if i >= len(closes) or closes[i] is None:
                continue
            candles.append(
                Candle(
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                    open=opens[i] if i < len(opens) and opens[i] is not None else closes[i],
                    high=highs[i] if i < len(highs) and highs[i] is not None else closes[i],
                    low=lows[i] if i < len(lows) and lows[i] is not None else closes[i],
                    close=closes[i],
                    volume=float(volumes[i]) if i < len(volumes) and volumes[i] is not None else 0.0,
                )
            )

        if timeframe == "H4":
            candles = self._aggregate_h4(candles)

        if not candles:
            return candles
        return candles[-count:]

    @staticmethod
    def _aggregate_h4(h1_candles: list[Candle]) -> list[Candle]:
        buckets: dict[datetime, list[Candle]] = {}
        for c in h1_candles:
            bucket_hour = (c.timestamp.hour // 4) * 4
            bucket_ts = c.timestamp.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
            buckets.setdefault(bucket_ts, []).append(c)

        h4_candles: list[Candle] = []
        for bucket_ts in sorted(buckets):
            members = sorted(buckets[bucket_ts], key=lambda c: c.timestamp)
            h4_candles.append(
                Candle(
                    timestamp=bucket_ts,
                    open=members[0].open,
                    high=max(m.high for m in members),
                    low=min(m.low for m in members),
                    close=members[-1].close,
                    volume=sum(m.volume for m in members),
                )
            )
        return h4_candles
