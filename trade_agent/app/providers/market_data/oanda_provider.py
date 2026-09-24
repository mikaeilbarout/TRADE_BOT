from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from app.models.market_data import Candle, Quote
from app.providers.market_data.base import MarketDataProvider, MarketDataProviderError

_PRACTICE_HOST = "https://api-fxpractice.oanda.com"
_LIVE_HOST = "https://api-fxtrade.oanda.com"

# Transient failures (2026-09-24): OANDA's practice API briefly answered a
# pricing call with "307 Temporary Redirect" and, hours later, a candles call
# with "401 Unauthorized" for a token that works -- and each single failure made
# the pipeline REJECT a real signal ("Market data unavailable, failing closed").
# Redirects are now followed, and transient statuses / transport errors are
# retried; anything else (e.g. 400/404) still fails at once.
_RETRY_STATUSES = frozenset({401, 408, 425, 429, 500, 502, 503, 504})
_ATTEMPTS = 3
_BACKOFF_SECONDS = (0.5, 1.5)

# OANDA quotes the real spot cross directly -- no futures-proxy mapping
# needed (unlike the Yahoo provider). Verified live 2026-09-14: pricing
# ~55s old, candles current to the last completed bar.
_SYMBOL_MAP = {
    "XAUUSD": "XAU_USD",
    "XAGUSD": "XAG_USD",
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
}

# OANDA's own granularity codes -- it natively supports H4, so unlike the
# Yahoo provider this needs no bar-merging.
_TIMEFRAME_MAP = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "D1": "D",
}


def _parse_time(value: str) -> datetime:
    # OANDA timestamps are RFC3339 with nanosecond fractions
    # ("2026-09-14T20:59:05.214995206Z"); datetime.fromisoformat only
    # accepts up to microseconds, so truncate the fraction first.
    if "." in value:
        head, frac = value.split(".", 1)
        frac = frac.rstrip("Z")[:6]
        value = f"{head}.{frac}+00:00"
    else:
        value = value.rstrip("Z") + "+00:00"
    return datetime.fromisoformat(value)


class OandaMarketDataProvider(MarketDataProvider):
    """Real, low-latency market data from OANDA's v20 REST API.

    Requires a free OANDA account (practice or live) and API token. Pricing
    is typically tens of seconds old (venue-reported), far fresher than the
    Yahoo provider's multi-minute lag on its unofficial chart endpoint --
    use this when MARKET_DATA_MAX_STALENESS_SECONDS needs to stay tight.
    """

    name = "oanda"

    def __init__(
        self,
        api_key: str,
        account_id: str,
        environment: str = "practice",
        timeout_seconds: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("OANDA_API_KEY is required")
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID is required")
        self._api_key = api_key
        self._account_id = account_id
        self._host = _LIVE_HOST if environment.lower() == "live" else _PRACTICE_HOST
        self._timeout = timeout_seconds
        self._transport = transport  # tests inject httpx.MockTransport
        self._sleep = sleep

    def _oanda_symbol(self, symbol: str) -> str:
        return _SYMBOL_MAP.get(symbol.upper(), symbol.upper())

    async def _get(self, path: str, params: dict) -> dict:
        url = f"{self._host}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(_ATTEMPTS):
            last = attempt == _ATTEMPTS - 1
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, follow_redirects=True, transport=self._transport
                ) as client:
                    resp = await client.get(url, params=params, headers=headers)
                if resp.status_code not in _RETRY_STATUSES or last:
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError as exc:  # timeouts, connection resets
                if last:
                    raise MarketDataProviderError(f"oanda fetch failed: {exc}") from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise MarketDataProviderError(f"oanda fetch failed: {exc}") from exc
            await self._sleep(_BACKOFF_SECONDS[attempt])
        raise MarketDataProviderError("oanda fetch failed: retries exhausted")  # unreachable

    async def get_quote(self, symbol: str) -> Quote:
        instrument = self._oanda_symbol(symbol)
        payload = await self._get(
            f"/v3/accounts/{self._account_id}/pricing",
            {"instruments": instrument},
        )
        prices = payload.get("prices") or []
        if not prices:
            raise MarketDataProviderError(f"oanda returned no price for {symbol!r}")
        quote = prices[0]
        bids = quote.get("bids") or []
        asks = quote.get("asks") or []
        if not bids or not asks:
            raise MarketDataProviderError(
                f"oanda quote for {symbol!r} has no two-sided price "
                f"(status={quote.get('status')!r}) -- instrument may be closed"
            )
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        timestamp = _parse_time(quote["time"])
        return Quote(bid=bid, ask=ask, timestamp=timestamp)

    async def get_candles(
        self, symbol: str, timeframe: str, count: int = 200
    ) -> list[Candle]:
        granularity = _TIMEFRAME_MAP.get(timeframe)
        if granularity is None:
            raise MarketDataProviderError(f"unsupported timeframe: {timeframe!r}")
        instrument = self._oanda_symbol(symbol)
        payload = await self._get(
            f"/v3/instruments/{instrument}/candles",
            {"granularity": granularity, "count": count, "price": "M"},
        )
        candles: list[Candle] = []
        for c in payload.get("candles") or []:
            if not c.get("complete"):
                # the in-progress bar; excluded so every returned candle is final
                continue
            mid = c.get("mid") or {}
            candles.append(
                Candle(
                    timestamp=_parse_time(c["time"]),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(c.get("volume") or 0.0),
                )
            )
        return candles
