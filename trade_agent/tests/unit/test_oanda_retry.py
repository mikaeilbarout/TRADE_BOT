"""OANDA provider: redirects are followed; transient failures are retried."""
import asyncio

import httpx
import pytest

from app.providers.market_data.base import MarketDataProviderError
from app.providers.market_data.oanda_provider import OandaMarketDataProvider

OK = {"prices": []}


def provider(responses, sleeps):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        status, body, headers = responses.pop(0)
        if isinstance(status, Exception):
            raise status
        return httpx.Response(status, json=body, headers=headers)

    async def sleep(seconds):
        sleeps.append(seconds)

    p = OandaMarketDataProvider("token", "101-004-1-001", transport=httpx.MockTransport(handler), sleep=sleep)
    return p, calls


def get(p):
    return asyncio.run(p._get("/v3/accounts/101-004-1-001/pricing", {"instruments": "XAU_USD"}))


def test_temporary_redirect_is_followed():
    sleeps = []
    p, calls = provider([(307, None, {"location": "/v3/accounts/101-004-1-001/pricing?instruments=XAU_USD"}), (200, OK, {})], sleeps)
    assert get(p) == OK and len(calls) == 2 and sleeps == []


def test_transient_401_is_retried_then_succeeds():
    sleeps = []
    p, calls = provider([(401, {"errorMessage": "x"}, {}), (200, OK, {})], sleeps)
    assert get(p) == OK and len(calls) == 2 and sleeps == [0.5]


def test_transport_error_is_retried():
    sleeps = []
    p, _ = provider([(httpx.ConnectTimeout("t"), None, {}), (503, None, {}), (200, OK, {})], sleeps)
    assert get(p) == OK and sleeps == [0.5, 1.5]


def test_persistent_401_fails_after_three_attempts():
    sleeps = []
    p, calls = provider([(401, {"errorMessage": "x"}, {})] * 3, sleeps)
    with pytest.raises(MarketDataProviderError, match="401"):
        get(p)
    assert len(calls) == 3


def test_non_transient_error_is_not_retried():
    sleeps = []
    p, calls = provider([(404, {"errorMessage": "no"}, {})], sleeps)
    with pytest.raises(MarketDataProviderError, match="404"):
        get(p)
    assert len(calls) == 1 and sleeps == []
