from __future__ import annotations

from datetime import datetime, timezone

import httpx

from app.models.sentiment import SentimentItem, SentimentKind, SourceQuality
from app.providers.sentiment.base import SentimentProvider, SentimentProviderError

_ENDPOINT = "https://newsapi.org/v2/everything"

# Established financial/wire outlets get MEDIUM weight; anything else is LOW.
# This mirrors SentimentProvider's requirement to label source quality from
# identity, not to have the agent guess it.
_HIGH_TRUST_SOURCES = {
    "reuters",
    "bloomberg",
    "the wall street journal",
    "cnbc",
    "financial times",
    "associated press",
}

# A small, transparent lexicon -- not a trained sentiment model. Words are
# weighted for how they typically move gold specifically (e.g. "rate cut" and
# "safe-haven" are bullish for XAUUSD even though they would not be generic
# positive-sentiment words for an equity).
_BULLISH_TERMS = [
    "rally", "surge", "soar", "safe-haven", "safe haven", "rate cut",
    "dovish", "inflation fears", "geopolitical tension", "gains",
    "record high", "demand for gold", "weaker dollar", "flight to safety",
]
_BEARISH_TERMS = [
    "rate hike", "hawkish", "sell-off", "selloff", "plunge", "tumble",
    "stronger dollar", "dollar strength", "profit-taking", "falls",
    "record low", "outflow",
]


def _score_text(text: str) -> float | None:
    lowered = text.lower()
    pos = sum(1 for term in _BULLISH_TERMS if term in lowered)
    neg = sum(1 for term in _BEARISH_TERMS if term in lowered)
    if pos == 0 and neg == 0:
        return None
    return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))


class NewsAPISentimentProvider(SentimentProvider):
    """Derives sentiment from real NewsAPI headlines using a small, visible
    keyword lexicon.

    This is NOT a dedicated sentiment/positioning vendor -- there is no free
    equivalent for gold-specific social or positioning data. It is offered as
    a real (non-mock) source: genuine, timestamped headline text scored by an
    inspectable heuristic, always labelled MEDIUM/LOW quality so the
    sentiment agent discounts it the way SentimentProvider's docstring
    requires. Replace with a dedicated positioning/social feed for
    production-grade sentiment.
    """

    name = "newsapi"

    def __init__(self, api_key: str, timeout_seconds: float = 8.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def fetch(
        self, symbol: str, query_terms: list[str], lookback_minutes: int
    ) -> list[SentimentItem]:
        terms = query_terms or [symbol, "gold"]
        query = " OR ".join(terms)
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": "25",
            "apiKey": self._api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_ENDPOINT, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SentimentProviderError(f"newsapi sentiment fetch failed: {exc}") from exc

        items: list[SentimentItem] = []
        for article in payload.get("articles", []):
            try:
                ts = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")
                )
            except (KeyError, ValueError):
                continue

            title = article.get("title") or ""
            description = article.get("description") or ""
            text = f"{title}. {description}".strip()
            source_name = (article.get("source") or {}).get("name", "unknown")

            items.append(
                SentimentItem(
                    source=f"newsapi:{source_name}",
                    kind=SentimentKind.NEWS_HEADLINE,
                    quality=(
                        SourceQuality.MEDIUM
                        if source_name.strip().lower() in _HIGH_TRUST_SOURCES
                        else SourceQuality.LOW
                    ),
                    text=text,
                    timestamp=ts,
                    score=_score_text(text),
                )
            )
        return items
