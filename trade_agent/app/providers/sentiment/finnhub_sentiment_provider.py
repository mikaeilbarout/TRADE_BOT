from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from app.models.sentiment import SentimentItem, SentimentKind, SourceQuality
from app.providers.sentiment.base import SentimentProvider, SentimentProviderError

_ENDPOINT = "https://finnhub.io/api/v1/news"
_CATEGORIES = ("general", "forex")

_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")

# Same idea as NewsAPISentimentProvider: a small, transparent lexicon, not a
# trained model, weighted for how each term typically moves gold specifically.
_HIGH_TRUST_SOURCES = {"reuters", "bloomberg", "cnbc", "ap", "forbes", "marketwatch"}
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


def _keywords(query_terms: list[str]) -> list[str]:
    out = []
    for term in query_terms:
        cleaned = _PARENTHETICAL.sub("", term).strip()
        if cleaned:
            out.append(cleaned.lower())
    return out


def _is_relevant(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _score_text(text: str) -> float | None:
    lowered = text.lower()
    pos = sum(1 for term in _BULLISH_TERMS if term in lowered)
    neg = sum(1 for term in _BEARISH_TERMS if term in lowered)
    if pos == 0 and neg == 0:
        return None
    return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))


class FinnhubSentimentProvider(SentimentProvider):
    """Derives sentiment from real finnhub.io headlines using the same
    visible keyword lexicon as NewsAPISentimentProvider.

    Not a dedicated positioning/social vendor -- see NewsAPISentimentProvider
    for the same caveat. Chosen over NewsAPI because finnhub's free tier has
    no publish-time embargo, so lookback_minutes windows of an hour or less
    actually return data instead of being filtered to nothing every time.
    """

    name = "finnhub"

    def __init__(self, api_key: str, timeout_seconds: float = 8.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def fetch(
        self, symbol: str, query_terms: list[str], lookback_minutes: int
    ) -> list[SentimentItem]:
        keywords = _keywords(query_terms or [symbol, "gold"])
        raw_articles: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for category in _CATEGORIES:
                    resp = await client.get(
                        _ENDPOINT,
                        params={"category": category, "token": self._api_key},
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    if isinstance(payload, list):
                        raw_articles.extend(payload)
        except (httpx.HTTPError, ValueError) as exc:
            raise SentimentProviderError(f"finnhub sentiment fetch failed: {exc}") from exc

        items: list[SentimentItem] = []
        for a in raw_articles:
            headline = a.get("headline") or ""
            summary = a.get("summary") or ""
            text = f"{headline}. {summary}".strip()
            if keywords and not _is_relevant(text, keywords):
                continue
            epoch = a.get("datetime")
            if not epoch:
                continue
            ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
            source_name = a.get("source") or "unknown"

            items.append(
                SentimentItem(
                    source=f"finnhub:{source_name}",
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
