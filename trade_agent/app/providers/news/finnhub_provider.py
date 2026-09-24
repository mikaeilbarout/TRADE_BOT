from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from app.models.enums import Bias, ImpactLevel
from app.models.news import NewsItem
from app.providers.news.base import NewsProvider, NewsProviderError

_ENDPOINT = "https://finnhub.io/api/v1/news"
# Finnhub's free /news endpoint has no keyword search -- it only serves the
# latest N items per fixed category. "general" covers Fed/inflation/macro,
# "forex" covers USD/rate-differential stories; XAUUSD news tends to land in
# one of the two. Relevance filtering therefore happens client-side below,
# unlike NewsAPIProvider which could push a `q` to the upstream API.
_CATEGORIES = ("general", "forex")

_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


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


class FinnhubNewsProvider(NewsProvider):
    """Free-tier finnhub.io news adapter -- no publish-time embargo (unlike
    this project's NewsAPI plan, which refuses anything published in the
    last ~24h), so results are genuinely usable for lookback windows of an
    hour or less. Retrieval/normalization only, same split of duties as
    NewsAPIProvider: this layer does not judge impact/bias, only fetches
    and tags breaking-ness.
    """

    name = "finnhub"

    def __init__(self, api_key: str, timeout_seconds: float = 8.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def fetch(
        self, query_terms: list[str], lookback_minutes: int
    ) -> list[NewsItem]:
        keywords = _keywords(query_terms)
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
            raise NewsProviderError(f"finnhub fetch failed: {exc}") from exc

        now = datetime.now(timezone.utc)
        items: list[NewsItem] = []
        for a in raw_articles:
            headline = a.get("headline") or ""
            summary = a.get("summary") or ""
            if keywords and not _is_relevant(f"{headline} {summary}", keywords):
                continue
            epoch = a.get("datetime")
            if not epoch:
                continue
            ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
            items.append(
                NewsItem(
                    title=headline or "(untitled)",
                    source=a.get("source") or "unknown",
                    url=a.get("url"),
                    timestamp=ts,
                    summary=summary,
                    impact=ImpactLevel.LOW,
                    bias=Bias.NEUTRAL,
                    category=a.get("category") or "general",
                    is_breaking=(now - ts).total_seconds() < 900,
                )
            )
        return items
