from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from app.models.enums import Bias, ImpactLevel
from app.models.news import NewsItem
from app.providers.news.base import NewsProvider, NewsProviderError

_ENDPOINT = "https://newsapi.org/v2/everything"

# Parenthetical annotations in asset_meta.relevant_factors (e.g. "inflation
# (CPI/PPI)") are for LLM prompt readability, not search syntax -- NewsAPI
# treats "(" / ")" as boolean grouping, so an exact-quoted parenthetical
# would almost never match a real headline. Stripped before building a query.
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


def _query_term(term: str) -> str:
    cleaned = _PARENTHETICAL.sub("", term).strip() or term
    # NewsAPI's `q` defaults to AND between bare words ("USD strength" would
    # become "USD AND strength"), which is far narrower than "either word is
    # fine" -- quoting makes a multi-word factor an exact phrase instead, and
    # every OR'd term between phrases still just needs one of them to match.
    return f'"{cleaned}"' if " " in cleaned else cleaned


class NewsAPIProvider(NewsProvider):
    """Thin adapter for newsapi.org. Only classifies breaking-vs-recent and
    leaves impact/bias classification to the news agent's LLM analysis --
    this layer's job is retrieval and normalization, not judgment (section
    21: 'do not allow agents to randomly browse the internet without
    controlling data quality', which cuts the other way too: this fetch
    layer should not silently invent a bias score either).
    """

    name = "newsapi"

    def __init__(self, api_key: str, timeout_seconds: float = 8.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def fetch(
        self, query_terms: list[str], lookback_minutes: int
    ) -> list[NewsItem]:
        query = " OR ".join(_query_term(t) for t in query_terms) if query_terms else "markets"
        # No `from`/`to` here -- verified empirically against the live API
        # (2026-09-16): this account's plan embargoes anything published in
        # roughly the last 24h (a `from` inside that window returns
        # totalResults=0 no matter how broad `q` is), so a `from` built from
        # lookback_minutes (typically ~60) would ask for a window the plan
        # can never serve and guarantee an empty bundle every single call --
        # worse than the unfiltered request, which at least returns whatever
        # the plan's newest available articles actually are. lookback_minutes
        # is intentionally unused here for that reason; a paid/real-time key
        # could reinstate a `from` window once that constraint no longer holds.
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
            raise NewsProviderError(f"newsapi fetch failed: {exc}") from exc

        articles = payload.get("articles", [])
        items: list[NewsItem] = []
        for a in articles:
            try:
                ts = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            items.append(
                NewsItem(
                    title=a.get("title") or "(untitled)",
                    source=(a.get("source") or {}).get("name", "unknown"),
                    url=a.get("url"),
                    timestamp=ts,
                    summary=a.get("description") or "",
                    impact=ImpactLevel.LOW,
                    bias=Bias.NEUTRAL,
                    category="general",
                    is_breaking=(datetime.now(timezone.utc) - ts).total_seconds() < 900,
                )
            )
        return items
