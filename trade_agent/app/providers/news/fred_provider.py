from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.models.enums import Bias, ImpactLevel
from app.models.news import NewsItem
from app.providers.news.base import NewsProvider, NewsProviderError

_FRED_BASE = "https://api.stlouisfed.org/fred"

_IMPORTANCE_TO_IMPACT = {
    "HIGH": ImpactLevel.HIGH,
    "MEDIUM": ImpactLevel.MEDIUM,
    "LOW": ImpactLevel.LOW,
}


@dataclass(frozen=True)
class FredSeriesSpec:
    series_id: str
    name: str
    importance: str  # HIGH | MEDIUM | LOW
    units_hint: str | None = None


# Deliberately NOT imported from research/data/ingest/fred.py's own
# DEFAULT_SERIES: that module lives under research/, which the API's own
# Dockerfile does not COPY into the container (only app/, fixtures/,
# scripts/) -- importing it would build fine locally but fail at runtime
# in the deployed service. A short, duplicated list here keeps the live
# service self-contained; same series, same rationale ("releases that
# actually move gold and the dollar").
DEFAULT_SERIES: tuple[FredSeriesSpec, ...] = (
    FredSeriesSpec("CPIAUCSL", "US CPI (all items, SA)", "HIGH", "index"),
    FredSeriesSpec("CPILFESL", "US Core CPI (ex food & energy, SA)", "HIGH", "index"),
    FredSeriesSpec("PAYEMS", "US Nonfarm Payrolls", "HIGH", "thousands of persons"),
    FredSeriesSpec("UNRATE", "US Unemployment Rate", "HIGH", "percent"),
    FredSeriesSpec("PCEPILFE", "US Core PCE Price Index", "HIGH", "index"),
    FredSeriesSpec("FEDFUNDS", "US Effective Federal Funds Rate (monthly)", "MEDIUM", "percent"),
    FredSeriesSpec("DFF", "US Effective Federal Funds Rate (daily)", "LOW", "percent"),
    FredSeriesSpec("GDPC1", "US Real GDP (quarterly)", "HIGH", "billions chained"),
    FredSeriesSpec("RSAFS", "US Retail Sales", "MEDIUM", "millions USD"),
    FredSeriesSpec("INDPRO", "US Industrial Production", "MEDIUM", "index"),
    FredSeriesSpec("DGS10", "US 10-Year Treasury Yield", "MEDIUM", "percent"),
    FredSeriesSpec("DGS2", "US 2-Year Treasury Yield", "LOW", "percent"),
    FredSeriesSpec("T10Y2Y", "US 10Y-2Y Treasury Spread", "LOW", "percent"),
    FredSeriesSpec("DTWEXBGS", "US Dollar Index (broad, goods & services)", "MEDIUM", "index"),
    FredSeriesSpec("UMCSENT", "University of Michigan Consumer Sentiment", "MEDIUM", "index"),
    FredSeriesSpec("ICSA", "US Initial Jobless Claims", "MEDIUM", "persons"),
    FredSeriesSpec("PPIACO", "US Producer Price Index (all commodities)", "MEDIUM", "index"),
    FredSeriesSpec("HOUST", "US Housing Starts", "LOW", "thousands of units"),
)


class FredNewsProvider(NewsProvider):
    """Live "what actually got released recently" check against FRED.

    Deliberately NOT a reuse of research/data/ingest/fred.py's
    FredCalendarSource -- that class is built for vintage-aware historical
    backtesting (ALFRED archival vintages, windowed fetches, an
    ArtifactCache) and would be the wrong shape for a single live "is
    there anything new right now" query. The DEFAULT_SERIES list above is
    the same curated set (which macro releases actually move gold/USD),
    duplicated rather than imported -- see that constant's own comment for
    why (research/ isn't in this service's Docker image).

    For each tracked series: check `last_updated` (a real timestamp, not
    just a date) via the lightweight /series endpoint; only for series
    updated within the lookback window, fetch the latest two observations
    to report the actual value against the prior one. This fills a real
    gap Finnhub's free headline feed can't: a CPI/NFP/Fed-funds print is
    exactly the kind of high-impact, gold-moving event the news_agent
    needs, and generic business headlines rarely cover it directly.

    Scope note: this surfaces releases that already happened, not the
    forward-looking release calendar (FRED's /release/dates endpoint,
    needs a release_id-per-series mapping) -- that would let
    high_impact_event_within_minutes anticipate an imminent print instead
    of only reacting after one lands. Left for a follow-up if the "already
    released" half proves useful.
    """

    name = "fred"

    def __init__(
        self,
        api_key: str,
        series: tuple[FredSeriesSpec, ...] = DEFAULT_SERIES,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._api_key = api_key
        self._series = series
        self._timeout = timeout_seconds

    async def _last_updated(self, client: httpx.AsyncClient, series_id: str) -> datetime | None:
        try:
            resp = await client.get(
                f"{_FRED_BASE}/series",
                params={"series_id": series_id, "api_key": self._api_key, "file_type": "json"},
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        seriess = payload.get("seriess") or []
        if not seriess:
            return None
        raw = seriess[0].get("last_updated")
        if not raw:
            return None
        try:
            # FRED format: "2026-09-15 08:35:02-05" (space-separated, tz offset).
            dt = datetime.fromisoformat(raw.replace(" ", "T", 1))
        except ValueError:
            return None
        return dt.astimezone(timezone.utc)

    async def _latest_observations(
        self, client: httpx.AsyncClient, series_id: str
    ) -> tuple[str | None, str | None]:
        try:
            resp = await client.get(
                f"{_FRED_BASE}/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 2,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError):
            return None, None
        obs = payload.get("observations") or []
        actual = obs[0].get("value") if len(obs) > 0 else None
        prior = obs[1].get("value") if len(obs) > 1 else None
        return actual, prior

    async def fetch(self, query_terms: list[str], lookback_minutes: int) -> list[NewsItem]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                updated_ats = await asyncio.gather(
                    *(self._last_updated(client, spec.series_id) for spec in self._series)
                )

                now = datetime.now(timezone.utc)
                fresh: list[tuple] = []
                for spec, updated_at in zip(self._series, updated_ats):
                    if updated_at is None:
                        continue
                    age_minutes = (now - updated_at).total_seconds() / 60
                    if age_minutes <= lookback_minutes:
                        fresh.append((spec, updated_at, age_minutes))

                if not fresh:
                    return []

                obs_results = await asyncio.gather(
                    *(self._latest_observations(client, spec.series_id) for spec, _, _ in fresh)
                )
        except httpx.HTTPError as exc:
            raise NewsProviderError(f"FRED fetch failed: {exc}") from exc

        items: list[NewsItem] = []
        for (spec, updated_at, age_minutes), (actual, prior) in zip(fresh, obs_results):
            if actual is None:
                continue
            summary = f"{spec.name}: {actual}"
            if prior is not None:
                summary += f" (prior: {prior})"
            if spec.units_hint:
                summary += f" [{spec.units_hint}]"
            items.append(
                NewsItem(
                    title=f"{spec.name} released",
                    source="FRED (Federal Reserve Bank of St. Louis)",
                    url=f"https://fred.stlouisfed.org/series/{spec.series_id}",
                    timestamp=updated_at,
                    summary=summary,
                    impact=_IMPORTANCE_TO_IMPACT.get(spec.importance, ImpactLevel.LOW),
                    bias=Bias.NEUTRAL,  # direction is the news_agent's judgment call, not ours
                    category="macro",
                    is_breaking=age_minutes < 15,
                )
            )
        return items
