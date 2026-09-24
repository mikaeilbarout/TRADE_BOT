from __future__ import annotations

import os

from research.data.ingest.base import SourceCost, SourceSpec
from research.data.ingest.fred import FredCalendarSource
from research.data.ingest.gdelt import GdeltGkgSource

"""The source catalogue: what was evaluated, what was chosen, and why.

Sources are listed here whether or not the pipeline uses them, so the decision
is inspectable rather than implicit in the imports. Each entry records the eight
things that actually decide whether a source is usable for this experiment:
historical coverage, timestamp accuracy, API availability, cost, reliability,
rate limits, licensing, and whether the dataset can be reproduced later.

Nothing in this file is aspirational. Every implemented source was built against
the provider's published contract; every rejected one says what disqualified it.
"""

# --- implemented -----------------------------------------------------------
IMPLEMENTED_SOURCES: tuple[SourceSpec, ...] = (
    GdeltGkgSource.spec,
    FredCalendarSource.spec,
)

# --- evaluated and not implemented -----------------------------------------
# Recorded so the choice can be challenged, and so a source that becomes viable
# (a paid plan, a changed free tier) can be picked up without re-doing the
# survey.
REJECTED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="newsapi",
        name="NewsAPI.org",
        kinds=("news",),
        hosts=("newsapi.org",),
        coverage_start="rolling 30 days on the free tier",
        coverage_note="Archive beyond a month requires a paid plan.",
        timestamp_granularity="publication timestamp to the second",
        cost=SourceCost.FREE_TIER_LIMITED,
        api_key_env="NEWSAPI_API_KEY",
        rate_limit="100 requests/day (free)",
        licensing="Free tier is development-only; commercial use needs a plan.",
        reproducible=False,
        implemented=False,
        not_implemented_reason=(
            "A 30-day window cannot cover a five-year backtest, and the free tier "
            "forbids the use this would be. Already wired into the LIVE service "
            "(app/providers/news) where a 30-day window is all that is needed."
        ),
        evaluation=(
            "Coverage 1/5 for this purpose, timestamps 5/5, cost 2/5, "
            "reproducibility 1/5 (results change as the index changes)."
        ),
    ),
    SourceSpec(
        key="alphavantage_news",
        name="Alpha Vantage NEWS_SENTIMENT",
        kinds=("news", "sentiment"),
        hosts=("www.alphavantage.co",),
        coverage_start="2022 (undocumented precisely)",
        coverage_note="time_from / time_to accept YYYYMMDDTHHMM windows.",
        timestamp_granularity="publication timestamp to the minute",
        cost=SourceCost.FREE_TIER_LIMITED,
        api_key_env="ALPHAVANTAGE_API_KEY",
        rate_limit="25 requests/day on the free tier",
        licensing="Free tier for personal use; redistribution restricted.",
        reproducible=True,
        implemented=False,
        not_implemented_reason=(
            "25 requests/day makes a five-year backfill take months of calendar "
            "time. Its sentiment scores also carry no statement of WHEN they were "
            "computed, so they cannot be claimed as POINT_IN_TIME_CAPTURE. Usable "
            "via the generic file importer if a paid plan is bought."
        ),
        evaluation=(
            "Coverage 3/5, timestamps 4/5, cost 2/5 at free tier, rate limits 1/5, "
            "provenance 2/5 (sentiment computation time not published)."
        ),
    ),
    SourceSpec(
        key="investing_forexfactory_calendars",
        name="Investing.com / ForexFactory economic calendars",
        kinds=("calendar",),
        hosts=("www.investing.com", "www.forexfactory.com"),
        coverage_start="multi-year",
        coverage_note=(
            "Carry consensus FORECASTS and exact release times, which FRED does "
            "not -- genuinely attractive data."
        ),
        timestamp_granularity="exact release time",
        cost=SourceCost.FREE,
        rate_limit="none published; scraping is rate-limited by the operator",
        licensing=(
            "Terms of service prohibit automated scraping and redistribution. No "
            "public API is offered."
        ),
        reproducible=False,
        implemented=False,
        not_implemented_reason=(
            "No API, and the terms of service forbid the scraping this would "
            "require. Not implemented on licensing grounds, not technical ones. "
            "They also serve a single current value per event, so an original "
            "release cannot be separated from a later revision -- the property "
            "this experiment most needs."
        ),
        evaluation=(
            "Coverage 5/5, timestamps 5/5, cost 5/5, licensing 0/5, "
            "provenance 1/5 (no vintages). Disqualified on licensing and "
            "provenance."
        ),
    ),
    SourceSpec(
        key="bls_api",
        name="US Bureau of Labor Statistics Public Data API v2",
        kinds=("calendar",),
        hosts=("api.bls.gov",),
        coverage_start="1913 for some series",
        coverage_note="Authoritative for CPI and employment.",
        timestamp_granularity="period, not release timestamp",
        cost=SourceCost.FREE_WITH_KEY,
        api_key_env="BLS_API_KEY",
        rate_limit="25 queries/day unregistered, 500/day registered",
        licensing="US government work, public domain.",
        reproducible=True,
        implemented=False,
        not_implemented_reason=(
            "Serves current values without release vintages, so an original "
            "release cannot be told from a revision. FRED/ALFRED carries the same "
            "BLS series WITH vintages, which is why it was chosen instead."
        ),
        evaluation=(
            "Coverage 5/5, authority 5/5, provenance 2/5 (no vintages). "
            "Superseded by ALFRED for this use."
        ),
    ),
    SourceSpec(
        key="gdelt_doc_api",
        name="GDELT DOC 2.0 full-text search API",
        kinds=("news",),
        hosts=("api.gdeltproject.org",),
        coverage_start="rolling ~3 months via the API",
        coverage_note="The same corpus as the archive files, through a search index.",
        timestamp_granularity="publication timestamp",
        cost=SourceCost.FREE,
        rate_limit="undocumented, throttled",
        licensing="open research use",
        reproducible=False,
        implemented=False,
        not_implemented_reason=(
            "Answers 'what does the index say today', which is exactly the "
            "reconstruction this experiment must avoid, and only reaches back "
            "about three months. The 15-minute archive files are used instead "
            "because their filenames ARE the point-in-time record."
        ),
        evaluation=(
            "Convenient and wrong for this purpose. Coverage 1/5, "
            "point-in-time fidelity 1/5."
        ),
    ),
    SourceSpec(
        key="common_crawl_news",
        name="Common Crawl CC-NEWS",
        kinds=("news",),
        hosts=("data.commoncrawl.org",),
        coverage_start="2016",
        coverage_note="WARC archives of crawled news, with crawl timestamps.",
        timestamp_granularity="crawl time; publication time inside the document",
        cost=SourceCost.FREE,
        rate_limit="none, but the corpus is tens of terabytes",
        licensing="open, subject to publishers' rights in the text",
        reproducible=True,
        implemented=False,
        not_implemented_reason=(
            "Genuinely point-in-time and free, but the volume (terabytes per year) "
            "and the full-text extraction needed make it a project in itself. A "
            "reasonable second source if GDELT's URL-derived headlines prove too "
            "thin."
        ),
        evaluation=(
            "Coverage 5/5, provenance 4/5, cost 5/5, engineering effort 1/5. "
            "Deferred, not rejected."
        ),
    ),
)

ALL_SOURCES: tuple[SourceSpec, ...] = IMPLEMENTED_SOURCES + REJECTED_SOURCES


def source_by_key(key: str) -> SourceSpec | None:
    return next((spec for spec in ALL_SOURCES if spec.key == key), None)


def required_keys() -> list[dict]:
    """API keys the implemented sources need, and whether they are present.

    Reads only whether the variable is SET -- never its value, and the value is
    never logged or written to any report.
    """
    out: list[dict] = []
    for spec in IMPLEMENTED_SOURCES:
        if not spec.requires_key():
            continue
        out.append(
            {
                "source": spec.key,
                "name": spec.name,
                "env_var": spec.api_key_env,
                "present": bool(os.environ.get(spec.api_key_env or "")),
                "cost": spec.cost.value,
                "signup": spec.key_signup_url,
                "rate_limit": spec.rate_limit,
            }
        )
    return out


def hosts_required() -> list[str]:
    """Every host the implemented sources must reach.

    Surfaced because in a network-restricted environment this list is exactly
    what has to be allowed, and a person should not have to read the code to
    find it.
    """
    hosts: list[str] = []
    for spec in IMPLEMENTED_SOURCES:
        for host in spec.hosts:
            if host not in hosts:
                hosts.append(host)
    return hosts


def catalogue_rows() -> list[dict]:
    return [
        {
            "key": spec.key,
            "name": spec.name,
            "kinds": ", ".join(spec.kinds),
            "implemented": spec.implemented,
            "coverage_start": spec.coverage_start,
            "timestamps": spec.timestamp_granularity,
            "cost": spec.cost.value,
            "api_key_env": spec.api_key_env or "-",
            "rate_limit": spec.rate_limit,
            "reproducible": spec.reproducible,
            "licensing": spec.licensing,
            "provenance": spec.provenance_support or "-",
            "evaluation": spec.evaluation,
            "why_not": spec.not_implemented_reason or "",
        }
        for spec in ALL_SOURCES
    ]
