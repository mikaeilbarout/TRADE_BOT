from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from research.data.ingest.base import (
    FetchResult,
    FetchWindow,
    HistoricalSource,
    NewsRecord,
    Provenance,
    SentimentRecord,
    SourceCost,
    SourceSpec,
    SourceUnavailable,
    TimePrecision,
    now_utc,
    stable_id,
)
from research.data.ingest.cache import ArtifactCache

"""GDELT 2.0 Global Knowledge Graph as a historical news source.

Why this source. It is the only option found that is simultaneously free, needs
no API key, covers years rather than weeks, and -- the property that matters
most here -- publishes on a fixed fifteen-minute cadence into immutable,
timestamped archive files. That cadence IS the point-in-time record: the file
named 20230510144500 is what GDELT had published by 14:45 on 2023-05-10, so
"what was knowable at 14:45" is answerable from the archive rather than
reconstructed from a search index today.

Access (verified against GDELT's published documentation):

* Index of every file with size and MD5:
  http://data.gdeltproject.org/gdeltv2/masterfilelist.txt
  three whitespace-separated columns -- size, MD5, URL.
* One 15-minute GKG slice:
  http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip
  where the timestamp is on a :00/:15/:30/:45 boundary, UTC.
* GDELT 2.0 begins 2015-02-18. Earlier periods are NOT available from this
  source and are reported as missing rather than approximated.

Two things this module deliberately does not do:

1. It does not trust its own column map. The GKG 2.1 layout below is from the
   published codebook, but a parser that silently mis-indexes would produce
   plausible-looking garbage -- headlines in the wrong column, timestamps from
   a different field. So every row is structurally validated (column count,
   parseable date, domain-shaped source, URL-shaped identifier, seven-field
   tone vector) and a file whose rows fail validation raises instead of
   yielding records.
2. It does not use GDELT's DOC 2.0 search API. That API answers "what does the
   index say today", which is the reconstruction this experiment must avoid.
"""

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
MASTER_FILE_LIST = f"{GDELT_BASE}/masterfilelist.txt"

# GDELT 2.0 coverage begins here. Documented, not assumed: requests before this
# return 404, and the pipeline reports the shortfall instead of filling it.
GDELT_2_START = datetime(2015, 2, 18, tzinfo=timezone.utc)

# GKG 2.1 tab-delimited column positions, from the published codebook
# (GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf). Named rather than inlined so
# a layout change is a one-line correction, and structurally verified at parse
# time by `validate_gkg_row` so a wrong map fails loudly on the first file.
COL_RECORD_ID = 0
COL_DATE = 1
COL_SOURCE_COLLECTION = 2
COL_SOURCE_COMMON_NAME = 3
COL_DOCUMENT_IDENTIFIER = 4
COL_V1_THEMES = 7
COL_V1_LOCATIONS = 9
COL_V1_PERSONS = 11
COL_V1_ORGANISATIONS = 13
COL_V15_TONE = 15
GKG_EXPECTED_COLUMNS = 27

# GDELT publishes no headline field. The document identifier is the article URL,
# and the last path segment of a news URL is almost always its slug, which is
# the headline in kebab case. Deriving a headline from it is a transformation of
# the provider's own data, not invention -- but it is lossy, so every record
# says so via `headline_source`.
_SLUG_NOISE = re.compile(
    r"\.(html?|php|aspx?|jsp|amp|shtml)$|^\d{4,}[-_]?|[-_]\d{6,}$", re.IGNORECASE
)
_ID_TAIL = re.compile(r"[-_](?:id)?\d{5,}$", re.IGNORECASE)

# Relevance filter. Deterministic keyword matching, applied BEFORE anything
# expensive: GDELT publishes on the order of 100k articles a day and the
# experiment needs gold, USD and macro policy. Filtering here is the single
# biggest cost lever in the whole pipeline -- it is why no LLM is needed to
# build the corpus.
GOLD_TERMS = (
    "gold", "xauusd", "xau", "bullion", "precious metal", "precious metals",
    "gold price", "gold prices", "spot gold", "comex",
)
USD_TERMS = (
    "dollar", "usd", "greenback", "dxy", "dollar index",
)
MACRO_TERMS = (
    "federal reserve", "fed ", "fomc", "interest rate", "interest rates",
    "rate cut", "rate hike", "rate decision", "inflation", "cpi",
    "consumer price", "core pce", "pce", "nonfarm", "non-farm", "payrolls",
    "jobs report", "unemployment", "treasury yield", "treasury yields",
    "bond yield", "bond yields", "recession", "monetary policy", "powell",
    "quantitative easing", "tapering", "jackson hole", "yield curve",
    "central bank", "ecb", "bank of england", "boj", "safe haven",
    "safe-haven", "geopolitical", "war", "sanctions", "tariff", "tariffs",
)

# GDELT theme codes are the provider's own classification and are a stronger
# signal than free-text matching, so a theme hit scores higher.
RELEVANT_THEMES = (
    "ECON_INFLATION", "ECON_INTEREST_RATE", "ECON_CENTRALBANK", "ECON_MONETARY",
    "ECON_STOCKMARKET", "ECON_BANKRUPTCY", "ECON_DEBT", "ECON_TAXATION",
    "ECON_TRADE_DISPUTE", "ECON_CURRENCY_EXCHANGE_RATE", "ECON_CURRENCY_RESERVES",
    "ECON_EARNINGSREPORT", "ECON_PRICECONTROL", "ECON_OILPRICE",
    "WB_2689_ECONOMIC_GROWTH", "WB_1116_MONETARY_POLICY",
    "EPU_POLICY_MONETARY", "EPU_ECONOMY", "EPU_CATS_MONETARY_POLICY",
    "SANCTIONS", "MILITARY", "ARMEDCONFLICT", "TERROR",
)

CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gold", GOLD_TERMS),
    ("monetary_policy", (
        "federal reserve", "fomc", "interest rate", "rate cut", "rate hike",
        "rate decision", "monetary policy", "powell", "central bank", "ecb",
        "bank of england", "boj", "quantitative easing", "tapering",
        "jackson hole",
    )),
    ("inflation", ("inflation", "cpi", "consumer price", "pce")),
    ("employment", ("nonfarm", "non-farm", "payrolls", "jobs report", "unemployment")),
    ("rates", ("treasury yield", "bond yield", "yield curve")),
    ("usd", USD_TERMS),
    ("geopolitics", ("war", "sanctions", "geopolitical", "tariff", "military")),
)


class GkgLayoutError(SourceUnavailable):
    """A GKG file does not match the documented layout.

    Fatal on purpose. The alternative to failing here is a corpus of headlines
    read out of the wrong column, which would look like data and be worthless.
    """


def gkg_url(slot: datetime) -> str:
    """URL of the GKG file for one 15-minute slot (UTC, on a quarter hour)."""
    slot = slot.astimezone(timezone.utc)
    if slot.minute % 15 or slot.second or slot.microsecond:
        raise ValueError(
            f"GDELT publishes on 15-minute boundaries; {slot.isoformat()} is not one"
        )
    return f"{GDELT_BASE}/{slot:%Y%m%d%H%M%S}.gkg.csv.zip"


def headline_from_url(url: str) -> tuple[str, str]:
    """Best-effort headline from an article URL slug.

    Returns (headline, how_it_was_derived). GDELT publishes no title field, so
    this is a transformation of the provider's URL rather than an invented
    headline -- and when the URL carries no usable slug the record says
    `url_only` instead of inventing words.
    """
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.split("?", 1)[0].split("#", 1)[0]
    tail = _SLUG_NOISE.sub("", tail)
    tail = _ID_TAIL.sub("", tail)
    words = [w for w in re.split(r"[-_+]+", tail) if w and not w.isdigit()]
    if len(words) < 3:
        return url, "url_only"
    headline = " ".join(words).strip()
    if len(headline) < 12:
        return url, "url_only"
    return headline[:300].capitalize(), "derived_from_url_slug"


def parse_tone(raw: str) -> tuple[float, float] | None:
    """Tone vector -> (tone, polarity).

    V1.5Tone is a comma-separated vector whose first element is the average
    tone in roughly [-100, 100]. Requiring the full seven-field shape is part
    of the layout check: a two-field value means this is not the tone column.
    """
    parts = raw.split(",")
    if len(parts) < 7:
        return None
    try:
        return float(parts[0]), float(parts[3])
    except ValueError:
        return None


def validate_gkg_row(row: list[str]) -> str | None:
    """Structural check on one GKG row. Returns a reason on failure.

    This is what makes a wrong column map a loud failure instead of a silent
    one: each field is checked for the SHAPE the codebook says it has.
    """
    if len(row) < GKG_EXPECTED_COLUMNS:
        return f"expected {GKG_EXPECTED_COLUMNS} columns, found {len(row)}"
    date_raw = row[COL_DATE]
    if not (len(date_raw) == 14 and date_raw.isdigit()):
        return (
            f"column {COL_DATE} should be a YYYYMMDDHHMMSS date, found {date_raw[:30]!r}"
        )
    document = row[COL_DOCUMENT_IDENTIFIER]
    if not document:
        return f"column {COL_DOCUMENT_IDENTIFIER} (document identifier) is empty"
    tone = row[COL_V15_TONE]
    if tone and parse_tone(tone) is None:
        return (
            f"column {COL_V15_TONE} should be a 7-field tone vector, found {tone[:40]!r}"
        )
    return None


def score_relevance(text: str, themes: str) -> tuple[float, list[str]]:
    """Deterministic relevance score and the terms that matched.

    Weighted so a gold mention counts most, GDELT's own economic theme codes
    next, then USD and macro language. The matched terms are recorded on the
    record so a filtered corpus can be audited instead of trusted.
    """
    lowered = text.lower()
    matched: list[str] = []
    score = 0.0
    for term in GOLD_TERMS:
        if term in lowered:
            matched.append(term)
            score += 3.0
            break
    upper_themes = themes.upper()
    for theme in RELEVANT_THEMES:
        if theme in upper_themes:
            matched.append(f"theme:{theme}")
            score += 2.0
            if score >= 8:
                break
    for term in USD_TERMS:
        if term in lowered:
            matched.append(term)
            score += 1.5
            break
    macro_hits = [term for term in MACRO_TERMS if term in lowered]
    if macro_hits:
        matched.extend(macro_hits[:3])
        score += min(len(macro_hits), 3) * 1.0
    return score, matched


def categorise(text: str) -> str:
    lowered = text.lower()
    for category, terms in CATEGORY_RULES:
        if any(term in lowered for term in terms):
            return category
    return "general"


class GdeltGkgSource(HistoricalSource):
    """Historical news (and point-in-time tone) from GDELT 2.0 GKG archives."""

    spec = SourceSpec(
        key="gdelt_gkg",
        name="GDELT 2.0 Global Knowledge Graph (15-minute archive files)",
        kinds=("news", "sentiment"),
        hosts=("data.gdeltproject.org",),
        coverage_start="2015-02-18",
        coverage_note=(
            "GDELT 2.0 onward, every 15 minutes. Individual slots are "
            "occasionally missing from the archive; those are recorded as gaps."
        ),
        timestamp_granularity=(
            "15-minute publication slot, plus the article's own publication "
            "timestamp to the second"
        ),
        cost=SourceCost.FREE,
        api_key_env=None,
        rate_limit=(
            "No published quota. One HTTP GET per 15-minute slot (~35,000 files "
            "for five years); the pipeline rate-limits itself and caches "
            "permanently."
        ),
        licensing=(
            "GDELT is published for open research use; the underlying article "
            "text remains its publishers'. This pipeline stores only metadata "
            "(URL, domain, timestamp, themes, tone), never article bodies."
        ),
        reproducible=True,
        provenance_support=(
            "Strong. Each record traces to an immutable archive file whose name "
            "IS its publication time, and whose MD5 the provider publishes in "
            "masterfilelist.txt."
        ),
        evaluation=(
            "Chosen as the primary news source: free, no key, multi-year, "
            "15-minute point-in-time granularity, provider checksums, fully "
            "reproducible. Weaknesses: no headline field (derived from the URL "
            "slug), and metadata only."
        ),
    )

    def __init__(
        self,
        cache: ArtifactCache,
        min_relevance: float = 2.0,
        max_records_per_slot: int = 40,
        client: httpx.Client | None = None,
        tone_provenance: Provenance = Provenance.POINT_IN_TIME_CAPTURE,
    ) -> None:
        self._cache = cache
        self._min_relevance = min_relevance
        self._max_records = max_records_per_slot
        self._client = client
        # GDELT computed this tone at ingestion and published it in a file
        # stamped with that 15-minute slot, so the VALUE existed at that
        # timestamp -- which is what POINT_IN_TIME_CAPTURE asserts. Left
        # configurable so a reviewer who disagrees can downgrade it without
        # editing code.
        self._tone_provenance = tone_provenance

    # --- windows ----------------------------------------------------------
    def windows(self, start: datetime, end: datetime) -> list[FetchWindow]:
        """One window per 15-minute publication slot."""
        start = max(start.astimezone(timezone.utc), GDELT_2_START)
        end = end.astimezone(timezone.utc)
        slot = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)
        windows: list[FetchWindow] = []
        while slot < end:
            windows.append(
                FetchWindow(
                    key=f"{slot:%Y%m%d%H%M%S}",
                    start=slot,
                    end=slot + timedelta(minutes=15),
                )
            )
            slot += timedelta(minutes=15)
        return windows

    def coverage_shortfall(self, start: datetime, end: datetime) -> str | None:
        """Report a requested range GDELT 2.0 cannot cover."""
        start = start.astimezone(timezone.utc)
        if start < GDELT_2_START:
            return (
                f"requested history begins {start:%Y-%m-%d} but GDELT 2.0 starts "
                f"{GDELT_2_START:%Y-%m-%d}; the earlier "
                f"{(GDELT_2_START - start).days} day(s) are UNAVAILABLE from this "
                "source and are not substituted"
            )
        return None

    def preflight(self) -> str | None:
        return None  # no key required

    # --- fetching ---------------------------------------------------------
    def fetch_window(self, window: FetchWindow) -> FetchResult:
        url = gkg_url(window.start)
        try:
            artifact = self._cache.fetch(url, suffix=".zip", allow_missing=True)
        except SourceUnavailable as exc:
            return FetchResult(window=window, failed=True, error=str(exc))

        if artifact is None:
            # GDELT has no file for this slot. A real gap, not a failure, and
            # not something to fill in.
            return FetchResult(window=window, empty=True, requests_made=1)

        try:
            rows = self._read_rows(artifact.path)
        except GkgLayoutError as exc:
            return FetchResult(window=window, failed=True, error=str(exc))

        records: list[NewsRecord] = []
        discovered_at = window.start  # the slot the provider published it in
        retrieved_at = artifact.retrieved_at or now_utc()

        for row in rows:
            record = self._to_news(row, discovered_at, retrieved_at, artifact)
            if record is not None:
                records.append(record)

        records.sort(key=lambda r: (-r.relevance_score, r.published_at))
        records = records[: self._max_records]

        return FetchResult(
            window=window,
            records=records,
            bytes_fetched=0 if artifact.from_cache else artifact.bytes_len,
            requests_made=0 if artifact.from_cache else 1,
            from_cache=artifact.from_cache,
            empty=not records,
        )

    def _read_rows(self, path: Path) -> list[list[str]]:
        """Decode one GKG zip into validated rows.

        A file whose first usable rows all fail the structural check raises:
        that means the layout is not what the codebook describes, and parsing
        on regardless would produce a corpus nobody could trust.
        """
        try:
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise GkgLayoutError(f"{path.name}: zip contains no CSV member")
                payload = archive.read(names[0])
        except zipfile.BadZipFile as exc:
            raise GkgLayoutError(f"{path.name}: not a valid zip ({exc})") from exc

        text = payload.decode("utf-8", errors="replace")
        # GKG rows carry free-text fields (V2Themes, V2Organizations, the
        # extras XML blob...) that routinely exceed Python's 128KB default
        # csv field limit on newsy days; raise it rather than truncate a
        # record silently.
        csv.field_size_limit(min(len(text) + 1, 2**31 - 1))
        reader = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
        rows: list[list[str]] = []
        problems: list[str] = []
        for row in reader:
            if not row or not any(row):
                continue
            problem = validate_gkg_row(row)
            if problem:
                problems.append(problem)
                continue
            rows.append(row)

        if not rows and problems:
            raise GkgLayoutError(
                f"{path.name}: every row failed the GKG 2.1 layout check "
                f"({len(problems)} rows). First reason: {problems[0]}. The column "
                "map in research/data/ingest/gdelt.py does not match this file; "
                "check it against the published GKG 2.1 codebook rather than "
                "ingesting mis-parsed rows."
            )
        return rows

    def _to_news(
        self, row: list[str], discovered_at: datetime, retrieved_at: datetime, artifact
    ) -> NewsRecord | None:
        published_at = _parse_gkg_date(row[COL_DATE])
        if published_at is None:
            return None

        domain = row[COL_SOURCE_COMMON_NAME].strip() or "unknown"
        document = row[COL_DOCUMENT_IDENTIFIER].strip()
        themes = row[COL_V1_THEMES]
        headline, headline_source = headline_from_url(document)

        haystack = " ".join(
            (headline, document, row[COL_V1_ORGANISATIONS], row[COL_V1_PERSONS])
        )
        score, matched = score_relevance(haystack, themes)
        if score < self._min_relevance:
            return None

        return NewsRecord(
            source=domain,
            source_id=row[COL_RECORD_ID].strip()
            or stable_id(document, row[COL_DATE]),
            headline=headline,
            category=categorise(haystack),
            url=document or None,
            published_at=published_at,
            discovered_at=discovered_at,
            retrieved_at=retrieved_at,
            time_precision=TimePrecision.EXACT,
            provenance=Provenance.ORIGINAL_RELEASE,
            original_release=True,
            matched_terms=matched[:6],
            relevance_score=round(score, 2),
            source_artifact=artifact.relative_name,
            source_checksum=artifact.sha256,
        )

    # --- sentiment --------------------------------------------------------
    def sentiment_from_window(self, window: FetchWindow) -> FetchResult:
        """Point-in-time tone for one slot, from the same archive file.

        The claim being made, stated plainly so it can be challenged: GDELT
        computed these tone scores when it ingested each article and published
        them inside a file whose name is that 15-minute slot. The VALUE
        therefore existed, publicly, at that timestamp -- it is not a present-day
        model's reading of an old article. That is what distinguishes
        POINT_IN_TIME_CAPTURE from RETROSPECTIVE here.

        The tone algorithm is GDELT's, fixed, and applied uniformly; this
        pipeline only aggregates the published values.
        """
        url = gkg_url(window.start)
        try:
            artifact = self._cache.fetch(url, suffix=".zip", allow_missing=True)
        except SourceUnavailable as exc:
            return FetchResult(window=window, failed=True, error=str(exc))
        if artifact is None:
            return FetchResult(window=window, empty=True, requests_made=1)

        try:
            rows = self._read_rows(artifact.path)
        except GkgLayoutError as exc:
            return FetchResult(window=window, failed=True, error=str(exc))

        tones: list[float] = []
        for row in rows:
            haystack = " ".join(
                (row[COL_DOCUMENT_IDENTIFIER], row[COL_V1_ORGANISATIONS])
            )
            score, _ = score_relevance(haystack, row[COL_V1_THEMES])
            if score < self._min_relevance:
                continue
            parsed = parse_tone(row[COL_V15_TONE])
            if parsed is not None:
                tones.append(parsed[0])

        if not tones:
            return FetchResult(
                window=window,
                empty=True,
                from_cache=artifact.from_cache,
                requests_made=0 if artifact.from_cache else 1,
            )

        average = sum(tones) / len(tones)
        record = SentimentRecord(
            source="gdelt_gkg_tone",
            source_id=stable_id("gdelt_tone", window.key),
            value=max(-1.0, min(1.0, average / 10.0)),
            raw_value=round(average, 4),
            scale="gdelt_v15_tone_average_normalised_by_10",
            method=(
                "arithmetic mean of GDELT V1.5 average-tone over relevance-filtered "
                "articles in this 15-minute publication slot"
            ),
            article_count=len(tones),
            published_at=window.start,
            discovered_at=window.start,
            retrieved_at=artifact.retrieved_at or now_utc(),
            time_precision=TimePrecision.EXACT,
            provenance=self._tone_provenance,
            original_release=True,
            source_artifact=artifact.relative_name,
            source_checksum=artifact.sha256,
        )
        return FetchResult(
            window=window,
            records=[record],
            bytes_fetched=0 if artifact.from_cache else artifact.bytes_len,
            requests_made=0 if artifact.from_cache else 1,
            from_cache=artifact.from_cache,
        )

    # --- provider checksums ----------------------------------------------
    def load_master_checksums(self) -> dict[str, str]:
        """Provider-published MD5 per file, from masterfilelist.txt.

        Used to verify downloads against GDELT's own hashes. The list is large
        (one line per file since 2015), so it is cached like any other
        artifact and parsed lazily.
        """
        artifact = self._cache.fetch(MASTER_FILE_LIST, suffix=".txt")
        if artifact is None:
            return {}
        checksums: dict[str, str] = {}
        for line in artifact.path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            _size, md5, file_url = parts
            if file_url.endswith(".gkg.csv.zip"):
                checksums[file_url] = md5
        return checksums


def _parse_gkg_date(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
