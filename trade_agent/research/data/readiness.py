from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from research.data.ingest.base import INADMISSIBLE_PROVENANCE, Provenance, TimePrecision
from research.data.ingest.checkpoint import IngestCheckpoint
from research.data.ingest.normalize import read_dataset
from research.data.ingest.pipeline import coverage_gaps
from research.data.ingest.registry import IMPLEMENTED_SOURCES, hosts_required, required_keys
from research.data.ingest.validate import (
    validate_calendar,
    validate_news,
    validate_sentiment,
)

"""The data readiness report.

One question: is there enough validated, point-in-time-safe history to begin the
development phase? The report answers it per dataset and then as a single
verdict, and it is generated from the data on disk rather than written by hand --
so re-running it after an ingestion run gives the true state, not a remembered
one.

A dataset is READY only when it exists, covers the requested period without
material gaps, passes every validation check, and carries provenance on every
row. Anything else is reported as the specific thing it is: UNAVAILABLE, PARTIAL,
or INVALID.
"""


class Readiness(BaseModel):
    status: str            # READY | PARTIAL | UNAVAILABLE | INVALID
    dataset: str
    path: str | None = None
    records: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    coverage_fraction: float | None = None
    missing_ranges: list[dict] = Field(default_factory=list)
    missing_ranges_truncated: int = 0
    provenance_counts: dict[str, int] = Field(default_factory=dict)
    time_precision_counts: dict[str, int] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    validation_passed: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status in ("READY", "PARTIAL")


class TickReadiness(BaseModel):
    status: str
    tick_files: int = 0
    tick_records: int | None = None
    candle_path: str | None = None
    candle_records: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    missing_ranges: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SplitReadiness(BaseModel):
    ready: bool = False
    reason: str = ""
    total_bars: int = 0
    development_bars: int = 0
    out_of_sample_bars: int = 0
    development_start: datetime | None = None
    development_end: datetime | None = None
    out_of_sample_start: datetime | None = None
    out_of_sample_end: datetime | None = None
    embargo_bars: int = 0
    sealed: bool = False


class DataReadinessReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    experiment: str
    requested_start: datetime
    requested_end: datetime
    calendar_release_time_policy: str

    ticks: TickReadiness
    news: Readiness
    calendar: Readiness
    sentiment: Readiness
    sentiment_retrospective: Readiness

    split: SplitReadiness
    hosts_required: list[str] = Field(default_factory=list)
    hosts_reachable: dict[str, str] = Field(default_factory=dict)
    api_keys: list[dict] = Field(default_factory=list)
    ingestion_progress: dict = Field(default_factory=dict)
    quality_problems: list[str] = Field(default_factory=list)
    leakage_risks: list[dict] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    verdict: str = "NOT_READY"
    verdict_reason: str = ""

    def save_json(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        return Path(path)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("UNKNOWN").astype(str).value_counts().items()
    }


def _coverage_fraction(
    frame: pd.DataFrame, start: datetime, end: datetime, gaps: list[dict]
) -> float | None:
    total = (end - start).total_seconds()
    if total <= 0 or frame.empty:
        return None
    missing = sum(gap["hours"] for gap in gaps) * 3600
    return round(max(0.0, min(1.0, 1.0 - missing / total)), 4)


def assess_dataset(
    dataset: str,
    path: Path | None,
    start: datetime,
    end: datetime,
    validator,
    max_gap_hours: float,
    expect_provenance: tuple[str, ...],
    point_in_time_only: bool = True,
) -> Readiness:
    """Assess one point-in-time dataset against the requested period."""
    if path is None or not Path(path).exists():
        return Readiness(
            status="UNAVAILABLE",
            dataset=dataset,
            path=str(path) if path else None,
            requested_start=start,
            requested_end=end,
            notes=[
                f"no {dataset} dataset on disk. Nothing is substituted; the "
                "dependent agent answers UNAVAILABLE and the fail-closed policy "
                "applies."
            ],
        )

    frame = read_dataset(Path(path))
    if frame.empty:
        return Readiness(
            status="UNAVAILABLE",
            dataset=dataset,
            path=str(path),
            requested_start=start,
            requested_end=end,
            notes=[f"{dataset} dataset exists but contains no rows"],
        )

    report = (
        validator(frame, point_in_time_only=point_in_time_only)
        if dataset.startswith("sentiment")
        else validator(frame)
    )
    gaps = coverage_gaps(frame, start, end, max_gap_hours=max_gap_hours)
    stamps = pd.to_datetime(frame["timestamp"], utc=True)

    readiness = Readiness(
        status="READY",
        dataset=dataset,
        path=str(path),
        records=len(frame),
        first_timestamp=stamps.min().to_pydatetime(),
        last_timestamp=stamps.max().to_pydatetime(),
        requested_start=start,
        requested_end=end,
        missing_ranges=gaps[:20],
        missing_ranges_truncated=max(0, len(gaps) - 20),
        coverage_fraction=_coverage_fraction(frame, start, end, gaps),
        provenance_counts=_counts(frame, "provenance"),
        time_precision_counts=_counts(frame, "time_precision"),
        sources=sorted(set(frame["source"].astype(str)))[:12]
        if "source" in frame.columns
        else [],
        validation_passed=report.passed,
        validation_errors=[f"{f.check}: {f.message}" for f in report.errors],
        validation_warnings=[f"{f.check}: {f.message}" for f in report.warnings],
    )

    unexpected = set(readiness.provenance_counts) - set(expect_provenance)
    if unexpected:
        readiness.notes.append(
            f"unexpected provenance value(s) present: {sorted(unexpected)}"
        )
    if not report.passed:
        readiness.status = "INVALID"
    elif gaps:
        readiness.status = "PARTIAL"
        readiness.notes.append(
            f"{len(gaps)} gap(s) longer than {max_gap_hours}h; timestamps inside a "
            "gap are served as UNAVAILABLE individually"
        )
    return readiness


def assess_ticks(
    tick_dir: Path, candle_path: Path, start: datetime, end: datetime, timeframe: int
) -> TickReadiness:
    """Assess the market-data side: raw ticks and the derived bars."""
    tick_files = (
        sorted(list(tick_dir.rglob("*.bi5")) + list(tick_dir.rglob("*.parquet")))
        if tick_dir.exists()
        else []
    )
    if not candle_path.exists():
        return TickReadiness(
            status="UNAVAILABLE",
            tick_files=len(tick_files),
            candle_path=str(candle_path),
            notes=[
                f"{len(tick_files)} raw tick file(s) cached; no M{timeframe} candle "
                "dataset built. Nothing is synthesised."
            ],
        )

    frame = read_dataset(candle_path)
    if frame.empty:
        return TickReadiness(
            status="UNAVAILABLE",
            tick_files=len(tick_files),
            candle_path=str(candle_path),
            notes=["candle dataset exists but is empty"],
        )

    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    # A weekend is a normal gap in FX; flag anything materially longer.
    gaps = coverage_gaps(frame, start, end, max_gap_hours=72.0)
    return TickReadiness(
        status="PARTIAL" if gaps else "READY",
        tick_files=len(tick_files),
        candle_records=len(frame),
        candle_path=str(candle_path),
        first_timestamp=stamps.min().to_pydatetime(),
        last_timestamp=stamps.max().to_pydatetime(),
        missing_ranges=gaps[:20],
        notes=(
            [f"{len(gaps)} gap(s) beyond a normal weekend break"] if gaps else []
        ),
    )


def assess_split(candle_path: Path, fraction: float, embargo: int, seal_path: Path) -> SplitReadiness:
    """Can the chronological 70/30 split be formed at all?"""
    if not candle_path.exists():
        return SplitReadiness(
            ready=False,
            reason=(
                "no candle dataset, so no chronological split exists yet. The split "
                "boundary is computed from the data's own bar count, so it cannot be "
                "established before the bars are built."
            ),
        )
    frame = read_dataset(candle_path)
    if frame.empty:
        return SplitReadiness(ready=False, reason="candle dataset is empty")

    from research.data.split import DataSplit

    try:
        split = DataSplit(
            frame,
            development_fraction=fraction,
            embargo_bars=embargo,
            seal_path=seal_path,
        )
    except ValueError as exc:
        return SplitReadiness(ready=False, reason=f"split cannot be formed: {exc}")

    summary = split.summary()
    return SplitReadiness(
        ready=True,
        reason="chronological split can be formed from the candle dataset",
        total_bars=summary["total_bars"],
        development_bars=summary["development_bars"],
        out_of_sample_bars=summary["out_of_sample_bars"],
        development_start=summary["development_start"],
        development_end=summary["development_end"],
        out_of_sample_start=summary["out_of_sample_start"],
        out_of_sample_end=summary["out_of_sample_end"],
        embargo_bars=summary["embargo_bars"],
        sealed=summary["sealed"],
    )


STANDING_LEAKAGE_RISKS: list[dict] = [
    {
        "risk": "MODEL_KNOWLEDGE_LEAKAGE",
        "severity": "CRITICAL",
        "status": "UNRESOLVED (not resolvable by data handling)",
        "detail": (
            "Point-in-time input data does not give point-in-time model knowledge. A "
            "model shown a 2023 setup may recognise the period from its own training "
            "data. No amount of correct data plumbing addresses this, and prompt "
            "instructions do not remove information from weights."
        ),
        "mitigation": (
            "Designed, not run: a date-stripped placebo arm (AI_STRIP_DATES), a "
            "shuffled-period control, and -- the only conclusive test -- forward "
            "paper trading past the model's training cutoff."
        ),
    },
    {
        "risk": "IMPUTED_RELEASE_TIMES",
        "severity": "MATERIAL",
        "status": "ACCEPTED BY CONFIGURATION (SCHEDULED_LOCAL)",
        "detail": (
            "ALFRED records a release DATE, not a clock time. Under SCHEDULED_LOCAL "
            "the time comes from the publisher's long-standing schedule (e.g. 08:30 "
            "America/New_York), converted to UTC with DST handled. If a particular "
            "release deviated from its usual time, the imputed timestamp could make "
            "a figure visible slightly before it printed."
        ),
        "mitigation": (
            "Every such record is marked IMPUTED_FROM_SCHEDULE and counted in this "
            "report, so the exposure is measurable. END_OF_DAY eliminates the risk "
            "at the cost of hiding the intraday reaction."
        ),
    },
    {
        "risk": "AGGREGATOR_INGEST_LAG",
        "severity": "CONTROLLED",
        "status": "MITIGATED",
        "detail": (
            "An article's publication time precedes the moment the aggregator "
            "exposed it. Filtering on publication time alone would grant up to 15 "
            "minutes of lookahead."
        ),
        "mitigation": (
            "Availability is max(published_at, discovered_at), and the point-in-time "
            "loader recomputes it from the component columns so a hand-edited "
            "timestamp cannot move it earlier."
        ),
    },
    {
        "risk": "REVISION_LEAKAGE",
        "severity": "CONTROLLED",
        "status": "MITIGATED",
        "detail": (
            "A revised economic figure presented as the value known at the original "
            "release would corrupt every decision at that timestamp."
        ),
        "mitigation": (
            "Original releases and revisions are separate records; every REVISED row "
            "is dropped at load time before any query runs; validation refuses a "
            "dataset in which a revision is not strictly later than its original."
        ),
    },
    {
        "risk": "RETROSPECTIVE_SENTIMENT",
        "severity": "CONTROLLED",
        "status": "MITIGATED",
        "detail": (
            "Sentiment produced today from an archived article is hindsight wearing a "
            "timestamp."
        ),
        "mitigation": (
            "Builders can emit only one hard-wired provenance value; RETROSPECTIVE "
            "records live in a separate file, are dropped at load, and fail "
            "validation if they appear in the main sentiment dataset."
        ),
    },
]


def build_report(
    experiment: str,
    start: datetime,
    end: datetime,
    release_time_policy: str,
    tick_dir: Path,
    candle_path: Path,
    news_path: Path,
    calendar_path: Path,
    sentiment_path: Path,
    sentiment_retrospective_path: Path,
    seal_path: Path,
    checkpoint_path: Path,
    timeframe_minutes: int = 15,
    development_fraction: float = 0.70,
    embargo_bars: int = 200,
    host_probe: dict[str, str] | None = None,
    news_max_gap_hours: float = 72.0,
    calendar_max_gap_hours: float = 336.0,
    sentiment_max_gap_hours: float = 72.0,
) -> DataReadinessReport:
    ticks = assess_ticks(tick_dir, candle_path, start, end, timeframe_minutes)
    news = assess_dataset(
        "news", news_path, start, end, validate_news, news_max_gap_hours,
        (Provenance.ORIGINAL_RELEASE.value,),
    )
    calendar = assess_dataset(
        "economic_calendar", calendar_path, start, end, validate_calendar,
        calendar_max_gap_hours,
        (Provenance.ORIGINAL_RELEASE.value, Provenance.REVISED.value),
    )
    sentiment = assess_dataset(
        "sentiment", sentiment_path, start, end, validate_sentiment,
        sentiment_max_gap_hours, (Provenance.POINT_IN_TIME_CAPTURE.value,),
        point_in_time_only=True,
    )
    retrospective = assess_dataset(
        "sentiment_retrospective", sentiment_retrospective_path, start, end,
        validate_sentiment, sentiment_max_gap_hours,
        (Provenance.RETROSPECTIVE.value,), point_in_time_only=False,
    )
    split = assess_split(candle_path, development_fraction, embargo_bars, seal_path)

    progress: dict = {}
    if Path(checkpoint_path).exists():
        checkpoint = IngestCheckpoint(Path(checkpoint_path))
        for spec in IMPLEMENTED_SOURCES:
            stats = checkpoint.stats(spec.key)
            if stats["windows"]:
                progress[spec.key] = stats

    report = DataReadinessReport(
        experiment=experiment,
        requested_start=start,
        requested_end=end,
        calendar_release_time_policy=release_time_policy,
        ticks=ticks,
        news=news,
        calendar=calendar,
        sentiment=sentiment,
        sentiment_retrospective=retrospective,
        split=split,
        hosts_required=hosts_required(),
        hosts_reachable=host_probe or {},
        api_keys=required_keys(),
        ingestion_progress=progress,
        leakage_risks=STANDING_LEAKAGE_RISKS,
    )
    _finalise(report)
    return report


def _finalise(report: DataReadinessReport) -> None:
    """Derive the quality problems, blockers and the single verdict."""
    problems: list[str] = []
    blockers: list[str] = []

    for readiness in (report.news, report.calendar, report.sentiment):
        problems.extend(
            f"{readiness.dataset}: {error}" for error in readiness.validation_errors
        )
        problems.extend(
            f"{readiness.dataset}: {warning}" for warning in readiness.validation_warnings
        )

    if report.ticks.status == "UNAVAILABLE":
        blockers.append(
            "No M15 candle dataset. Without bars there is no chronological split, no "
            "signals, and therefore no development phase."
        )
    if report.news.status == "UNAVAILABLE":
        blockers.append(
            "No news dataset. The news agent would answer UNAVAILABLE on every "
            "signal, so the experiment would measure a three-agent chain."
        )
    if report.calendar.status == "UNAVAILABLE":
        blockers.append(
            "No economic-calendar dataset. Event-risk blackout rules cannot fire."
        )
    if report.sentiment.status == "UNAVAILABLE":
        blockers.append(
            "No point-in-time sentiment dataset. The sentiment agent would answer "
            "UNAVAILABLE on every signal."
        )
    for readiness in (report.news, report.calendar, report.sentiment, report.ticks):
        status = getattr(readiness, "status", None)
        if status == "INVALID":
            blockers.append(
                f"{getattr(readiness, 'dataset', 'ticks')} failed validation; the "
                "pipeline fails closed rather than running on it."
            )

    unreachable = [
        host for host, state in report.hosts_reachable.items() if not state.startswith("OK")
    ]
    if unreachable:
        blockers.append(
            "Host(s) not reachable from this environment: "
            + ", ".join(f"{h} ({report.hosts_reachable[h]})" for h in unreachable)
        )
    missing_keys = [k["env_var"] for k in report.api_keys if not k["present"]]
    if missing_keys:
        blockers.append(
            "API key(s) not set: " + ", ".join(missing_keys)
        )

    imputed = report.calendar.time_precision_counts.get(
        TimePrecision.IMPUTED_FROM_SCHEDULE.value, 0
    )
    if report.calendar.usable and imputed == 0 and report.calendar.records:
        problems.append(
            "calendar: SCHEDULED_LOCAL was requested but no record is marked "
            "IMPUTED_FROM_SCHEDULE; the policy may not have been applied"
        )

    report.quality_problems = problems
    report.blockers = blockers

    datasets_ready = all(
        r.usable for r in (report.news, report.calendar, report.sentiment)
    )
    if blockers:
        report.verdict = "NOT_READY"
        report.verdict_reason = (
            f"{len(blockers)} blocker(s) remain; the development phase cannot begin. "
            "Nothing was fabricated to fill the gaps."
        )
    elif not report.split.ready:
        report.verdict = "NOT_READY"
        report.verdict_reason = report.split.reason
    elif not datasets_ready:
        report.verdict = "PARTIAL"
        report.verdict_reason = (
            "market data and the split are ready, but at least one point-in-time "
            "dataset is unusable; the experiment would measure a reduced agent chain."
        )
    else:
        report.verdict = "READY"
        report.verdict_reason = (
            "market data, the chronological split and all three point-in-time "
            "datasets are present, validated and provenance-tagged."
        )


# --- rendering -------------------------------------------------------------
def _fmt(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "-"


def render_markdown(report: DataReadinessReport) -> str:
    r = report
    lines = [
        "# Data readiness report",
        "",
        f"Generated {r.generated_at:%Y-%m-%d %H:%M} UTC for `{r.experiment}`.",
        f"Requested period: **{_fmt(r.requested_start)} .. {_fmt(r.requested_end)}**",
        f"Calendar release-time policy: **{r.calendar_release_time_policy}**",
        "",
        f"## Verdict: {r.verdict}",
        "",
        r.verdict_reason,
        "",
        "## Coverage",
        "",
        "| Dataset | Status | Records | First | Last | Coverage | Gaps |",
        "|---|---|---|---|---|---|---|",
        f"| Tick / M15 candles | {r.ticks.status} | {r.ticks.candle_records} | "
        f"{_fmt(r.ticks.first_timestamp)} | {_fmt(r.ticks.last_timestamp)} | - | "
        f"{len(r.ticks.missing_ranges)} |",
    ]
    for readiness in (r.news, r.calendar, r.sentiment, r.sentiment_retrospective):
        coverage = (
            f"{readiness.coverage_fraction * 100:.1f}%"
            if readiness.coverage_fraction is not None
            else "-"
        )
        lines.append(
            f"| {readiness.dataset} | {readiness.status} | {readiness.records} | "
            f"{_fmt(readiness.first_timestamp)} | {_fmt(readiness.last_timestamp)} | "
            f"{coverage} | {len(readiness.missing_ranges)}"
            + (f" (+{readiness.missing_ranges_truncated})" if readiness.missing_ranges_truncated else "")
            + " |"
        )
    lines.append("")

    lines += ["## Provenance", "", "| Dataset | Provenance | Records |", "|---|---|---|"]
    any_provenance = False
    for readiness in (r.news, r.calendar, r.sentiment, r.sentiment_retrospective):
        for provenance, count in sorted(readiness.provenance_counts.items()):
            marker = " **(inadmissible)**" if provenance in INADMISSIBLE_PROVENANCE else ""
            lines.append(f"| {readiness.dataset} | `{provenance}`{marker} | {count} |")
            any_provenance = True
    if not any_provenance:
        lines.append("| - | no records of any provenance | 0 |")
    lines.append("")

    imputed = r.calendar.time_precision_counts.get(
        TimePrecision.IMPUTED_FROM_SCHEDULE.value, 0
    )
    lines += [
        "### Calendar timestamp precision",
        "",
        f"- `IMPUTED_FROM_SCHEDULE`: **{imputed}** record(s)",
    ]
    for precision, count in sorted(r.calendar.time_precision_counts.items()):
        if precision != TimePrecision.IMPUTED_FROM_SCHEDULE.value:
            lines.append(f"- `{precision}`: {count} record(s)")
    lines.append("")

    lines += [
        "### Sentiment provenance",
        "",
        f"- `POINT_IN_TIME_CAPTURE`: "
        f"{r.sentiment.provenance_counts.get(Provenance.POINT_IN_TIME_CAPTURE.value, 0)}",
        f"- `RETROSPECTIVE`: "
        f"{r.sentiment_retrospective.provenance_counts.get(Provenance.RETROSPECTIVE.value, 0)} "
        "(separate dataset, inadmissible for the leakage-safe experiment)",
        f"- `UNAVAILABLE`: "
        + (
            "the whole period"
            if r.sentiment.status == "UNAVAILABLE"
            else f"{len(r.sentiment.missing_ranges)} gap period(s)"
        ),
        "",
    ]

    if any(x.missing_ranges for x in (r.news, r.calendar, r.sentiment)) or r.ticks.missing_ranges:
        lines += ["## Missing date ranges", ""]
        for label, ranges in (
            ("ticks/candles", r.ticks.missing_ranges),
            ("news", r.news.missing_ranges),
            ("calendar", r.calendar.missing_ranges),
            ("sentiment", r.sentiment.missing_ranges),
        ):
            if not ranges:
                continue
            lines.append(f"**{label}**")
            for gap in ranges[:10]:
                lines.append(
                    f"- {gap['start'][:16]} .. {gap['end'][:16]} "
                    f"({gap['hours']}h) — {gap['reason']}"
                )
            lines.append("")

    lines += ["## Sources", "", "| Dataset | Sources seen in the data |", "|---|---|"]
    for readiness in (r.news, r.calendar, r.sentiment):
        lines.append(
            f"| {readiness.dataset} | "
            + (", ".join(readiness.sources) if readiness.sources else "none") + " |"
        )
    lines.append("")

    lines += ["## 70/30 split", ""]
    if r.split.ready:
        lines += [
            f"- Total bars: {r.split.total_bars}",
            f"- Development (first 70%): {r.split.development_bars} bars, "
            f"{_fmt(r.split.development_start)} .. {_fmt(r.split.development_end)}",
            f"- Out-of-sample (final 30%): {r.split.out_of_sample_bars} bars, "
            f"{_fmt(r.split.out_of_sample_start)} .. {_fmt(r.split.out_of_sample_end)}",
            f"- Embargo at the boundary: {r.split.embargo_bars} bars",
            f"- Strategy sealed: {'yes' if r.split.sealed else 'no (out-of-sample locked)'}",
        ]
    else:
        lines.append(f"**Cannot be formed.** {r.split.reason}")
    lines.append("")

    if r.quality_problems:
        lines += ["## Remaining data-quality problems", ""]
        lines += [f"- {problem}" for problem in r.quality_problems]
        lines.append("")
    else:
        lines += ["## Remaining data-quality problems", "", "None detected.", ""]

    lines += ["## Look-ahead / leakage risks", ""]
    for risk in r.leakage_risks:
        lines += [
            f"**{risk['risk']}** — {risk['severity']} — {risk['status']}",
            "",
            risk["detail"],
            "",
            f"_Mitigation:_ {risk['mitigation']}",
            "",
        ]

    lines += ["## Environment", "", "| Host | Reachable |", "|---|---|"]
    for host in r.hosts_required:
        lines.append(f"| `{host}` | {r.hosts_reachable.get(host, 'not probed')} |")
    lines += ["", "| API key | Set |", "|---|---|"]
    for key in r.api_keys:
        lines.append(f"| `{key['env_var']}` | {'yes' if key['present'] else 'NO'} |")
    lines.append("")

    if r.blockers:
        lines += ["## Blockers", ""]
        lines += [f"{i}. {blocker}" for i, blocker in enumerate(r.blockers, 1)]
        lines.append("")

    return "\n".join(lines)


def write_reports(report: DataReadinessReport, directory: Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = report.save_json(directory / "data_readiness.json")
    md_path = directory / "data_readiness.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
