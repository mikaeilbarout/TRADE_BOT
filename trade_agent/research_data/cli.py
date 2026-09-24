from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from research.data.ingest.base import Provenance, SourceCost
from research.data.ingest.cache import ArtifactCache
from research.data.ingest.checkpoint import IngestCheckpoint
from research.data.ingest.env import load_env_file
from research.data.ingest.fred import (
    DEFAULT_SERIES,
    FredCalendarSource,
    ReleaseTimePolicy,
)
from research.data.ingest.gdelt import GDELT_2_START, GdeltGkgSource
from research.data.ingest.normalize import read_dataset, to_frame, write_dataset
from research.data.ingest.pipeline import IngestPipeline, coverage_gaps
from research.data.ingest.registry import (
    ALL_SOURCES,
    IMPLEMENTED_SOURCES,
    catalogue_rows,
    hosts_required,
    required_keys,
)
from research.data.ingest.validate import (
    ValidationFailed,
    fail_closed,
    validate_calendar,
    validate_news,
    validate_sentiment,
)
from research.experiment import ExperimentConfig, load_experiment

"""`python -m research_data <command>` -- automated historical data ingestion.

Commands:

    sources          what sources exist, what they cost, what keys they need
    fetch-news       GDELT 2.0 GKG archives -> news dataset            (resumable)
    fetch-calendar   FRED/ALFRED vintages -> calendar dataset          (resumable)
    build-sentiment  point-in-time tone, or retrospective LLM scoring  (resumable)
    validate         run every check; fail closed
    status           what exists, what is missing, what to run next

Every fetch command is resumable: progress is committed per window, and a
re-run skips what is already settled. Nothing here fabricates data -- a period
that cannot be fetched is reported as a gap.
"""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MISSING_INPUT = 2
EXIT_VALIDATION_FAILED = 3


def _paths(config: ExperimentConfig) -> dict[str, Path]:
    data = config.backtest.data_dir
    results = config.backtest.results_dir
    return {
        "news": data / "news" / "news.parquet",
        "calendar": data / "calendar" / "calendar.parquet",
        "sentiment": data / "sentiment" / "sentiment.parquet",
        "sentiment_retrospective": data / "sentiment" / "sentiment_retrospective.parquet",
        "cache": data / "ingest_cache",
        "checkpoint": data / "ingest_checkpoint.sqlite",
        "llm_cache": data / "sentiment" / "llm_scores.json",
        "reports": results / "ingest",
    }


def _range(config: ExperimentConfig, args) -> tuple[datetime, datetime]:
    start = (
        datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        if getattr(args, "start", None)
        else datetime.combine(
            config.backtest.start_date, datetime.min.time(), timezone.utc
        )
    )
    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if getattr(args, "end", None)
        else datetime.combine(config.backtest.end_date, datetime.min.time(), timezone.utc)
    )
    return start, end


def _config(args) -> ExperimentConfig:
    config = load_experiment(Path(args.config) if getattr(args, "config", None) else None)
    if getattr(args, "data_dir", None):
        config.backtest.data_dir = Path(args.data_dir)
    return config


# --- commands --------------------------------------------------------------
def cmd_sources(args) -> int:
    """Report every evaluated source, and what is needed to use it."""
    if getattr(args, "json", False):
        # Machine-readable mode emits JSON and nothing else, so it can be piped.
        print(
            json.dumps(
                {
                    "hosts_required": hosts_required(),
                    "api_keys": required_keys(),
                    "sources": catalogue_rows(),
                },
                indent=2,
            )
        )
        return EXIT_OK

    print("HOSTS that must be reachable for the implemented sources:")
    for host in hosts_required():
        print(f"  - {host}")
    print()

    print("API KEYS required:")
    keys = required_keys()
    if not keys:
        print("  none")
    for key in keys:
        state = "SET" if key["present"] else "NOT SET"
        print(f"  - {key['env_var']}  [{state}]  ({key['cost']}) for {key['name']}")
        if not key["present"] and key["signup"]:
            print(f"      free signup: {key['signup']}")
    print()

    print("IMPLEMENTED sources:")
    for spec in IMPLEMENTED_SOURCES:
        print(f"  {spec.key}: {spec.name}")
        print(f"    kinds      : {', '.join(spec.kinds)}")
        print(f"    coverage   : from {spec.coverage_start} -- {spec.coverage_note}")
        print(f"    timestamps : {spec.timestamp_granularity}")
        print(f"    cost       : {spec.cost.value}")
        print(f"    rate limit : {spec.rate_limit}")
        print(f"    licensing  : {spec.licensing}")
        print(f"    provenance : {spec.provenance_support}")
    print()

    print("EVALUATED and NOT used:")
    for spec in ALL_SOURCES:
        if spec.implemented:
            continue
        print(f"  {spec.key}: {spec.not_implemented_reason}")
    return EXIT_OK


def cmd_fetch_news(args) -> int:
    config = _config(args)
    p = _paths(config)
    start, end = _range(config, args)

    source = GdeltGkgSource(
        cache=ArtifactCache(p["cache"]),
        min_relevance=args.min_relevance,
        max_records_per_slot=args.max_per_slot,
    )
    shortfall = source.coverage_shortfall(start, end)
    if shortfall:
        print(f"NOTE: {shortfall}")

    pipeline = IngestPipeline(
        source,
        IngestCheckpoint(p["checkpoint"]),
        kind="news",
        requests_per_second=args.rate,
    )
    print(
        f"fetching news {start:%Y-%m-%d} .. {end:%Y-%m-%d} from {source.spec.name}\n"
        f"  cache: {p['cache']}\n  dataset: {p['news']}"
    )
    outcome = pipeline.run(
        start, end, p["news"], max_windows=args.max_windows
    )
    _report_outcome(outcome, p["reports"], "news")
    if outcome.stopped_early:
        print(f"\nSTOPPED: {outcome.stopped_reason}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    return EXIT_OK


def cmd_fetch_calendar(args) -> int:
    config = _config(args)
    p = _paths(config)
    start, end = _range(config, args)

    source = FredCalendarSource(
        cache=ArtifactCache(p["cache"]),
        release_time_policy=ReleaseTimePolicy(args.release_time_policy),
    )
    blocked = source.preflight()
    if blocked:
        print(f"ERROR: {blocked}", file=sys.stderr)
        return EXIT_MISSING_INPUT

    pipeline = IngestPipeline(
        source,
        IngestCheckpoint(p["checkpoint"]),
        kind="calendar",
        requests_per_second=args.rate,
    )
    print(
        f"fetching calendar {start:%Y-%m-%d} .. {end:%Y-%m-%d} from {source.spec.name}\n"
        f"  series: {len(DEFAULT_SERIES)} tracked\n"
        f"  release-time policy: {args.release_time_policy}\n"
        f"  vintages: initial releases AND revisions, stored separately\n"
        f"  dataset: {p['calendar']}"
    )
    outcome = pipeline.run(start, end, p["calendar"], max_windows=args.max_windows)
    _report_outcome(outcome, p["reports"], "calendar")
    if outcome.stopped_early:
        print(f"\nSTOPPED: {outcome.stopped_reason}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    return EXIT_OK


def cmd_build_sentiment(args) -> int:
    """Build sentiment, point-in-time by default.

    The two modes are different datasets with different admissibility, so they
    are written to different files and never merged.
    """
    config = _config(args)
    p = _paths(config)
    start, end = _range(config, args)

    if args.mode == "point-in-time":
        source = GdeltGkgSource(cache=ArtifactCache(p["cache"]))
        pipeline = IngestPipeline(
            source,
            IngestCheckpoint(p["checkpoint"]),
            kind="sentiment",
            requests_per_second=args.rate,
        )
        print(
            "building POINT_IN_TIME_CAPTURE sentiment from GDELT tone values that "
            "were published in each 15-minute archive file.\n"
            "  No model is called; nothing is recomputed today.\n"
            f"  dataset: {p['sentiment']}"
        )
        outcome = pipeline.run(
            start,
            end,
            p["sentiment"],
            max_windows=args.max_windows,
            sentiment_mode=True,
        )
        _report_outcome(outcome, p["reports"], "sentiment")
        if outcome.stopped_early:
            print(f"\nSTOPPED: {outcome.stopped_reason}", file=sys.stderr)
            return EXIT_MISSING_INPUT
        return EXIT_OK

    # --- retrospective ---------------------------------------------------
    from research.data.ingest.sentiment import RetrospectiveLlmBuilder

    news = read_dataset(p["news"])
    if news.empty:
        print(
            f"ERROR: no news dataset at {p['news']}. Retrospective sentiment is "
            "scored FROM the news corpus, so fetch news first "
            "(python -m research_data fetch-news).",
            file=sys.stderr,
        )
        return EXIT_MISSING_INPUT

    print(
        "WARNING: retrospective mode scores archived headlines with a model running\n"
        "         TODAY. Every record is labelled RETROSPECTIVE, the leakage-safe\n"
        "         experiment refuses it, and it must not be presented as\n"
        "         point-in-time sentiment. It exists as a diagnostic comparison.\n"
    )

    from research.data.ingest.base import NewsRecord, now_utc

    records = [
        NewsRecord(
            source=str(row.get("source", "unknown")),
            source_id=str(row.get("source_id", "")),
            headline=str(row.get("headline", "")),
            category=str(row.get("category", "general")),
            published_at=row["timestamp"],
            retrieved_at=now_utc(),
            provenance=Provenance.ORIGINAL_RELEASE,
        )
        for _, row in news.iterrows()
    ]
    builder = RetrospectiveLlmBuilder(model=args.model, cache_path=p["llm_cache"])
    batches = builder.batches(records)
    estimate = builder.cost_estimate(batches)
    print(json.dumps(estimate, indent=2))

    if not args.execute:
        print(
            "\nDry run. Nothing was scored and nothing was spent. Re-run with "
            "--execute to score the pending batches."
        )
        return EXIT_OK

    print(
        "\nERROR: scoring requires an ANTHROPIC_API_KEY and is not run by this "
        "command automatically. The batches, the cache and the cost estimate are "
        "ready; wire the scoring call once you have decided to spend, so that no "
        "command in this pipeline can charge your account as a side effect.",
        file=sys.stderr,
    )
    return EXIT_MISSING_INPUT


def cmd_validate(args) -> int:
    config = _config(args)
    p = _paths(config)
    start, end = _range(config, args)

    reports = []
    frames = {
        "news": (read_dataset(p["news"]), validate_news),
        "calendar": (read_dataset(p["calendar"]), validate_calendar),
        "sentiment": (read_dataset(p["sentiment"]), validate_sentiment),
    }
    for name, (frame, validator) in frames.items():
        report = validator(frame)
        reports.append(report)
        print(f"{report.summary()}")
        for finding in report.findings:
            print(f"  [{finding.severity.value}] {finding.check}: {finding.message}")
            for example in finding.examples[:2]:
                print(f"      e.g. {example}")
        gaps = coverage_gaps(frame, start, end, max_gap_hours=args.max_gap_hours)
        if gaps:
            print(f"  missing periods ({len(gaps)}):")
            for gap in gaps[:5]:
                print(
                    f"    {gap['start'][:16]} .. {gap['end'][:16]} "
                    f"({gap['hours']}h) -- {gap['reason']}"
                )
            if len(gaps) > 5:
                print(f"    ... and {len(gaps) - 5} more")

    p["reports"].mkdir(parents=True, exist_ok=True)
    (p["reports"] / "validation.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in reports], indent=2, default=str)
    )
    print(f"\nwritten -> {p['reports'] / 'validation.json'}")

    try:
        fail_closed(reports)
    except ValidationFailed as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILED

    # An empty dataset raises no ERROR, but "passed validation" would be a
    # misleading thing to say about a file with nothing in it. Report the
    # distinction and exit non-zero, so a caller cannot read silence as success.
    empty = [report.dataset for report in reports if report.rows == 0]
    populated = [report for report in reports if report.rows > 0]
    if empty:
        print(
            f"\n{len(empty)} dataset(s) are EMPTY and therefore not validated: "
            f"{', '.join(empty)}."
        )
        if populated:
            print(
                f"{len(populated)} populated dataset(s) passed every check: "
                f"{', '.join(r.dataset for r in populated)}."
            )
        else:
            print(
                "No dataset contains any rows, so nothing was actually validated. "
                "Run the fetch commands first; see `python -m research_data readiness` "
                "for the full picture."
            )
        return EXIT_MISSING_INPUT

    print(f"\nall {len(populated)} dataset(s) passed validation")
    return EXIT_OK


def _probe_hosts(hosts: list[str], timeout: float = 12.0) -> dict[str, str]:
    """Check each required host, reporting WHY it is unreachable.

    A policy denial (403/407 from the proxy) is reported as such rather than as a
    generic failure, because the remedy is completely different from a network
    outage: the host has to be allowed, not retried.
    """
    import httpx

    probes = {
        "data.gdeltproject.org": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
        "api.stlouisfed.org": "https://api.stlouisfed.org/fred/releases?file_type=json",
    }
    out: dict[str, str] = {}
    for host in hosts:
        url = probes.get(host, f"https://{host}/")
        try:
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
            if response.status_code in (401, 403, 407):
                out[host] = (
                    f"BLOCKED ({response.status_code}) -- not permitted by the "
                    "environment's network egress policy, or credentials rejected"
                )
            elif response.status_code >= 500:
                out[host] = f"UNAVAILABLE (provider returned {response.status_code})"
            else:
                out[host] = f"OK ({response.status_code})"
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            detail = str(exc)
            if "403" in detail or "CONNECT" in detail.upper():
                out[host] = (
                    "BLOCKED (proxy refused CONNECT) -- not permitted by the "
                    "environment's network egress policy"
                )
            else:
                out[host] = f"UNREACHABLE ({type(exc).__name__}: {detail[:80]})"
    return out


def cmd_import_candles(args) -> int:
    """Import a supplied OHLC export as the candle dataset, and derive the
    higher timeframe the strategy's trend filter needs."""
    from research.data.candle_import import (
        CandleImportError,
        load_candle_csv,
        resample_candles,
        write_candles,
    )

    config = _config(args)
    p = _paths(config)
    try:
        frame, report = load_candle_csv(
            Path(args.csv),
            symbol=config.backtest.symbol,
            timeframe_minutes=config.backtest.timeframe_minutes,
            source_timezone=args.timezone,
        )
    except CandleImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT

    entry_path = write_candles(frame, config.backtest.candle_path)
    trend_minutes = args.trend_minutes
    trend = resample_candles(frame, trend_minutes, config.backtest.timeframe_minutes)
    trend_path = write_candles(
        trend,
        config.backtest.data_dir / "candles"
        / f"{config.backtest.symbol}_M{trend_minutes}.parquet",
    )

    print(
        f"imported {report.rows_out:,} M{report.timeframe_minutes} bars "
        f"({report.first_timestamp:%Y-%m-%d} .. {report.last_timestamp:%Y-%m-%d})"
    )
    print(f"  source sha256 : {report.source_sha256[:16]}...")
    print(f"  timezone      : {report.source_timezone}")
    print(f"  spread        : {'observed' if report.spread_observed else 'NOT observed -- modelled'}")
    print(f"  session gaps  : {report.session_gaps} (largest {report.largest_gap_hours}h)")
    for note in report.notes:
        print(f"  - {note}")
    print(f"\nderived {len(trend):,} M{trend_minutes} trend bars -> {trend_path}")

    p["reports"].mkdir(parents=True, exist_ok=True)
    (p["reports"] / "candle_import.json").write_text(
        json.dumps(report.as_dict(), indent=2, default=str)
    )
    print(f"report -> {p['reports'] / 'candle_import.json'}")
    return EXIT_OK


def cmd_readiness(args) -> int:
    """Produce the data readiness report from what is actually on disk."""
    from research.data.readiness import build_report, render_markdown, write_reports

    config = _config(args)
    p = _paths(config)
    start, end = _range(config, args)

    host_probe = {} if args.no_probe else _probe_hosts(hosts_required())

    report = build_report(
        experiment=config.name,
        start=start,
        end=end,
        release_time_policy=config.calendar_release_time_policy,
        tick_dir=config.backtest.tick_dir,
        candle_path=config.backtest.candle_path,
        news_path=p["news"],
        calendar_path=p["calendar"],
        sentiment_path=p["sentiment"],
        sentiment_retrospective_path=p["sentiment_retrospective"],
        seal_path=config.backtest.seal_path,
        checkpoint_path=p["checkpoint"],
        timeframe_minutes=config.backtest.timeframe_minutes,
        development_fraction=config.backtest.split.development_fraction,
        embargo_bars=config.backtest.split.embargo_bars,
        host_probe=host_probe,
    )
    written = write_reports(report, p["reports"])
    print(render_markdown(report))
    print(f"written -> {written['json']}")
    print(f"         -> {written['markdown']}")
    return EXIT_OK if report.verdict == "READY" else EXIT_MISSING_INPUT


def cmd_status(args) -> int:
    config = _config(args)
    p = _paths(config)
    start, end = _range(config, args)
    checkpoint = IngestCheckpoint(p["checkpoint"]) if p["checkpoint"].exists() else None

    print(f"experiment period: {start:%Y-%m-%d} .. {end:%Y-%m-%d}\n")
    print("datasets:")
    for label, path, kind in (
        ("news", p["news"], "news"),
        ("calendar", p["calendar"], "economic_calendar"),
        ("sentiment (point-in-time)", p["sentiment"], "sentiment"),
        ("sentiment (retrospective)", p["sentiment_retrospective"], "sentiment"),
    ):
        frame = read_dataset(path)
        if frame.empty:
            print(f"  [ ] {label:28} UNAVAILABLE  ({path})")
            continue
        stamps = frame["timestamp"]
        provenances = (
            sorted(set(frame["provenance"].astype(str)))
            if "provenance" in frame.columns
            else ["-"]
        )
        print(
            f"  [x] {label:28} {len(frame):>7} rows  "
            f"{stamps.min():%Y-%m-%d} .. {stamps.max():%Y-%m-%d}  {provenances}"
        )

    if checkpoint:
        print("\ningestion progress (resumable):")
        for spec in IMPLEMENTED_SOURCES:
            stats = checkpoint.stats(spec.key)
            if not stats["windows"]:
                continue
            print(
                f"  {spec.key}: {stats['by_status']} over {stats['windows']} windows, "
                f"{stats['records']} records, {stats['requests_made']} requests"
            )
            failed = checkpoint.failed_windows(spec.key)
            if failed:
                print(
                    f"    {len(failed)} failed window(s) will be RETRIED on the next run; "
                    f"first: {failed[0].window_key} ({(failed[0].error or '')[:80]})"
                )
    else:
        print("\nno ingestion has run yet")

    cache = ArtifactCache(p["cache"])
    print(f"\ncache: {cache.summary()}")

    print("\nkeys:")
    for key in required_keys():
        print(f"  {key['env_var']}: {'SET' if key['present'] else 'NOT SET'}")

    print("\nnext:")
    if not read_dataset(p["news"]).empty:
        pass
    if read_dataset(p["news"]).empty:
        print("  python -m research_data fetch-news")
    elif read_dataset(p["calendar"]).empty:
        print("  python -m research_data fetch-calendar   (needs FRED_API_KEY)")
    elif read_dataset(p["sentiment"]).empty:
        print("  python -m research_data build-sentiment")
    else:
        print("  python -m research_data validate")
    return EXIT_OK


def _report_outcome(outcome, reports_dir: Path, label: str) -> None:
    print(
        f"\n  windows: {outcome.windows_total} total, "
        f"{outcome.windows_settled_before} already settled, "
        f"{outcome.windows_fetched} fetched, {outcome.windows_empty} empty, "
        f"{outcome.windows_failed} failed"
    )
    print(
        f"  records in dataset: {outcome.records_written} "
        f"({outcome.duplicates_removed} duplicates removed)"
    )
    print(
        f"  network: {outcome.requests_made} requests, "
        f"{outcome.bytes_fetched / 1e6:.1f} MB"
    )
    if outcome.errors:
        print(f"  first error: {outcome.errors[0][:200]}")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{label}_ingest.json").write_text(
        json.dumps(outcome.as_dict(), indent=2, default=str)
    )


# --- wiring ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> argparse.ArgumentParser:
        target.add_argument("--config", help="experiment config JSON")
        target.add_argument("--data-dir", help="override the data directory")
        target.add_argument("--start", help="ISO date; defaults to the experiment start")
        target.add_argument("--end", help="ISO date; defaults to the experiment end")
        return target

    sources = sub.add_parser("sources", help="list evaluated sources, costs and keys")
    sources.add_argument("--json", action="store_true")
    sources.set_defaults(func=cmd_sources)

    news = common(sub.add_parser("fetch-news", help="download historical news (resumable)"))
    news.add_argument("--rate", type=float, default=4.0, help="requests per second")
    news.add_argument("--max-windows", type=int, help="stop after N windows")
    news.add_argument(
        "--min-relevance",
        type=float,
        default=2.0,
        help="drop articles scoring below this (deterministic keyword/theme filter)",
    )
    news.add_argument("--max-per-slot", type=int, default=40)
    news.set_defaults(func=cmd_fetch_news)

    calendar = common(
        sub.add_parser("fetch-calendar", help="download historical releases (resumable)")
    )
    calendar.add_argument("--rate", type=float, default=2.0)
    calendar.add_argument("--max-windows", type=int)
    calendar.add_argument(
        "--release-time-policy",
        default=ReleaseTimePolicy.SCHEDULED_LOCAL.value,
        choices=[p.value for p in ReleaseTimePolicy],
        help=(
            "SCHEDULED_LOCAL (default, chosen for this experiment: the publisher's "
            "long-standing release clock time in US/Eastern converted to UTC with "
            "DST handled, and every such record marked IMPUTED_FROM_SCHEDULE) or "
            "END_OF_DAY (a release is visible only after its date ends -- more "
            "conservative, but it hides the intraday reaction)"
        ),
    )
    calendar.set_defaults(func=cmd_fetch_calendar)

    sentiment = common(sub.add_parser("build-sentiment", help="build a sentiment dataset"))
    sentiment.add_argument(
        "--mode",
        default="point-in-time",
        choices=["point-in-time", "retrospective"],
        help=(
            "point-in-time: aggregate tone values that were published at the time "
            "(no model, no cost). retrospective: score archived headlines with a "
            "model today, labelled RETROSPECTIVE and inadmissible."
        ),
    )
    sentiment.add_argument("--rate", type=float, default=4.0)
    sentiment.add_argument("--max-windows", type=int)
    sentiment.add_argument("--model", default="claude-haiku-4-5")
    sentiment.add_argument(
        "--execute", action="store_true", help="retrospective mode: actually spend"
    )
    sentiment.set_defaults(func=cmd_build_sentiment)

    validate = common(sub.add_parser("validate", help="validate datasets; fail closed"))
    validate.add_argument("--max-gap-hours", type=float, default=72.0)
    validate.set_defaults(func=cmd_validate)

    status = common(sub.add_parser("status", help="what exists and what to run next"))
    status.set_defaults(func=cmd_status)

    import_candles = common(
        sub.add_parser(
            "import-candles", help="import a supplied OHLC export as the candle dataset"
        )
    )
    import_candles.add_argument("--csv", required=True, help="path to the OHLC export")
    import_candles.add_argument(
        "--timezone",
        default="UTC",
        help="timezone of naive source timestamps (default UTC; a broker export in "
        "server time shifts every bar if this is wrong)",
    )
    import_candles.add_argument(
        "--trend-minutes",
        type=int,
        default=240,
        help="higher timeframe to derive for the trend filter (default 240 = H4)",
    )
    import_candles.set_defaults(func=cmd_import_candles)

    readiness = common(
        sub.add_parser("readiness", help="full data readiness report (JSON + Markdown)")
    )
    readiness.add_argument(
        "--no-probe", action="store_true", help="skip the host reachability probe"
    )
    readiness.set_defaults(func=cmd_readiness)
    return parser


def main(argv: list[str] | None = None, load_env: bool = True) -> int:
    """CLI entry point.

    `load_env=False` skips reading `.env`, which is how tests stay isolated from
    whatever credentials happen to exist on the machine running them. A test
    that silently picks up a real key starts making real network calls.
    """
    if load_env:
        # Credentials live in .env by documented convention; load it before any
        # source checks for a key. An exported variable still wins.
        load_env_file()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
