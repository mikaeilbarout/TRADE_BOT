from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.config import BacktestConfig
from research.experiment import (
    ExperimentConfig,
    ManifestMismatch,
    assert_ab_identical,
    load_experiment,
    save_experiment,
)
from research.data.split import LeakageError
from research.manifest import RunManifest, sha256_file

"""`python -m research.cli <command>` -- the whole experiment, reproducibly.

Every stage is a command, every command reads the same `ExperimentConfig`, and
each writes its outputs where the next one expects them. The point is that
reproducing a result means re-running a listed sequence of commands, not
knowing which internal functions to call in which order from a REPL.

    fetch      download / import tick data
    candles    tick data -> M15 bars
    split      report the chronological 70/30 boundary
    datasets   audit news / sentiment / calendar availability and provenance
    develop    walk-forward parameter search on the first 70% ONLY
    freeze     write the strategy seal (unlocks out-of-sample access)
    baseline   Experiment A on the sealed out-of-sample period
    pilot      Experiment B on a small sample, with a measured cost report
    compare    A vs B, with the counterfactual analysis
    status     what exists, what is missing, what can run next

Ordering is enforced by the artifacts, not by documentation: `baseline` cannot
read out-of-sample data until `freeze` has written a seal, and `freeze` refuses
to run without a `develop` report.
"""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MISSING_INPUT = 2
# A guard refused the operation because it would have leaked out-of-sample
# information. Its own code so a script can tell "you skipped a step" from
# "that step would have invalidated the experiment".
EXIT_LEAKAGE_REFUSED = 3


# --- paths ----------------------------------------------------------------
def _dataset_path(directory: Path, stem: str) -> Path:
    """Prefer an ingested parquet dataset, fall back to a supplied CSV.

    Returns the parquet path when neither exists, so an error message points at
    the file the pipeline would produce rather than one nobody was asked for.
    """
    parquet = directory / f"{stem}.parquet"
    csv = directory / f"{stem}.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    return parquet


def paths(config: ExperimentConfig) -> dict[str, Path]:
    backtest = config.backtest
    results = backtest.results_dir
    return {
        "ticks": backtest.tick_dir,
        "candles": backtest.candle_path,
        "seal": backtest.seal_path,
        "results": results,
        "develop": results / "optimization_report.json",
        "baseline": results / "baseline_result.json",
        "baseline_manifest": results / "baseline_manifest.json",
        "ai": results / "ai_result.json",
        "ai_manifest": results / "ai_manifest.json",
        "decisions": results / "ai_decisions.json",
        "datasets": results / "dataset_report.json",
        # The ingestion pipeline (python -m research_data) writes parquet;
        # a hand-supplied CSV at the same stem still works and is used when no
        # ingested dataset exists.
        "news": _dataset_path(backtest.data_dir / "news", "news"),
        "sentiment": _dataset_path(backtest.data_dir / "sentiment", "sentiment"),
        "calendar": _dataset_path(backtest.data_dir / "calendar", "calendar"),
        # Deliberately not a fixed key here: the checkpoint file is scoped
        # per-run_id in cmd_pilot, since resume is keyed only by signal_id
        # and a shared file across runs would replay an earlier run's
        # decisions at zero cost instead of re-deciding under new settings.
    }


def _load_config(args) -> ExperimentConfig:
    config = load_experiment(Path(args.config) if args.config else None)
    if args.symbol:
        config.backtest.symbol = args.symbol
    return config


def _require(path: Path, what: str, remedy: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found at {path}. {remedy}")


def _load_candles(config: ExperimentConfig) -> pd.DataFrame:
    path = paths(config)["candles"]
    _require(
        path,
        "candle dataset",
        "Build it first: `python -m research.cli fetch` then "
        "`python -m research.cli candles`. No synthetic series is substituted.",
    )
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _split(config: ExperimentConfig, candles: pd.DataFrame):
    from research.data.split import DataSplit

    return DataSplit(
        candles,
        development_fraction=config.backtest.split.development_fraction,
        embargo_bars=config.backtest.split.embargo_bars,
        seal_path=paths(config)["seal"],
    )


# --- commands -------------------------------------------------------------
def cmd_fetch(args) -> int:
    """Download or import tick data. Never fabricates it."""
    from research.data.sources.base import TickDataUnavailableError

    config = _load_config(args)
    destination = paths(config)["ticks"]
    destination.mkdir(parents=True, exist_ok=True)

    start = datetime.combine(config.backtest.start_date, datetime.min.time(), timezone.utc)
    end = datetime.combine(config.backtest.end_date, datetime.min.time(), timezone.utc)

    if args.csv:
        from research.data.sources.csv_source import CsvTickSource

        source = CsvTickSource(
            path=Path(args.csv),
            price_column=args.price_column,
            assumed_spread=args.assumed_spread,
            source_timezone=args.timezone,
        )
        try:
            frame = source.load_all()
        except TickDataUnavailableError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_MISSING_INPUT
        out = destination / "ticks.parquet"
        frame.to_parquet(out, index=False)
        print(
            f"imported {len(frame):,} ticks from {args.csv} "
            f"({frame['timestamp'].min()} .. {frame['timestamp'].max()}) -> {out}"
        )
        return EXIT_OK

    from research.data.sources.dukascopy import DukascopyTickSource

    source = DukascopyTickSource(
        cache_dir=destination,
        point_divisor=config.backtest.instrument.point_divisor,
    )
    hours = total = 0
    empty_hours = 0
    try:
        for hour in source.iter_hours(start, end):
            frame = source.fetch_hour(config.backtest.symbol, hour)
            hours += 1
            total += len(frame)
            if frame.empty:
                empty_hours += 1
            if args.limit_hours and hours >= args.limit_hours:
                break
            if hours % 500 == 0:
                print(f"  {hours} hours, {total:,} ticks so far", flush=True)
    except TickDataUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT

    print(
        f"fetched {hours} hours, {total:,} ticks ({empty_hours} empty hours) "
        f"cached under {destination}"
    )
    return EXIT_OK


def cmd_candles(args) -> int:
    """Aggregate cached ticks into M15 bars."""
    from research.data.candles import build_candles, candle_coverage

    config = _load_config(args)
    p = paths(config)
    tick_files = sorted(p["ticks"].rglob("*.parquet")) + sorted(p["ticks"].rglob("*.bi5"))
    if not tick_files:
        print(
            f"ERROR: no tick data under {p['ticks']}. Run `fetch` first, or import your "
            "own export with `fetch --csv <path>`. Nothing is generated.",
            file=sys.stderr,
        )
        return EXIT_MISSING_INPUT

    parquet_files = [f for f in tick_files if f.suffix == ".parquet"]
    if parquet_files:
        ticks = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    else:
        from research.data.sources.dukascopy import DukascopyTickSource, decode_bi5

        source = DukascopyTickSource(
            cache_dir=p["ticks"],
            point_divisor=config.backtest.instrument.point_divisor,
        )
        frames = []
        for path in tick_files:
            # .../YYYY/MM/DD/HHh_ticks.bi5
            hour = datetime(
                int(path.parents[2].name),
                int(path.parents[1].name),
                int(path.parents[0].name),
                int(path.name[:2]),
                tzinfo=timezone.utc,
            )
            frames.append(
                decode_bi5(
                    path.read_bytes(), hour, config.backtest.instrument.point_divisor
                )
            )
        ticks = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    ticks = ticks.sort_values("timestamp").reset_index(drop=True)
    candles = build_candles(ticks, timeframe_minutes=config.backtest.timeframe_minutes)
    p["candles"].parent.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(p["candles"], index=False)
    coverage = candle_coverage(candles, config.backtest.timeframe_minutes)
    print(f"built {len(candles):,} M{config.backtest.timeframe_minutes} bars -> {p['candles']}")
    print(json.dumps(coverage, indent=2, default=str))
    return EXIT_OK


def cmd_split(args) -> int:
    """Report the chronological boundary. Reads no out-of-sample prices."""
    config = _load_config(args)
    candles = _load_candles(config)
    summary = _split(config, candles).summary()
    print(json.dumps(summary, indent=2, default=str))
    if not summary["sealed"]:
        print(
            "\nOut-of-sample data is LOCKED: run `develop` then `freeze` before any "
            "command can read past the boundary."
        )
    return EXIT_OK


def cmd_datasets(args) -> int:
    """Audit the point-in-time datasets. Reports absence as absence."""
    from research.data.datasets import candle_dataset_report, inspect_all

    config = _load_config(args)
    p = paths(config)
    start = datetime.combine(config.backtest.start_date, datetime.min.time(), timezone.utc)
    end = datetime.combine(config.backtest.end_date, datetime.min.time(), timezone.utc)

    bundle, _ = inspect_all(p["news"], p["sentiment"], p["calendar"], start, end)
    bundle.candles = candle_dataset_report(
        p["candles"] if p["candles"].exists() else None,
        config.backtest.timeframe_minutes,
    )

    for row in bundle.summary_rows():
        print(
            f"{row['dataset']:<18} {row['availability']:<12} rows={row['rows']:<8} "
            f"{row['coverage']:<26} precision={row['precision']:<10} "
            f"provenance={row['provenance']}"
        )
    for report in bundle.reports():
        for note in report.notes:
            print(f"  - {report.name}: {note}")

    p["results"].mkdir(parents=True, exist_ok=True)
    p["datasets"].write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nwritten -> {p['datasets']}")
    missing = bundle.unavailable_names()
    if missing:
        print(
            f"\nUNAVAILABLE: {', '.join(missing)}. Dependent agents will answer "
            "UNAVAILABLE and the fail-closed policy applies. No data is invented."
        )
    return EXIT_OK


def cmd_develop(args) -> int:
    """Walk-forward parameter search on the development set ONLY."""
    from research.strategy.optimizer import WalkForwardOptimizer
    from research.strategy.donchian_scalp import parameter_grid

    config = _load_config(args)
    candles = _load_candles(config)
    split = _split(config, candles)
    p = paths(config)

    optimizer = WalkForwardOptimizer(
        config=config.backtest, optimizer_config=config.optimizer
    )
    grid = parameter_grid()
    if args.limit_candidates:
        # A smoke-test aid: rehearse the whole pipeline on a slice of the grid
        # before committing to the full search. A seal written from a limited
        # search records the reduced candidate count, so no report can
        # overstate how wide the search actually was.
        grid = grid[: args.limit_candidates]
    print(
        f"searching {len(grid)} candidates over {config.optimizer.folds} folds of "
        f"{split.boundary_index:,} development bars "
        f"({split.development_start:%Y-%m-%d} .. {split.development_end:%Y-%m-%d})"
    )
    report = optimizer.optimize(
        split, candidates=grid, dataset_hash=sha256_file(p["candles"])
    )
    p["results"].mkdir(parents=True, exist_ok=True)
    p["develop"].write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nevaluated {report.candidates_evaluated}, eligible {report.eligible_candidates}")
    for candidate in report.top(5):
        print(
            f"  #{candidate.rank} objective={candidate.objective_score:.4f} "
            f"trades={candidate.total_trades} "
            f"profitable_folds={candidate.profitable_folds}/{len(candidate.folds)} "
            f"eligible={candidate.eligible}"
        )
    print(f"\n{report.selection_reason}")
    print(f"written -> {p['develop']}")
    if report.selected_params is None:
        return EXIT_ERROR
    return EXIT_OK


def cmd_freeze(args) -> int:
    """Write the strategy seal. After this, out-of-sample access opens."""
    from research.data.split import guard_reseal
    from research.strategy.optimizer import OptimizationReport, freeze_strategy

    config = _load_config(args)
    p = paths(config)
    _require(
        p["develop"],
        "optimization report",
        "Run `python -m research.cli develop` first: a seal must record a real "
        "development-only selection.",
    )
    report = OptimizationReport.model_validate_json(p["develop"].read_text(encoding="utf-8"))

    # Refuses to re-freeze after the test set has been touched unless the
    # caller explicitly accepts that it invalidates the claim.
    guard_reseal(p["seal"], reason=args.reason or "manual reseal", allow=args.allow_reseal)

    candles = _load_candles(config)
    split = _split(config, candles)
    seal = freeze_strategy(report, split, p["seal"], config.optimizer)
    print(f"sealed {seal.strategy_name} params_hash={seal.params_hash[:16]}")
    print(f"  development: {seal.development_start:%Y-%m-%d} .. {seal.development_end:%Y-%m-%d}")
    print(
        f"  out-of-sample (now unlocked): {seal.out_of_sample_start:%Y-%m-%d} .. "
        f"{seal.out_of_sample_end:%Y-%m-%d}"
    )
    print(f"  criterion: {seal.selection_criterion}")
    print(f"written -> {p['seal']}")
    return EXIT_OK


def _sealed_signals(config: ExperimentConfig):
    """Out-of-sample bars and signals, at the sealed parameters."""
    from research.data.split import StrategySeal
    from research.strategy.donchian_scalp import DonchianParams, DonchianScalpStrategy

    p = paths(config)
    _require(
        p["seal"],
        "strategy seal",
        "Run `develop` then `freeze`: out-of-sample data stays locked until the "
        "strategy is frozen.",
    )
    seal = StrategySeal.load(p["seal"])
    seal.verify_dataset(sha256_file(p["candles"]))

    candles = _load_candles(config)
    split = _split(config, candles)
    params = DonchianParams.model_validate(seal.params)
    # Raises unless these are exactly the sealed parameters.
    frame = split.out_of_sample(params.to_dict())

    strategy = DonchianScalpStrategy(params, symbol=config.backtest.symbol)
    prepared = strategy.prepare(frame)
    prepared["is_test"] = frame["is_test"].values
    signals = strategy.generate(prepared)
    return seal, prepared, signals


def _with_ai_indicators(
    bars: pd.DataFrame, trend_timeframe_minutes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the indicators the AI technical agent's payload reads.

    `DonchianScalpStrategy.prepare()` only computes what the strategy itself
    needs (the Donchian channel, ATR, and the H4 trend verdict) -- it never
    computed EMA50/200, RSI or MACD, because those were only ever produced by
    `research.strategy.seventy_thirty`, the superseded placeholder nothing in
    the experiment path imports. `research.ai.agents.technical_block` and
    `htf_block` read those columns via `bar.get(...)`, which returns None
    when the column is simply absent -- so every technical (and, via
    `htf_block`, every higher-timeframe) verdict was silently built from an
    empty indicator set for every signal, not just ones with genuinely
    missing data. This computes them here, once, right before the AI run,
    without disturbing the strategy's own `atr`/`trend` columns that the
    signals were actually generated from.
    """
    from research.backtest.indicators import compute_indicator_frame
    from research.data.candle_import import resample_candles

    computed = compute_indicator_frame(bars, ema_fast=12, ema_slow=26)
    new_cols = [c for c in computed.columns if c not in bars.columns]
    enriched = bars.copy()
    for col in new_cols:
        enriched[col] = computed[col]

    entry_minutes = int(
        (bars["timestamp"].iloc[1] - bars["timestamp"].iloc[0]).total_seconds() // 60
    )
    htf = resample_candles(
        bars[["timestamp", "open", "high", "low", "close", "volume"]],
        trend_timeframe_minutes,
        entry_minutes,
    )
    htf = compute_indicator_frame(htf, ema_fast=12, ema_slow=26)
    return enriched, htf


def cmd_baseline(args) -> int:
    """Experiment A on the out-of-sample period."""
    from research.data.datasets import candle_dataset_report
    from research.report.limitations import limitations_for

    config = _load_config(args)
    p = paths(config)
    seal, bars, signals = _sealed_signals(config)
    config.strategy_params = type(config.strategy_params).model_validate(seal.params)

    executor = config.executor()
    result = executor.run_baseline(bars, signals, run_id=args.run_id or "baseline")
    manifest = config.manifest(
        run_id=result.run_id,
        run_kind="baseline",
        # The candle dataset is declared by BOTH arms so the equality check can
        # prove they ran on the same bars.
        datasets=[
            candle_dataset_report(
                p["candles"], config.backtest.timeframe_minutes
            ).to_dataset_version()
        ],
        seal_hash=seal.params_hash,
        limitations=[l.model_dump(mode="json") for l in limitations_for("baseline")],
    )

    p["results"].mkdir(parents=True, exist_ok=True)
    p["baseline"].write_text(result.model_dump_json(indent=2), encoding="utf-8")
    manifest.save(p["baseline_manifest"])

    from research.backtest.metrics import compute_metrics

    metrics = compute_metrics(result)
    print(
        f"Experiment A: {metrics.total_trades} trades from {len(signals)} signals, "
        f"net {metrics.net_profit:,.2f} ({metrics.return_pct:.2f}%), "
        f"win rate {metrics.win_rate:.1f}%, max DD {metrics.max_drawdown_pct:.2f}%"
    )
    print(f"  skipped: {len(result.skipped)} | {executor.summary.as_dict()}")
    print(f"written -> {p['baseline']}, {p['baseline_manifest']}")
    return EXIT_OK


def cmd_pilot(args) -> int:
    """Experiment B on a bounded sample, with a measured cost report.

    Requires an API key unless `--mock` is passed. `--mock` uses the canned
    client: it exercises the whole pipeline and produces a report whose costs
    are SYNTHETIC, which the report states.
    """
    from research.ai.checkpoint import CheckpointStore
    from research.ai.client import (
        AnthropicAgentClient,
        MockAgentClient,
        default_mock_verdicts,
    )
    from research.ai.pit_store import CalendarPitStore, NewsPitStore, SentimentPitStore
    from research.ai.prompts import all_prompt_versions
    from research.ai.runner import AIBacktestRunner, select_pilot_signals
    from research.data.datasets import candle_dataset_report, inspect_all
    from research.report.cost_report import build_cost_report, write_reports
    from research.report.limitations import limitations_for
    from app.services.risk_service import AccountState

    config = _load_config(args)
    p = paths(config)
    ai_settings = config.ai_settings()

    if not args.mock and not ai_settings.anthropic_api_key:
        print(
            "ERROR: no ANTHROPIC_API_KEY is set, so no real pilot can run. Set it in "
            ".env, or run `pilot --mock` to exercise the pipeline with the canned "
            "client (its costs are synthetic and the report says so).",
            file=sys.stderr,
        )
        return EXIT_MISSING_INPUT

    seal, bars, signals = _sealed_signals(config)
    config.strategy_params = type(config.strategy_params).model_validate(seal.params)
    bars, htf_bars = _with_ai_indicators(bars, config.strategy_params.trend_timeframe_minutes)

    sample = select_pilot_signals(signals, args.count, method=args.method)
    print(
        f"pilot: {len(sample)} of {len(signals)} out-of-sample signals "
        f"({args.method}), budget ${ai_settings.ai_cost_limit_usd}"
    )

    start = datetime.combine(config.backtest.start_date, datetime.min.time(), timezone.utc)
    end = datetime.combine(config.backtest.end_date, datetime.min.time(), timezone.utc)
    bundle, frames = inspect_all(p["news"], p["sentiment"], p["calendar"], start, end)
    for row in bundle.summary_rows():
        print(f"  dataset {row['dataset']}: {row['availability']}")

    client = (
        MockAgentClient(verdicts=default_mock_verdicts())
        if args.mock
        else AnthropicAgentClient(ai_settings)
    )
    # Resolved here (not left to runner.run's own default) so the checkpoint
    # file can be scoped to it below -- the checkpoint's resume-by-signal_id
    # logic has no other way to tell one run's decisions from another's, so a
    # shared file would silently replay an earlier run's (possibly
    # differently-prompted) verdicts at zero cost instead of re-deciding.
    run_id = args.run_id or str(uuid.uuid4())[:12]
    checkpoint = CheckpointStore(p["results"] / f"ai_checkpoint_{run_id}.sqlite")
    runner = AIBacktestRunner(
        config=config.backtest,
        ai_settings=ai_settings,
        client=client,
        gate=config.gate(),
        risk_service=config.risk_service(),
        engine=config.engine(),
        checkpoint=checkpoint,
        news_store=NewsPitStore(frames["news"]),
        sentiment_store=SentimentPitStore(frames["sentiment"]),
        calendar_store=CalendarPitStore(frames["economic_calendar"]),
        blackout_minutes=config.policy.high_impact_news_blackout_minutes,
        policy_thresholds=config.policy_thresholds(),
    )

    outcome = asyncio.run(
        runner.run(
            sample,
            bars,
            AccountState(balance=config.backtest.risk.initial_balance, market_open=True),
            htf_bars=htf_bars,
            run_id=run_id,
        )
    )

    executor = config.executor()
    result = executor.run_with_decisions(
        bars, sample, outcome.decisions, run_id=outcome.run_id, run_kind="ai"
    )

    skipped_deterministic = sum(1 for d in outcome.decisions if not d.gate_passed)
    report = build_cost_report(
        run_id=outcome.run_id,
        records=outcome.records,
        total_signals=len(sample),
        processed_signals=len(outcome.decisions) - skipped_deterministic,
        skipped_deterministic=skipped_deterministic,
        models_used={
            agent: ai_settings.model_for(agent)
            for agent in ("technical", "news", "sentiment", "final")
        },
        resumed_signals=outcome.resumed_count,
        failed_closed_signals=sum(1 for d in outcome.decisions if d.failed_closed),
        agent_calls_skipped=sum(len(d.skip_reasons) for d in outcome.decisions),
        decision_counts=outcome.decision_counts(),
        ai_result=result,
        full_oos_signal_count=len(signals),
        budget_usd=ai_settings.ai_cost_limit_usd,
        budget_stopped=outcome.budget_stopped,
        stopped_reason=outcome.stopped_reason,
        prompt_cache_enabled=ai_settings.ai_use_prompt_cache,
        limitations=limitations_for("ai", bundle.unavailable_names()),
    )

    manifest = config.manifest(
        run_id=outcome.run_id,
        run_kind="ai",
        datasets=[
            candle_dataset_report(
                p["candles"], config.backtest.timeframe_minutes
            ).to_dataset_version(),
            *bundle.dataset_versions(),
        ],
        seal_hash=seal.params_hash,
        prompts_hash=json.dumps(all_prompt_versions(), sort_keys=True),
        limitations=[
            l.model_dump(mode="json")
            for l in limitations_for("ai", bundle.unavailable_names())
        ],
    )

    p["results"].mkdir(parents=True, exist_ok=True)
    p["ai"].write_text(result.model_dump_json(indent=2), encoding="utf-8")
    p["decisions"].write_text(
        json.dumps([d.model_dump(mode="json") for d in outcome.decisions], indent=2, default=str),
        encoding="utf-8",
    )
    manifest.save(p["ai_manifest"])
    written = write_reports(report, p["results"])

    print(
        f"\ndecisions: {outcome.decision_counts()} | executed {len(result.trades)} | "
        f"measured cost ${report.total_cost_usd:.4f}"
        + ("  [MOCK: synthetic costs]" if args.mock else "")
    )
    print(f"written -> {p['ai']}, {written['json']}, {written['markdown']}")
    if outcome.budget_stopped:
        print(f"NOTE: stopped on the budget limit: {outcome.stopped_reason}")
    return EXIT_OK


def cmd_compare(args) -> int:
    """A vs B, with the counterfactual analysis. Refuses mismatched configs."""
    from research.backtest.engine import BacktestResult
    from research.report.compare import compare, write_reports
    from research.report.cost_report import CostReport
    from research.report.limitations import limitations_for

    config = _load_config(args)
    p = paths(config)
    _require(p["baseline"], "baseline result", "Run `python -m research.cli baseline`.")
    _require(p["ai"], "AI result", "Run `python -m research.cli pilot`.")

    baseline = BacktestResult.model_validate_json(p["baseline"].read_text(encoding="utf-8"))
    ai = BacktestResult.model_validate_json(p["ai"].read_text(encoding="utf-8"))
    baseline_manifest = (
        RunManifest.model_validate_json(p["baseline_manifest"].read_text(encoding="utf-8"))
        if p["baseline_manifest"].exists()
        else None
    )
    ai_manifest = (
        RunManifest.model_validate_json(p["ai_manifest"].read_text(encoding="utf-8"))
        if p["ai_manifest"].exists()
        else None
    )
    cost_path = p["results"] / "cost_report.json"
    cost = (
        CostReport.model_validate_json(cost_path.read_text(encoding="utf-8"))
        if cost_path.exists()
        else None
    )

    try:
        report = compare(
            config,
            baseline,
            ai,
            baseline_manifest=baseline_manifest,
            ai_manifest=ai_manifest,
            cost=cost,
            limitations=limitations_for("ai"),
            restrict_to_shared=not args.full_universe,
        )
    except ManifestMismatch as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    written = write_reports(report, p["results"])
    print(
        f"A net {report.baseline_net_profit:,.2f} | B net "
        f"{report.ai_metrics.net_profit:,.2f} | AI cost ${report.ai_cost_usd:.4f} | "
        f"B after cost {report.ai_net_profit_after_cost:,.2f}"
    )
    if report.counterfactual:
        print(f"  {report.counterfactual.headline()}")
    print(f"written -> {written['json']}, {written['markdown']}")
    return EXIT_OK


def cmd_status(args) -> int:
    """What exists, what is missing, and what can run next."""
    config = _load_config(args)
    p = paths(config)
    rows = [
        ("tick data", p["ticks"], any(p["ticks"].rglob("*")) if p["ticks"].exists() else False),
        ("M15 candles", p["candles"], p["candles"].exists()),
        ("optimization report", p["develop"], p["develop"].exists()),
        ("strategy seal", p["seal"], p["seal"].exists()),
        ("baseline result", p["baseline"], p["baseline"].exists()),
        ("AI result", p["ai"], p["ai"].exists()),
        ("news dataset", p["news"], p["news"].exists()),
        ("sentiment dataset", p["sentiment"], p["sentiment"].exists()),
        ("calendar dataset", p["calendar"], p["calendar"].exists()),
    ]
    for label, path, present in rows:
        print(f"  [{'x' if present else ' '}] {label:<20} {path}")

    key = "set" if config.ai_settings().anthropic_api_key else "NOT set"
    print(f"  ANTHROPIC_API_KEY: {key}")

    print("\nnext:")
    if not rows[0][2]:
        print("  `fetch` (or `fetch --csv <path>`) -- no tick data yet")
    elif not rows[1][2]:
        print("  `candles`")
    elif not rows[2][2]:
        print("  `develop`")
    elif not rows[3][2]:
        print("  `freeze`")
    elif not rows[4][2]:
        print("  `baseline`")
    elif not rows[5][2]:
        print("  `pilot --mock` to rehearse, then `pilot` once the key is set")
    else:
        print("  `compare`")
    return EXIT_OK


def cmd_config(args) -> int:
    """Write the default experiment config, or validate an existing one."""
    config = _load_config(args)
    target = Path(args.config or "results/experiment.json")
    assert_ab_identical(config, config)
    save_experiment(config, target)
    print(f"experiment config written -> {target}")
    print(f"  fingerprint {config.fingerprint_hash()[:16]}")
    print(f"  both arms will be built from this file by every command")
    return EXIT_OK


def cmd_dashboard(args) -> int:
    """Deferred by instruction until the pilot infrastructure is proven."""
    print(
        "The dashboard is not implemented: it was explicitly deferred until the pilot "
        "infrastructure is proven. `compare` already writes comparison.json and "
        "comparison.md, which contain everything a dashboard would plot.",
        file=sys.stderr,
    )
    return EXIT_MISSING_INPUT


# --- wiring ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="path to the experiment config JSON")
    parser.add_argument("--symbol", help="override the instrument symbol")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Accept --config/--symbol after the subcommand as well as before it.

        `cli --config x.json develop` and `cli develop --config x.json` both
        work: an experiment config that is easy to pass in the wrong position
        is an experiment config people will forget to pass.
        """
        target.add_argument("--config", help="path to the experiment config JSON")
        target.add_argument("--symbol", help="override the instrument symbol")
        return target

    fetch = add_common(sub.add_parser("fetch", help="download or import tick data"))
    fetch.add_argument("--csv", help="import your own tick export instead of downloading")
    fetch.add_argument("--price-column", help="single price column, if your export has no bid/ask")
    fetch.add_argument("--assumed-spread", type=float, help="spread to apply to a price-only export")
    fetch.add_argument("--timezone", default="UTC", help="timezone of the export's timestamps")
    fetch.add_argument("--limit-hours", type=int, help="stop after N hours (for a smoke test)")
    fetch.set_defaults(func=cmd_fetch)

    add_common(sub.add_parser("candles", help="tick data -> M15 bars")).set_defaults(
        func=cmd_candles
    )
    add_common(sub.add_parser("split", help="report the 70/30 boundary")).set_defaults(
        func=cmd_split
    )
    add_common(
        sub.add_parser("datasets", help="audit point-in-time data availability")
    ).set_defaults(func=cmd_datasets)
    develop_parser = add_common(
        sub.add_parser("develop", help="walk-forward search on the first 70%% only")
    )
    develop_parser.add_argument(
        "--limit-candidates",
        type=int,
        help="evaluate only the first N grid candidates (for rehearsing the pipeline; "
        "the seal records the reduced count)",
    )
    develop_parser.set_defaults(func=cmd_develop)

    freeze = add_common(sub.add_parser("freeze", help="write the strategy seal"))
    freeze.add_argument(
        "--allow-reseal",
        action="store_true",
        help="re-freeze after a seal exists; recorded in the seal history and invalidates "
        "the out-of-sample claim",
    )
    freeze.add_argument("--reason", help="why a reseal is being allowed")
    freeze.set_defaults(func=cmd_freeze)

    baseline = add_common(
        sub.add_parser("baseline", help="Experiment A on the out-of-sample period")
    )
    baseline.add_argument("--run-id")
    baseline.set_defaults(func=cmd_baseline)

    pilot = add_common(sub.add_parser("pilot", help="Experiment B on a bounded sample"))
    pilot.add_argument("--count", type=int, default=100, help="signals to decide on")
    pilot.add_argument(
        "--method", default="chronological", choices=["chronological", "evenly_spaced"]
    )
    pilot.add_argument(
        "--mock",
        action="store_true",
        help="use the canned client: no API calls, no spend, synthetic costs",
    )
    pilot.add_argument("--run-id")
    pilot.set_defaults(func=cmd_pilot)

    compare_parser = add_common(
        sub.add_parser("compare", help="A vs B with counterfactual analysis")
    )
    compare_parser.add_argument(
        "--full-universe",
        action="store_true",
        help="do NOT restrict both arms to the shared signal set (only honest when the "
        "AI arm decided on every baseline signal)",
    )
    compare_parser.set_defaults(func=cmd_compare)

    add_common(sub.add_parser("status", help="what exists and what runs next")).set_defaults(
        func=cmd_status
    )
    add_common(
        sub.add_parser("config", help="write or validate the experiment config")
    ).set_defaults(func=cmd_config)
    add_common(sub.add_parser("dashboard", help="(deferred)")).set_defaults(
        func=cmd_dashboard
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LeakageError as exc:
        print(f"REFUSED (leakage guard): {exc}", file=sys.stderr)
        return EXIT_LEAKAGE_REFUSED
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_MISSING_INPUT
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
