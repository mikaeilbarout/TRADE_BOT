from __future__ import annotations

import asyncio
import time

from app.agents.base import AgentError, AgentRun
from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.news_agent import NewsAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.config.assets import get_asset_meta
from app.config.settings import Settings
from app.models.enums import FinalDecision, GateDecision, PipelineStage
from app.models.news import NewsBundle
from app.models.pipeline_result import (
    AgentTrace,
    DataSources,
    PipelineResult,
    StageLatency,
)
from app.models.sentiment import SentimentBundle
from app.models.signal import TradeSignal
from app.observability.logging import get_logger
from app.services.clock import Clock, system_clock
from app.services.decision_policy import DecisionPolicy
from app.services.market_data import MarketDataService, MarketDataUnavailableError
from app.services.news_service import NewsService, NewsUnavailableError
from app.services.risk_service import AccountState, RiskService
from app.services.sentiment_service import SentimentService, SentimentUnavailableError

logger = get_logger(__name__)


class DecisionPipeline:
    """Runs one signal through the agent chain and the deterministic guards.

    Parallelized 2026-09-18 (was a strict sequential chain where sentiment
    saw news's findings and technical saw both): the news/sentiment/
    technical specialists now run concurrently, each independent, and only
    the final agent synthesizes all three. Roughly halves the agent-chain
    latency (~80s -> ~40s observed), which matters because the ORIGINAL
    signal's entry goes stale while it waits -- live data showed this
    staleness was the single most common reason the final agent had to
    MODIFY a trade rather than approve it as-is. The trade-off: sentiment
    and technical no longer explicitly critique the prior agent's
    conclusion (agreement_with_previous is NOT_APPLICABLE for all three
    now, like news_agent already was) -- that cross-referencing job now
    rests entirely on final_agent's own chain_conflicts resolution, which
    already did most of the real synthesis work anyway.

        hard risk pre-check
          -> market data snapshot
          -> Agents 1-3 news / sentiment / technical, IN PARALLEL (signal + market only)
          -> Agent 4 final       (all three findings + risk context)
          -> deterministic decision policy (vetoes, thresholds)
          -> deterministic execution guard (hard risk rules)

    Every stage fails closed: a missing, invalid, or timed-out dependency
    yields REJECT or WAIT, never a silent APPROVE.
    """

    def __init__(
        self,
        settings: Settings,
        market_data_service: MarketDataService,
        news_service: NewsService,
        sentiment_service: SentimentService,
        risk_service: RiskService,
        decision_policy: DecisionPolicy,
        news_agent: NewsAgent,
        sentiment_agent: SentimentAgent,
        technical_agent: TechnicalAgent,
        final_agent: FinalDecisionAgent,
        clock: Clock = system_clock,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._market_data = market_data_service
        self._news_service = news_service
        self._sentiment_service = sentiment_service
        self._risk = risk_service
        self._policy = decision_policy
        self._news_agent = news_agent
        self._sentiment_agent = sentiment_agent
        self._technical_agent = technical_agent
        self._final_agent = final_agent

    async def run(
        self, signal: TradeSignal, account: AccountState | None = None
    ) -> PipelineResult:
        account = account or AccountState()
        # One "now" for the whole decision, from the injected clock: live it
        # is the wall clock; in a historical replay it is the signal's own
        # decision time, so ages, freshness and session are computed as of
        # then rather than as of today (see app/services/clock.py).
        now = self._clock()
        start = time.monotonic()
        latencies: list[StageLatency] = []
        errors: list[str] = []
        traces: list[AgentTrace] = []
        sources = DataSources(
            llm_provider=self._news_agent.provider_name,
            llm_model=self._settings.llm_model,
            market_data_provider=self._settings.market_data_provider,
            news_provider=self._settings.news_provider,
            sentiment_provider=self._settings.sentiment_provider,
        )

        def stage(name: str, started: float) -> None:
            latencies.append(StageLatency(stage=name, seconds=time.monotonic() - started))

        def finish(
            decision: FinalDecision,
            reason: str,
            stage_reached: PipelineStage,
            degraded: bool = False,
            **extra,
        ) -> PipelineResult:
            # Every early return here is a fail-closed path (REJECT, or a
            # WAIT for an imminent critical event), so execution is blocked
            # unless a caller explicitly says otherwise -- defaulting the
            # other way would reintroduce exactly the silent-passthrough
            # this flag exists to prevent.
            extra.setdefault("execution_blocked", decision != FinalDecision.APPROVE)
            result = PipelineResult(
                signal_id=signal.signal_id,
                signal=signal,
                decision=decision,
                confidence=extra.pop("confidence", 0.0),
                reason=reason,
                stage_reached=stage_reached,
                degraded=degraded,
                errors=errors,
                latencies=latencies,
                agent_traces=traces,
                data_sources=sources,
                total_latency_seconds=time.monotonic() - start,
                **extra,
            )
            logger.info(
                "decision complete",
                extra={
                    "signal_id": signal.signal_id,
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "decision": result.decision.value,
                    "ai_decision": result.ai_decision.value if result.ai_decision else None,
                    "confidence": result.confidence,
                    "weighted_score": round(result.weighted_score, 2),
                    "stage_reached": result.stage_reached.value,
                    "veto_triggered": result.veto_triggered,
                    "degraded": result.degraded,
                    "latency_seconds": round(result.total_latency_seconds, 4),
                    "error_count": len(result.errors),
                },
            )
            return result

        def record_agent_failure(exc: AgentError) -> None:
            errors.append(str(exc))
            traces.append(
                AgentTrace(
                    agent_name=exc.agent_name,
                    input_snapshot={},
                    error=str(exc),
                    latency_seconds=exc.latency_seconds,
                    model=self._settings.llm_model,
                    model_version=self._news_agent.provider_name,
                )
            )

        def record_agent_run(agent_name: str, run: AgentRun) -> None:
            decision_value = getattr(run.result, "decision", None)
            traces.append(
                AgentTrace(
                    agent_name=agent_name,
                    input_snapshot=_jsonable(run.input_snapshot),
                    output=run.result.model_dump(mode="json"),
                    latency_seconds=run.latency_seconds,
                    attempts=run.attempts,
                    model=self._settings.llm_model,
                    model_version=self._news_agent.provider_name,
                    decision=decision_value.value if decision_value else None,
                )
            )
            latencies.append(StageLatency(stage=agent_name, seconds=run.latency_seconds))

        logger.info(
            "signal received",
            extra={
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "side": signal.side.value,
                "strategy": signal.strategy,
            },
        )

        # 1. Hard risk pre-check (cheap, deterministic, no LLM cost yet).
        stage_start = time.monotonic()
        precheck = self._risk.pre_check(signal, account, market=None, now=now)
        stage(PipelineStage.HARD_RISK_PRECHECK.value, stage_start)
        if not precheck.passed:
            return finish(
                FinalDecision.REJECT,
                "Hard risk pre-check failed: " + "; ".join(precheck.violations),
                PipelineStage.HARD_RISK_PRECHECK,
                guard_violations=precheck.violations,
            )

        # 2. Market data: one shared snapshot every agent in the chain reads.
        stage_start = time.monotonic()
        try:
            market = await self._market_data.get_snapshot(
                signal.symbol, self._settings.all_timeframes
            )
        except MarketDataUnavailableError as exc:
            errors.append(f"market_data: {exc}")
            return finish(
                FinalDecision.REJECT,
                f"Market data unavailable, failing closed: {exc}",
                PipelineStage.MARKET_DATA,
                degraded=True,
            )
        stage(PipelineStage.MARKET_DATA.value, stage_start)
        sources.market_data_stale = market.is_stale

        # Re-run the pre-check now that live spread/volatility/staleness exist.
        market_precheck = self._risk.pre_check(signal, account, market=market, now=now)
        if not market_precheck.passed:
            return finish(
                FinalDecision.REJECT,
                "Hard risk pre-check failed against live market data: "
                + "; ".join(market_precheck.violations),
                PipelineStage.HARD_RISK_PRECHECK,
                market_snapshot=market,
                guard_violations=market_precheck.violations,
            )

        asset_meta = get_asset_meta(signal.symbol)

        # 3. Fetch both data bundles concurrently -- independent of each
        # other and of the agents below.
        # gather(return_exceptions=True) rather than awaiting one then the
        # other: awaiting sequentially meant that if the first raised, the
        # second task was cancelled but never awaited (and on any exception
        # type the first handler did not expect, simply abandoned) -- which
        # surfaces later as "Task exception was never retrieved".
        news_bundle, sentiment_bundle = await asyncio.gather(
            self._news_service.get_news(signal.symbol),
            self._sentiment_service.get_sentiment(signal.symbol),
            return_exceptions=True,
        )
        for label, stage_enum, bundle in (
            ("News", PipelineStage.NEWS_AGENT, news_bundle),
            ("Sentiment", PipelineStage.SENTIMENT_AGENT, sentiment_bundle),
        ):
            if isinstance(bundle, (NewsUnavailableError, SentimentUnavailableError)):
                errors.append(f"{label.lower()}_service: {bundle}")
                return finish(
                    FinalDecision.REJECT,
                    f"{label} data unavailable, failing closed: {bundle}",
                    stage_enum,
                    market_snapshot=market,
                    degraded=True,
                )
            if isinstance(bundle, BaseException):
                raise bundle
        _record_news_sources(sources, news_bundle)
        _record_sentiment_sources(sources, sentiment_bundle)

        # 4. Agents 1-3 -- news, sentiment, technical, IN PARALLEL. Each is
        # independent (signal + market + its own data only); return_
        # exceptions=True so one AgentError doesn't cancel the other two
        # calls mid-flight (they're already paid for/in flight either way).
        news_run, sentiment_run, technical_run = await asyncio.gather(
            self._news_agent.run(signal, market, news_bundle, asset_meta, now=now),
            self._sentiment_agent.run(signal, market, sentiment_bundle, now=now),
            self._technical_agent.run(signal, market, now=now),
            return_exceptions=True,
        )

        for stage_enum, label, run in (
            (PipelineStage.NEWS_AGENT, "News agent", news_run),
            (PipelineStage.SENTIMENT_AGENT, "Sentiment agent", sentiment_run),
            (PipelineStage.TECHNICAL_AGENT, "Technical agent", technical_run),
        ):
            if isinstance(run, AgentError):
                record_agent_failure(run)
                return finish(
                    FinalDecision.REJECT,
                    f"{label} failed, failing closed: {run}",
                    stage_enum,
                    market_snapshot=market,
                    degraded=True,
                )
            if isinstance(run, BaseException):
                raise run

        news_result = news_run.result
        record_agent_run(PipelineStage.NEWS_AGENT.value, news_run)
        sentiment_result = sentiment_run.result
        record_agent_run(PipelineStage.SENTIMENT_AGENT.value, sentiment_run)
        technical_result = technical_run.result
        record_agent_run(PipelineStage.TECHNICAL_AGENT.value, technical_run)

        # A critical, imminent event still invalidates the trade regardless
        # of what the other two dimensions found -- checked after the
        # parallel gather (can no longer skip paying for sentiment/
        # technical the way the old sequential short-circuit did, since
        # all three are already in flight together by the time news
        # resolves).
        if (
            self._settings.short_circuit_on_critical_news
            and news_result.decision == GateDecision.BLOCK
            and news_result.high_impact_event_within_minutes
        ):
            return finish(
                FinalDecision.WAIT,
                "News agent reports a critical high-impact event inside the "
                f"danger window. {news_result.summary}",
                PipelineStage.NEWS_AGENT,
                confidence=news_result.confidence,
                market_snapshot=market,
                news_result=news_result,
                sentiment_result=sentiment_result,
                technical_result=technical_result,
                short_circuited=True,
                veto_triggered=True,
                veto_reasons=["critical high-impact news event imminent"],
            )

        # 6. Agent 4 -- final decision over the complete chain.
        try:
            final_run = await self._final_agent.run(
                signal,
                market,
                news_result,
                sentiment_result,
                technical_result,
                account,
                self._settings,
                now=now,
            )
        except AgentError as exc:
            record_agent_failure(exc)
            return finish(
                FinalDecision.REJECT,
                f"Final decision agent failed, failing closed: {exc}",
                PipelineStage.FINAL_DECISION_AGENT,
                market_snapshot=market,
                news_result=news_result,
                sentiment_result=sentiment_result,
                technical_result=technical_result,
                degraded=True,
            )
        final_result = final_run.result
        record_agent_run(PipelineStage.FINAL_DECISION_AGENT.value, final_run)

        # 7. Deterministic policy: vetoes and thresholds the LLM cannot skip.
        stage_start = time.monotonic()
        outcome = self._policy.apply(
            signal,
            final_result,
            news_result,
            sentiment_result,
            technical_result,
            market,
            # The deterministic fact, not the LLM's self-report: a bundle
            # served from a stale cache because the provider was down is
            # degraded whether or not the model chose to say so.
            data_degraded=news_bundle.is_degraded or sentiment_bundle.is_degraded,
        )
        stage(PipelineStage.DECISION_POLICY.value, stage_start)

        decision = outcome.decision
        reason = outcome.reason
        execution_blocked = outcome.execution_blocked
        guard_violations: list[str] = []

        # 8. Execution guard: re-validate the exact trade about to be executed.
        # That is always the bot's own signal now -- the AI can approve or
        # refuse it, never redraw it (MODIFY was removed 2026-09-19).
        if decision == FinalDecision.APPROVE:
            stage_start = time.monotonic()
            guard = self._risk.final_guard(
                signal, account, market=market, original_signal=signal
            )
            stage(PipelineStage.EXECUTION_GUARD.value, stage_start)
            if not guard.passed:
                guard_violations = guard.violations
                decision = FinalDecision.REJECT
                execution_blocked = True
                reason = "Execution guard overrode AI decision: " + "; ".join(guard.violations)

        return finish(
            decision,
            reason,
            PipelineStage.COMPLETE,
            execution_blocked=execution_blocked,
            confidence=final_result.confidence,
            market_snapshot=market,
            news_result=news_result,
            sentiment_result=sentiment_result,
            technical_result=technical_result,
            final_result=final_result,
            ai_decision=final_result.decision,
            weighted_score=outcome.weighted_score,
            veto_triggered=outcome.veto_triggered,
            veto_reasons=outcome.veto_reasons,
            policy_warnings=outcome.warnings,
            guard_violations=guard_violations,
            degraded=bool(errors),
        )


def _record_news_sources(sources: DataSources, bundle: NewsBundle) -> None:
    sources.news_item_count = len(bundle.items)
    sources.news_titles = [item.title for item in bundle.items]
    sources.news_degraded = bundle.is_degraded


def _record_sentiment_sources(sources: DataSources, bundle: SentimentBundle) -> None:
    sources.sentiment_item_count = len(bundle.items)
    sources.sentiment_sources = sorted({item.source for item in bundle.items})
    sources.sentiment_degraded = bundle.is_degraded


def _jsonable(payload: dict) -> dict:
    """Agent payloads contain datetimes; normalize them for JSON columns."""
    import json

    return json.loads(json.dumps(payload, default=str))
