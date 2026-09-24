from __future__ import annotations

import asyncio
import time

from app.agents.base import AgentError
from app.agents.unified_agent import UnifiedAgent
from app.config.settings import Settings
from app.models.enums import FinalDecision, PipelineStage
from app.models.pipeline_result import (
    AgentTrace,
    DataSources,
    PipelineResult,
    StageLatency,
)
from app.observability.logging import get_logger
from app.services.clock import Clock, system_clock
from app.services.market_data import MarketDataService, MarketDataUnavailableError
from app.services.news_service import NewsService, NewsUnavailableError
from app.services.policy_core import (
    PolicyAction,
    PolicyInputs,
    PolicyThresholds,
    evaluate_policy,
)
from app.services.risk_service import AccountState, RiskService
from app.services.sentiment_service import SentimentService, SentimentUnavailableError

logger = get_logger(__name__)


class SingleAgentPipeline:
    """Same deterministic scaffolding as DecisionPipeline (hard risk
    pre-check, shared market snapshot, news/sentiment fetch, blackout
    window, execution guard) but ONE agent call (UnifiedAgent) in place of
    the news/sentiment/technical/final four-agent chain.

    Reuses `app.services.policy_core.evaluate_policy` -- the exact same
    deterministic safety rules the four-agent pipeline enforces (min
    confidence, blackout window, degraded-data WAIT, stale-market WAIT) --
    with no gate vetoes (there are no specialists to veto) and no
    weighted-score floor (there are no per-dimension scores to weigh).
    """

    def __init__(
        self,
        settings: Settings,
        market_data_service: MarketDataService,
        news_service: NewsService,
        sentiment_service: SentimentService,
        risk_service: RiskService,
        unified_agent: UnifiedAgent,
        clock: Clock = system_clock,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._market_data = market_data_service
        self._news_service = news_service
        self._sentiment_service = sentiment_service
        self._risk = risk_service
        self._agent = unified_agent

    async def run(
        self, signal, account: AccountState | None = None
    ) -> PipelineResult:
        account = account or AccountState()
        now = self._clock()
        start = time.monotonic()
        latencies: list[StageLatency] = []
        errors: list[str] = []
        traces: list[AgentTrace] = []
        sources = DataSources(
            llm_provider=self._agent.provider_name,
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
                    "stage_reached": result.stage_reached.value,
                    "veto_triggered": result.veto_triggered,
                    "degraded": result.degraded,
                    "latency_seconds": round(result.total_latency_seconds, 4),
                    "error_count": len(result.errors),
                },
            )
            return result

        # 1. Hard risk pre-check.
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

        # 2. Market data.
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

        # 3. News + sentiment bundles, concurrently.
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
        sources.news_item_count = len(news_bundle.items)
        sources.news_titles = [item.title for item in news_bundle.items]
        sources.news_degraded = news_bundle.is_degraded
        sources.sentiment_item_count = len(sentiment_bundle.items)
        sources.sentiment_sources = sorted({item.source for item in sentiment_bundle.items})
        sources.sentiment_degraded = sentiment_bundle.is_degraded

        # 4. The one agent call.
        try:
            run = await self._agent.run(
                signal, market, news_bundle, sentiment_bundle, account, self._settings, now=now
            )
        except AgentError as exc:
            errors.append(str(exc))
            traces.append(
                AgentTrace(
                    agent_name=exc.agent_name,
                    input_snapshot={},
                    error=str(exc),
                    latency_seconds=exc.latency_seconds,
                    model=self._settings.llm_model,
                    model_version=self._agent.provider_name,
                )
            )
            return finish(
                FinalDecision.REJECT,
                f"Unified agent failed, failing closed: {exc}",
                PipelineStage.FINAL_DECISION_AGENT,
                market_snapshot=market,
                degraded=True,
            )

        agent_result = run.result
        traces.append(
            AgentTrace(
                agent_name=self._agent.agent_name,
                input_snapshot=_jsonable(run.input_snapshot),
                output=agent_result.model_dump(mode="json"),
                latency_seconds=run.latency_seconds,
                attempts=run.attempts,
                model=self._settings.llm_model,
                model_version=self._agent.provider_name,
                decision=agent_result.decision.value,
            )
        )
        latencies.append(StageLatency(stage=self._agent.agent_name, seconds=run.latency_seconds))

        # 4.5. Deterministic override, user-requested 2026-09-19: the LLM's
        # own judgment defaults heavily toward APPROVE (see unified_agent.md),
        # which is deliberate -- but a full 5-year stability check found one
        # combination whose direction held in every year with enough data
        # (see settings.py): a counter-daily-trend trade during a strongly
        # trending D1 market loses noticeably more often. This rule catches
        # it in code instead of relying on the agent to weigh it
        # consistently every time.
        #
        # The threshold is looked up by THIS signal's own strategy
        # (settings.py explains why it is per-strategy, not one global
        # number): a profile with no entry there -- H1, tested and found
        # not to hold -- gets no override at all, never a threshold
        # borrowed from a different profile's validation.
        adx_threshold = self._settings.single_agent_strong_trend_adx_thresholds.get(
            signal.strategy
        )
        d1_indicators = market.indicators.get("D1")
        d1_adx = d1_indicators.adx_14 if d1_indicators else None
        counter_trend_in_strong_trend = (
            adx_threshold is not None
            and agent_result.decision == FinalDecision.APPROVE
            and agent_result.trade_vs_daily_trend == "AGAINST"
            and d1_adx is not None
            and d1_adx >= adx_threshold
        )
        if counter_trend_in_strong_trend:
            return finish(
                FinalDecision.REJECT,
                (
                    f"Deterministic override: D1 ADX(14) is {d1_adx:.1f} (>= "
                    f"{adx_threshold:.1f} for {signal.strategy}, a strongly "
                    "trending market) and the trade runs against the daily trend. "
                    f"Agent reasoning: {agent_result.summary}"
                ),
                PipelineStage.DECISION_POLICY,
                execution_blocked=True,
                confidence=agent_result.confidence,
                market_snapshot=market,
                unified_result=agent_result,
                ai_decision=agent_result.decision,
                weighted_score=0.0,
                veto_triggered=True,
                veto_reasons=["counter-trend during a strongly trending D1 market (ADX)"],
                degraded=bool(errors),
            )

        # 5. Deterministic policy -- the shared core, no gate vetoes (no
        # specialists to veto) and no weighted-score floor (no per-dimension
        # scores to weigh); confidence floor, blackout window and degraded/
        # stale WAIT still apply exactly as they do for the four-agent path.
        thresholds = PolicyThresholds(
            min_confidence=self._settings.min_confidence,
            veto_on_news_block=False,
            veto_on_sentiment_block=False,
            veto_on_technical_block=False,
            allow_modify=False,
        )
        inputs = PolicyInputs(
            action=PolicyAction(agent_result.decision.value),
            confidence=agent_result.confidence,
            high_impact_within_window=agent_result.high_impact_event_within_minutes,
            degraded=(
                news_bundle.is_degraded
                or sentiment_bundle.is_degraded
                or agent_result.is_degraded
            ),
            market_stale=market.is_stale,
            has_modified_levels=True,
            component_scores=None,
        )
        outcome = evaluate_policy(inputs, thresholds)

        decision = FinalDecision(outcome.action.value)
        reason = (
            agent_result.summary
            if outcome.deciding_rule == "AGENT_DECISION"
            else f"{outcome.reason}. Agent reasoning: {agent_result.summary}"
        )
        execution_blocked = outcome.execution_blocked
        guard_violations: list[str] = []

        # 6. Execution guard: re-validate the exact trade about to be
        # executed. The AI can approve or refuse it, never redraw it.
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
            confidence=agent_result.confidence,
            market_snapshot=market,
            unified_result=agent_result,
            ai_decision=agent_result.decision,
            weighted_score=0.0,
            veto_triggered=outcome.veto_triggered,
            veto_reasons=outcome.veto_reasons,
            guard_violations=guard_violations,
            degraded=bool(errors),
        )


def _jsonable(payload: dict) -> dict:
    """Agent payloads contain datetimes; normalize them for JSON columns."""
    import json

    return json.loads(json.dumps(payload, default=str))
