from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from app.models.enums import Side
from app.models.trade import ModifiedTrade
from app.services.policy_core import (
    ComponentScores,
    GateSignal,
    PolicyAction,
    PolicyInputs,
    PolicyThresholds,
    evaluate_policy,
)
from app.services.risk_service import AccountState, RiskService
from research.ai.agents import (
    build_final_request,
    build_news_request,
    build_sentiment_request,
    build_technical_request,
)
from research.ai.checkpoint import CheckpointStore, SignalProgress
from research.ai.client import AgentCallError, BaseAgentClient
from research.ai.decision import SignalDecision
from research.ai.cost import BudgetExceeded, CallRecord, CostLedger, TokenUsage
from research.ai.gate import DeterministicGate, GateResult
from research.ai.pit_store import CalendarPitStore, NewsPitStore, PitQueryResult, SentimentPitStore
from research.ai.prompts import all_prompt_versions
from research.ai.schemas import (
    FinalAction,
    FinalVerdict,
    Gate,
    NewsVerdict,
    SentimentVerdict,
    TechnicalVerdict,
)
from research.ai.settings import AISettings, FailClosedPolicy
from research.backtest.engine import BacktestEngine
from research.config import BacktestConfig
from research.strategy.base import StrategySignal


@dataclass
class RunOutcome:
    run_id: str
    decisions: list[SignalDecision] = field(default_factory=list)
    records: list[CallRecord] = field(default_factory=list)
    budget_stopped: bool = False
    stopped_reason: str | None = None
    signals_considered: int = 0
    resumed_count: int = 0

    def decision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.action.value] = counts.get(decision.action.value, 0) + 1
        return counts


def select_pilot_signals(
    signals: list[StrategySignal], count: int, method: str = "chronological"
) -> list[StrategySignal]:
    """Pick the pilot sample reproducibly and WITHOUT outcome knowledge.

    Two allowed methods, neither of which can see whether a trade won:

    * `chronological` (default) -- the first N signals of the out-of-sample
      period. Fully deterministic and the closest analogue to going live at
      the start of the test period.
    * `evenly_spaced` -- N signals spread across the whole period, which
      samples more market regimes at the cost of not being a contiguous run.

    Selection deliberately has no access to trade results, so it cannot
    cherry-pick profitable signals.
    """
    # Validate the method BEFORE any early return, so a typo'd method name is
    # never silently accepted just because the sample happens to be small.
    if method not in {"chronological", "evenly_spaced"}:
        raise ValueError(
            f"unknown sampling method {method!r}; use 'chronological' or 'evenly_spaced'"
        )

    ordered = sorted(signals, key=lambda s: s.signal_time)
    if count >= len(ordered):
        return ordered
    if method == "chronological":
        return ordered[:count]

    step = len(ordered) / count
    return [ordered[int(i * step)] for i in range(count)]


class AIBacktestRunner:
    """Runs the AI decision layer over historical signals, cost-first.

    Order of operations per signal, chosen so that the cheapest rejection
    path runs first:

      1. Resume check -- already decided? zero cost.
      2. Deterministic gate -- zero cost.
      3. Data availability -- an agent whose point-in-time store has nothing
         can only answer UNAVAILABLE, so it is skipped rather than paid for.
      4. Technical -> News -> Sentiment on the cheap model, chained.
      5. Final adjudication on the capable model.
      6. Deterministic execution guard over the resulting trade.

    Any failure in 4-6 fails closed to the configured policy.
    """

    def __init__(
        self,
        config: BacktestConfig,
        ai_settings: AISettings,
        client: BaseAgentClient,
        gate: DeterministicGate,
        risk_service: RiskService,
        engine: BacktestEngine,
        checkpoint: CheckpointStore,
        news_store: NewsPitStore,
        sentiment_store: SentimentPitStore,
        calendar_store: CalendarPitStore | None = None,
        blackout_minutes: int = 15,
        policy_thresholds: PolicyThresholds | None = None,
    ) -> None:
        self._config = config
        self._settings = ai_settings
        self._client = client
        self._gate = gate
        self._risk = risk_service
        self._engine = engine
        self._checkpoint = checkpoint
        self._news_store = news_store
        self._sentiment_store = sentiment_store
        self._calendar_store = calendar_store
        self._blackout = blackout_minutes
        # The deterministic policy is shared with the live service. Passing
        # thresholds explicitly is how `research.experiment.ExperimentConfig`
        # guarantees both paths are configured from one source; the fallback
        # derives them from AISettings for standalone use.
        self._thresholds = policy_thresholds or thresholds_from_ai_settings(ai_settings)

        self._ledger = CostLedger(ai_settings.ai_cost_limit_usd)
        # Recover prior spend so a restart cannot silently double the budget.
        prior = checkpoint.total_spend()
        if prior:
            self._ledger.add_external(prior)

    @property
    def ledger(self) -> CostLedger:
        return self._ledger

    async def run(
        self,
        signals: list[StrategySignal],
        bars: pd.DataFrame,
        account: AccountState,
        htf_bars: pd.DataFrame | None = None,
        run_id: str | None = None,
    ) -> RunOutcome:
        run_id = run_id or str(uuid.uuid4())[:12]
        outcome = RunOutcome(run_id=run_id, signals_considered=len(signals))
        already_done = self._checkpoint.completed_signal_ids()
        prompt_versions = all_prompt_versions()

        for signal in sorted(signals, key=lambda s: s.signal_time):
            if signal.signal_id in already_done:
                outcome.resumed_count += 1
                stored = self._checkpoint.get_progress(signal.signal_id)
                if stored and stored.payload:
                    outcome.decisions.append(SignalDecision.model_validate(stored.payload))
                continue

            try:
                decision = await self._process_signal(
                    signal, bars, account, htf_bars, outcome
                )
            except BudgetExceeded as exc:
                outcome.budget_stopped = True
                outcome.stopped_reason = str(exc)
                self._checkpoint.save_progress(
                    SignalProgress(
                        signal_id=signal.signal_id,
                        status="budget_stopped",
                        run_id=run_id,
                    )
                )
                break

            outcome.decisions.append(decision)
            self._checkpoint.save_progress(
                SignalProgress(
                    signal_id=signal.signal_id,
                    status="skipped_deterministic" if not decision.gate_passed else "done",
                    decision=decision.action.value,
                    confidence=decision.confidence,
                    payload=decision.model_dump(mode="json"),
                    cost_usd=decision.cost_usd,
                    usage=decision.usage,
                    prompt_versions=prompt_versions,
                    models={
                        agent: self._settings.model_for(agent)
                        for agent in ("technical", "news", "sentiment", "final")
                    },
                    run_id=run_id,
                )
            )

        outcome.records = self._ledger.records
        return outcome

    # --- per-signal pipeline ---------------------------------------------
    async def _process_signal(
        self,
        signal: StrategySignal,
        bars: pd.DataFrame,
        account: AccountState,
        htf_bars: pd.DataFrame | None,
        outcome: RunOutcome,
    ) -> SignalDecision:
        # 1-2. Deterministic gate: free rejection.
        gate_result = self._gate.check(signal, bars, account)
        if not gate_result.passed:
            return SignalDecision(
                signal_id=signal.signal_id,
                signal_time=signal.signal_time,
                action=FinalAction.REJECT,
                confidence=1.0,
                reason=f"Deterministic pre-AI gate rejected: {gate_result.reason}",
                reason_codes=["DETERMINISTIC_REJECT"],
                gate_passed=False,
                gate_violations=gate_result.violations,
                deciding_rule="DETERMINISTIC_GATE",
                blocking_agent="deterministic_gate",
            )

        market = gate_result.market_snapshot
        volume = gate_result.sized_volume
        decision = SignalDecision(
            signal_id=signal.signal_id,
            signal_time=signal.signal_time,
            action=FinalAction.REJECT,
            confidence=0.0,
            reason="not evaluated",
        )
        usage_total = TokenUsage()
        skip_reasons: dict[str, str | None] = {}

        # 3. Technical agent (always has data: the bars themselves).
        technical = await self._call_agent(
            build_technical_request(
                signal, market, bars, htf_bars, self._settings, volume
            ),
            decision,
            outcome,
        )
        if isinstance(technical, str):
            return self._fail_closed(decision, f"technical agent unavailable: {technical}")
        usage_total = usage_total + _last_usage(outcome)

        # 4. News agent -- skipped when the point-in-time store has nothing.
        news_query = self._news_store.query(
            signal.signal_time, self._settings.ai_news_max_items * 60, self._settings.ai_news_max_items
        )
        if self._calendar_store is not None:
            calendar = self._calendar_store.query_events(signal.signal_time, self._blackout * 4)
            if calendar.available:
                news_query = news_query.model_copy(
                    update={
                        "available": True,
                        "events": calendar.events,
                        "reason": news_query.reason,
                    }
                )

        news: NewsVerdict | None = None
        if self._should_skip(news_query):
            skip_reasons["news"] = news_query.reason or "no point-in-time news data"
        else:
            result = await self._call_agent(
                build_news_request(
                    signal, market, news_query, self._blackout, self._settings
                ),
                decision,
                outcome,
            )
            if isinstance(result, str):
                return self._fail_closed(decision, f"news agent unavailable: {result}")
            news = result
            usage_total = usage_total + _last_usage(outcome)

        # 5. Sentiment agent -- same availability rule.
        sentiment_query = self._sentiment_store.query(
            signal.signal_time,
            self._settings.ai_sentiment_max_items * 60,
            self._settings.ai_sentiment_max_items,
        )
        sentiment: SentimentVerdict | None = None
        if self._should_skip(sentiment_query):
            skip_reasons["sentiment"] = sentiment_query.reason or "no point-in-time sentiment data"
        else:
            result = await self._call_agent(
                build_sentiment_request(
                    signal, market, sentiment_query, news, skip_reasons.get("news"), self._settings
                ),
                decision,
                outcome,
            )
            if isinstance(result, str):
                return self._fail_closed(decision, f"sentiment agent unavailable: {result}")
            sentiment = result
            usage_total = usage_total + _last_usage(outcome)

        # 6. Final adjudication.
        risk_context = {
            "balance": account.balance,
            "sized_volume": round(volume, 2),
            "open_positions": account.open_positions,
            "daily_loss_pct": account.daily_loss_pct,
            "min_rr": self._config.risk.risk_per_trade_pct,
        }
        final = await self._call_agent(
            build_final_request(
                signal, market, technical, news, sentiment, skip_reasons,
                self._settings, risk_context, volume,
            ),
            decision,
            outcome,
        )
        if isinstance(final, str):
            return self._fail_closed(decision, f"final agent unavailable: {final}")
        usage_total = usage_total + _last_usage(outcome)

        decision.technical = technical.model_dump(mode="json") if technical else None
        decision.news = news.model_dump(mode="json") if news else None
        decision.sentiment = sentiment.model_dump(mode="json") if sentiment else None
        decision.final = final.model_dump(mode="json")
        decision.skip_reasons = skip_reasons
        decision.usage = usage_total
        decision.cost_usd = sum(
            r.cost_usd for r in outcome.records if r.signal_id == signal.signal_id
        ) or sum(r.cost_usd for r in self._ledger.records if r.signal_id == signal.signal_id)
        decision.reason_codes = list(final.reason_codes)

        return self._resolve(decision, signal, final, technical, news, sentiment, gate_result, account)

    def _should_skip(self, query: PitQueryResult) -> bool:
        """Skip an agent whose data source genuinely has nothing.

        Paying a model to tell us 'UNAVAILABLE' about data we already know is
        absent is pure waste; the skip is recorded so the final agent still
        sees the absence explicitly.
        """
        if not self._settings.ai_skip_agent_when_data_unavailable:
            return False
        return not query.available or query.is_empty

    async def _call_agent(self, request, decision: SignalDecision, outcome: RunOutcome):
        """Call one agent, enforcing the budget first. Returns the verdict, or
        an error string for the caller to fail closed on."""
        # Reserve against the budget using the most recent observed cost for
        # this agent, so the run stops BEFORE exceeding the limit.
        projected = _projected_cost(self._ledger, request.agent)
        self._ledger.check_budget(projected)

        try:
            verdict, record = await self._client.call(request)
        except AgentCallError as exc:
            failure = CallRecord(
                signal_id=request.signal_id,
                agent=request.agent,
                model=request.model,
                usage=TokenUsage(),
                cost_usd=0.0,
                latency_seconds=0.0,
                prompt_version=request.prompt_version,
                error=str(exc),
            )
            self._ledger.record(failure)
            self._checkpoint.log_call(failure)
            outcome.records.append(failure)
            return str(exc)

        self._ledger.record(record)
        self._checkpoint.log_call(record)
        outcome.records.append(record)
        decision.agents_called.append(request.agent)
        return verdict

    def _fail_closed(self, decision: SignalDecision, reason: str) -> SignalDecision:
        action = (
            FinalAction.WAIT
            if self._settings.ai_fail_closed_policy == FailClosedPolicy.WAIT
            else FinalAction.REJECT
        )
        decision.action = action
        decision.confidence = 0.0
        decision.reason = f"Failed closed ({self._settings.ai_fail_closed_policy.value}): {reason}"
        decision.reason_codes = ["FAIL_CLOSED"]
        decision.failed_closed = True
        decision.deciding_rule = "FAIL_CLOSED"
        decision.blocking_agent = "fail_closed"
        return decision

    def _resolve(
        self,
        decision: SignalDecision,
        signal: StrategySignal,
        final: FinalVerdict,
        technical: TechnicalVerdict | None,
        news: NewsVerdict | None,
        sentiment: SentimentVerdict | None,
        gate_result: GateResult,
        account: AccountState,
    ) -> SignalDecision:
        """Apply the shared deterministic policy and the execution guard to
        the final agent's answer. The AI never has the last word.

        Every veto and threshold rule lives in `app.services.policy_core` and
        is the same code the live service runs. What stays here is the part
        that genuinely cannot be shared: turning a MODIFY into a concrete
        trade and re-running the risk engine against it.
        """
        # Validate the modification BEFORE calling the policy, so the policy's
        # "MODIFY without usable levels" rule sees the real answer rather than
        # trusting that three floats are present.
        modified = (
            self._validate_modification(signal, final)
            if final.action == FinalAction.MODIFY
            else None
        )

        scores = final.component_scores()
        inputs = PolicyInputs(
            action=PolicyAction(final.action.value),
            confidence=final.confidence,
            news_gate=_gate_signal(news),
            sentiment_gate=_gate_signal(sentiment),
            technical_gate=_gate_signal(technical),
            high_impact_within_window=bool(
                news is not None and news.high_impact_within_window
            ),
            # An agent skipped for missing point-in-time data reports
            # UNAVAILABLE through the gate mapping above and carries no WAIT
            # of its own; "degraded" is the separate case of an agent that
            # DID answer but flagged its own inputs as thin/stale via
            # `AgentVerdict.is_degraded`, mirroring
            # app/services/decision_policy.py's identical rule.
            degraded=bool(
                (technical is not None and technical.is_degraded)
                or (news is not None and news.is_degraded)
                or (sentiment is not None and sentiment.is_degraded)
            ),
            # Offline exception: wall-clock staleness is meaningless on
            # historical bars, where every quote is years old by definition.
            # The bar timestamps themselves enforce causality instead -- see
            # research.ai.gate.replay_risk_settings for the same reasoning
            # applied to the risk engine. This is the ONLY policy input the
            # research path populates differently from the live path.
            market_stale=False,
            has_modified_levels=(
                final.action != FinalAction.MODIFY or modified is not None
            ),
            component_scores=(
                ComponentScores(
                    news_score=scores[0],
                    sentiment_score=scores[1],
                    technical_score=scores[2],
                    risk_score=scores[3],
                )
                if scores is not None
                else None
            ),
        )

        result = evaluate_policy(inputs, self._thresholds)
        decision.confidence = final.confidence
        decision.weighted_score = result.weighted_score
        decision.deciding_rule = result.deciding_rule

        if result.action not in (PolicyAction.APPROVE, PolicyAction.MODIFY):
            decision.action = FinalAction(result.action.value)
            decision.reason = result.reason
            decision.blocking_agent = _attribute(result, inputs)
            return decision

        # --- execution guard: the same risk engine, run again -------------
        if result.action == PolicyAction.MODIFY:
            assert modified is not None  # the policy rejects the None case above
            guard = self._risk.final_guard(
                modified,
                account,
                market=gate_result.market_snapshot,
                original_signal=gate_result.trade_signal,
            )
            if not guard.passed:
                decision.action = FinalAction.REJECT
                decision.guard_violations = guard.violations
                decision.deciding_rule = "EXECUTION_GUARD"
                decision.blocking_agent = "risk_engine"
                decision.reason = "Execution guard rejected modification: " + "; ".join(
                    guard.violations
                )
                return decision

            decision.action = FinalAction.MODIFY
            decision.was_modified = True
            decision.modified_entry = modified.entry
            decision.modified_sl = modified.stop_loss
            decision.modified_tp = modified.take_profit
            decision.reason = final.note or "; ".join(final.reason_codes)
            return decision

        guard = self._risk.final_guard(
            gate_result.trade_signal,
            account,
            market=gate_result.market_snapshot,
            original_signal=gate_result.trade_signal,
        )
        if not guard.passed:
            decision.action = FinalAction.REJECT
            decision.guard_violations = guard.violations
            decision.deciding_rule = "EXECUTION_GUARD"
            decision.blocking_agent = "risk_engine"
            decision.reason = "Execution guard rejected approval: " + "; ".join(
                guard.violations
            )
            return decision

        decision.action = FinalAction.APPROVE
        decision.reason = final.note or "; ".join(final.reason_codes)
        return decision

    def _validate_modification(
        self, signal: StrategySignal, final: FinalVerdict
    ) -> ModifiedTrade | None:
        if final.entry is None or final.stop_loss is None or final.take_profit is None:
            return None
        try:
            return ModifiedTrade(
                symbol=signal.symbol,
                side=signal.side,
                entry=final.entry,
                stop_loss=final.stop_loss,
                take_profit=final.take_profit,
                volume=None,
                order_type="LIMIT" if final.entry != signal.entry else "MARKET",
            )
        except ValueError:
            # Incoherent levels (e.g. BUY with stop above entry) never reach
            # execution -- the schema refuses them.
            return None


def thresholds_from_ai_settings(settings: AISettings) -> PolicyThresholds:
    """The research path's view of the shared policy numbers.

    `app.services.decision_policy.thresholds_from_settings` builds the live
    path's view from the live Settings, and
    `tests/research/test_policy_parity.py` asserts the two agree when both
    are derived from one `ExperimentConfig`.
    """
    return PolicyThresholds(
        min_confidence=settings.ai_min_final_confidence,
        min_weighted_score=settings.ai_min_weighted_score,
        veto_on_news_block=settings.ai_veto_on_news_block,
        veto_on_sentiment_block=settings.ai_veto_on_sentiment_block,
        veto_on_technical_block=settings.ai_veto_on_technical_block,
        allow_modify=settings.ai_allow_modify,
        weight_news=settings.ai_weight_news,
        weight_sentiment=settings.ai_weight_sentiment,
        weight_technical=settings.ai_weight_technical,
        weight_risk=settings.ai_weight_risk,
    )


def _gate_signal(verdict) -> GateSignal:
    """Map an agent verdict -- or its absence -- onto a policy gate.

    A skipped agent becomes UNAVAILABLE rather than PASS: "we have no data"
    must never read as "this dimension is fine".
    """
    if verdict is None:
        return GateSignal.UNAVAILABLE
    return GateSignal(verdict.decision.value)


def _attribute(result, inputs: PolicyInputs) -> str | None:
    """Name who is answerable for a non-approval.

    Used by the counterfactual report to answer "which agent caused the most
    false rejections", which needs an attribution recorded at decision time
    rather than inferred later from prose.
    """
    if result.deciding_rule == "BLOCK_VETO":
        if inputs.technical_gate == GateSignal.BLOCK:
            return "technical"
        if inputs.news_gate == GateSignal.BLOCK:
            return "news"
        if inputs.sentiment_gate == GateSignal.BLOCK:
            return "sentiment"
        return "chain"
    if result.deciding_rule == "TIME_SENSITIVE_VETO":
        if inputs.high_impact_within_window:
            return "news"
        if inputs.degraded:
            return "chain"
        return "market_data"
    return "final"


def _projected_cost(ledger: CostLedger, agent: str) -> float:
    """Estimate the next call's cost from this agent's observed history.

    Using a measured value rather than a guess means the budget reservation
    tightens as the run proceeds instead of relying on a prior.
    """
    costs = [r.cost_usd for r in ledger.records if r.agent == agent and r.error is None]
    if not costs:
        return 0.0
    return max(costs)


def _last_usage(outcome: RunOutcome) -> TokenUsage:
    return outcome.records[-1].usage if outcome.records else TokenUsage()


def side_of(signal: StrategySignal) -> Side:
    return signal.side
