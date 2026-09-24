from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.news_agent import NewsAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.unified_agent import UnifiedAgent
from app.backtest.historical_providers import (
    HistoricalMarketDataProvider,
    HistoricalNewsProvider,
    HistoricalSentimentProvider,
)
from app.backtest.simulate import simulate_outcome
from app.config.settings import Settings
from app.models.enums import FinalDecision
from app.models.market_data import Candle
from app.models.news import NewsItem
from app.models.sentiment import SentimentItem
from app.models.signal import TradeSignal
from app.providers.llm.base import LLMProvider
from app.services.decision_policy import DecisionPolicy
from app.services.market_data import MarketDataService
from app.services.news_service import NewsService
from app.services.pipeline import DecisionPipeline
from app.services.risk_service import AccountState, RiskService
from app.services.sentiment_service import SentimentService
from app.services.single_agent_pipeline import SingleAgentPipeline


class HistoricalSignalFixture(BaseModel):
    """One row of a backtest dataset: a historical signal plus exactly the
    market/news/sentiment data that would have been available at that
    moment, plus the future candles used only to score the outcome."""

    signal: TradeSignal
    market_history: dict[str, list[Candle]]
    future_candles: list[Candle]
    news_items: list[NewsItem] = Field(default_factory=list)
    sentiment_items: list[SentimentItem] = Field(default_factory=list)


@dataclass
class RecordOutcome:
    signal_id: str
    symbol: str
    side: str
    ai_decision: str
    ai_confidence: float
    raw_r: float
    ai_filtered_r: float
    ai_filtered_taken: bool
    ai_modified_r: float
    ai_modified_taken: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class VariantStats:
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def avg_r(self) -> float:
        return self.total_r / self.trades_taken if self.trades_taken else 0.0

    def to_dict(self) -> dict:
        return {
            "trades_taken": self.trades_taken,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "total_r": round(self.total_r, 2),
            "avg_r": round(self.avg_r, 3),
        }


@dataclass
class BacktestReport:
    records: list[RecordOutcome]
    raw: VariantStats
    ai_filtered: VariantStats
    ai_filtered_and_modified: VariantStats
    false_rejections: int  # raw was a winner, AI-filtered strategy skipped it
    avoided_losses: int  # raw was a loser, AI-filtered strategy skipped it

    def to_dict(self) -> dict:
        decisions: dict[str, int] = {}
        for r in self.records:
            decisions[r.ai_decision] = decisions.get(r.ai_decision, 0) + 1
        return {
            "signal_count": len(self.records),
            "decision_breakdown": decisions,
            "raw_strategy": self.raw.to_dict(),
            "strategy_plus_ai_filter": self.ai_filtered.to_dict(),
            "strategy_plus_ai_filter_and_modify": self.ai_filtered_and_modified.to_dict(),
            "false_rejections": self.false_rejections,
            "avoided_losses": self.avoided_losses,
        }


def load_fixtures(path: str | Path) -> list[HistoricalSignalFixture]:
    data = json.loads(Path(path).read_text())
    return [HistoricalSignalFixture.model_validate(row) for row in data]


def _build_pipeline_for_record(
    settings: Settings, llm: LLMProvider, fixture: HistoricalSignalFixture
) -> DecisionPipeline:
    as_of = fixture.signal.timestamp
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # The replay's clock: every age/freshness/session computation on the
    # decision path reads "now" from here, so the agents see the signal as
    # it was at decision time -- not as a year-old signal with all of its
    # news filtered out as stale, which is exactly what happened before
    # this existed and was only caught by diffing a replayed payload
    # against a live one.
    def clock():
        return as_of

    # Half the project's canonical XAUUSD spread ($0.30/oz) either side of
    # the entry, not +/-0.01: the agents and the risk pre-check both see
    # the spread, and 0.02 is nothing like what the live feed shows.
    half_spread = 0.15
    market_provider = HistoricalMarketDataProvider(
        as_of=as_of,
        candles_by_timeframe=fixture.market_history,
        bid=fixture.signal.entry - half_spread,
        ask=fixture.signal.entry + half_spread,
    )

    def agent(cls):
        return cls(llm, settings.agent_timeout_seconds, settings.llm_model, settings.llm_max_retries)

    return DecisionPipeline(
        settings=settings,
        market_data_service=MarketDataService(market_provider, settings, clock=clock),
        news_service=NewsService(
            HistoricalNewsProvider(as_of, fixture.news_items), settings, clock=clock
        ),
        sentiment_service=SentimentService(
            HistoricalSentimentProvider(as_of, fixture.sentiment_items), settings, clock=clock
        ),
        risk_service=RiskService(settings),
        decision_policy=DecisionPolicy(settings),
        news_agent=agent(NewsAgent),
        sentiment_agent=agent(SentimentAgent),
        technical_agent=agent(TechnicalAgent),
        final_agent=agent(FinalDecisionAgent),
        clock=clock,
    )


def _build_single_agent_pipeline_for_record(
    settings: Settings, llm: LLMProvider, fixture: HistoricalSignalFixture
) -> SingleAgentPipeline:
    """Same point-in-time replay wiring as `_build_pipeline_for_record`, but
    for `SingleAgentPipeline` (one UnifiedAgent call instead of the
    news/sentiment/technical/final chain)."""
    as_of = fixture.signal.timestamp
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    def clock():
        return as_of

    half_spread = 0.15
    market_provider = HistoricalMarketDataProvider(
        as_of=as_of,
        candles_by_timeframe=fixture.market_history,
        bid=fixture.signal.entry - half_spread,
        ask=fixture.signal.entry + half_spread,
    )

    return SingleAgentPipeline(
        settings=settings,
        market_data_service=MarketDataService(market_provider, settings, clock=clock),
        news_service=NewsService(
            HistoricalNewsProvider(as_of, fixture.news_items), settings, clock=clock
        ),
        sentiment_service=SentimentService(
            HistoricalSentimentProvider(as_of, fixture.sentiment_items), settings, clock=clock
        ),
        risk_service=RiskService(settings),
        unified_agent=UnifiedAgent(
            llm, settings.agent_timeout_seconds, settings.llm_model, settings.llm_max_retries
        ),
        clock=clock,
    )


def backtest_settings(settings: Settings) -> Settings:
    """Disable the live-trading-only freshness rules for replay.

    Signal age and market-data staleness are execution-time safety rules
    (section 32); on historical data they are always tripped and would
    reject every signal before any agent ran. Everything else -- risk
    limits, vetoes, thresholds, the execution guard -- stays fully active.
    """
    return settings.model_copy(
        update={
            "max_signal_age_seconds": float(10**9),
            "market_data_max_staleness_seconds": float(10**9),
        }
    )


async def run_backtest(
    fixtures: list[HistoricalSignalFixture],
    settings: Settings,
    llm: LLMProvider,
    account: AccountState | None = None,
) -> BacktestReport:
    account = account or AccountState()
    settings = backtest_settings(settings)
    records: list[RecordOutcome] = []

    for fixture in fixtures:
        signal = fixture.signal
        raw_outcome = simulate_outcome(
            signal.side.value, signal.entry, signal.stop_loss, signal.take_profit,
            fixture.future_candles,
        )

        pipeline = _build_pipeline_for_record(settings, llm, fixture)
        result = await pipeline.run(signal, account)

        ai_filtered_taken = result.decision == FinalDecision.APPROVE
        ai_filtered_r = raw_outcome.r_multiple if ai_filtered_taken else 0.0

        # MODIFY no longer exists (2026-09-19); this variant is kept only so
        # the report shape is stable, and now equals the APPROVE filter.
        ai_modified_taken = ai_filtered_taken
        ai_modified_r = ai_filtered_r

        records.append(
            RecordOutcome(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                side=signal.side.value,
                ai_decision=result.decision.value,
                ai_confidence=result.confidence,
                raw_r=raw_outcome.r_multiple,
                ai_filtered_r=ai_filtered_r,
                ai_filtered_taken=ai_filtered_taken,
                ai_modified_r=ai_modified_r,
                ai_modified_taken=ai_modified_taken,
                errors=result.errors,
            )
        )

    return _aggregate(records)


def _aggregate(records: list[RecordOutcome]) -> BacktestReport:
    raw_stats = VariantStats()
    filtered_stats = VariantStats()
    modified_stats = VariantStats()
    false_rejections = 0
    avoided_losses = 0

    def tally(stats: VariantStats, r: float) -> None:
        stats.trades_taken += 1
        stats.total_r += r
        if r > 0:
            stats.wins += 1
        elif r < 0:
            stats.losses += 1

    for record in records:
        tally(raw_stats, record.raw_r)

        if record.ai_filtered_taken:
            tally(filtered_stats, record.ai_filtered_r)
        elif record.raw_r > 0:
            false_rejections += 1
        elif record.raw_r < 0:
            avoided_losses += 1

        if record.ai_modified_taken:
            tally(modified_stats, record.ai_modified_r)

    return BacktestReport(
        records=records,
        raw=raw_stats,
        ai_filtered=filtered_stats,
        ai_filtered_and_modified=modified_stats,
        false_rejections=false_rejections,
        avoided_losses=avoided_losses,
    )
