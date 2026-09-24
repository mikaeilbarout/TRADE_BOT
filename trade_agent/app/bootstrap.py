from __future__ import annotations

from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.news_agent import NewsAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.unified_agent import UnifiedAgent
from app.api.deps import AppContainer
from app.config.settings import Settings, get_settings
from app.database.session import build_engine, build_session_factory
from app.providers.factory import (
    build_market_data_provider,
    build_news_provider,
    build_sentiment_provider,
)
from app.providers.llm.base import LLMProvider
from app.providers.llm.factory import build_llm_provider
from app.services.cache import build_cache
from app.services.decision_policy import DecisionPolicy
from app.services.execution_service import ExecutionService
from app.services.market_data import MarketDataService
from app.services.news_service import NewsService
from app.services.pipeline import DecisionPipeline
from app.services.risk_service import RiskService
from app.services.sentiment_service import SentimentService
from app.services.single_agent_pipeline import SingleAgentPipeline


def build_pipeline(
    settings: Settings, llm: LLMProvider
) -> tuple[DecisionPipeline | SingleAgentPipeline, RiskService]:
    """Wire the decision pipeline. Separate from container construction so
    tests and the backtest harness build the exact same chain the API
    serves, without needing a database.

    `settings.pipeline_mode` picks the four-agent chain or the single
    UnifiedAgent -- everything upstream of the agent(s) (market data, news,
    sentiment, risk) is identical either way.
    """
    cache = build_cache(settings.redis_url)

    market_data_service = MarketDataService(build_market_data_provider(settings), settings)
    news_service = NewsService(build_news_provider(settings), settings, cache)
    sentiment_service = SentimentService(build_sentiment_provider(settings), settings, cache)
    risk_service = RiskService(settings)

    def agent(cls):
        return cls(
            llm,
            settings.agent_timeout_seconds,
            settings.llm_model,
            settings.llm_max_retries,
        )

    if settings.pipeline_mode == "single_agent":
        pipeline: DecisionPipeline | SingleAgentPipeline = SingleAgentPipeline(
            settings=settings,
            market_data_service=market_data_service,
            news_service=news_service,
            sentiment_service=sentiment_service,
            risk_service=risk_service,
            unified_agent=agent(UnifiedAgent),
        )
        return pipeline, risk_service

    if settings.pipeline_mode != "four_agent":
        raise ValueError(
            f"unknown PIPELINE_MODE {settings.pipeline_mode!r}; use 'four_agent' or 'single_agent'"
        )

    decision_policy = DecisionPolicy(settings)
    pipeline = DecisionPipeline(
        settings=settings,
        market_data_service=market_data_service,
        news_service=news_service,
        sentiment_service=sentiment_service,
        risk_service=risk_service,
        decision_policy=decision_policy,
        news_agent=agent(NewsAgent),
        sentiment_agent=agent(SentimentAgent),
        technical_agent=agent(TechnicalAgent),
        final_agent=agent(FinalDecisionAgent),
    )
    return pipeline, risk_service


def build_container(
    settings: Settings | None = None, llm_override: LLMProvider | None = None
) -> AppContainer:
    """Single place that assembles the whole application from Settings.

    Centralizing this (rather than scattering construction across main.py)
    means tests can build an identical container but swap in a
    MockLLMProvider or a fresh in-memory database without duplicating
    wiring logic.
    """
    settings = settings or get_settings()
    llm = llm_override or build_llm_provider(settings)
    pipeline, risk_service = build_pipeline(settings, llm)

    engine = build_engine(settings)
    return AppContainer(
        settings=settings,
        pipeline=pipeline,
        execution_service=ExecutionService(settings),
        risk_service=risk_service,
        engine=engine,
        session_factory=build_session_factory(engine),
    )
