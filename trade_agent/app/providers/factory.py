from __future__ import annotations

from app.config.settings import Settings
from app.providers.market_data.base import MarketDataProvider
from app.providers.market_data.mock_provider import MockMarketDataProvider
from app.providers.news.base import NewsProvider
from app.providers.news.mock_provider import MockNewsProvider
from app.providers.sentiment.base import SentimentProvider
from app.providers.sentiment.mock_provider import MockSentimentProvider


def build_market_data_provider(settings: Settings) -> MarketDataProvider:
    provider = settings.market_data_provider.lower()
    if provider == "mock":
        return MockMarketDataProvider()
    if provider == "yahoo":
        from app.providers.market_data.yahoo_provider import YahooFinanceMarketDataProvider

        return YahooFinanceMarketDataProvider()
    if provider == "oanda":
        from app.providers.market_data.oanda_provider import OandaMarketDataProvider

        return OandaMarketDataProvider(
            api_key=settings.oanda_api_key or "",
            account_id=settings.oanda_account_id or "",
            environment=settings.oanda_environment,
        )
    raise ValueError(
        f"unknown market_data_provider: {settings.market_data_provider!r}. "
        "Implement MarketDataProvider and register it here to add a real feed."
    )


def build_news_provider(settings: Settings) -> NewsProvider:
    provider = settings.news_provider.lower()
    if provider == "mock":
        return MockNewsProvider()
    if provider == "newsapi":
        if not settings.newsapi_api_key:
            raise ValueError("NEWSAPI_API_KEY is not configured")
        from app.providers.news.newsapi_provider import NewsAPIProvider

        return NewsAPIProvider(api_key=settings.newsapi_api_key)
    if provider == "finnhub":
        if not settings.finnhub_api_key:
            raise ValueError("FINNHUB_API_KEY is not configured")
        from app.providers.news.finnhub_provider import FinnhubNewsProvider

        return FinnhubNewsProvider(api_key=settings.finnhub_api_key)
    if provider == "fred":
        if not settings.fred_api_key:
            raise ValueError("FRED_API_KEY is not configured")
        from app.providers.news.fred_provider import FredNewsProvider

        return FredNewsProvider(api_key=settings.fred_api_key)
    if provider == "finnhub+fred":
        if not settings.finnhub_api_key:
            raise ValueError("FINNHUB_API_KEY is not configured")
        if not settings.fred_api_key:
            raise ValueError("FRED_API_KEY is not configured")
        from app.providers.news.composite_provider import CompositeNewsProvider
        from app.providers.news.finnhub_provider import FinnhubNewsProvider
        from app.providers.news.fred_provider import FredNewsProvider

        return CompositeNewsProvider(
            [
                FinnhubNewsProvider(api_key=settings.finnhub_api_key),
                FredNewsProvider(api_key=settings.fred_api_key),
            ]
        )
    raise ValueError(f"unknown news_provider: {settings.news_provider!r}")


def build_sentiment_provider(settings: Settings) -> SentimentProvider:
    provider = settings.sentiment_provider.lower()
    if provider == "mock":
        return MockSentimentProvider()
    if provider == "newsapi":
        if not settings.newsapi_api_key:
            raise ValueError("NEWSAPI_API_KEY is not configured")
        from app.providers.sentiment.newsapi_sentiment_provider import NewsAPISentimentProvider

        return NewsAPISentimentProvider(api_key=settings.newsapi_api_key)
    if provider == "finnhub":
        if not settings.finnhub_api_key:
            raise ValueError("FINNHUB_API_KEY is not configured")
        from app.providers.sentiment.finnhub_sentiment_provider import FinnhubSentimentProvider

        return FinnhubSentimentProvider(api_key=settings.finnhub_api_key)
    raise ValueError(
        f"unknown sentiment_provider: {settings.sentiment_provider!r}. "
        "Implement SentimentProvider and register it here to add a real feed."
    )
