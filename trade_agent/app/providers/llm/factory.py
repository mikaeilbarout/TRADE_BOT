from __future__ import annotations

from app.config.settings import Settings
from app.providers.llm.base import LLMProvider, LLMProviderError
from app.providers.llm.mock_provider import MockLLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Single place that decides which concrete LLMProvider to instantiate.
    Everything downstream depends only on the LLMProvider interface."""

    provider = settings.llm_provider.lower()

    if provider == "mock":
        return MockLLMProvider()

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY is not configured")
        from app.providers.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            use_prompt_cache=settings.llm_use_prompt_cache,
            cache_ttl=settings.llm_cache_ttl,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured")
        from app.providers.llm.anthropic_provider import OpenAIProvider

        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.llm_model)

    raise LLMProviderError(f"unknown llm_provider: {settings.llm_provider!r}")
