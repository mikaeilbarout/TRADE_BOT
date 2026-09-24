from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel

from app.providers.llm.base import LLMProvider, LLMProviderError

T = TypeVar("T", bound=BaseModel)

ResponseFactory = Callable[[str, str, type[BaseModel]], BaseModel | Awaitable[BaseModel]]


class MockLLMProvider(LLMProvider):
    """Deterministic, network-free provider for tests and local dev without
    API keys. Pass a `responses` map keyed by response_model.__name__ to
    canned instances, or a `factory` callable for dynamic behavior (e.g. to
    simulate timeouts/malformed output in failure tests).
    """

    name = "mock"

    def __init__(
        self,
        responses: dict[str, BaseModel] | None = None,
        factory: ResponseFactory | None = None,
        fail: bool = False,
    ) -> None:
        self._responses = responses or {}
        self._factory = factory
        self._fail = fail

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> T:
        if self._fail:
            raise LLMProviderError("mock provider configured to fail")

        if self._factory is not None:
            result = self._factory(system_prompt, user_prompt, response_model)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment]
            if not isinstance(result, response_model):
                raise LLMProviderError("mock factory returned wrong type")
            return result  # type: ignore[return-value]

        canned = self._responses.get(response_model.__name__)
        if canned is None:
            raise LLMProviderError(
                f"mock provider has no canned response for {response_model.__name__}"
            )
        if not isinstance(canned, response_model):
            raise LLMProviderError("canned response type mismatch")
        return canned
