from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProviderError(Exception):
    """Raised for any provider-level failure: network, auth, rate limit,
    malformed response, or schema validation failure. Callers (agents) must
    treat this as fail-closed, never as an implicit approval."""


class LLMProvider(ABC):
    """Abstraction over a chat/completion backend that can be coerced into
    returning a structured, schema-validated JSON object.

    Swapping Claude <-> OpenAI <-> Gemini <-> a local model means writing one
    new subclass; nothing else in the system (agents, prompts, pipeline)
    changes.
    """

    name: str = "base"

    @abstractmethod
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T:
        """Return a validated instance of `response_model`.

        Implementations MUST raise LLMProviderError (not return partial or
        best-effort data) on any failure: network error, timeout, non-JSON
        response, or schema validation failure. There is no silent fallback.
        """
        raise NotImplementedError
