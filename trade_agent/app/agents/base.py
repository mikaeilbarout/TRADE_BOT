from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.observability.logging import get_logger
from app.providers.llm.base import LLMProvider, LLMProviderError

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class AgentError(Exception):
    """Raised whenever an agent cannot produce a trustworthy result: LLM
    timeout, provider error, or schema validation failure. The pipeline
    must treat this as fail-closed -- never substitute a default 'PASS'."""

    def __init__(self, agent_name: str, message: str, latency_seconds: float = 0.0):
        super().__init__(f"[{agent_name}] {message}")
        self.agent_name = agent_name
        self.latency_seconds = latency_seconds


@dataclass
class AgentRun:
    """One agent call: its validated result plus everything needed for the
    audit trail (section 17 wants the exact input, not a paraphrase)."""

    result: Any
    latency_seconds: float
    input_snapshot: dict
    attempts: int


@lru_cache(maxsize=None)
def load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


class BaseAgent:
    """Shared plumbing every agent uses: load its own versioned prompt file,
    call the LLM with a hard timeout and bounded retries, validate the
    structured result, and hand back the exact input it was given so the
    decision can be reconstructed later.

    Each concrete agent supplies only its prompt filename, response schema,
    and how to build its input payload from typed inputs.
    """

    prompt_file: str
    response_model: type[BaseModel]
    agent_name: str

    def __init__(
        self,
        llm: LLMProvider,
        timeout_seconds: float,
        model_name: str | None = None,
        max_retries: int = 1,
    ):
        self._llm = llm
        self._timeout = timeout_seconds
        self._model_name = model_name
        self._max_retries = max(0, max_retries)

    @property
    def system_prompt(self) -> str:
        return load_prompt(self.prompt_file)

    @property
    def provider_name(self) -> str:
        return getattr(self._llm, "name", "unknown")

    def render_user_prompt(self, payload: dict) -> str:
        """Default rendering of a typed payload into the user message. Agents
        override `build_payload`, not this, so every agent's prompt body is
        built the same way and the payload doubles as the audit snapshot."""
        import json

        # Compact, not indent=2: indentation is pure whitespace to the model
        # and was a large share of every agent's input tokens (the technical
        # agent's 360-candle block most of all). The audit snapshot is the
        # payload dict itself, so nothing human-readable is lost.
        return (
            f"{self.task_instruction}\n\n"
            "INPUT (JSON):\n"
            f"{json.dumps(payload, separators=(',', ':'), default=str)}"
        )

    task_instruction: str = "Analyze the following input."

    async def _call(self, payload: dict) -> AgentRun:
        user_prompt = self.render_user_prompt(payload)
        attempts = 0
        last_error: Exception | None = None
        start = time.monotonic()

        # One attempt plus `max_retries` retries: an LLM blip or a single
        # malformed response should not silently become a rejected trade,
        # but the budget is bounded because latency matters (section 23).
        while attempts <= self._max_retries:
            attempts += 1
            attempt_start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self._llm.complete_structured(
                        system_prompt=self.system_prompt,
                        user_prompt=user_prompt,
                        response_model=self.response_model,
                    ),
                    timeout=self._timeout,
                )
                if hasattr(result, "model") and getattr(result, "model", None) is None:
                    result = result.model_copy(update={"model": self._model_name})
                return AgentRun(
                    result=result,
                    latency_seconds=time.monotonic() - start,
                    input_snapshot=payload,
                    attempts=attempts,
                )
            except (asyncio.TimeoutError, LLMProviderError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "agent call failed",
                    extra={
                        "agent": self.agent_name,
                        "attempt": attempts,
                        "max_attempts": self._max_retries + 1,
                        "latency_seconds": round(time.monotonic() - attempt_start, 4),
                        "error": str(exc) or type(exc).__name__,
                    },
                )

        latency = time.monotonic() - start
        detail = (
            f"timed out after {self._timeout}s"
            if isinstance(last_error, asyncio.TimeoutError)
            else str(last_error)
        )
        raise AgentError(
            self.agent_name, f"{detail} (after {attempts} attempt(s))", latency
        )
