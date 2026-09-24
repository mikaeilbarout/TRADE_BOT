from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from research.ai.cost import CallRecord, TokenUsage, compute_cost
from research.ai.models import get_pricing
from research.ai.schemas import MAX_NOTE_CHARS
from research.ai.settings import AISettings

T = TypeVar("T", bound=BaseModel)


class AgentCallError(Exception):
    """Any failure that means this agent produced no trustworthy verdict:
    API error, timeout, malformed output, schema violation. Callers must
    fail closed on it (spec section 11)."""


@dataclass
class AgentRequest:
    """One agent call, split into a STATIC prefix and VOLATILE content.

    The split is the whole cost strategy: `static_system` is byte-identical
    across every signal for a given agent, so it caches; `user_payload`
    carries only the per-signal numbers. Anything time-varying that leaked
    into the static half would silently destroy the cache hit rate.
    """

    agent: str
    signal_id: str
    static_system: str
    user_payload: dict
    response_model: type[BaseModel]
    prompt_version: str
    model: str
    max_output_tokens: int = 400
    extra_static_system: str | None = None

    def rendered_user_text(self) -> str:
        # sort_keys keeps the serialization deterministic; an unsorted dict
        # would produce different bytes for identical data.
        return json.dumps(self.user_payload, sort_keys=True, separators=(",", ":"), default=str)


_UNSUPPORTED_CONSTRAINTS_BY_TYPE = {
    # Anthropic's strict tool schema validator rejects ALL of these
    # validation-only keywords, for every type -- confirmed against the live
    # API one type at a time as each was hit in turn:
    #   "tools.0.custom: For 'number' type, properties maximum, minimum are
    #    not supported"
    #   "tools.0.custom: For 'integer' type, properties maximum, minimum are
    #    not supported"
    #   "tools.0.custom: For 'array' type, property 'maxItems' is not
    #    supported"
    # "string" (minLength/maxLength/pattern) follows the same pattern by
    # extrapolation, not yet individually confirmed by an API error.
    "number": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"),
    "integer": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"),
    "array": ("minItems", "maxItems"),
    "string": ("minLength", "maxLength", "pattern"),
}


def _strip_unsupported_constraints(node: Any) -> None:
    """Remove JSON-schema validation keywords Anthropic's strict tool schema
    rejects for a given `type`, recursively. Pydantic-side validation still
    enforces the real constraint when the response is parsed back into the
    model -- only the schema advertised to the API loses it.
    """
    if isinstance(node, dict):
        keys_to_strip = _UNSUPPORTED_CONSTRAINTS_BY_TYPE.get(node.get("type"))
        if keys_to_strip:
            for key in keys_to_strip:
                node.pop(key, None)
        for value in node.values():
            _strip_unsupported_constraints(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_constraints(item)


def build_strict_tool(response_model: type[BaseModel], tool_name: str = "emit") -> dict:
    """Turn a Pydantic model into a strict tool schema.

    `strict: true` plus `additionalProperties: false` guarantees the tool
    input validates against the schema, which is what makes short structured
    outputs reliable enough to parse without retries.
    """
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False
    _strip_unsupported_constraints(schema)
    return {
        "name": tool_name,
        "description": "Return the structured verdict. No prose outside this tool.",
        "input_schema": schema,
        "strict": True,
    }


class BaseAgentClient(ABC):
    @abstractmethod
    async def call(self, request: AgentRequest) -> tuple[BaseModel, CallRecord]: ...

    @abstractmethod
    def build_batch_params(self, request: AgentRequest) -> dict:
        """The `params` body for a Message Batches request entry."""


class AnthropicAgentClient(BaseAgentClient):
    """Anthropic Messages API client tuned for cost.

    Three cost levers are applied here:
      * The static system prefix carries `cache_control`, so tools + system
        are cached and only the per-signal payload is billed at full rate.
        Render order is tools -> system -> messages, so placing the
        breakpoint on the last system block covers both.
      * Outputs are capped hard (`max_output_tokens`) and the schema is
        reason codes rather than prose.
      * Per-agent model routing means the cheap model runs the three
        analysts and only the adjudicator uses the expensive one.
    """

    def __init__(self, settings: AISettings, client: Any | None = None) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
        else:
            if not settings.anthropic_api_key:
                raise AgentCallError(
                    "ANTHROPIC_API_KEY is not set. Export it (never commit it) before "
                    "running the AI backtest."
                )
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.ai_timeout_seconds,
                max_retries=settings.ai_max_retries,
            )

    # --- request assembly -------------------------------------------------
    def _system_blocks(self, request: AgentRequest) -> list[dict]:
        blocks: list[dict] = [{"type": "text", "text": request.static_system}]
        if request.extra_static_system:
            blocks.append({"type": "text", "text": request.extra_static_system})
        if self._settings.ai_use_prompt_cache:
            # Breakpoint on the LAST static block: caches tools + all system
            # text. Volatile content lives in `messages`, after this point.
            blocks[-1]["cache_control"] = {
                "type": "ephemeral",
                "ttl": self._settings.ai_cache_ttl,
            }
        return blocks

    def _request_body(self, request: AgentRequest) -> dict:
        pricing = get_pricing(request.model)
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "system": self._system_blocks(request),
            "messages": [{"role": "user", "content": request.rendered_user_text()}],
            "tools": [build_strict_tool(request.response_model)],
            "tool_choice": {"type": "tool", "name": "emit"},
        }
        # Only send effort where the model accepts it: Haiku 4.5 rejects
        # `output_config.effort`, so sending it unconditionally would 400 on
        # exactly the cheap path this design depends on.
        if pricing.supports_effort:
            body["output_config"] = {"effort": "low"}
        return body

    def build_batch_params(self, request: AgentRequest) -> dict:
        return self._request_body(request)

    # --- live call --------------------------------------------------------
    async def call(self, request: AgentRequest) -> tuple[BaseModel, CallRecord]:
        started = time.monotonic()
        try:
            response = await self._client.messages.create(**self._request_body(request))
        except Exception as exc:
            raise AgentCallError(
                f"{request.agent} call failed for signal {request.signal_id}: {exc}"
            ) from exc

        latency = time.monotonic() - started
        usage = extract_usage(response)
        record = CallRecord(
            signal_id=request.signal_id,
            agent=request.agent,
            model=request.model,
            usage=usage,
            cost_usd=compute_cost(usage, request.model, batch=False),
            latency_seconds=latency,
            batch=False,
            prompt_version=request.prompt_version,
        )

        try:
            verdict = parse_tool_result(response, request.response_model)
        except AgentCallError as exc:
            record.error = str(exc)
            raise

        return verdict, record


def extract_usage(response: Any) -> TokenUsage:
    """Pull the four token fields, tolerating None (the API omits cache
    fields entirely when caching is not in play)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )


def parse_tool_result(response: Any, response_model: type[T]) -> T:
    """Validate the forced tool call's input against the schema.

    A refusal or a missing tool_use block is an error, not an empty verdict --
    the caller must fail closed rather than treat silence as approval.
    """
    if getattr(response, "stop_reason", None) == "refusal":
        raise AgentCallError("model refused the request (stop_reason=refusal)")

    content = getattr(response, "content", None) or []
    tool_block = next((b for b in content if getattr(b, "type", None) == "tool_use"), None)
    if tool_block is None:
        raise AgentCallError("response contained no tool_use block")

    payload = tool_block.input
    if isinstance(payload, str):
        # Tool inputs are JSON; never string-match them.
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AgentCallError(f"tool input was not valid JSON: {exc}") from exc

    # `note` carries the model's free-text rationale, capped at MAX_NOTE_CHARS
    # by the schema -- but that cap is only a prose hint in the payload sent
    # to the model (Anthropic's strict tool schema rejects `maxLength` on the
    # wire; see _strip_unsupported_constraints), and a model given genuinely
    # rich indicator/sentiment data to discuss overruns a prose hint far more
    # often than one with nothing to say. Truncating here is strictly safer
    # than failing the whole call closed over a rationale that is display-only
    # and plays no role in the decision itself.
    if isinstance(payload, dict) and isinstance(payload.get("note"), str):
        if len(payload["note"]) > MAX_NOTE_CHARS:
            payload["note"] = payload["note"][:MAX_NOTE_CHARS]

    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise AgentCallError(f"tool input failed schema validation: {exc}") from exc


def default_mock_verdicts() -> dict[str, BaseModel]:
    """Neutral canned verdicts, for rehearsing the pipeline without spending.

    Deliberately bland: every analyst returns WARN at middling confidence and
    the adjudicator approves at the threshold. These are NOT analysis and a run
    using them proves only that the plumbing works -- which is why the cost
    report labels a mock run's costs synthetic.
    """
    from research.ai.schemas import (
        Bias,
        FinalAction,
        FinalVerdict,
        Gate,
        NewsVerdict,
        RiskLevel,
        SentimentVerdict,
        TechnicalVerdict,
    )

    return {
        "technical": TechnicalVerdict(
            decision=Gate.WARN,
            confidence=0.6,
            bias=Bias.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            reason_codes=["MOCK_RUN"],
            note="mock verdict: not analysis",
            htf_aligned=False,
        ),
        "news": NewsVerdict(
            decision=Gate.WARN,
            confidence=0.6,
            reason_codes=["MOCK_RUN"],
            note="mock verdict: not analysis",
        ),
        "sentiment": SentimentVerdict(
            decision=Gate.WARN,
            confidence=0.6,
            reason_codes=["MOCK_RUN"],
            note="mock verdict: not analysis",
        ),
        "final": FinalVerdict(
            action=FinalAction.APPROVE,
            confidence=0.75,
            reason_codes=["MOCK_RUN"],
            note="mock verdict: not analysis",
            news_score=60,
            sentiment_score=60,
            technical_score=65,
            risk_score=60,
        ),
    }


@dataclass
class MockAgentClient(BaseAgentClient):
    """Deterministic, network-free client for tests and dry runs.

    Returns canned verdicts per agent and synthesizes plausible token counts
    so the cost pipeline itself can be tested without spending anything.
    """

    verdicts: dict[str, BaseModel] = field(default_factory=dict)
    fail_agents: set[str] = field(default_factory=set)
    simulated_usage: TokenUsage | None = None
    calls: list[AgentRequest] = field(default_factory=list)
    cache_after_first_call: bool = True

    def build_batch_params(self, request: AgentRequest) -> dict:
        return {"model": request.model, "max_tokens": request.max_output_tokens}

    async def call(self, request: AgentRequest) -> tuple[BaseModel, CallRecord]:
        self.calls.append(request)
        if request.agent in self.fail_agents:
            raise AgentCallError(f"mock failure for {request.agent}")

        verdict = self.verdicts.get(request.agent)
        if verdict is None:
            raise AgentCallError(f"mock has no canned verdict for {request.agent}")
        if not isinstance(verdict, request.response_model):
            raise AgentCallError("canned verdict type mismatch")

        usage = self.simulated_usage or self._synthesize_usage(request)
        return verdict, CallRecord(
            signal_id=request.signal_id,
            agent=request.agent,
            model=request.model,
            usage=usage,
            cost_usd=compute_cost(usage, request.model),
            latency_seconds=0.01,
            prompt_version=request.prompt_version,
        )

    def _synthesize_usage(self, request: AgentRequest) -> TokenUsage:
        static_tokens = max(1, len(request.static_system) // 4)
        volatile_tokens = max(1, len(request.rendered_user_text()) // 4)
        prior_calls = sum(1 for c in self.calls[:-1] if c.agent == request.agent)
        warm = self.cache_after_first_call and prior_calls > 0
        return TokenUsage(
            input_tokens=volatile_tokens,
            cache_creation_tokens=0 if warm else static_tokens,
            cache_read_tokens=static_tokens if warm else 0,
            output_tokens=60,
        )
