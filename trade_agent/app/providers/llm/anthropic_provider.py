from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.providers.llm.base import LLMProvider, LLMProviderError

T = TypeVar("T", bound=BaseModel)

# Even inside a forced tool_use call the model occasionally lets a stray
# closing tag leak into a string field's own content -- observed live
# 2026-09-18 in a final_decision_agent response: summary ended with
# "...validated technical reframe.</summary>\n</invoke>\n". Real trading
# analysis prose never legitimately contains an XML/HTML-style closing tag,
# so stripping this pattern from every string in the tool call's input is
# safe and has no false-positive cost.
_LEAKED_CLOSING_TAG = re.compile(r"\s*</[A-Za-z_][\w:-]*\s*>\s*")


def _strip_leaked_tags(node: Any) -> Any:
    if isinstance(node, str):
        return _LEAKED_CLOSING_TAG.sub(" ", node).strip()
    if isinstance(node, dict):
        return {k: _strip_leaked_tags(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_leaked_tags(v) for v in node]
    return node


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


class AnthropicProvider(LLMProvider):
    """Structured-output backend using Claude via tool-use.

    We force the model to call a single synthetic tool whose input schema is
    the target Pydantic model's JSON schema, then validate the tool call's
    input against that same model. This avoids fragile prose parsing.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        use_prompt_cache: bool = True,
        cache_ttl: str = "1h",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMProviderError(
                "anthropic package not installed; add it to requirements.txt"
            ) from exc
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._use_prompt_cache = use_prompt_cache
        self._cache_ttl = cache_ttl

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> T:
        schema = response_model.model_json_schema()
        _strip_unsupported_constraints(schema)
        tool_name = "emit_result"
        # Each agent's system_prompt is the same .md file byte-for-byte on
        # every call; only user_prompt (the per-signal payload) changes.
        # Anthropic renders tools before system, so a cache_control
        # breakpoint on this one system block covers both -- cost/latency
        # only, never changes what the model is shown. Same mechanism as
        # research/ai/client.py, just ported here for the live path.
        system: str | list[dict] = system_prompt
        if self._use_prompt_cache:
            system = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral", "ttl": self._cache_ttl},
                }
            ]
        try:
            # temperature intentionally omitted -- newer Claude models reject
            # it outright ("`temperature` is deprecated for this model"),
            # confirmed via a live 400 from the API. `temperature` stays in
            # this method's signature for interface compatibility with
            # LLMProvider/OpenAIProvider, just unused here.
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": tool_name,
                        "description": "Return the structured analysis result.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception as exc:  # network, auth, rate limit, etc.
            raise LLMProviderError(f"anthropic call failed: {exc}") from exc

        tool_use = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_use is None:
            raise LLMProviderError("anthropic response contained no tool_use block")

        try:
            return response_model.model_validate(_strip_leaked_tags(tool_use.input))
        except ValidationError as exc:
            raise LLMProviderError(
                f"anthropic tool_use input failed schema validation: {exc}"
            ) from exc


class OpenAIProvider(LLMProvider):
    """Structured-output backend using OpenAI's JSON-schema response format."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMProviderError(
                "openai package not installed; add it to requirements.txt"
            ) from exc
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> T:
        schema = response_model.model_json_schema()
        schema["additionalProperties"] = False
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": False,
                    },
                },
            )
        except Exception as exc:
            raise LLMProviderError(f"openai call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("openai response had empty content")
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMProviderError(f"openai response failed validation: {exc}") from exc
