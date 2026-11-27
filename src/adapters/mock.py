"""Deterministic provider adapter for tests and demos."""

import json
from collections.abc import AsyncIterator
from datetime import datetime

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.tool_calls import JsonObject, request_seed, stable_call_id
from router.schemas import ChatMessage, ProviderResponse, ToolCall, ToolCallFunction, ToolConfig

_RESULT_PREVIEW_CHARS = 120


class MockProviderAdapter(BaseProviderAdapter):
    """In-memory provider adapter with configurable failures.

    Tool calling is emulated deterministically, so tests and demos run the
    whole round trip offline. When tools are offered and ``tool_choice`` is
    not ``"none"``, the mock calls the function a named ``tool_choice`` forces,
    otherwise the first declared tool, with arguments built from its JSON
    Schema (see :func:`schema_example_arguments`). With ``tool_choice: "auto"``
    a conversation that ends with tool results gets a text answer quoting them
    instead. Like every adapter, the mock never executes a tool.
    """

    def __init__(self, provider_name: str, fail: bool = False, output_tokens: int = 16) -> None:
        """Initialize mock adapter.

        Args:
            provider_name: Provider name returned by the adapter.
            fail: Whether calls should fail.
            output_tokens: Length of the mock answer in tokens; a smaller
                ``max_tokens`` truncates it and reports ``finish_reason="length"``.
        """
        self.provider_name = provider_name
        self._fail = fail
        self._output_tokens = output_tokens

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> ProviderResponse:
        """Return a deterministic mock completion (or tool call)."""
        if self._fail:
            raise ProviderError(f"mock provider failed: {self.provider_name}")
        prompt = "\n".join(message.content for message in messages)
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = min(max_tokens, self._output_tokens)
        truncated = max_tokens < self._output_tokens
        tool_calls = _emulated_tool_calls(model, messages, tool_config)
        if tool_calls:
            return ProviderResponse(
                content="",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
                finish_reason="length" if truncated else "tool_calls",
                tool_calls=tool_calls,
            )
        return ProviderResponse(
            content=_mock_answer(model, messages),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
            finish_reason="length" if truncated else "stop",
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> AsyncIterator[str | ProviderResponse]:
        """Stream deterministic mock chunks, then the final response."""
        response = await self._complete_once(model, messages, max_tokens, tool_config)
        # Split into a few chunks so SSE clients exercise multi-frame parsing.
        words = response.content.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            yield f"{word}{suffix}"
        yield response

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int, *, at: datetime | None = None
    ) -> float:
        """Estimate mock completion cost: one flat rate for every model and instant."""
        return ((input_tokens + output_tokens) / 1000.0) * 0.001

    async def health_check(self) -> bool:
        """Return mock provider health."""
        return not self._fail


def _emulated_tool_calls(
    model: str, messages: list[ChatMessage], tool_config: ToolConfig | None
) -> list[ToolCall]:
    """Return the single tool call the mock makes for this turn (none for a text answer)."""
    if tool_config is None or tool_config.tool_choice == "none":
        return []
    must_call = tool_config.tool_choice == "required" or tool_config.forced_function is not None
    if not must_call and messages and messages[-1].role == "tool":
        return []
    name = tool_config.forced_function or tool_config.tools[0].function.name
    function = next(tool.function for tool in tool_config.tools if tool.function.name == name)
    arguments = json.dumps(schema_example_arguments(function.parameters_schema()))
    seed = request_seed(
        {"model": model, "messages": [message.model_dump(mode="json") for message in messages]}
    )
    return [
        ToolCall(
            id=stable_call_id(seed, 0, name, arguments),
            function=ToolCallFunction(name=name, arguments=arguments),
        )
    ]


def schema_example_arguments(schema: JsonObject) -> JsonObject:
    """Build deterministic example arguments for an object JSON Schema.

    Every ``required`` property is filled (every declared property when the
    schema lists none as required) with the first of its ``const``, ``enum``,
    ``default``, or ``examples`` values, else a placeholder for its ``type``:
    ``"mock-<name>"`` for strings, the ``minimum`` (or 1) for numbers, ``true``,
    ``[]``, or a nested object.

    Args:
        schema: ``tools[].function.parameters``.

    Returns:
        Arguments object.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = schema.get("required")
    names = (
        [name for name in required if name in properties]
        if isinstance(required, list) and required
        else list(properties)
    )
    return {str(name): _example_value(str(name), properties[name]) for name in names}


def _example_value(name: str, schema: object) -> object:
    """Return a deterministic example value for one property schema."""
    if not isinstance(schema, dict):
        return f"mock-{name}"
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    if "default" in schema:
        return schema["default"]
    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        return examples[0]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((kind for kind in schema_type if kind != "null"), None)
    if schema_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        return minimum if isinstance(minimum, int | float) else 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return []
    if schema_type == "object":
        return schema_example_arguments(schema)
    if schema_type == "null":
        return None
    return f"mock-{name}"


def _mock_answer(model: str, messages: list[ChatMessage]) -> str:
    """Return the mock text answer, quoting the tool results the conversation ends with."""
    results = _trailing_tool_results(messages)
    if not results:
        return f"[{model}] routed response"
    summary = "; ".join(f"{name} returned {preview}" for name, preview in results)
    return f"[{model}] answer from tool results: {summary}"


def _trailing_tool_results(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    """Return ``(function name, result preview)`` for the tool messages ending the conversation."""
    names_by_id = {
        call.id: call.function.name for message in messages for call in message.tool_calls or ()
    }
    results: list[tuple[str, str]] = []
    for message in reversed(messages):
        if message.role != "tool":
            break
        preview = " ".join(message.content.split())
        if len(preview) > _RESULT_PREVIEW_CHARS:
            preview = preview[: _RESULT_PREVIEW_CHARS - 3] + "..."
        results.append((names_by_id.get(message.tool_call_id or "", "tool"), preview))
    return results[::-1]
