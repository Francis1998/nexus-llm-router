"""Google Gemini provider adapter."""

import json
from bisect import bisect_right
from collections.abc import AsyncIterator
from datetime import datetime

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import JsonObject, json_object, nested_int
from adapters.tool_calls import (
    DUMMY_THOUGHT_SIGNATURE,
    arguments_object,
    decode_thought_signature,
    encode_thought_signature,
    request_seed,
    stable_call_id,
    tool_finish_reason,
)
from router.config import list_rates
from router.schemas import (
    ChatMessage,
    NamedToolChoice,
    ProviderResponse,
    ToolCall,
    ToolCallFunction,
    ToolConfig,
)


def _extract_gemini_text(candidates: object) -> str:
    """Concatenate answer text from a Gemini ``candidates`` list.

    A candidate's ``content.parts`` is an ordered list of typed parts. When a
    thinking model (the Gemini 2.5 and 3 series) is asked for thought summaries,
    the response interleaves *thought* parts \u2014 marked ``{"thought": true}`` and
    carrying the model's internal reasoning \u2014 with the answer parts. Joining
    every part with a ``text`` field therefore leaked that reasoning into the
    user-facing completion. Thought parts are skipped so only the answer text is
    returned, mirroring the Anthropic adapter's handling of ``thinking`` blocks.

    Args:
        candidates: The ``candidates`` field of a Gemini response body.

    Returns:
        The concatenated answer text, or an empty string when absent.
    """
    if not isinstance(candidates, list) or not candidates:
        return ""
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        return ""
    candidate_content = first_candidate.get("content", {})
    if not isinstance(candidate_content, dict):
        return ""
    parts = candidate_content.get("parts", [])
    if not isinstance(parts, list):
        return ""
    segments: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("thought") is True:
            continue
        text_value = part.get("text")
        if isinstance(text_value, str):
            segments.append(text_value)
    return "".join(segments)


class GoogleGeminiAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini generateContent."""

    provider_name = "google"

    def __init__(self, api_key: str | None, timeout_seconds: float) -> None:
        """Initialize the Gemini adapter.

        Args:
            api_key: Google API key.
            timeout_seconds: Provider request timeout.
        """
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        # USD per 1K tokens (input, output): static list prices, verified 2026-11-27.
        # Announced changes are router.config.PRICE_SCHEDULE entries, applied from
        # their effective instant: Gemini 4 Carbon and Argon rise to (0.004, 0.020)
        # on 2026-12-01, Gemini 3.8 Flash to (0.0015, 0.0075) on 2027-01-01.
        # Gemini 4 Neon is paid tier only.
        self._costs: dict[str, tuple[float, float]] = {
            "gemini-4-carbon": (0.002, 0.010),
            "gemini-4-argon": (0.002, 0.010),
            "gemini-4-neon": (0.003, 0.015),
            "gemini-4-hydrogen": (0.0015, 0.0075),
            "gemini-3.8-flash": (0.00075, 0.00375),
            "gemini-3.5-flash-lite": (0.0003, 0.0025),
            "gemini-3.1-pro-preview": (0.0035, 0.0105),
            "gemini-3.5-flash": (0.0015, 0.009),
        }

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> ProviderResponse:
        """Return a normalized Gemini completion."""
        if not self._api_key:
            raise ProviderError("GOOGLE_API_KEY is not configured")
        payload = _build_gemini_payload(messages, max_tokens, tool_config)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        # Send the key as a header, never as ``?key=``: httpx logs every request
        # URL at INFO, which would write the key to the gateway logs.
        headers = {"x-goog-api-key": self._api_key}
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise ProviderError(
                f"google request failed with status {response.status_code}",
                status_code=response.status_code,
            )
        body = json_object(response)
        candidates = body.get("candidates", [])
        content = _extract_gemini_text(candidates)
        tool_calls = _extract_gemini_tool_calls(candidates, request_seed(payload))
        if tool_config is not None and tool_config.parallel_tool_calls is False:
            # Gemini has no switch for parallel calls; honor the client's request.
            tool_calls = tool_calls[:1]
        stop_reason: object = None
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            stop_reason = candidates[0].get("finishReason")
        input_tokens = nested_int(body, ["usageMetadata", "promptTokenCount"])
        output_tokens = nested_int(body, ["usageMetadata", "candidatesTokenCount"])
        return ProviderResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
            finish_reason=tool_finish_reason(stop_reason, {"MAX_TOKENS"}, tool_calls),
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> AsyncIterator[str | ProviderResponse]:
        """Stream normalized Gemini text chunks, then the final response."""
        response = await self._complete_once(model, messages, max_tokens, tool_config)
        yield response.content
        yield response

    def has_price(self, model: str) -> bool:
        """Return whether the Gemini price table lists ``model``."""
        return model in self._costs

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int, *, at: datetime | None = None
    ) -> float:
        """Estimate Gemini completion cost at the list prices in effect at ``at``."""
        input_cost, output_cost = list_rates(model, self._costs.get(model, (0.0, 0.0)), at)
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether Google credentials are configured."""
        return self._api_key is not None


def _build_gemini_payload(
    messages: list[ChatMessage], max_tokens: int, tool_config: ToolConfig | None = None
) -> dict[str, object]:
    """Build a Gemini generateContent payload from normalized chat messages."""
    system_text = "\n\n".join(message.content for message in messages if message.role == "system")
    payload: dict[str, object] = {
        "contents": _gemini_contents(messages),
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    if tool_config is not None:
        payload.update(_gemini_tool_fields(tool_config))
    return payload


def _gemini_contents(messages: list[ChatMessage]) -> list[JsonObject]:
    """Translate chat turns, assistant tool calls, and tool results into Gemini ``contents``.

    Assistant ``tool_calls`` become ``functionCall`` parts carrying the thought
    signature embedded in their ids (Gemini 3 and later reject replayed calls
    without one; a call from another provider gets Gemini's documented
    placeholder). ``tool`` results become ``functionResponse`` parts named
    after the call they answer, and consecutive results share one turn, as
    Gemini expects for parallel calls. Gemini pairs these name-only responses
    with the calls by position, so they follow the order of the preceding
    assistant turn's calls, whatever order the client sent them in.
    """
    contents: list[JsonObject] = []
    names_by_id: dict[str, str] = {}
    call_positions: dict[str, int] = {}
    turn_positions: list[int] = []
    previous_was_tool_result = False
    for message in messages:
        if message.role == "system":
            continue
        name = names_by_id.get(message.tool_call_id or "")
        if message.role == "tool" and name is not None:
            part: JsonObject = {
                "functionResponse": {"name": name, "response": _function_response(message.content)}
            }
            position = call_positions.get(message.tool_call_id or "", len(call_positions))
            last = contents[-1] if contents else None
            if previous_was_tool_result and last is not None and isinstance(last["parts"], list):
                slot = bisect_right(turn_positions, position)
                turn_positions.insert(slot, position)
                last["parts"].insert(slot, part)
            else:
                turn_positions = [position]
                contents.append({"role": "user", "parts": [part]})
            previous_was_tool_result = True
            continue
        previous_was_tool_result = False
        if message.role == "assistant" and message.tool_calls:
            parts: list[JsonObject] = [{"text": message.content}] if message.content else []
            signatures = [decode_thought_signature(call.id) for call in message.tool_calls]
            if not any(signatures):
                signatures[0] = DUMMY_THOUGHT_SIGNATURE
            call_positions = {call.id: index for index, call in enumerate(message.tool_calls)}
            for call, signature in zip(message.tool_calls, signatures, strict=True):
                names_by_id[call.id] = call.function.name
                function_part: JsonObject = {
                    "functionCall": {
                        "name": call.function.name,
                        "args": arguments_object(call.function.arguments),
                    }
                }
                if signature is not None:
                    function_part["thoughtSignature"] = signature
                parts.append(function_part)
            contents.append({"role": "model", "parts": parts})
            continue
        contents.append(
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
        )
    return contents


def _function_response(content: str) -> JsonObject:
    """Return a ``functionResponse.response`` object for a tool result.

    A JSON-object result is passed as-is; any other text is wrapped as
    ``{"output": ...}``, the key Gemini documents for function output.
    """
    try:
        decoded = json.loads(content)
    except ValueError:
        decoded = None
    return decoded if isinstance(decoded, dict) else {"output": content}


def _gemini_tool_fields(tool_config: ToolConfig) -> JsonObject:
    """Return Gemini ``tools`` (function declarations) and ``toolConfig``."""
    declarations: list[JsonObject] = []
    for tool in tool_config.tools:
        declaration: JsonObject = {
            "name": tool.function.name,
            "parametersJsonSchema": tool.function.parameters_schema(),
        }
        if tool.function.description:
            declaration["description"] = tool.function.description
        declarations.append(declaration)
    fields: JsonObject = {"tools": [{"functionDeclarations": declarations}]}
    choice = tool_config.tool_choice
    if isinstance(choice, NamedToolChoice):
        calling: JsonObject = {"mode": "ANY", "allowedFunctionNames": [choice.function.name]}
    elif choice is None:
        return fields
    else:
        calling = {"mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}[choice]}
    fields["toolConfig"] = {"functionCallingConfig": calling}
    return fields


def _extract_gemini_tool_calls(candidates: object, seed: str) -> list[ToolCall]:
    """Return the ``functionCall`` parts of the first candidate as OpenAI tool calls.

    Gemini ids are used when present; otherwise ids are derived from the request
    digest, so they are stable for one request and unique across turns. A part's
    ``thoughtSignature`` is embedded in the id for the next turn.
    """
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return []
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    calls: list[ToolCall] = []
    for index, part in enumerate(parts if isinstance(parts, list) else []):
        function_call = part.get("functionCall") if isinstance(part, dict) else None
        if not isinstance(function_call, dict) or not isinstance(function_call.get("name"), str):
            continue
        name = function_call["name"]
        arguments = json.dumps(function_call.get("args") or {})
        provider_id = function_call.get("id")
        base_id = (
            provider_id
            if isinstance(provider_id, str) and provider_id
            else stable_call_id(seed, index, name, arguments)
        )
        call_id = encode_thought_signature(base_id, part.get("thoughtSignature"))
        calls.append(
            ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))
        )
    return calls
