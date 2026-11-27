"""OpenAI provider adapter.

Text-only requests use Chat Completions. Requests that offer tools or replay
tool calls use the Responses API, which every GPT-6.x model supports and
GPT-6.2 requires for tool calling.
"""

from collections.abc import AsyncIterator
from datetime import datetime

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import JsonObject, finish_reason, json_object, message_text, nested_int
from adapters.tool_calls import openai_chat_messages, provider_safe_id, tool_finish_reason
from router.config import list_rates
from router.model_ids import GPT6_LONG_PROMPT_THRESHOLD_TOKENS
from router.schemas import (
    MIN_TOOL_OUTPUT_TOKENS,
    ChatMessage,
    LongPromptPricing,
    NamedToolChoice,
    ProviderResponse,
    ToolCall,
    ToolCallFunction,
    ToolConfig,
)

_LEGACY_MAX_TOKENS_PREFIXES = ("gpt-4", "gpt-3.5")
_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIAdapter(BaseProviderAdapter):
    """Adapter for OpenAI chat completions."""

    provider_name = "openai"

    def __init__(self, api_key: str | None, timeout_seconds: float) -> None:
        """Initialize the OpenAI adapter.

        Args:
            api_key: OpenAI API key.
            timeout_seconds: Provider request timeout.
        """
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        # USD per 1K tokens (input, output); GPT-6.x list prices as of 2026-11-19.
        # GPT-6.2 Astra and Sol supersede GPT-6 Astra and GPT-6.1 Sol, which stay
        # priced for existing callers.
        self._costs: dict[str, tuple[float, float]] = {
            "gpt-6.2-astra": (0.008, 0.040),
            "gpt-6.2-sol": (0.002, 0.010),
            "gpt-6-astra": (0.010, 0.050),
            "gpt-6.1-sol": (0.002, 0.010),
            "gpt-6.1-terra": (0.001, 0.005),
            "gpt-6-luna": (0.0001, 0.0005),
            "gpt-5.5": (0.006, 0.018),
            "gpt-4.1-mini": (0.0002, 0.0008),
        }
        # GPT-6.x bills the whole request at 2x input / 1.5x output above 272K input tokens.
        self._long_prompt_pricing: dict[str, LongPromptPricing] = {
            model: LongPromptPricing(
                threshold_tokens=GPT6_LONG_PROMPT_THRESHOLD_TOKENS,
                input_cost_per_1k=input_cost * 2,
                output_cost_per_1k=output_cost * 1.5,
            )
            for model, (input_cost, output_cost) in self._costs.items()
            if model.startswith("gpt-6")
        }

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> ProviderResponse:
        """Return a normalized OpenAI completion."""
        if not self._api_key:
            raise ProviderError("OPENAI_API_KEY is not configured")
        if tool_config is not None or _replays_tools(messages):
            if max_tokens < MIN_TOOL_OUTPUT_TOKENS:
                # Raising the cap would exceed the client's limit and the budget estimate.
                raise ProviderError(
                    "openai tool calls go through the Responses API, which needs "
                    f"max_output_tokens >= {MIN_TOOL_OUTPUT_TOKENS}; got max_tokens={max_tokens}",
                    status_code=400,
                )
            payload = _responses_payload(model, messages, max_tokens, tool_config)
            body = await self._post(_RESPONSES_URL, payload)
            return self._normalize_responses_body(model, body)
        chat_payload: dict[str, object] = {
            "model": model,
            "messages": openai_chat_messages(messages),
        }
        # Reasoning-era models (GPT-5.x, GPT-6, o-series) reject ``max_tokens``
        # and require ``max_completion_tokens``. Only legacy GPT-4.x / GPT-3.5
        # SKUs still expect ``max_tokens``.
        if model.startswith(_LEGACY_MAX_TOKENS_PREFIXES):
            chat_payload["max_tokens"] = max_tokens
        else:
            chat_payload["max_completion_tokens"] = max_tokens
        body = await self._post(_CHAT_COMPLETIONS_URL, chat_payload)
        choices = body.get("choices", [])
        content = ""
        stop_reason: object = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            content = message_text(choices[0].get("message"))
            stop_reason = choices[0].get("finish_reason")
        input_tokens = nested_int(body, ["usage", "prompt_tokens"])
        output_tokens = nested_int(body, ["usage", "completion_tokens"])
        return ProviderResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
            finish_reason=finish_reason(stop_reason, {"length"}),
        )

    async def _post(self, url: str, payload: dict[str, object]) -> JsonObject:
        """POST ``payload`` to an OpenAI endpoint and return the JSON body."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise ProviderError(
                f"openai request failed with status {response.status_code}",
                status_code=response.status_code,
            )
        return json_object(response)

    def _normalize_responses_body(self, model: str, body: JsonObject) -> ProviderResponse:
        """Normalize a Responses API ``response`` (text, function calls, usage)."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        output = body.get("output")
        for item in output if isinstance(output, list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                text_parts.extend(_output_text(item.get("content")))
            elif item.get("type") == "function_call":
                call = _function_call(item)
                if call is not None:
                    tool_calls.append(call)
        incomplete = body.get("incomplete_details")
        reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
        input_tokens = nested_int(body, ["usage", "input_tokens"])
        output_tokens = nested_int(body, ["usage", "output_tokens"])
        return ProviderResponse(
            content="".join(text_parts),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
            finish_reason=tool_finish_reason(reason, {"max_output_tokens"}, tool_calls),
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> AsyncIterator[str | ProviderResponse]:
        """Stream normalized OpenAI text chunks, then the final response."""
        response = await self._complete_once(model, messages, max_tokens, tool_config)
        yield response.content
        yield response

    def has_price(self, model: str) -> bool:
        """Return whether the OpenAI price table lists ``model``."""
        return model in self._costs

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int, *, at: datetime | None = None
    ) -> float:
        """Estimate OpenAI completion cost, including GPT-6 long-prompt pricing."""
        input_cost, output_cost = list_rates(model, self._costs.get(model, (0.0, 0.0)), at)
        long_prompt_pricing = self._long_prompt_pricing.get(model)
        if long_prompt_pricing is not None:
            input_cost, output_cost = long_prompt_pricing.rates_for(
                input_tokens, (input_cost, output_cost)
            )
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether OpenAI credentials are configured."""
        return self._api_key is not None


def _replays_tools(messages: list[ChatMessage]) -> bool:
    """Return whether the conversation carries tool calls or tool results."""
    return any(message.tool_calls or message.role == "tool" for message in messages)


def _responses_payload(
    model: str, messages: list[ChatMessage], max_tokens: int, tool_config: ToolConfig | None
) -> dict[str, object]:
    """Translate a chat conversation with tools into a Responses API request.

    System messages become ``developer`` input messages in place, assistant
    ``tool_calls`` become ``function_call`` items, and ``tool`` results become
    ``function_call_output`` items paired by ``call_id``. ``store`` is false:
    nexus is stateless and resends the conversation every turn.
    """
    payload: dict[str, object] = {
        "model": model,
        "input": _responses_input(messages),
        "max_output_tokens": max_tokens,
        "store": False,
    }
    if tool_config is None:
        return payload
    payload["tools"] = [
        {
            "type": "function",
            "name": tool.function.name,
            "description": tool.function.description or "",
            "parameters": tool.function.parameters_schema(),
            # The Responses API defaults ``strict`` to true; Chat Completions to false.
            "strict": bool(tool.function.strict),
        }
        for tool in tool_config.tools
    ]
    choice = tool_config.tool_choice
    if isinstance(choice, NamedToolChoice):
        payload["tool_choice"] = {"type": "function", "name": choice.function.name}
    elif choice is not None:
        payload["tool_choice"] = choice
    if tool_config.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = tool_config.parallel_tool_calls
    return payload


def _responses_input(messages: list[ChatMessage]) -> list[JsonObject]:
    """Return Responses API ``input`` items for router chat messages."""
    items: list[JsonObject] = []
    for message in messages:
        if message.role == "tool" and message.tool_call_id is not None:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": provider_safe_id(message.tool_call_id),
                    "output": message.content,
                }
            )
            continue
        role = {"system": "developer", "tool": "user"}.get(message.role, message.role)
        if message.content or not message.tool_calls:
            items.append({"type": "message", "role": role, "content": message.content})
        for call in message.tool_calls or ():
            items.append(
                {
                    "type": "function_call",
                    "call_id": provider_safe_id(call.id),
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            )
    return items


def _output_text(content: object) -> list[str]:
    """Return the text of a Responses ``message`` item's ``output_text`` / ``refusal`` parts."""
    texts: list[str] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        field = {"output_text": "text", "refusal": "refusal"}.get(str(part.get("type")))
        value = part.get(field) if field else None
        if isinstance(value, str):
            texts.append(value)
    return texts


def _function_call(item: JsonObject) -> ToolCall | None:
    """Return the tool call of a Responses ``function_call`` item (``None`` if malformed)."""
    call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str):
        return None
    return ToolCall(
        id=call_id,
        function=ToolCallFunction(
            name=name, arguments=arguments if isinstance(arguments, str) else "{}"
        ),
    )
