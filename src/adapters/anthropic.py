"""Anthropic provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import finish_reason, json_object, nested_int
from router.model_ids import HAIKU_5_5_LONG_PROMPT_THRESHOLD_TOKENS
from router.schemas import ChatMessage, LongPromptPricing, ProviderResponse


def _extract_anthropic_text(content_blocks: object) -> str:
    """Concatenate text from all text blocks in an Anthropic response.

    The Messages API returns ``content`` as an ordered list of typed blocks. A
    response may contain several ``text`` blocks, and a non-text block (for
    example a ``thinking`` block when extended thinking is enabled, or a
    ``tool_use`` block) can precede the answer. Reading only ``content[0].text``
    therefore returned an empty string whenever the first block was not text and
    truncated multi-block answers to their first segment. All ``text`` blocks
    are joined in order to faithfully reconstruct the completion.

    Args:
        content_blocks: The ``content`` field of an Anthropic response body.

    Returns:
        The concatenated text of every text block, or an empty string.
    """
    if not isinstance(content_blocks, list):
        return ""
    segments: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in (None, "text"):
            continue
        text_value = block.get("text")
        if isinstance(text_value, str):
            segments.append(text_value)
    return "".join(segments)


def _build_anthropic_payload(
    model: str, messages: list[ChatMessage], max_tokens: int
) -> dict[str, object]:
    """Build Anthropic Messages API payload with system text extracted.

    The Anthropic Messages API accepts only ``user`` and ``assistant`` turn
    roles; tool results are carried on a ``user`` turn. An incoming ``tool``
    message was previously forwarded verbatim, producing a ``{"role": "tool"}``
    entry that Anthropic rejects with a 400 and fails the whole request. Tool
    turns are therefore mapped to the ``user`` role here.
    """
    system_parts = [message.content for message in messages if message.role == "system"]
    conversation_messages = [
        {
            "role": "user" if message.role == "tool" else message.role,
            "content": message.content,
        }
        for message in messages
        if message.role in {"user", "assistant", "tool"}
    ]
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": conversation_messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return payload


class AnthropicAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Claude messages."""

    provider_name = "anthropic"

    def __init__(self, api_key: str | None, timeout_seconds: float) -> None:
        """Initialize the Anthropic adapter.

        Args:
            api_key: Anthropic API key.
            timeout_seconds: Provider request timeout.
        """
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        # USD per 1K tokens (input, output); Claude list prices as of 2026-11-12.
        # Fable 5.2 supersedes Fable 5.1, which stays priced for existing callers.
        self._costs: dict[str, tuple[float, float]] = {
            "claude-fable-5-2": (0.008, 0.040),
            "claude-fable-5-1": (0.010, 0.050),
            "claude-opus-5-5": (0.004, 0.020),
            "claude-sonnet-5-5": (0.002, 0.010),
            "claude-haiku-5-5": (0.0001, 0.0005),
            "claude-sonnet-4-6": (0.003, 0.015),
            "claude-haiku-4-5": (0.0008, 0.004),
        }
        # Haiku 5.5 bills the whole request at $0.50 / $2.50 per 1M above 100K prompt tokens.
        self._long_prompt_pricing: dict[str, LongPromptPricing] = {
            "claude-haiku-5-5": LongPromptPricing(
                threshold_tokens=HAIKU_5_5_LONG_PROMPT_THRESHOLD_TOKENS,
                input_cost_per_1k=0.0005,
                output_cost_per_1k=0.0025,
            ),
        }

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a normalized Anthropic completion."""
        if not self._api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured")
        payload = _build_anthropic_payload(model, messages, max_tokens)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"anthropic request failed with status {response.status_code}",
                status_code=response.status_code,
            )
        body = json_object(response)
        content = _extract_anthropic_text(body.get("content", []))
        input_tokens = nested_int(body, ["usage", "input_tokens"])
        output_tokens = nested_int(body, ["usage", "output_tokens"])
        return ProviderResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
            finish_reason=finish_reason(body.get("stop_reason"), {"max_tokens"}),
        )

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str | ProviderResponse]:
        """Stream normalized Anthropic text chunks, then the final response."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content
        yield response

    def has_price(self, model: str) -> bool:
        """Return whether the Anthropic price table lists ``model``."""
        return model in self._costs

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate Anthropic completion cost, including Haiku 5.5 long-prompt pricing."""
        input_cost, output_cost = self._costs.get(model, (0.0, 0.0))
        long_prompt_pricing = self._long_prompt_pricing.get(model)
        if long_prompt_pricing is not None:
            input_cost, output_cost = long_prompt_pricing.rates_for(
                input_tokens, (input_cost, output_cost)
            )
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether Anthropic credentials are configured."""
        return self._api_key is not None
