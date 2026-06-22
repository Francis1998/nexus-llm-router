"""Anthropic provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import json_object, nested_int
from router.schemas import ChatMessage, ProviderResponse


def _build_anthropic_payload(
    model: str, messages: list[ChatMessage], max_tokens: int
) -> dict[str, object]:
    """Build Anthropic Messages API payload with system text extracted."""
    system_parts = [message.content for message in messages if message.role == "system"]
    conversation_messages = [
        {"role": message.role, "content": message.content}
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
        self._costs: dict[str, tuple[float, float]] = {
            "claude-sonnet-4-6": (0.003, 0.015),
            "claude-haiku-4-5": (0.0008, 0.004),
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
            raise ProviderError(f"anthropic request failed with status {response.status_code}")
        body = json_object(response)
        content_blocks = body.get("content", [])
        content = ""
        if (
            isinstance(content_blocks, list)
            and content_blocks
            and isinstance(content_blocks[0], dict)
        ):
            raw_text = content_blocks[0].get("text", "")
            content = raw_text if isinstance(raw_text, str) else ""
        input_tokens = nested_int(body, ["usage", "input_tokens"])
        output_tokens = nested_int(body, ["usage", "output_tokens"])
        return ProviderResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
        )

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream normalized Anthropic text chunks."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate Anthropic completion cost."""
        input_cost, output_cost = self._costs.get(model, (0.0, 0.0))
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether Anthropic credentials are configured."""
        return self._api_key is not None
