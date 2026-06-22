"""OpenAI provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import json_object, nested_int
from router.schemas import ChatMessage, ProviderResponse


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
        self._costs: dict[str, tuple[float, float]] = {
            "gpt-5.5": (0.006, 0.018),
            "gpt-4.1-mini": (0.0002, 0.0008),
        }

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a normalized OpenAI completion."""
        if not self._api_key:
            raise ProviderError("OPENAI_API_KEY is not configured")
        payload = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
        if response.status_code >= 400:
            raise ProviderError(f"openai request failed with status {response.status_code}")
        body = json_object(response)
        choices = body.get("choices", [])
        content = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                raw_content = message.get("content", "")
                content = raw_content if isinstance(raw_content, str) else ""
        input_tokens = nested_int(body, ["usage", "prompt_tokens"])
        output_tokens = nested_int(body, ["usage", "completion_tokens"])
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
        """Stream normalized OpenAI text chunks."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate OpenAI completion cost."""
        input_cost, output_cost = self._costs.get(model, (0.0, 0.0))
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether OpenAI credentials are configured."""
        return self._api_key is not None
