"""Moonshot Kimi OpenAI-compatible provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import json_object, message_text, nested_int
from router.schemas import ChatMessage, ProviderResponse


class MoonshotAdapter(BaseProviderAdapter):
    """Adapter for Moonshot Kimi through an OpenAI-compatible API."""

    provider_name = "moonshot"

    def __init__(self, api_key: str | None, base_url: str, timeout_seconds: float) -> None:
        """Initialize the Moonshot adapter.

        Args:
            api_key: Moonshot API key.
            base_url: OpenAI-compatible base URL.
            timeout_seconds: Provider request timeout.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._costs: dict[str, tuple[float, float]] = {"kimi-k2": (0.0005, 0.002)}

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a normalized Moonshot completion."""
        if not self._api_key:
            raise ProviderError("MOONSHOT_API_KEY is not configured")
        payload = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
        if response.status_code >= 400:
            raise ProviderError(f"moonshot request failed with status {response.status_code}")
        body = json_object(response)
        choices = body.get("choices", [])
        content = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            content = message_text(choices[0].get("message"))
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
        """Stream normalized Moonshot text chunks."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate Moonshot completion cost."""
        input_cost, output_cost = self._costs.get(model, (0.0, 0.0))
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether Moonshot credentials are configured."""
        return self._api_key is not None
