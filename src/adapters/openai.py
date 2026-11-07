"""OpenAI provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import json_object, message_text, nested_int
from router.schemas import ChatMessage, ProviderResponse

_LEGACY_MAX_TOKENS_PREFIXES = ("gpt-4", "gpt-3.5")


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
        # USD per 1K tokens (input, output); GPT-6 list prices as of 2026-11-06.
        self._costs: dict[str, tuple[float, float]] = {
            "gpt-6-astra": (0.010, 0.050),
            "gpt-6.1-sol": (0.002, 0.010),
            "gpt-6.1-terra": (0.001, 0.005),
            "gpt-6-luna": (0.0001, 0.0005),
            "gpt-5.5": (0.006, 0.018),
            "gpt-4.1-mini": (0.0002, 0.0008),
        }

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a normalized OpenAI completion."""
        if not self._api_key:
            raise ProviderError("OPENAI_API_KEY is not configured")
        payload: dict[str, object] = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
        }
        # Reasoning-era models (GPT-5.x, GPT-6, o-series) reject ``max_tokens``
        # and require ``max_completion_tokens``. Only legacy GPT-4.x / GPT-3.5
        # SKUs still expect ``max_tokens``.
        if model.startswith(_LEGACY_MAX_TOKENS_PREFIXES):
            payload["max_tokens"] = max_tokens
        else:
            payload["max_completion_tokens"] = max_tokens
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
