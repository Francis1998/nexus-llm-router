"""Google Gemini provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import json_object, nested_int
from router.schemas import ChatMessage, ProviderResponse


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
        self._costs: dict[str, tuple[float, float]] = {
            "gemini-3.1-pro-preview": (0.0035, 0.0105),
            "gemini-3.5-flash": (0.0015, 0.009),
        }

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a normalized Gemini completion."""
        if not self._api_key:
            raise ProviderError("GOOGLE_API_KEY is not configured")
        payload = {
            "contents": [
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
                for message in messages
            ],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self._api_key}"
        )
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(url, json=payload)
        if response.status_code >= 400:
            raise ProviderError(f"google request failed with status {response.status_code}")
        body = json_object(response)
        candidates = body.get("candidates", [])
        content = ""
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            candidate_content = candidates[0].get("content", {})
            if isinstance(candidate_content, dict):
                parts = candidate_content.get("parts", [])
                if isinstance(parts, list) and parts and isinstance(parts[0], dict):
                    raw_text = parts[0].get("text", "")
                    content = raw_text if isinstance(raw_text, str) else ""
        input_tokens = nested_int(body, ["usageMetadata", "promptTokenCount"])
        output_tokens = nested_int(body, ["usageMetadata", "candidatesTokenCount"])
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
        """Stream normalized Gemini text chunks."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate Gemini completion cost."""
        input_cost, output_cost = self._costs.get(model, (0.0, 0.0))
        return (input_tokens / 1000.0) * input_cost + (output_tokens / 1000.0) * output_cost

    async def health_check(self) -> bool:
        """Return whether Google credentials are configured."""
        return self._api_key is not None
