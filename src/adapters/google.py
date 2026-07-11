"""Google Gemini provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import json_object, nested_int
from router.schemas import ChatMessage, ProviderResponse


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
        payload = _build_gemini_payload(messages, max_tokens)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self._api_key}"
        )
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(url, json=payload)
        if response.status_code >= 400:
            raise ProviderError(f"google request failed with status {response.status_code}")
        body = json_object(response)
        content = _extract_gemini_text(body.get("candidates", []))
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


def _build_gemini_payload(messages: list[ChatMessage], max_tokens: int) -> dict[str, object]:
    """Build a Gemini generateContent payload from normalized chat messages."""
    system_text = "\n\n".join(message.content for message in messages if message.role == "system")
    payload: dict[str, object] = {
        "contents": [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    return payload
