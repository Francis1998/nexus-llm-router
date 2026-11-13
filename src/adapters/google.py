"""Google Gemini provider adapter."""

from collections.abc import AsyncIterator

import httpx

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.http_utils import finish_reason, json_object, nested_int
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
        # USD per 1K tokens (input, output) as of 2026-11-06. Gemini 4 Argon rises
        # to (0.004, 0.020) on 2026-12-01; Gemini 3.8 Flash to (0.0015, 0.0075)
        # on 2027-01-01.
        self._costs: dict[str, tuple[float, float]] = {
            "gemini-4-argon": (0.002, 0.010),
            "gemini-3.8-flash": (0.00075, 0.00375),
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
            finish_reason=finish_reason(stop_reason, {"MAX_TOKENS"}),
        )

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str | ProviderResponse]:
        """Stream normalized Gemini text chunks, then the final response."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content
        yield response

    def has_price(self, model: str) -> bool:
        """Return whether the Gemini price table lists ``model``."""
        return model in self._costs

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
