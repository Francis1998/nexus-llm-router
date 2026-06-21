"""Deterministic provider adapter for tests and demos."""

from collections.abc import AsyncIterator

from adapters.base import BaseProviderAdapter, ProviderError
from router.schemas import ChatMessage, ProviderResponse


class MockProviderAdapter(BaseProviderAdapter):
    """In-memory provider adapter with configurable failures."""

    def __init__(self, provider_name: str, fail: bool = False) -> None:
        """Initialize mock adapter.

        Args:
            provider_name: Provider name returned by the adapter.
            fail: Whether calls should fail.
        """
        self.provider_name = provider_name
        self._fail = fail

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a deterministic mock completion."""
        if self._fail:
            raise ProviderError(f"mock provider failed: {self.provider_name}")
        prompt = "\n".join(message.content for message in messages)
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = min(max_tokens, 16)
        return ProviderResponse(
            content=f"[{model}] routed response",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self.estimate_cost(model, input_tokens, output_tokens),
        )

    async def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream deterministic mock chunks."""
        response = await self.complete(model, messages, max_tokens)
        yield response.content

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate mock completion cost."""
        return ((input_tokens + output_tokens) / 1000.0) * 0.001

    async def health_check(self) -> bool:
        """Return mock provider health."""
        return not self._fail
