"""Base provider adapter interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from router.schemas import ChatMessage, ProviderResponse


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


class BaseProviderAdapter(ABC):
    """Provider adapter contract for model completion backends."""

    provider_name: str

    @abstractmethod
    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        """Return a non-streaming completion.

        Args:
            model: Provider model name.
            messages: Chat messages.
            max_tokens: Maximum completion tokens.

        Returns:
            Normalized provider response.
        """

    @abstractmethod
    def stream(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream completion chunks.

        Args:
            model: Provider model name.
            messages: Chat messages.
            max_tokens: Maximum completion tokens.

        Yields:
            Text chunks from the provider.
        """

    @abstractmethod
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate completion cost.

        Args:
            model: Provider model name.
            input_tokens: Estimated input tokens.
            output_tokens: Estimated output tokens.

        Returns:
            Estimated cost in USD.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider availability.

        Returns:
            True when the provider is available.
        """
