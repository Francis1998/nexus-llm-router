"""Base provider adapter interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from router.schemas import ChatMessage, ProviderResponse, ToolConfig


class ProviderError(RuntimeError):
    """Raised when a provider request fails.

    ``status_code`` is the provider's HTTP status when it answered with an error,
    or ``None`` when no response was received (missing credentials, network).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable failure description.
            status_code: Provider HTTP status, when the provider answered.
        """
        super().__init__(message)
        self.status_code = status_code


class BaseProviderAdapter(ABC):
    """Provider adapter contract for model completion backends.

    ``tool_config`` is passed only for requests that offer tools (as a keyword
    argument), so adapters written before tool calling keep serving tool-free
    requests. Adapters translate it, assistant ``tool_calls``, and ``tool``
    results to their provider's function-calling format and report the model's
    calls in ``ProviderResponse.tool_calls``. Adapters never execute tools.
    """

    provider_name: str

    async def _complete_once(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None,
    ) -> ProviderResponse:
        """Call :meth:`complete`, passing ``tool_config`` only when the request has tools.

        ``stream`` implementations that wrap ``complete`` use this so subclasses
        overriding ``complete`` without ``tool_config`` keep working.
        """
        if tool_config is None:
            return await self.complete(model, messages, max_tokens)
        return await self.complete(model, messages, max_tokens, tool_config=tool_config)

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> ProviderResponse:
        """Return a non-streaming completion.

        Args:
            model: Provider model name.
            messages: Chat messages.
            max_tokens: Maximum completion tokens.
            tool_config: Tools offered to the model, when the request has any.

        Returns:
            Normalized provider response.
        """

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        tool_config: ToolConfig | None = None,
    ) -> AsyncIterator[str | ProviderResponse]:
        """Stream completion chunks.

        Args:
            model: Provider model name.
            messages: Chat messages.
            max_tokens: Maximum completion tokens.
            tool_config: Tools offered to the model, when the request has any.

        Yields:
            Text chunks from the provider, optionally followed by one final
            ``ProviderResponse`` carrying provider-reported usage, cost, and stop
            reason. Streams that end without it are accounted from estimates.
            Tool calls arrive only in that final ``ProviderResponse``.
        """

    @abstractmethod
    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int, *, at: datetime | None = None
    ) -> float:
        """Estimate completion cost.

        Args:
            model: Provider model name.
            input_tokens: Estimated input tokens.
            output_tokens: Estimated output tokens.
            at: Instant whose list prices apply: a ``router.config.PRICE_SCHEDULE``
                change in effect then replaces the static price. Defaults to now.

        Returns:
            Estimated cost in USD.
        """

    def has_price(self, model: str) -> bool:
        """Return whether ``estimate_cost`` knows the price of ``model``.

        Adapters with a static price table return ``False`` for models missing
        from it, for which ``estimate_cost`` returns 0.0; the router then bills
        the model's catalog prices instead. The default trusts ``estimate_cost``.

        Args:
            model: Provider model name.

        Returns:
            ``True`` when the adapter prices ``model``.
        """
        del model
        return True

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider availability.

        Returns:
            True when the provider is available.
        """
