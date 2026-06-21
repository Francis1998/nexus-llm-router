"""Provider adapter registry."""

from adapters.anthropic import AnthropicAdapter
from adapters.base import BaseProviderAdapter
from adapters.google import GoogleGeminiAdapter
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from router.config import ProviderSettings


class AdapterRegistry:
    """Lookup table for provider adapters."""

    def __init__(self, adapters: dict[str, BaseProviderAdapter]) -> None:
        """Initialize registry.

        Args:
            adapters: Provider adapters keyed by provider name.
        """
        self._adapters = adapters

    def get(self, provider: str) -> BaseProviderAdapter:
        """Return an adapter by provider name.

        Args:
            provider: Provider name.

        Returns:
            Provider adapter.

        Raises:
            KeyError: If no adapter exists for the provider.
        """
        return self._adapters[provider]

    def items(self) -> list[tuple[str, BaseProviderAdapter]]:
        """Return registered provider adapter entries.

        Returns:
            Provider name and adapter tuples.
        """
        return list(self._adapters.items())


def build_adapter_registry(provider_settings: ProviderSettings) -> AdapterRegistry:
    """Build the default provider adapter registry.

    Args:
        provider_settings: Provider API settings.

    Returns:
        Adapter registry.
    """
    timeout_seconds = provider_settings.request_timeout_seconds
    return AdapterRegistry(
        {
            "openai": OpenAIAdapter(provider_settings.openai_api_key, timeout_seconds),
            "anthropic": AnthropicAdapter(provider_settings.anthropic_api_key, timeout_seconds),
            "google": GoogleGeminiAdapter(provider_settings.google_api_key, timeout_seconds),
            "moonshot": MoonshotAdapter(
                provider_settings.moonshot_api_key,
                provider_settings.moonshot_base_url,
                timeout_seconds,
            ),
        },
    )
