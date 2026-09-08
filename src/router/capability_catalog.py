"""Mutable, refreshable model capability catalog for routing strategies."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping

from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)

# Seed capabilities for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
# Mirrors the static ``_KNOWN_MODEL_CAPABILITIES`` entries strategies use today,
# but lives in a mutable catalog that can be refreshed without process restart.
_STATIC_SEED: dict[str, frozenset[str]] = {
    OPENAI_FRONTIER_MODEL: frozenset(
        {
            "agents_api",
            "audio",
            "background_mode",
            "deep_research",
            "json",
            "logprobs",
            "long_context",
            "mcp",
            "memory_tool",
            "parallel_tool_calls",
            "realtime_api",
            "streaming",
            "tools",
            "vision",
            "webhook",
        }
    ),
    ANTHROPIC_SAFETY_MODEL: frozenset(
        {
            "deep_research",
            "json",
            "long_context",
            "mcp",
            "memory_tool",
            "parallel_tool_calls",
            "streaming",
            "tools",
            "vision",
            "webhook",
        }
    ),
    GEMINI_FLASH_MODEL: frozenset({"vision", "tools", "json", "streaming", "audio"}),
    MOONSHOT_BALANCED_MODEL: frozenset({"tools", "long_context", "json", "streaming"}),
}


class ModelCapabilityCatalog:
    """Thread-safe mutable catalog of model capability frozensets.

    Strategies historically read a module-level ``_KNOWN_MODEL_CAPABILITIES``
    dict that cannot be refreshed without a restart. This catalog seeds from
    known models (``gpt-5.5``, ``claude-sonnet-4-6``, ``gemini-3.5-flash``,
    ``kimi-k2``), supports live ``upsert``, and can ``refresh_from_static``
    to restore the seed map.
    """

    def __init__(
        self,
        seed: Mapping[str, Iterable[str]] | None = None,
        *,
        auto_seed: bool = True,
    ) -> None:
        """Initialize the catalog.

        Args:
            seed: Optional initial mapping of model → capabilities.
            auto_seed: When ``True`` and ``seed`` is ``None``, load the static
                known-model seed. When ``False`` and ``seed`` is ``None``,
                start empty.
        """
        self._lock = threading.RLock()
        self._capabilities: dict[str, frozenset[str]] = {}
        if seed is not None:
            with self._lock:
                for model, caps in seed.items():
                    self._capabilities[str(model)] = frozenset(caps)
        elif auto_seed:
            self.refresh_from_static()

    def get(self, model: str) -> frozenset[str] | None:
        """Return the capability set for ``model``, or ``None`` if unknown."""
        with self._lock:
            caps = self._capabilities.get(model)
            return None if caps is None else frozenset(caps)

    def upsert(self, model: str, capabilities: Iterable[str]) -> None:
        """Insert or replace the capability set for ``model``.

        Args:
            model: Model identifier (for example ``gpt-5.5``).
            capabilities: Capability names to store as a frozenset.
        """
        if not model:
            raise ValueError("model must be non-empty")
        frozen = frozenset(capabilities)
        with self._lock:
            self._capabilities[model] = frozen

    def refresh_from_static(self) -> int:
        """Replace the catalog contents with the static known-model seed.

        Returns:
            Number of models loaded from the static seed.
        """
        with self._lock:
            self._capabilities = {model: frozenset(caps) for model, caps in _STATIC_SEED.items()}
            return len(self._capabilities)

    def snapshot(self) -> dict[str, frozenset[str]]:
        """Return a shallow copy of the current model → capabilities map.

        Strategies can hold the returned dict (or pass it as
        ``capability_map``) without racing subsequent catalog mutations.
        """
        with self._lock:
            return {model: frozenset(caps) for model, caps in self._capabilities.items()}

    def __len__(self) -> int:
        """Return how many models are currently catalogued."""
        with self._lock:
            return len(self._capabilities)

    def __contains__(self, model: object) -> bool:
        """Return whether ``model`` is present in the catalog."""
        if not isinstance(model, str):
            return False
        with self._lock:
            return model in self._capabilities
