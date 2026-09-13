"""Session-key → provider sticky affinity for conversation continuity."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StickyProviderBinding:
    """One session → provider sticky mapping."""

    session_key: str
    provider: str


class StickyProviderAffinity:
    """Pin a provider to a conversation/session key for continuity.

    Closes the LiteLLM / Portkey / OpenRouter sticky-provider gap for offline
    Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    after the first successful provider selection, ``bind(session_key,
    provider)`` so subsequent turns ``get(session_key)`` the same provider
    (prompt-cache warmth, tool-state continuity, billing consistency).

    Distinct from ``ProviderErrorBudgetShed`` (sliding-window error % shedding)
    and from sticky *routing strategies* in ``strategies.py`` — this is a
    reusable in-memory affinity map with explicit ``get`` / ``bind`` /
    ``clear`` for gateway or middleware callers.
    """

    def __init__(self) -> None:
        """Initialize an empty session → provider map."""
        self._lock = threading.Lock()
        self._bindings: dict[str, str] = {}

    def bind(self, session_key: str, provider: str) -> StickyProviderBinding:
        """Create or overwrite the sticky provider for ``session_key``.

        Args:
            session_key: Conversation / session identifier (non-empty).
            provider: Provider id to pin (non-empty), e.g. ``openai``.

        Returns:
            The stored ``StickyProviderBinding``.

        Raises:
            ValueError: If ``session_key`` or ``provider`` is empty.
        """
        if not session_key:
            raise ValueError("session_key must be non-empty")
        if not provider:
            raise ValueError("provider must be non-empty")
        with self._lock:
            self._bindings[session_key] = provider
        return StickyProviderBinding(session_key=session_key, provider=provider)

    def get(self, session_key: str) -> str | None:
        """Return the sticky provider for ``session_key``, or ``None``.

        Args:
            session_key: Conversation / session identifier (non-empty).

        Returns:
            Provider id when bound, else ``None``.

        Raises:
            ValueError: If ``session_key`` is empty.
        """
        if not session_key:
            raise ValueError("session_key must be non-empty")
        with self._lock:
            return self._bindings.get(session_key)

    def clear(self, session_key: str) -> bool:
        """Remove the sticky binding for ``session_key``.

        Args:
            session_key: Conversation / session identifier (non-empty).

        Returns:
            ``True`` if a binding was removed, ``False`` if unknown.

        Raises:
            ValueError: If ``session_key`` is empty.
        """
        if not session_key:
            raise ValueError("session_key must be non-empty")
        with self._lock:
            return self._bindings.pop(session_key, None) is not None

    def sessions(self) -> list[str]:
        """Return sorted session keys currently sticky-bound."""
        with self._lock:
            return sorted(self._bindings)
