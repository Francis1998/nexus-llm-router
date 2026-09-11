"""Sticky prompt-prefix → model affinity table for provider cache hits.

Distinct from the ``prompt-prefix-cache`` *routing strategy*, which
deterministically hashes a long system-prompt prefix onto a catalog bucket.
``PromptCacheAffinityRouter`` is an explicit remember/choose table that
callers wire *after* observing a provider prompt-cache hit (or other
cache-friendly outcome), so subsequent turns with the same prefix stay on
the same ``model_id``.

Also distinct from ``cache-hit-sticky-warm-pool``, ``semantic-cache-ttl-affinity``,
and ``prompt-caching-prefer`` strategies: this module does not participate in
``build_strategies``; it is a reusable library callers compose around the
engine.

Closes the LiteLLM / OpenRouter gap where process-local sticky affinity after
an observed cache hit is left to application glue, for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence


def fingerprint(prompt: str, *, chars: int = 256) -> str:
    """Return a stable SHA-256 hex digest of the first ``chars`` of ``prompt``.

    Args:
        prompt: Full prompt text (system + user, or any caller-defined slice).
        chars: Number of leading characters to hash (must be ``>= 1``).

    Returns:
        Lowercase hex SHA-256 digest of ``prompt[:chars]`` encoded as UTF-8.

    Raises:
        ValueError: If ``chars`` is less than 1.
    """
    if chars < 1:
        raise ValueError(f"chars must be >= 1, got {chars}")
    prefix = prompt[:chars]
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


class PromptCacheAffinityRouter:
    """Map a stable prompt-prefix fingerprint → preferred ``model_id``.

    Callers typically:

    1. Compute ``fingerprint(prompt)`` (or supply their own digest).
    2. On a cache miss / first turn, route normally, then ``remember`` the
       chosen model when a provider cache hit (or warm-cache signal) is
       observed.
    3. On later turns, ``choose`` among healthy candidates; if the sticky
       model is still eligible, reuse it to maximize provider KV-cache hits.

    Thread-safe for concurrent remember/choose/snapshot from request workers.
    """

    def __init__(self) -> None:
        """Initialize an empty affinity table."""
        self._affinity: dict[str, str] = {}
        self._lock = threading.Lock()

    def remember(self, prefix_fingerprint: str, model_id: str) -> None:
        """Pin ``model_id`` as the preferred model for ``prefix_fingerprint``.

        Args:
            prefix_fingerprint: Stable prefix digest (for example from
                :func:`fingerprint`).
            model_id: Model identifier to stick to (for example ``gpt-5.5``).

        Raises:
            ValueError: If either argument is empty.
        """
        if not prefix_fingerprint:
            raise ValueError("prefix_fingerprint must be non-empty")
        if not model_id:
            raise ValueError("model_id must be non-empty")
        with self._lock:
            self._affinity[prefix_fingerprint] = model_id

    def choose(
        self,
        prefix_fingerprint: str,
        candidates: Sequence[str],
    ) -> str | None:
        """Return the sticky model if still present in ``candidates``.

        Args:
            prefix_fingerprint: Stable prefix digest previously remembered.
            candidates: Currently eligible model ids (order ignored).

        Returns:
            The remembered ``model_id`` when it is in ``candidates``, else
            ``None`` (unknown fingerprint, empty candidates, or sticky model
            no longer eligible).

        Raises:
            ValueError: If ``prefix_fingerprint`` is empty.
        """
        if not prefix_fingerprint:
            raise ValueError("prefix_fingerprint must be non-empty")
        with self._lock:
            preferred = self._affinity.get(prefix_fingerprint)
        if preferred is None:
            return None
        if preferred in candidates:
            return preferred
        return None

    def snapshot(self) -> Mapping[str, str]:
        """Return a copy of the fingerprint → model affinity table."""
        with self._lock:
            return dict(self._affinity)

    def forget(self, prefix_fingerprint: str) -> bool:
        """Remove one affinity entry.

        Args:
            prefix_fingerprint: Digest to clear.

        Returns:
            ``True`` if an entry was removed, else ``False``.

        Raises:
            ValueError: If ``prefix_fingerprint`` is empty.
        """
        if not prefix_fingerprint:
            raise ValueError("prefix_fingerprint must be non-empty")
        with self._lock:
            return self._affinity.pop(prefix_fingerprint, None) is not None

    def clear(self) -> None:
        """Remove all affinity entries."""
        with self._lock:
            self._affinity.clear()

    def __len__(self) -> int:
        """Return the number of remembered prefix affinities."""
        with self._lock:
            return len(self._affinity)
