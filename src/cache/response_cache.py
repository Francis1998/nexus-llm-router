"""In-memory exact-match response cache with TTL and optional tenant namespaces."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _CacheEntry:
    """Stored cache value with absolute expiry time."""

    value: object
    expires_at: float


class ResponseCache:
    """Exact-match response cache keyed by every generation-affecting input.

    Keys cover model, messages (including assistant tool calls and tool
    results), offered tools and ``tool_choice``, temperature, ``max_tokens``,
    and the routing strategy, and may be optionally namespaced by tenant so
    multi-tenant gateways do not share completions across customers. Entries
    expire after ``ttl_seconds``.
    """

    def __init__(self, ttl_seconds: float = 300.0, *, enabled: bool = True) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for each entry in seconds.
            enabled: When False, get always misses and set is a no-op.
        """
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        self._ttl_seconds = float(ttl_seconds)
        self._enabled = enabled
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """Return whether the cache accepts lookups and writes."""
        return self._enabled

    @property
    def ttl_seconds(self) -> float:
        """Return the configured TTL in seconds."""
        return self._ttl_seconds

    @staticmethod
    def make_key(
        model: str | None,
        messages: Sequence[object],
        temperature: float,
        tenant: str | None = None,
        *,
        max_tokens: int | None = None,
        strategy: str | None = None,
        tools: Mapping[str, Any] | None = None,
    ) -> str:
        """Build a stable SHA-256 cache key.

        Args:
            model: Requested model id (or None when the router will choose).
            messages: Chat messages as dicts or objects with role/content and,
                for tool calling, ``tool_calls`` / ``tool_call_id``.
            temperature: Sampling temperature included in the fingerprint.
            tenant: Optional tenant namespace for isolation.
            max_tokens: Output token limit; a completion truncated at a small
                limit must not be replayed for a request allowing more tokens.
            strategy: Effective routing strategy; different strategies may pick
                different models for the same prompt.
            tools: Canonical tool settings (``ToolConfig.canonical()``): a
                completion produced for one set of tools or ``tool_choice``
                must not be replayed for another.

        Returns:
            Hex digest cache key. Requests without tools keep the keys earlier
            builds produced.
        """
        normalized_messages: list[dict[str, object]] = []
        for message in messages:
            fields = message if isinstance(message, dict) else None
            role = str(_field(message, fields, "role", "user"))
            content = str(_field(message, fields, "content", ""))
            normalized: dict[str, object] = {"role": role, "content": content}
            tool_calls = _field(message, fields, "tool_calls", None)
            if isinstance(tool_calls, Sequence) and tool_calls:
                normalized["tool_calls"] = [_tool_call_fingerprint(call) for call in tool_calls]
            tool_call_id = _field(message, fields, "tool_call_id", None)
            if tool_call_id is not None:
                normalized["tool_call_id"] = str(tool_call_id)
            normalized_messages.append(normalized)
        payload: dict[str, object] = {
            "tenant": tenant or "",
            "model": model or "",
            "messages": normalized_messages,
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "strategy": strategy or "",
        }
        if tools is not None:
            payload["tools"] = dict(tools)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> object | None:
        """Return a cached value when present and unexpired.

        Args:
            key: Cache key from :meth:`make_key`.

        Returns:
            Cached value, or None on miss / disabled / expiry.
        """
        if not self._enabled:
            self.misses += 1
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def set(self, key: str, value: object) -> None:
        """Store a value under ``key`` with the configured TTL.

        Args:
            key: Cache key from :meth:`make_key`.
            value: Value to store (typically a RouterResponse).
        """
        if not self._enabled:
            return
        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)

    def clear(self) -> None:
        """Drop all entries and reset hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        """Return the number of in-memory entries (including expired)."""
        with self._lock:
            return len(self._entries)


def _field(message: object, fields: dict[Any, Any] | None, name: str, default: object) -> object:
    """Read ``name`` from a message given as a dict or as an object."""
    if fields is not None:
        return fields.get(name, default)
    return getattr(message, name, default)


def _tool_call_fingerprint(call: object) -> object:
    """Return a JSON-serializable form of one assistant tool call."""
    model_dump = getattr(call, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return call
