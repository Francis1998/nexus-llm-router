"""Response cache TTL advisor.

Advises short/medium/long cache TTL bands from request class. Closes the
Helicone / Portkey / LiteLLM cache-TTL gap. Distinct from
``PromptCacheHitAdvisor`` (prefix cache hits) and
``CostAttributionTagLedger`` (cost tags). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

_ALLOWED = frozenset({"ephemeral", "session", "stable", "static"})


@dataclass(frozen=True, slots=True)
class CacheTtlAdvice:
    """TTL advice for one request class."""

    request_class: str
    ttl_seconds: int
    band: str


class ResponseCacheTtlAdvisor:
    """Advise response cache TTL from request class."""

    def advise(self, *, request_class: str) -> CacheTtlAdvice:
        """Return TTL band for ``request_class``.

        Args:
            request_class: One of ``ephemeral`` / ``session`` / ``stable`` /
                ``static``.

        Returns:
            CacheTtlAdvice with band ``short`` / ``medium`` / ``long``.
        """

        klass = request_class.strip().lower()
        if klass not in _ALLOWED:
            raise ValueError("request_class must be one of ephemeral/session/stable/static")

        if klass == "ephemeral":
            return CacheTtlAdvice(request_class=klass, ttl_seconds=30, band="short")
        if klass == "session":
            return CacheTtlAdvice(request_class=klass, ttl_seconds=300, band="short")
        if klass == "stable":
            return CacheTtlAdvice(request_class=klass, ttl_seconds=3_600, band="medium")
        return CacheTtlAdvice(request_class=klass, ttl_seconds=86_400, band="long")
