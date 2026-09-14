"""Ordered provider fallback chain planner (preference-based, offline)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FallbackChainPlan:
    """Structured ordered fallback plan for one routing attempt."""

    primary: str
    chain: list[str]
    skipped_unavailable: list[str]
    rationale: str

    def ordered(self) -> list[str]:
        """Return ``[primary, *chain]`` as a concrete attempt list."""
        return [self.primary, *self.chain]


class ProviderFallbackChainPlanner:
    """Plan an ordered provider fallback chain from explicit preferences.

    Closes the LiteLLM / OpenRouter *static preference-chain* gap for Nexus
    gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    operators need a deterministic planner that orders providers by declared
    preference, skips unavailable ones, and optionally anchors on an explicit
    primary — without requiring live health observations.

    Distinct from ``ProviderFallbackScoreboard`` (outcome-based health ranking
    via ``record_outcome`` / ``rank()``). This planner never mutates state and
    never observes latency/error rates — it only orders candidates.
    """

    def __init__(self, *, preference_order: Sequence[str] | None = None) -> None:
        """Initialize with an optional default preference order.

        Args:
            preference_order: Preferred provider ids (earlier = higher priority).
                Empty strings are rejected. When omitted, candidate input order
                is preserved (after optional primary anchoring).

        Raises:
            ValueError: If any preference entry is empty.
        """
        order = tuple(preference_order or ())
        if any(not item for item in order):
            raise ValueError("preference_order entries must be non-empty")
        self._preference_order = order

    @property
    def preference_order(self) -> tuple[str, ...]:
        """Return the configured default preference order."""
        return self._preference_order

    def plan(
        self,
        *,
        candidates: Sequence[str],
        unavailable: set[str] | frozenset[str] | None = None,
        primary: str | None = None,
        preference_order: Sequence[str] | None = None,
    ) -> FallbackChainPlan:
        """Build an ordered fallback plan from candidates and preferences.

        Args:
            candidates: Provider ids eligible for this request (non-empty).
            unavailable: Providers to skip (open circuits, missing adapters).
            primary: Optional forced primary; must appear in available
                candidates when set.
            preference_order: Optional per-call override of the default order.

        Returns:
            Immutable ``FallbackChainPlan`` with primary, remaining chain,
            skipped unavailable providers, and a human-readable rationale.

        Raises:
            ValueError: If candidates are empty, primary is invalid, or no
                providers remain available.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if primary is not None and not primary:
            raise ValueError("primary must be non-empty when provided")

        blocked = set(unavailable or ())
        # Preserve first-seen candidate order for unknowns / ties.
        seen: set[str] = set()
        unique_candidates: list[str] = []
        for name in candidates:
            if not name:
                raise ValueError("candidate provider ids must be non-empty")
            if name in seen:
                continue
            seen.add(name)
            unique_candidates.append(name)

        skipped = [name for name in unique_candidates if name in blocked]
        available = [name for name in unique_candidates if name not in blocked]
        if not available:
            raise ValueError("no available providers after applying unavailable set")

        prefs = tuple(preference_order) if preference_order is not None else self._preference_order
        if any(not item for item in prefs):
            raise ValueError("preference_order entries must be non-empty")

        rank: Mapping[str, int] = {name: index for index, name in enumerate(prefs)}
        preferred = [name for name in prefs if name in available]
        unknown = [name for name in available if name not in rank]
        ordered = preferred + unknown

        if primary is not None:
            if primary not in available:
                raise ValueError("primary must be an available candidate")
            ordered = [primary, *[name for name in ordered if name != primary]]

        head, *tail = ordered
        rationale = (
            f"primary={head}; chain={list(tail)}; "
            f"preference_order={list(prefs)}; skipped_unavailable={skipped}"
        )
        return FallbackChainPlan(
            primary=head,
            chain=list(tail),
            skipped_unavailable=list(skipped),
            rationale=rationale,
        )
