"""Cost attribution tag ledger (Helicone-style tags).

Records offline cost attribution tags per request for later rollups. Closes
the Helicone / Portkey / LiteLLM cost-tag gap. Distinct from
``SpendLedgerGuard`` (hard spend caps) and ``TenantSpendQuotaGuard`` (tenant
quotas). Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostAttributionEntry:
    """One attributed cost record."""

    request_id: str
    tags: tuple[str, ...]
    cost_usd: float


@dataclass(frozen=True, slots=True)
class CostAttributionRollup:
    """Rollup for one tag."""

    tag: str
    request_count: int
    total_cost_usd: float


class CostAttributionTagLedger:
    """Accumulate tagged request costs and roll up by tag."""

    def __init__(self) -> None:
        self._entries: list[CostAttributionEntry] = []

    def record(
        self,
        request_id: str,
        *,
        tags: tuple[str, ...] | list[str],
        cost_usd: float,
    ) -> CostAttributionEntry:
        """Record cost for ``request_id`` under ``tags``."""

        if not request_id:
            raise ValueError("request_id must be non-empty")
        clean = tuple(sorted({t.strip() for t in tags if t.strip()}))
        if not clean:
            raise ValueError("tags must be non-empty")
        if cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        entry = CostAttributionEntry(
            request_id=request_id,
            tags=clean,
            cost_usd=float(cost_usd),
        )
        self._entries.append(entry)
        return entry

    def rollup(self) -> tuple[CostAttributionRollup, ...]:
        """Return per-tag rollups sorted by tag name."""

        totals: dict[str, list[float]] = {}
        for entry in self._entries:
            for tag in entry.tags:
                totals.setdefault(tag, []).append(entry.cost_usd)
        return tuple(
            CostAttributionRollup(
                tag=tag,
                request_count=len(costs),
                total_cost_usd=round(sum(costs), 6),
            )
            for tag, costs in sorted(totals.items())
        )
