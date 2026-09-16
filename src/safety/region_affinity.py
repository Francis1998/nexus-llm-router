"""Advisory provider ranking by preferred-region affinity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegionAffinityRank:
    """One provider's ranked region-affinity score."""

    provider: str
    affinity_score: float
    matched_regions: tuple[str, ...]
    provider_regions: tuple[str, ...]
    rank: int
    advisory: str


@dataclass(frozen=True, slots=True)
class RegionAffinityAdvice:
    """Ranked region-affinity advice across candidate providers."""

    preferred_regions: tuple[str, ...]
    rankings: tuple[RegionAffinityRank, ...]
    advisory: str


class ProviderRegionAffinityAdvisor:
    """Rank providers by affinity to a preferred-region list.

    Closes the OpenRouter / LiteLLM / Portkey *region / data-residency
    preference* gap for offline Nexus gateways serving GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2: given ``preferred_regions``
    and a ``provider → regions`` map, return ranked affinity advice so
    callers can prefer nearer / residency-aligned providers.

    Distinct from ``ProviderHealthScoreboard`` (success/error health bands)
    and ``StickyProviderAffinity`` (session_key → provider sticky map).
    This advisor never rejects traffic and never mutates sticky state —
    callers reorder or log from ``rankings``.
    """

    def advise(
        self,
        *,
        preferred_regions: list[str] | tuple[str, ...],
        provider_regions: dict[str, list[str] | tuple[str, ...]],
    ) -> RegionAffinityAdvice:
        """Return providers ranked by region affinity score.

        Affinity scoring (higher is better):

        - Exact match on the first preferred region contributes ``1.0``.
        - Each additional preferred-region overlap contributes
          ``0.5 / max(len(preferred_regions) - 1, 1)``.
        - Providers with no overlap score ``0.0`` and still appear last.

        Ties break by provider id ascending. Rank is 1-based.

        Args:
            preferred_regions: Ordered preference list (non-empty strings).
            provider_regions: Map of provider id → regions it serves.

        Returns:
            Immutable ``RegionAffinityAdvice`` with ranked providers.

        Raises:
            ValueError: If preferred list or provider map entries are empty /
                invalid.
        """
        if not preferred_regions:
            raise ValueError("preferred_regions must be non-empty")
        prefs: list[str] = []
        for region in preferred_regions:
            if not region or not str(region).strip():
                raise ValueError("preferred_regions entries must be non-empty")
            prefs.append(str(region).strip())
        if not provider_regions:
            raise ValueError("provider_regions must be non-empty")

        primary = prefs[0]
        extra_weight = 0.5 / float(max(len(prefs) - 1, 1))
        scored: list[tuple[float, str, tuple[str, ...], tuple[str, ...]]] = []

        for provider, regions in provider_regions.items():
            if not provider or not str(provider).strip():
                raise ValueError("provider ids must be non-empty")
            if not regions:
                raise ValueError(f"provider_regions[{provider!r}] must be non-empty")
            cleaned: list[str] = []
            for region in regions:
                if not region or not str(region).strip():
                    raise ValueError(
                        f"provider_regions[{provider!r}] entries must be non-empty"
                    )
                cleaned.append(str(region).strip())
            region_set = {r.casefold() for r in cleaned}
            pref_cf = [p.casefold() for p in prefs]
            matched = tuple(p for p in prefs if p.casefold() in region_set)
            score = 0.0
            if primary.casefold() in region_set:
                score += 1.0
            extras = sum(1 for p in pref_cf[1:] if p in region_set)
            score += extras * extra_weight
            scored.append((score, str(provider).strip(), matched, tuple(cleaned)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        rankings: list[RegionAffinityRank] = []
        for idx, (score, provider, matched, regions_tuple) in enumerate(scored, start=1):
            if score >= 1.0:
                note = f"primary region {primary!r} matched"
            elif score > 0.0:
                note = f"secondary overlap={matched}"
            else:
                note = "no preferred-region overlap"
            rankings.append(
                RegionAffinityRank(
                    provider=provider,
                    affinity_score=float(score),
                    matched_regions=matched,
                    provider_regions=regions_tuple,
                    rank=idx,
                    advisory=f"rank={idx} score={score:.3f}: {note}",
                )
            )

        top = rankings[0]
        summary = (
            f"ranked {len(rankings)} providers; top={top.provider} "
            f"score={top.affinity_score:.3f} preferred={prefs}"
        )
        return RegionAffinityAdvice(
            preferred_regions=tuple(prefs),
            rankings=tuple(rankings),
            advisory=summary,
        )
