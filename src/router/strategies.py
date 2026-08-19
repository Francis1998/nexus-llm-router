"""Pluggable routing strategies for the decide phase."""

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.schemas import (
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    RoutingDecision,
    RoutingStrategyName,
    TaskSignals,
)


class ProviderHealth(Protocol):
    """Read-only view of provider circuit health used by reliability routing."""

    def is_available(self, provider: str) -> bool:
        """Return whether a provider may currently be routed to.

            Args:
                provider: Provider name.

                token_bucket_tenant_rate: Per-tenant request-token refill rate
                per second for token-bucket-tenant routing.
        Returns:
                True when the provider is routable.
        """


class RoutingStrategy(ABC):
    """Base class for all routing strategies."""

    strategy_name: RoutingStrategyName

    def __init__(self, model_catalog: Mapping[str, ModelCandidate]) -> None:
        """Initialize strategy with a model catalog.

        Args:
            model_catalog: Available model candidates by model name.
        """
        self._model_catalog = model_catalog

    @abstractmethod
    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a model for a request.

        Args:
            request: Router request.
            signals: Observed task signals.

        Returns:
            Routing decision with rationale and fallback chain.
        """

    def _fallback_chain(self, chosen_model: str) -> list[str]:
        """Build a fallback chain excluding the selected model.

        Args:
            chosen_model: Primary selected model.

        Returns:
            Ordered fallback model names.
        """
        candidates = sorted(
            self._model_catalog.values(),
            key=lambda candidate: candidate.quality_score,
            reverse=True,
        )
        return [candidate.model for candidate in candidates if candidate.model != chosen_model][:3]

    def _decision(self, model: str, rationale: str) -> RoutingDecision:
        """Create a routing decision for a model.

        Args:
            model: Selected model name.
            rationale: Human-readable routing rationale.

        Returns:
            Routing decision.
        """
        candidate = self._model_catalog[model]
        return RoutingDecision(
            chosen_model=model,
            provider=candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=self._fallback_chain(model),
        )


class RuleBasedStrategy(RoutingStrategy):
    """Route using an explicit priority matrix."""

    strategy_name = RoutingStrategyName.RULE_BASED

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a model from deterministic domain and complexity rules."""
        if signals.domain_tag is DomainTag.MEDICAL:
            return self._decision(
                ANTHROPIC_SAFETY_MODEL, "medical domain requires highest safety prior"
            )
        if signals.domain_tag is DomainTag.LEGAL:
            return self._decision(
                ANTHROPIC_SAFETY_MODEL, "legal domain favors Claude policy reasoning"
            )
        if signals.domain_tag is DomainTag.CODE and signals.complexity_score >= 0.55:
            return self._decision(
                OPENAI_FRONTIER_MODEL, "complex code prompt favors GPT-5.5 quality"
            )
        if (
            signals.complexity_score <= 0.35
            and signals.latency_requirement is LatencyRequirement.REALTIME
        ):
            return self._decision(GEMINI_FLASH_MODEL, "simple realtime prompt favors low latency")
        requested_model = request.requested_model
        if requested_model and requested_model in self._model_catalog:
            return self._decision(requested_model, "explicit compatible model request honored")
        return self._decision(
            OPENAI_BALANCED_MODEL, "general prompt routed to balanced low-cost model"
        )


class ClassifierStrategy(RoutingStrategy):
    """Route using classifier-derived complexity and domain signals."""

    strategy_name = RoutingStrategyName.CLASSIFIER

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a model based on classifier scores."""
        if signals.complexity_score >= 0.8:
            return self._decision(ANTHROPIC_SAFETY_MODEL, "classifier marked task high complexity")
        if signals.domain_tag is DomainTag.CODE:
            return self._decision(OPENAI_FRONTIER_MODEL, "classifier detected code domain")
        if signals.complexity_score <= 0.4:
            return self._decision(
                MOONSHOT_BALANCED_MODEL, "classifier marked task simple and cost-sensitive"
            )
        return self._decision(OPENAI_BALANCED_MODEL, "classifier selected balanced middle tier")


class CostOptimalStrategy(RoutingStrategy):
    """Minimize cost subject to a configurable quality floor."""

    strategy_name = RoutingStrategyName.COST_OPTIMAL

    def __init__(self, model_catalog: Mapping[str, ModelCandidate], quality_floor: float) -> None:
        """Initialize the cost optimizer.

        Args:
            model_catalog: Available model candidates by model name.
            quality_floor: Minimum acceptable quality score.
        """
        super().__init__(model_catalog)
        self._quality_floor = quality_floor

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Solve the one-hot model assignment LP for minimum cost."""
        feasible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if candidate.quality_score >= self._quality_floor
            and signals.domain_tag in candidate.supports_domains
            and (
                candidate.supports_realtime
                or signals.latency_requirement is LatencyRequirement.BATCH
            )
        ]
        if not feasible_candidates:
            highest_quality = max(
                self._model_catalog.values(),
                key=lambda candidate: candidate.quality_score,
            )
            return self._decision(
                highest_quality.model,
                "quality floor forced highest-quality fallback",
            )
        selected_candidate = min(
            feasible_candidates,
            key=lambda candidate: candidate.estimate_cost(
                signals.prompt_tokens_estimate,
                request.max_tokens,
            ),
        )
        estimated_cost = selected_candidate.estimate_cost(
            signals.prompt_tokens_estimate,
            request.max_tokens,
        )
        rationale = (
            "LP objective minimized estimated cost "
            f"${estimated_cost:.6f} with quality floor {self._quality_floor:.2f}"
        )
        return self._decision(selected_candidate.model, rationale)


class LatencyStats:
    """Rolling latency summary used by latency-aware routing."""

    def __init__(self) -> None:
        """Initialize empty latency observations."""
        self._observations: dict[str, list[float]] = {}

    def observe(self, provider: str, latency_ms: float) -> None:
        """Record a latency observation.

        Args:
            provider: Provider name.
            latency_ms: Observed latency in milliseconds.
        """
        provider_observations = self._observations.setdefault(provider, [])
        provider_observations.append(latency_ms)
        if len(provider_observations) > 100:
            del provider_observations[0]

    def p95(self, provider: str) -> float:
        """Return rolling p95 latency for a provider.

        Args:
            provider: Provider name.

        Returns:
            Provider p95 latency in milliseconds.
        """
        observations = sorted(self._observations.get(provider, []))
        if not observations:
            return 0.0
        index = min(len(observations) - 1, int(0.95 * (len(observations) - 1)))
        return observations[index]

    def p50(self, provider: str) -> float:
        """Return rolling p50 latency for a provider.

        Args:
            provider: Provider name.

        Returns:
            Provider p50 latency in milliseconds.
        """
        observations = sorted(self._observations.get(provider, []))
        if not observations:
            return 0.0
        index = min(len(observations) - 1, int(0.50 * (len(observations) - 1)))
        return observations[index]

    def p99(self, provider: str) -> float:
        """Return rolling p99 latency for a provider.

        Args:
            provider: Provider name.

        Returns:
            Provider p99 latency in milliseconds.
        """
        observations = sorted(self._observations.get(provider, []))
        if not observations:
            return 0.0
        index = min(len(observations) - 1, int(0.99 * (len(observations) - 1)))
        return observations[index]


class InflightStats:
    """Provider in-flight counters used by least-busy routing."""

    def __init__(self) -> None:
        """Initialize empty in-flight counters."""
        self._inflight: dict[str, int] = {}

    def begin(self, provider: str) -> None:
        """Record that a provider attempt has started.

        Args:
            provider: Provider name.
        """
        self._inflight[provider] = self._inflight.get(provider, 0) + 1

    def finish(self, provider: str) -> None:
        """Record that a provider attempt has finished.

        Args:
            provider: Provider name.
        """
        current = self._inflight.get(provider, 0)
        if current <= 1:
            self._inflight.pop(provider, None)
        else:
            self._inflight[provider] = current - 1

    def load_score(self, provider: str) -> int:
        """Return the provider's current in-flight load score.

        Args:
            provider: Provider name.

        Returns:
            Number of live attempts currently dispatched to the provider.
        """
        return self._inflight.get(provider, 0)

    @staticmethod
    def tenant_provider_key(tenant_key: str, provider: str) -> str:
        """Build a composite InflightStats key for tenant-scoped load.

        Args:
            tenant_key: Tenant or session identity.
            provider: Provider name.

        Returns:
            Composite key used by tenant concurrency lease tracking.
        """
        return f"tenant:{tenant_key}|provider:{provider}"

    def begin_for_tenant(self, tenant_key: str, provider: str) -> None:
        """Record a tenant-scoped in-flight attempt for a provider."""
        self.begin(self.tenant_provider_key(tenant_key, provider))

    def finish_for_tenant(self, tenant_key: str, provider: str) -> None:
        """Clear a tenant-scoped in-flight attempt for a provider."""
        self.finish(self.tenant_provider_key(tenant_key, provider))

    def tenant_load_score(self, tenant_key: str, provider: str) -> int:
        """Return live in-flight count for a tenant on a provider."""
        return self.load_score(self.tenant_provider_key(tenant_key, provider))


class RateLimitStats:
    """Rolling provider rate-limit observations used by soft-rate-limit routing."""

    def __init__(self, max_observations: int = 100) -> None:
        """Initialize empty rate-limit observation windows.

        Args:
            max_observations: Maximum recent observations kept per provider.

        Raises:
            ValueError: If the observation window is not positive.
        """
        if max_observations < 1:
            raise ValueError(f"max_observations must be positive, got {max_observations}")
        self._max_observations = max_observations
        self._observations: dict[str, list[bool]] = {}

    def observe(self, provider: str, *, rate_limited: bool) -> None:
        """Record whether a provider attempt hit a rate limit.

        Args:
            provider: Provider name.
            rate_limited: Whether the attempt failed with a 429/rate-limit signal.
        """
        provider_observations = self._observations.setdefault(provider, [])
        provider_observations.append(rate_limited)
        if len(provider_observations) > self._max_observations:
            del provider_observations[0]

    def rate_limit_count(self, provider: str) -> int:
        """Return recent rate-limit hits for a provider."""
        return sum(self._observations.get(provider, []))

    def rate_limit_rate(self, provider: str) -> float:
        """Return the recent rate-limit hit rate for a provider.

        Providers with no observations are treated as having no rate pressure.
        """
        observations = self._observations.get(provider, [])
        if not observations:
            return 0.0
        return self.rate_limit_count(provider) / len(observations)


@dataclass
class _ProviderTokenBucket:
    """Mutable per-provider token bucket state."""

    tokens: float
    updated_at: float


class TokenBucketStats:
    """Per-provider token buckets used by token-bucket-burst routing."""

    def __init__(self, capacity: int, refill_per_second: float) -> None:
        """Initialize token-bucket settings shared across routing decisions.

        Args:
            capacity: Maximum tokens per provider bucket.
            refill_per_second: Token refill rate per second.

        Raises:
            ValueError: If capacity or refill rate is invalid.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_per_second <= 0:
            raise ValueError(f"refill_per_second must be positive, got {refill_per_second}")
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._buckets: dict[str, _ProviderTokenBucket] = {}

    @property
    def capacity(self) -> float:
        """Return the configured bucket capacity."""
        return self._capacity

    def available_tokens(self, provider: str) -> float:
        """Return the provider's current token balance after refill."""
        return self._refill(provider)

    def remaining_fraction(self, provider: str) -> float:
        """Return the provider's token balance as a fraction of capacity."""
        return self.available_tokens(provider) / self._capacity

    def consume(self, provider: str, tokens: int = 1) -> None:
        """Consume tokens from a provider bucket after routing selection.

        Args:
            provider: Provider name.
            tokens: Tokens to consume.
        """
        available = self._refill(provider)
        bucket = self._buckets[provider]
        bucket.tokens = max(0.0, available - tokens)

    def _refill(self, provider: str) -> float:
        now = time.monotonic()
        bucket = self._buckets.setdefault(
            provider,
            _ProviderTokenBucket(tokens=self._capacity, updated_at=now),
        )
        elapsed_seconds = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(
            self._capacity,
            bucket.tokens + elapsed_seconds * self._refill_per_second,
        )
        bucket.updated_at = now
        return bucket.tokens


class ModelTier(StrEnum):
    """Inferred capability tier used by model-tier-rate-limit routing."""

    FRONTIER = "frontier"
    MID = "mid"
    ECONOMY = "economy"


def infer_model_tier(model_name: str) -> ModelTier:
    """Infer a model's capability tier from its name using catalog heuristics.

    Economy markers (mini, nano, flash, etc.) are checked first so names such as
    ``gemini-3.5-flash`` classify as economy rather than frontier.
    """
    normalized = model_name.lower()
    economy_markers = ("-mini", "nano", "flash", "haiku-light", "-lite", "lite-")
    if any(marker in normalized for marker in economy_markers):
        return ModelTier.ECONOMY
    frontier_markers = (
        "gpt-5",
        "claude-sonnet-4",
        "claude-opus",
        "gemini-3",
        "kimi-k2",
        "o3",
    )
    if any(marker in normalized for marker in frontier_markers):
        return ModelTier.FRONTIER
    mid_markers = ("gpt-4.1", "claude-haiku", "gemini-2")
    if any(marker in normalized for marker in mid_markers):
        return ModelTier.MID
    return ModelTier.MID


def infer_target_tier(complexity_score: float) -> ModelTier:
    """Map a classifier complexity score onto a target capability tier.

    High-complexity prompts prefer frontier SKUs (GPT-5.5, Claude Sonnet 4.6,
    Gemini 3.x, Kimi K2); medium prompts prefer mid-tier; low-complexity prompts
    prefer economy SKUs. Thresholds are fixed so routing stays deterministic and
    needs no extra ``NEXUS_*`` knobs.
    """
    if complexity_score >= 0.7:
        return ModelTier.FRONTIER
    if complexity_score >= 0.35:
        return ModelTier.MID
    return ModelTier.ECONOMY


class TierRequestStats:
    """Rolling per-provider request timestamps for tier-aware soft rate limits."""

    def __init__(self, window_seconds: float = 60.0, max_timestamps: int = 500) -> None:
        """Initialize empty provider request windows.

        Args:
            window_seconds: Rolling window length in seconds.
            max_timestamps: Maximum timestamps retained per provider.

        Raises:
            ValueError: If the window or retention cap is invalid.
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if max_timestamps < 1:
            raise ValueError(f"max_timestamps must be positive, got {max_timestamps}")
        self._window_seconds = window_seconds
        self._max_timestamps = max_timestamps
        self._timestamps: dict[str, list[float]] = {}

    def record(self, provider: str, *, now: float | None = None) -> None:
        """Record a routing decision against a provider's rolling window."""
        timestamp = time.monotonic() if now is None else now
        provider_timestamps = self._timestamps.setdefault(provider, [])
        provider_timestamps.append(timestamp)
        self._prune(provider, timestamp)

    def request_count(self, provider: str, *, now: float | None = None) -> int:
        """Return recent request count for a provider inside the rolling window."""
        timestamp = time.monotonic() if now is None else now
        self._prune(provider, timestamp)
        return len(self._timestamps.get(provider, []))

    def saturation_fraction(
        self, provider: str, rpm_limit: int, *, now: float | None = None
    ) -> float:
        """Return recent load as a fraction of the tier RPM limit."""
        if rpm_limit < 1:
            raise ValueError(f"rpm_limit must be positive, got {rpm_limit}")
        return self.request_count(provider, now=now) / rpm_limit

    def is_under_limit(self, provider: str, rpm_limit: int, *, now: float | None = None) -> bool:
        """Return whether the provider is below its tier RPM limit."""
        return self.request_count(provider, now=now) < rpm_limit

    def _prune(self, provider: str, now: float) -> None:
        timestamps = self._timestamps.get(provider, [])
        cutoff = now - self._window_seconds
        while timestamps and timestamps[0] < cutoff:
            del timestamps[0]
        if len(timestamps) > self._max_timestamps:
            del timestamps[: len(timestamps) - self._max_timestamps]


class LatencyAwareStrategy(RoutingStrategy):
    """Route to low-latency models while penalizing degraded providers."""

    strategy_name = RoutingStrategyName.LATENCY_AWARE

    def __init__(
        self, model_catalog: Mapping[str, ModelCandidate], latency_stats: LatencyStats
    ) -> None:
        """Initialize latency-aware strategy.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
        """
        super().__init__(model_catalog)
        self._latency_stats = latency_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a model using quality, cost, and provider p95 latency."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ]
        if not eligible_candidates:
            highest_quality = max(
                self._model_catalog.values(),
                key=lambda candidate: candidate.quality_score,
            )
            return self._decision(
                highest_quality.model,
                "no model supports requested domain; "
                "latency-aware fell back to highest-quality model",
            )
        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (
                self._latency_stats.p95(candidate.provider)
                + (1.0 - candidate.quality_score) * 1000.0
                + candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens)
                * 1000.0
            ),
        )
        p95_latency = self._latency_stats.p95(selected_candidate.provider)
        rationale = f"latency-aware score favored provider p95={p95_latency:.1f}ms"
        return self._decision(selected_candidate.model, rationale)


class CostLatencyParetoStrategy(RoutingStrategy):
    """Pick a Pareto-optimal candidate on estimated cost and provider latency.

    Single-objective routers either chase the cheapest model or the fastest
    provider and can discard strong trade-offs that sit on the cost/latency
    frontier. This strategy mirrors LiteLLM/Portkey-style multi-objective
    routing: it keeps only non-dominated domain-eligible candidates under
    ascending cost and ascending rolling p95 latency, then breaks remaining
    ties by higher quality (then lower cost, lower latency, and model name).
    """

    strategy_name = RoutingStrategyName.COST_LATENCY_PARETO

    def __init__(
        self, model_catalog: Mapping[str, ModelCandidate], latency_stats: LatencyStats
    ) -> None:
        """Initialize cost/latency Pareto routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
        """
        super().__init__(model_catalog)
        self._latency_stats = latency_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best Pareto-optimal cost/latency candidate by quality."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate,
                request.max_tokens,
            )
            for candidate in eligible_candidates
        }
        latencies = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        pareto_front = [
            candidate
            for candidate in eligible_candidates
            if not any(
                other.model != candidate.model
                and costs[other.model] <= costs[candidate.model]
                and latencies[other.model] <= latencies[candidate.model]
                and (
                    costs[other.model] < costs[candidate.model]
                    or latencies[other.model] < latencies[candidate.model]
                )
                for other in eligible_candidates
            )
        ]
        selected_candidate = min(
            pareto_front,
            key=lambda candidate: (
                -candidate.quality_score,
                costs[candidate.model],
                latencies[candidate.model],
                candidate.model,
            ),
        )
        rationale = (
            "cost-latency-pareto selected non-dominated candidate "
            f"{selected_candidate.model} "
            f"(est ${costs[selected_candidate.model]:.6f}, "
            f"provider p95 {latencies[selected_candidate.model]:.1f}ms, "
            f"quality {selected_candidate.quality_score:.2f}; "
            f"front size {len(pareto_front)})"
        )
        return self._decision(selected_candidate.model, rationale)


class LeastBusyStrategy(RoutingStrategy):
    """Route to the best model on the least-loaded eligible provider.

    Quality-first strategies can converge on one high-scoring provider even while
    it is already saturated. This strategy consults live in-flight counters and
    first chooses among providers with the lowest current load score, then breaks
    ties by candidate quality and estimated request cost.
    """

    strategy_name = RoutingStrategyName.LEAST_BUSY

    def __init__(
        self, model_catalog: Mapping[str, ModelCandidate], inflight_stats: InflightStats
    ) -> None:
        """Initialize the least-busy strategy.

        Args:
            model_catalog: Available model candidates by model name.
            inflight_stats: Live provider in-flight attempt counters.
        """
        super().__init__(model_catalog)
        self._inflight_stats = inflight_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the highest-quality candidate on the least-busy provider."""
        ordered = self._load_ordered_candidates(request, signals)
        selected_candidate = ordered[0]
        load_score = self._inflight_stats.load_score(selected_candidate.provider)
        estimated_cost = selected_candidate.estimate_cost(
            signals.prompt_tokens_estimate,
            request.max_tokens,
        )
        rationale = (
            f"least-busy selected provider {selected_candidate.provider} "
            f"with load {load_score}; picked highest-quality eligible model "
            f"{selected_candidate.model} (quality {selected_candidate.quality_score:.2f}, "
            f"est ${estimated_cost:.6f})"
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in ordered[1:4]],
        )

    def _load_ordered_candidates(
        self, request: RouterRequest, signals: TaskSignals
    ) -> list[ModelCandidate]:
        """Return eligible candidates ordered by load, quality, and cost.

        Args:
            request: Router request.
            signals: Observed task signals.

        Returns:
            Domain-eligible candidates ordered by (load asc, quality desc,
            estimated cost asc, model name).
        """
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        return sorted(
            eligible_candidates,
            key=lambda candidate: (
                self._inflight_stats.load_score(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )


class ConcurrencyCapStrategy(RoutingStrategy):
    """Route to the highest-quality model whose provider is below a live cap.

    ``LeastBusyStrategy`` spreads traffic toward providers with the lowest current
    load. This strategy is a hard-cap sibling: providers whose live in-flight
    attempt count is at or above the configured cap are skipped for primary
    selection. Among the remaining candidates, quality wins first so premium
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic still goes to the
    strongest available provider without piling onto a saturated one.
    """

    strategy_name = RoutingStrategyName.CONCURRENCY_CAP

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        inflight_stats: InflightStats,
        per_provider_cap: int,
    ) -> None:
        """Initialize the concurrency-cap strategy.

        Args:
            model_catalog: Available model candidates by model name.
            inflight_stats: Live provider in-flight attempt counters.
            per_provider_cap: Maximum live attempts allowed per provider before
                that provider is skipped for primary selection.

        Raises:
            ValueError: If the cap is below one.
        """
        super().__init__(model_catalog)
        if per_provider_cap < 1:
            raise ValueError(f"per_provider_cap must be >= 1, got {per_provider_cap}")
        self._inflight_stats = inflight_stats
        self._per_provider_cap = per_provider_cap

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the highest-quality eligible model below the provider cap."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        under_cap_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._inflight_stats.load_score(candidate.provider) < self._per_provider_cap
        ]

        if under_cap_candidates:
            selected_candidate = max(
                under_cap_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            load_score = self._inflight_stats.load_score(selected_candidate.provider)
            rationale = (
                f"concurrency-cap selected below cap {self._per_provider_cap}; "
                f"{selected_candidate.provider} load {load_score}/"
                f"{self._per_provider_cap} with highest eligible quality "
                f"{selected_candidate.quality_score:.2f}"
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    self._inflight_stats.load_score(candidate.provider),
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            load_score = self._inflight_stats.load_score(selected_candidate.provider)
            rationale = (
                f"concurrency-cap found every eligible provider at or above cap "
                f"{self._per_provider_cap}; routed to least-loaded fallback "
                f"{selected_candidate.provider} load {load_score}/{self._per_provider_cap}"
            )

        fallback_candidates = self._fallback_candidates(
            selected_candidate.model,
            eligible_candidates,
            request,
            signals,
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )

    def _fallback_candidates(
        self,
        chosen_model: str,
        eligible_candidates: list[ModelCandidate],
        request: RouterRequest,
        signals: TaskSignals,
    ) -> list[ModelCandidate]:
        """Order fallbacks so below-cap providers are attempted before saturated ones."""
        return sorted(
            [candidate for candidate in eligible_candidates if candidate.model != chosen_model],
            key=lambda candidate: (
                self._inflight_stats.load_score(candidate.provider) >= self._per_provider_cap,
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )


class ReliabilityAwareStrategy(RoutingStrategy):
    """Route to healthy providers first, avoiding open circuit breakers.

    Cost- and latency-aware strategies optimize for price or speed but can keep
    selecting a provider whose circuit breaker has tripped, wasting the primary
    attempt on a known-unhealthy provider before the engine falls back. This
    strategy consults live circuit-breaker health: it selects the
    highest-quality model whose provider is currently available, and orders the
    fallback chain healthy-providers-first so recovery attempts prefer working
    providers.
    """

    strategy_name = RoutingStrategyName.RELIABILITY_AWARE

    def __init__(
        self, model_catalog: Mapping[str, ModelCandidate], provider_health: ProviderHealth
    ) -> None:
        """Initialize the reliability-aware strategy.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
        """
        super().__init__(model_catalog)
        self._provider_health = provider_health

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model whose provider circuit is closed."""
        domain_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        healthy_candidates = [
            candidate
            for candidate in domain_candidates
            if self._provider_health.is_available(candidate.provider)
        ]
        selection_pool = healthy_candidates or domain_candidates
        selected_candidate = max(
            selection_pool,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            ),
        )
        if healthy_candidates:
            rationale = (
                f"reliability-aware selected healthy provider {selected_candidate.provider} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
        else:
            rationale = (
                "reliability-aware found no healthy provider for the domain; "
                "routed to the highest-quality candidate"
            )
        return self._decision(selected_candidate.model, rationale)

    def _fallback_chain(self, chosen_model: str) -> list[str]:
        """Order the fallback chain by provider health, then quality.

        Args:
            chosen_model: Primary selected model.

        Returns:
            Ordered fallback model names preferring healthy providers.
        """
        candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if candidate.model != chosen_model
        ]
        candidates.sort(
            key=lambda candidate: (
                not self._provider_health.is_available(candidate.provider),
                -candidate.quality_score,
            )
        )
        return [candidate.model for candidate in candidates][:3]


class SoftRateLimitStrategy(RoutingStrategy):
    """Prefer healthy providers with fewer recent rate-limit observations.

    Hard circuit breakers react after repeated provider failures, but gateways
    such as LiteLLM and Portkey can expose softer pressure earlier through 429 or
    "rate limit" errors. This strategy uses those observations as a routing hint:
    among domain-eligible healthy providers it picks the candidate on the provider
    with the fewest recent rate-limit hits, then breaks ties by model quality and
    estimated request cost. If every provider is unhealthy, it still returns the
    lowest-rate-limit candidate so decide-time remains deterministic and the
    engine's normal fallback/guardrail path can continue.
    """

    strategy_name = RoutingStrategyName.SOFT_RATE_LIMIT

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        rate_limit_stats: RateLimitStats,
    ) -> None:
        """Initialize soft-rate-limit routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            rate_limit_stats: Recent provider rate-limit observations.
        """
        super().__init__(model_catalog)
        self._provider_health = provider_health
        self._rate_limit_stats = rate_limit_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best candidate with the lowest recent rate-limit pressure."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        ordered = self._rate_limit_ordered_candidates(request, signals, eligible_candidates)
        selected_candidate = ordered[0]
        rate_limit_count = self._rate_limit_stats.rate_limit_count(selected_candidate.provider)
        rate_limit_rate = self._rate_limit_stats.rate_limit_rate(selected_candidate.provider)
        health_context = (
            "healthy providers considered first"
            if any(self._provider_health.is_available(candidate.provider) for candidate in ordered)
            else "no healthy providers; using soft rate-limit ordering"
        )
        estimated_cost = selected_candidate.estimate_cost(
            signals.prompt_tokens_estimate,
            request.max_tokens,
        )
        rationale = (
            f"soft-rate-limit selected provider {selected_candidate.provider} "
            f"with {rate_limit_count} recent rate-limit hit(s) "
            f"({rate_limit_rate:.2%}); {health_context}; picked {selected_candidate.model} "
            f"(quality {selected_candidate.quality_score:.2f}, est ${estimated_cost:.6f})"
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in ordered[1:4]],
        )

    def _rate_limit_ordered_candidates(
        self,
        request: RouterRequest,
        signals: TaskSignals,
        eligible_candidates: list[ModelCandidate],
    ) -> list[ModelCandidate]:
        """Order candidates by health, rate-limit pressure, quality, and cost."""
        healthy_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
        ]
        primary_pool = healthy_candidates or eligible_candidates
        unhealthy_remainder = [
            candidate for candidate in eligible_candidates if candidate not in primary_pool
        ]

        def ordering_key(candidate: ModelCandidate) -> tuple[int, float, float, float, str]:
            return (
                self._rate_limit_stats.rate_limit_count(candidate.provider),
                self._rate_limit_stats.rate_limit_rate(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            )

        return [
            *sorted(primary_pool, key=ordering_key),
            *sorted(unhealthy_remainder, key=ordering_key),
        ]


class TokenBucketBurstStrategy(RoutingStrategy):
    """Route using per-provider token buckets that allow bursty traffic.

    LiteLLM, Portkey, and OpenRouter expose provider-side token buckets that
    refill over time. This strategy mirrors that pattern: each provider keeps a
    bucket with capacity ``C`` refilling at rate ``R``. Primary selection prefers
    providers with at least one token available (after refill), then breaks ties
    by quality and estimated cost. When every bucket is empty it falls back to
    the highest remaining token fraction, then to the cheapest eligible model so
    GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, and Kimi K2 traffic still drains
    gradually instead of hard-blocking.
    """

    strategy_name = RoutingStrategyName.TOKEN_BUCKET_BURST

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        token_bucket_stats: TokenBucketStats,
    ) -> None:
        """Initialize token-bucket-burst routing.

        Args:
            model_catalog: Available model candidates by model name.
            token_bucket_stats: Shared per-provider token bucket state.
        """
        super().__init__(model_catalog)
        self._token_bucket_stats = token_bucket_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a provider with available burst tokens, or the best empty bucket."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        with_tokens = [
            candidate
            for candidate in eligible_candidates
            if self._token_bucket_stats.available_tokens(candidate.provider) >= 1.0
        ]

        if with_tokens:
            selected_candidate = max(
                with_tokens,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            available_tokens = self._token_bucket_stats.available_tokens(
                selected_candidate.provider
            )
            rationale = (
                f"token-bucket-burst selected provider {selected_candidate.provider} "
                f"with {available_tokens:.1f}/{self._token_bucket_stats.capacity:.0f} tokens; "
                f"picked highest-quality eligible model {selected_candidate.model} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
        else:
            selected_candidate = max(
                eligible_candidates,
                key=lambda candidate: (
                    self._token_bucket_stats.remaining_fraction(candidate.provider),
                    -candidate.estimate_cost(
                        signals.prompt_tokens_estimate,
                        request.max_tokens,
                    ),
                    candidate.quality_score,
                    candidate.model,
                ),
            )
            remaining_fraction = self._token_bucket_stats.remaining_fraction(
                selected_candidate.provider
            )
            estimated_cost = selected_candidate.estimate_cost(
                signals.prompt_tokens_estimate,
                request.max_tokens,
            )
            rationale = (
                "token-bucket-burst found every eligible provider bucket empty; "
                f"routed to highest remaining fraction {remaining_fraction:.2%} "
                f"({selected_candidate.provider}) with lowest cost "
                f"${estimated_cost:.6f}"
            )

        self._token_bucket_stats.consume(selected_candidate.provider)
        ordered_fallbacks = self._ordered_fallback_candidates(
            selected_candidate.model,
            eligible_candidates,
            request,
            signals,
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in ordered_fallbacks[:3]],
        )

    def _ordered_fallback_candidates(
        self,
        chosen_model: str,
        eligible_candidates: list[ModelCandidate],
        request: RouterRequest,
        signals: TaskSignals,
    ) -> list[ModelCandidate]:
        """Order fallbacks so providers with burst tokens are attempted first."""
        return sorted(
            [candidate for candidate in eligible_candidates if candidate.model != chosen_model],
            key=lambda candidate: (
                self._token_bucket_stats.available_tokens(candidate.provider) < 1.0,
                -self._token_bucket_stats.remaining_fraction(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )


class ModelTierRateLimitStrategy(RoutingStrategy):
    """Route using tier-specific soft RPM limits inferred from model names.

    Frontier models (GPT-5.5, Claude Sonnet 4.6, Gemini 3.x Pro, Kimi K2, o3)
    carry tighter per-provider RPM ceilings than mid-tier or economy SKUs. This
    strategy tracks rolling request timestamps per provider, infers each
    candidate's tier from its model name, and prefers providers that are still
    under that tier's configured RPM. When every eligible provider is saturated
    it falls back to the least-saturated provider, then to quality and cost.
    """

    strategy_name = RoutingStrategyName.MODEL_TIER_RATE_LIMIT

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        tier_request_stats: TierRequestStats,
        frontier_rpm: int,
        mid_rpm: int,
        economy_rpm: int,
    ) -> None:
        """Initialize model-tier-rate-limit routing.

        Args:
            model_catalog: Available model candidates by model name.
            tier_request_stats: Shared per-provider rolling request timestamps.
            frontier_rpm: RPM ceiling for frontier-tier models.
            mid_rpm: RPM ceiling for mid-tier models.
            economy_rpm: RPM ceiling for economy-tier models.

        Raises:
            ValueError: If any RPM limit is not positive.
        """
        super().__init__(model_catalog)
        rpm_limits = (frontier_rpm, mid_rpm, economy_rpm)
        if any(limit < 1 for limit in rpm_limits):
            raise ValueError(f"tier RPM limits must be positive, got {rpm_limits}")
        self._tier_request_stats = tier_request_stats
        self._rpm_limits = {
            ModelTier.FRONTIER: frontier_rpm,
            ModelTier.MID: mid_rpm,
            ModelTier.ECONOMY: economy_rpm,
        }

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a provider under its model tier RPM, or the least saturated."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        under_limit = [
            candidate
            for candidate in eligible_candidates
            if self._is_candidate_under_limit(candidate)
        ]
        primary_pool = under_limit or eligible_candidates
        saturated_fallback = under_limit == []

        def ordering_key(candidate: ModelCandidate) -> tuple[float, float, float, str]:
            tier = infer_model_tier(candidate.model)
            rpm_limit = self._rpm_limits[tier]
            return (
                self._tier_request_stats.saturation_fraction(candidate.provider, rpm_limit),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            )

        ordered = sorted(primary_pool, key=ordering_key)
        selected_candidate = ordered[0]
        tier = infer_model_tier(selected_candidate.model)
        rpm_limit = self._rpm_limits[tier]
        recent_count = self._tier_request_stats.request_count(selected_candidate.provider)
        saturation = self._tier_request_stats.saturation_fraction(
            selected_candidate.provider,
            rpm_limit,
        )
        estimated_cost = selected_candidate.estimate_cost(
            signals.prompt_tokens_estimate,
            request.max_tokens,
        )
        limit_context = (
            "all eligible providers saturated; routed to least-saturated"
            if saturated_fallback
            else f"under {tier.value} tier RPM {rpm_limit}"
        )
        rationale = (
            f"model-tier-rate-limit selected provider {selected_candidate.provider} "
            f"({recent_count}/{rpm_limit} recent requests, {saturation:.2%} saturated, "
            f"{tier.value} tier); {limit_context}; picked {selected_candidate.model} "
            f"(quality {selected_candidate.quality_score:.2f}, est ${estimated_cost:.6f})"
        )
        self._tier_request_stats.record(selected_candidate.provider)
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in ordered[1:4]],
        )

    def _is_candidate_under_limit(self, candidate: ModelCandidate) -> bool:
        tier = infer_model_tier(candidate.model)
        rpm_limit = self._rpm_limits[tier]
        return self._tier_request_stats.is_under_limit(candidate.provider, rpm_limit)


class WeightedBlendStrategy(RoutingStrategy):
    """Route by a tunable weighted blend of quality, cost, and latency.

    The rule-based, cost-optimal, and latency-aware strategies each optimize a
    single axis (or a hard-coded mix). Operators frequently want an explicit,
    tunable trade-off instead: "favour quality but keep cost and latency in the
    picture". This strategy computes, for every domain-eligible candidate, a
    composite score from three normalized components and selects the highest:

    * quality: the candidate ``quality_score`` (already in ``[0, 1]``);
    * cost: min-max normalized so the cheapest candidate scores ``1.0``;
    * latency: min-max normalized so the lowest rolling p95 scores ``1.0``.

    Weights are normalized to sum to one, so only their ratios matter. When all
    weights are zero the strategy falls back to pure quality.
    """

    strategy_name = RoutingStrategyName.WEIGHTED_BLEND

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        quality_weight: float,
        cost_weight: float,
        latency_weight: float,
    ) -> None:
        """Initialize the weighted-blend strategy.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            quality_weight: Non-negative weight for the quality component.
            cost_weight: Non-negative weight for the (inverted) cost component.
            latency_weight: Non-negative weight for the (inverted) latency
                component.

        Raises:
            ValueError: If any weight is negative.
        """
        super().__init__(model_catalog)
        weights = (quality_weight, cost_weight, latency_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"weighted-blend weights must be non-negative, got {weights}")
        self._latency_stats = latency_stats
        total_weight = sum(weights)
        if total_weight <= 0.0:
            self._quality_weight, self._cost_weight, self._latency_weight = 1.0, 0.0, 0.0
        else:
            self._quality_weight = quality_weight / total_weight
            self._cost_weight = cost_weight / total_weight
            self._latency_weight = latency_weight / total_weight

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the candidate with the highest weighted composite score."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        latencies = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        cost_scores = _inverse_min_max(costs)
        latency_scores = _inverse_min_max(latencies)

        def blended_score(candidate: ModelCandidate) -> float:
            return (
                self._quality_weight * candidate.quality_score
                + self._cost_weight * cost_scores[candidate.model]
                + self._latency_weight * latency_scores[candidate.model]
            )

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                blended_score(candidate),
                candidate.quality_score,
                -costs[candidate.model],
            ),
        )
        rationale = (
            f"weighted-blend score {blended_score(selected_candidate):.3f} "
            f"(quality={self._quality_weight:.2f}, cost={self._cost_weight:.2f}, "
            f"latency={self._latency_weight:.2f})"
        )
        return self._decision(selected_candidate.model, rationale)


class ProviderHealthScoreBlendStrategy(RoutingStrategy):
    """Route by blending provider health, reliability, latency, quality, and cost.

    This is a health-aware sibling of :class:`WeightedBlendStrategy`: it keeps the
    tunable normalized quality/cost/latency scoring model, adds rolling provider
    success rate from :class:`SuccessStats`, and consults live provider circuit
    availability before scoring. Open circuits are a hard gate when at least one
    domain-eligible candidate has a closed circuit, so a known-bad provider does
    not win on cheap cost or stale latency stats. If every eligible provider is
    open, the strategy still returns the best scored model to keep decide-time
    deterministic; the engine's guardrails will skip unavailable attempts.
    """

    strategy_name = RoutingStrategyName.PROVIDER_HEALTH_SCORE_BLEND

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        success_stats: "SuccessStats",
        latency_stats: LatencyStats,
        success_weight: float,
        latency_weight: float,
        quality_weight: float,
        cost_weight: float,
    ) -> None:
        """Initialize the provider-health score blend strategy.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            success_stats: Rolling provider success observations.
            latency_stats: Rolling provider latency observations.
            success_weight: Non-negative rolling success-rate component weight.
            latency_weight: Non-negative inverse-p95-latency component weight.
            quality_weight: Non-negative model quality component weight.
            cost_weight: Non-negative inverse-cost component weight.

        Raises:
            ValueError: If any weight is negative.
        """
        super().__init__(model_catalog)
        weights = (success_weight, latency_weight, quality_weight, cost_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                f"provider-health-score-blend weights must be non-negative, got {weights}"
            )
        self._provider_health = provider_health
        self._success_stats = success_stats
        self._latency_stats = latency_stats
        total_weight = sum(weights)
        if total_weight <= 0.0:
            (
                self._success_weight,
                self._latency_weight,
                self._quality_weight,
                self._cost_weight,
            ) = (0.0, 0.0, 1.0, 0.0)
        else:
            self._success_weight = success_weight / total_weight
            self._latency_weight = latency_weight / total_weight
            self._quality_weight = quality_weight / total_weight
            self._cost_weight = cost_weight / total_weight

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the highest health-blended score among eligible candidates."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        availability = {
            candidate.model: self._availability_score(candidate.provider)
            for candidate in eligible_candidates
        }
        healthy_candidates = [
            candidate for candidate in eligible_candidates if availability[candidate.model] > 0.0
        ]
        selection_pool = healthy_candidates or eligible_candidates
        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        latencies = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        success_rates = {
            candidate.model: self._success_stats.success_rate(candidate.provider)
            for candidate in eligible_candidates
        }
        cost_scores = _inverse_min_max(costs)
        latency_scores = _inverse_min_max(latencies)

        def blended_score(candidate: ModelCandidate) -> float:
            return (
                self._success_weight * success_rates[candidate.model]
                + self._latency_weight * latency_scores[candidate.model]
                + self._quality_weight * candidate.quality_score
                + self._cost_weight * cost_scores[candidate.model]
            )

        def health_score(candidate: ModelCandidate) -> float:
            if healthy_candidates:
                return availability[candidate.model] * blended_score(candidate)
            return blended_score(candidate)

        def ranking_key(candidate: ModelCandidate) -> tuple[float, float, float, float]:
            return (
                health_score(candidate),
                success_rates[candidate.model],
                candidate.quality_score,
                -costs[candidate.model],
            )

        selected_candidate = max(selection_pool, key=ranking_key)
        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                availability[candidate.model] <= 0.0,
                -health_score(candidate),
                -success_rates[candidate.model],
                -candidate.quality_score,
                costs[candidate.model],
            ),
        )
        if healthy_candidates:
            health_context = "open circuits excluded from primary scoring"
        else:
            health_context = "no closed circuits; scored all eligible candidates"
        rationale = (
            f"provider-health-score-blend score {health_score(selected_candidate):.3f} "
            f"for {selected_candidate.provider} ({health_context}; "
            f"availability={availability[selected_candidate.model]:.0f}, "
            f"success={success_rates[selected_candidate.model]:.2%}, "
            f"p95={latencies[selected_candidate.model]:.1f}ms, "
            f"quality={selected_candidate.quality_score:.2f}, "
            f"est_cost=${costs[selected_candidate.model]:.6f}; "
            f"weights success={self._success_weight:.2f}, "
            f"latency={self._latency_weight:.2f}, "
            f"quality={self._quality_weight:.2f}, cost={self._cost_weight:.2f})"
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )

    def _availability_score(self, provider: str) -> float:
        """Return ``1.0`` for closed circuits and ``0.0`` for open circuits."""
        return 1.0 if self._provider_health.is_available(provider) else 0.0


class HealthCostLatencyStrategy(RoutingStrategy):
    """Route by a tunable weighted blend of provider health, cost, and latency.

    This is a simpler ternary sibling of :class:`ProviderHealthScoreBlendStrategy`:
    it scores domain-eligible candidates from rolling provider success rate
    (:class:`SuccessStats`), inverse normalized estimated request cost, and
    inverse normalized rolling provider p95 latency (:class:`LatencyStats`) with
    no circuit-breaker gating or quality component. Operators tune the trade-off
    across GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic via
    ``NEXUS_HCL_*`` weights.
    """

    strategy_name = RoutingStrategyName.HEALTH_COST_LATENCY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        success_stats: "SuccessStats",
        latency_stats: LatencyStats,
        health_weight: float,
        cost_weight: float,
        latency_weight: float,
    ) -> None:
        """Initialize the health/cost/latency blend strategy.

        Args:
            model_catalog: Available model candidates by model name.
            success_stats: Rolling provider success observations.
            latency_stats: Rolling provider latency observations.
            health_weight: Non-negative rolling success-rate component weight.
            cost_weight: Non-negative inverse-cost component weight.
            latency_weight: Non-negative inverse-p95-latency component weight.

        Raises:
            ValueError: If any weight is negative.
        """
        super().__init__(model_catalog)
        weights = (health_weight, cost_weight, latency_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"health-cost-latency weights must be non-negative, got {weights}")
        self._success_stats = success_stats
        self._latency_stats = latency_stats
        total_weight = sum(weights)
        if total_weight <= 0.0:
            self._health_weight, self._cost_weight, self._latency_weight = 1.0, 0.0, 0.0
        else:
            self._health_weight = health_weight / total_weight
            self._cost_weight = cost_weight / total_weight
            self._latency_weight = latency_weight / total_weight

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the highest health/cost/latency blended score among eligible candidates."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        latencies = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        health_rates = {
            candidate.model: self._success_stats.success_rate(candidate.provider)
            for candidate in eligible_candidates
        }
        cost_scores = _inverse_min_max(costs)
        latency_scores = _inverse_min_max(latencies)

        def blended_score(candidate: ModelCandidate) -> float:
            return (
                self._health_weight * health_rates[candidate.model]
                + self._cost_weight * cost_scores[candidate.model]
                + self._latency_weight * latency_scores[candidate.model]
            )

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                blended_score(candidate),
                health_rates[candidate.model],
                -costs[candidate.model],
                -latencies[candidate.model],
            ),
        )
        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                -blended_score(candidate),
                -health_rates[candidate.model],
                costs[candidate.model],
                latencies[candidate.model],
            ),
        )
        rationale = (
            f"health-cost-latency score {blended_score(selected_candidate):.3f} "
            f"for {selected_candidate.provider} ("
            f"health={health_rates[selected_candidate.model]:.2%}, "
            f"p95={latencies[selected_candidate.model]:.1f}ms, "
            f"est_cost=${costs[selected_candidate.model]:.6f}; "
            f"weights health={self._health_weight:.2f}, "
            f"cost={self._cost_weight:.2f}, latency={self._latency_weight:.2f})"
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


def _inverse_min_max(values: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalize values so the smallest maps to ``1.0``.

    Lower is better for cost and latency, so the minimum value is the most
    desirable and scores ``1.0`` while the maximum scores ``0.0``. When every
    value is equal, all candidates are equally good and score ``1.0`` (a neutral
    component that lets the other weighted terms decide the winner).

    Args:
        values: Mapping of model name to a lower-is-better metric.

    Returns:
        Mapping of model name to a normalized score in ``[0.0, 1.0]``.
    """
    if not values:
        return {}
    lowest = min(values.values())
    highest = max(values.values())
    if highest == lowest:
        return dict.fromkeys(values, 1.0)
    span = highest - lowest
    return {model: (highest - value) / span for model, value in values.items()}


class BudgetAwareStrategy(RoutingStrategy):
    """Route to the highest-quality model within a per-request cost ceiling.

    ``CostOptimalStrategy`` minimizes cost subject to a *quality floor*; this
    strategy is its dual: it maximizes quality subject to a hard *cost ceiling*.
    Given a per-request USD ceiling, it selects the highest-quality
    domain-eligible candidate whose estimated request cost stays within the
    ceiling. When no candidate fits the ceiling (for example a very long prompt
    or a domain served only by premium models), it falls back to the cheapest
    eligible candidate and records that the ceiling could not be met, so the
    request still routes deterministically rather than failing.
    """

    strategy_name = RoutingStrategyName.BUDGET_AWARE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        request_cost_ceiling_usd: float,
    ) -> None:
        """Initialize the budget-aware strategy.

        Args:
            model_catalog: Available model candidates by model name.
            request_cost_ceiling_usd: Maximum acceptable estimated cost per
                request, in USD.

        Raises:
            ValueError: If the cost ceiling is negative.
        """
        super().__init__(model_catalog)
        if request_cost_ceiling_usd < 0.0:
            raise ValueError(
                f"request_cost_ceiling_usd must be non-negative, got {request_cost_ceiling_usd}"
            )
        self._request_cost_ceiling_usd = request_cost_ceiling_usd

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model whose estimated cost fits the ceiling."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        affordable_candidates = [
            candidate
            for candidate in eligible_candidates
            if costs[candidate.model] <= self._request_cost_ceiling_usd
        ]
        if affordable_candidates:
            selected_candidate = max(
                affordable_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    -costs[candidate.model],
                ),
            )
            rationale = (
                f"budget-aware selected highest quality {selected_candidate.quality_score:.2f} "
                f"within ${self._request_cost_ceiling_usd:.4f} ceiling "
                f"(est ${costs[selected_candidate.model]:.6f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (costs[candidate.model], -candidate.quality_score),
        )
        rationale = (
            f"budget-aware found no model within ${self._request_cost_ceiling_usd:.4f} ceiling; "
            f"routed to cheapest eligible model (est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class ProviderFamilyCostCeilingStrategy(RoutingStrategy):
    """Route to the highest-quality model within its provider-family cost ceiling.

    OpenRouter/LiteLLM-style multi-provider setups often assign spend budgets per
    provider family (``openai``, ``anthropic``, ``google``, ``moonshot``). This
    strategy enforces those ceilings at decide time: each domain-eligible
    candidate is checked against the ceiling for its provider family, then the
    highest-quality affordable candidate wins. When no family has an affordable
    eligible model, the strategy falls back across families to the cheapest
    domain-eligible model so routing stays deterministic.

    A single default ceiling (``NEXUS_PROVIDER_FAMILY_COST_CEILING_USD``) applies
    to every family; optional per-family overrides can tighten or relax one
    provider without changing the others.
    """

    strategy_name = RoutingStrategyName.PROVIDER_FAMILY_COST_CEILING

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_family_cost_ceiling_usd: float = 0.05,
        family_ceilings_usd: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize the provider-family cost-ceiling strategy.

        Args:
            model_catalog: Available model candidates by model name.
            provider_family_cost_ceiling_usd: Default maximum estimated cost per
                request (USD) applied to every provider family.
            family_ceilings_usd: Optional per-provider-family ceiling overrides
                (for example ``{"openai": 0.01, "anthropic": 0.05}``).

        Raises:
            ValueError: If the default ceiling or any override is negative.
        """
        super().__init__(model_catalog)
        if provider_family_cost_ceiling_usd < 0.0:
            raise ValueError(
                "provider_family_cost_ceiling_usd must be non-negative, "
                f"got {provider_family_cost_ceiling_usd}"
            )
        overrides = dict(family_ceilings_usd or {})
        for family, ceiling in overrides.items():
            if ceiling < 0.0:
                raise ValueError(
                    f"family ceiling for {family!r} must be non-negative, got {ceiling}"
                )
        self._default_ceiling_usd = provider_family_cost_ceiling_usd
        self._family_ceilings_usd = overrides

    def _ceiling_for(self, provider: str) -> float:
        """Return the cost ceiling for a provider family.

        Args:
            provider: Provider family name (``openai``, ``anthropic``, …).

        Returns:
            Ceiling in USD for that family.
        """
        return self._family_ceilings_usd.get(provider, self._default_ceiling_usd)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model within its provider-family ceiling."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        affordable_candidates = [
            candidate
            for candidate in eligible_candidates
            if costs[candidate.model] <= self._ceiling_for(candidate.provider)
        ]
        if affordable_candidates:
            selected_candidate = max(
                affordable_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    -costs[candidate.model],
                ),
            )
            family_ceiling = self._ceiling_for(selected_candidate.provider)
            rationale = (
                "provider-family-cost-ceiling selected "
                f"{selected_candidate.provider} family "
                f"quality {selected_candidate.quality_score:.2f} "
                f"within ${family_ceiling:.4f} ceiling "
                f"(est ${costs[selected_candidate.model]:.6f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (costs[candidate.model], -candidate.quality_score),
        )
        rationale = (
            "provider-family-cost-ceiling found no model within any family ceiling; "
            f"fell back across families to cheapest eligible "
            f"{selected_candidate.provider}/{selected_candidate.model} "
            f"(est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class LatencyBudgetStrategy(RoutingStrategy):
    """Route to the highest-quality model within a rolling latency SLA.

    This strategy is the latency-domain dual of :class:`BudgetAwareStrategy`.
    Where ``BudgetAwareStrategy`` maximizes quality subject to a hard *cost*
    ceiling and :class:`LatencyAwareStrategy` simply minimizes a blended latency
    score (picking the fastest option regardless of how much quality it gives
    up), this strategy maximizes *quality* subject to a hard *latency* ceiling:
    it selects the highest-quality domain-eligible model whose provider rolling
    p95 latency stays within a configured SLA, so a request only trades quality
    for speed when the SLA actually requires it.

    Providers with no recorded latency yet report a p95 of ``0.0`` and are
    treated as within the SLA, so a cold start still routes to the best model and
    the SLA tightens as observations accrue. When no candidate meets the SLA (for
    example every provider is degraded), it falls back to the lowest-p95 eligible
    candidate and records that the SLA could not be met, so the request still
    routes deterministically rather than failing.
    """

    strategy_name = RoutingStrategyName.LATENCY_BUDGET

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        latency_sla_ms: float,
    ) -> None:
        """Initialize the latency-budget strategy.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            latency_sla_ms: Maximum acceptable provider p95 latency per request,
                in milliseconds.

        Raises:
            ValueError: If the latency SLA is negative.
        """
        super().__init__(model_catalog)
        if latency_sla_ms < 0.0:
            raise ValueError(f"latency_sla_ms must be non-negative, got {latency_sla_ms}")
        self._latency_stats = latency_stats
        self._latency_sla_ms = latency_sla_ms

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model whose provider p95 fits the SLA."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        p95_by_model = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        within_sla = [
            candidate
            for candidate in eligible_candidates
            if p95_by_model[candidate.model] <= self._latency_sla_ms
        ]
        if within_sla:
            selected_candidate = max(
                within_sla,
                key=lambda candidate: (
                    candidate.quality_score,
                    -p95_by_model[candidate.model],
                ),
            )
            rationale = (
                f"latency-budget selected highest quality {selected_candidate.quality_score:.2f} "
                f"within {self._latency_sla_ms:.0f}ms SLA "
                f"(provider p95 {p95_by_model[selected_candidate.model]:.1f}ms)"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (p95_by_model[candidate.model], -candidate.quality_score),
        )
        rationale = (
            f"latency-budget found no provider within {self._latency_sla_ms:.0f}ms SLA; "
            f"routed to lowest-latency model "
            f"(provider p95 {p95_by_model[selected_candidate.model]:.1f}ms)"
        )
        return self._decision(selected_candidate.model, rationale)


class StickySessionStrategy(RoutingStrategy):
    """Pin every request in a session to one model via consistent hashing.

    Multi-turn conversations degrade when successive turns hop between models:
    context handling shifts, provider prompt-caches miss, and rationale traces
    become harder to audit. Cost-, latency-, and quality-optimizing strategies
    all make per-request decisions and can switch models mid-session.

    This strategy instead deterministically maps a request's ``session_id`` onto
    one of the domain-eligible candidates using a stable hash. Two properties
    follow: every request sharing a ``session_id`` routes to the same model
    (intra-session consistency and prompt-cache affinity), while distinct
    sessions spread roughly uniformly across the eligible pool (session-level
    load balancing). Candidates are ordered deterministically by model name so
    the mapping is stable across processes and does not depend on catalog
    iteration order.
    """

    strategy_name = RoutingStrategyName.STICKY_SESSION

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pin the request's session to a deterministically hashed model."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        ordered_candidates = sorted(eligible_candidates, key=lambda candidate: candidate.model)
        digest = sha256(request.session_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % len(ordered_candidates)
        selected_candidate = ordered_candidates[bucket]
        rationale = (
            f"sticky-session pinned session '{request.session_id}' to "
            f"{selected_candidate.model} (bucket {bucket}/{len(ordered_candidates)})"
        )
        return self._decision(selected_candidate.model, rationale)


class ValueStrategy(RoutingStrategy):
    """Route to the model with the best quality-per-dollar (value) ratio.

    ``CostOptimalStrategy`` minimizes cost subject to a quality *floor* and
    ``BudgetAwareStrategy`` maximizes quality subject to a cost *ceiling*. Both
    require the operator to pick a threshold. This strategy needs no threshold:
    it maximizes the *efficiency* of the spend by selecting the domain-eligible
    candidate with the highest ``quality_score`` per estimated request dollar.

    This favours models that deliver strong quality cheaply (for example a
    balanced mid-tier model on a general prompt) over premium models whose
    marginal quality gain does not justify their marginal cost, while still
    preferring a premium model when nothing cheaper is close in quality. Costs
    are floored by a small epsilon so a zero-token estimate cannot divide by
    zero, and ties break toward higher quality then lower cost.
    """

    strategy_name = RoutingStrategyName.VALUE

    _COST_EPSILON_USD = 1e-9

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the candidate with the highest quality-per-dollar ratio."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }

        def value_ratio(candidate: ModelCandidate) -> float:
            return candidate.quality_score / max(costs[candidate.model], self._COST_EPSILON_USD)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                value_ratio(candidate),
                candidate.quality_score,
                -costs[candidate.model],
            ),
        )
        rationale = (
            f"value routing selected best quality-per-dollar "
            f"(quality {selected_candidate.quality_score:.2f}, "
            f"est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class ComplexityTierStrategy(RoutingStrategy):
    """Escalate model quality in proportion to the task's complexity score.

    The rule-based and classifier strategies also branch on complexity, but they
    do so against *hard-coded model names*, so they silently misroute whenever
    the catalog changes (a renamed or dropped model breaks the branch). This
    strategy instead treats the classifier's ``complexity_score`` (already in
    ``[0, 1]``) directly as a *required quality target* and picks the **cheapest**
    domain-eligible candidate whose ``quality_score`` meets that target. A
    trivial prompt (complexity near ``0``) accepts any model and therefore routes
    to the cheapest, while a hard prompt (complexity near ``1``) admits only the
    top-quality models — a deterministic quality-for-cost escalation ladder that
    adapts to whatever catalog is configured, with no thresholds to tune.

    When no candidate meets the target (for example a very hard prompt in a
    catalog with a modest ceiling), it falls back to the single highest-quality
    eligible candidate, so the request still routes deterministically. Ties among
    admitted candidates break toward higher quality, then lower cost.
    """

    strategy_name = RoutingStrategyName.COMPLEXITY_TIER

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the cheapest model whose quality meets the complexity target."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        required_quality = signals.complexity_score
        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        admitted = [
            candidate
            for candidate in eligible_candidates
            if candidate.quality_score >= required_quality
        ]
        if admitted:
            selected_candidate = min(
                admitted,
                key=lambda candidate: (
                    costs[candidate.model],
                    -candidate.quality_score,
                ),
            )
            rationale = (
                f"complexity-tier admitted quality>={required_quality:.2f} and picked cheapest "
                f"(quality {selected_candidate.quality_score:.2f}, "
                f"est ${costs[selected_candidate.model]:.6f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (candidate.quality_score, -costs[candidate.model]),
        )
        rationale = (
            f"complexity-tier found no model meeting quality>={required_quality:.2f}; "
            f"routed to highest-quality candidate (quality {selected_candidate.quality_score:.2f})"
        )
        return self._decision(selected_candidate.model, rationale)


class CanaryStrategy(RoutingStrategy):
    """Roll traffic onto a canary model gradually, pausing on ill health.

    Progressive delivery routes a small, deterministic fraction of traffic to a
    new *canary* model while the rest stays on the proven *stable* model, so a
    regression is caught on a slice of requests before a full cutover. Unlike
    :class:`ABRoutingStrategy` (a symmetric experiment that always honours its
    split), this strategy is health-gated: whenever the canary provider's
    circuit breaker is open it routes **all** traffic to the stable model, so a
    failing canary cannot keep drawing its share of live traffic. Bucketing is a
    stable hash of ``request_id`` so a given request always lands on the same
    arm, and the fallback chain is anchored on the stable model.
    """

    strategy_name = RoutingStrategyName.CANARY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        stable_model: str,
        canary_model: str,
        canary_weight: float,
    ) -> None:
        """Initialize the canary strategy.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            stable_model: Proven model that serves the majority of traffic.
            canary_model: New model under gradual rollout.
            canary_weight: Fraction of traffic sent to the canary, within
                ``[0.0, 1.0]``.

        Raises:
            ValueError: If a model is missing from the catalog or the weight is
                outside the ``[0.0, 1.0]`` range.
        """
        super().__init__(model_catalog)
        unknown_models = [
            model for model in (stable_model, canary_model) if model not in model_catalog
        ]
        if unknown_models:
            raise ValueError(
                f"canary models not in model catalog: {', '.join(sorted(unknown_models))}"
            )
        if not 0.0 <= canary_weight <= 1.0:
            raise ValueError(f"canary_weight must be within [0.0, 1.0], got {canary_weight}")
        self._provider_health = provider_health
        self._stable_model = stable_model
        self._canary_model = canary_model
        self._canary_weight = canary_weight

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Route to the canary for its traffic slice unless it is unhealthy."""
        canary_provider = self._model_catalog[self._canary_model].provider
        if not self._provider_health.is_available(canary_provider):
            return self._decision(
                self._stable_model,
                f"canary paused: provider {canary_provider} is unhealthy; routed to stable model",
            )
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < self._canary_weight:
            return self._decision(
                self._canary_model,
                f"canary bucket={bucket:.4f} < weight {self._canary_weight:.2f}; "
                "routed to canary model",
            )
        return self._decision(
            self._stable_model,
            f"canary bucket={bucket:.4f} >= weight {self._canary_weight:.2f}; "
            "routed to stable model",
        )

    def _fallback_chain(self, chosen_model: str) -> list[str]:
        """Anchor the fallback chain on the stable model.

        When the canary is chosen, the stable model is the safest first
        fallback; otherwise fall back by quality as usual.

        Args:
            chosen_model: Primary selected model.

        Returns:
            Ordered fallback model names.
        """
        quality_ordered = super()._fallback_chain(chosen_model)
        if chosen_model == self._canary_model and self._stable_model != chosen_model:
            remainder = [model for model in quality_ordered if model != self._stable_model]
            return [self._stable_model, *remainder][:3]
        return quality_ordered


class RoundRobinStrategy(RoutingStrategy):
    """Spread traffic evenly across every domain-eligible provider.

    The cost-, latency-, quality-, and value-optimizing strategies all converge
    on whichever single model currently scores best, so under steady traffic
    they hammer one provider — concentrating rate-limit pressure and correlated
    failure on it while other configured providers sit idle. This strategy is a
    load-balancer: it distributes requests as evenly as possible across the
    distinct providers that offer a domain-eligible model, then routes each
    request to that provider's highest-quality eligible model.

    Balancing uses a stable hash of ``request_id`` rather than a mutable
    round-robin counter so the mapping is **deterministic and replayable** — the
    same request always resolves to the same provider, which keeps audit records
    reproducible and avoids shared cross-request state — while distinct requests
    still spread uniformly across the provider pool (consistent-hash balancing).
    Providers are ordered deterministically by name so the bucketing is stable
    across processes and independent of catalog iteration order.
    """

    strategy_name = RoutingStrategyName.ROUND_ROBIN

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Route to a hash-balanced provider's best domain-eligible model."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        providers = sorted({candidate.provider for candidate in eligible_candidates})
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % len(providers)
        selected_provider = providers[bucket]

        provider_candidates = [
            candidate
            for candidate in eligible_candidates
            if candidate.provider == selected_provider
        ]
        selected_candidate = max(
            provider_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            ),
        )
        rationale = (
            f"round-robin balanced to provider '{selected_provider}' "
            f"(bucket {bucket}/{len(providers)}); picked its best eligible model "
            f"{selected_candidate.model} (quality {selected_candidate.quality_score:.2f})"
        )
        return self._decision(selected_candidate.model, rationale)


class CascadeStrategy(RoutingStrategy):
    """Route to the cheapest capable model with a cost-ascending escalation ladder.

    ``CostOptimalStrategy`` also starts from the cheapest model, but it enforces
    a fixed operator quality *floor* and then inherits the base quality-ordered
    fallback chain, so a first-attempt failure jumps straight to the globally
    highest-quality (and usually priciest) model. This strategy instead models a
    *cascade*: it routes the primary attempt to the cheapest domain-eligible
    model and then orders the fallback chain by **ascending cost**, so each retry
    escalates one rung up the price/capability ladder rather than leaping to the
    top. That minimizes expected spend on the common (first-attempt-succeeds)
    path while still climbing toward stronger models when a cheaper one fails,
    with no thresholds to tune.

    Eligibility mirrors the sibling optimizers: candidates must support the
    request domain and (unless the request is batch) real-time serving; if none
    qualify the filter relaxes to domain-only, then to the whole catalog, so a
    request always routes deterministically. Costs are estimated from the prompt
    and ``max_tokens``; ties break toward higher quality then model name so the
    ladder is stable across processes.
    """

    strategy_name = RoutingStrategyName.CASCADE

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Route to the cheapest eligible model with a cost-ascending ladder."""
        ordered = self._cost_ordered_candidates(request, signals)
        selected_candidate = ordered[0]
        cost = selected_candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens)
        rationale = (
            f"cascade routed to cheapest eligible model {selected_candidate.model} "
            f"(quality {selected_candidate.quality_score:.2f}, est ${cost:.6f}); "
            f"fallback escalates by ascending cost"
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in ordered[1:4]],
        )

    def _cost_ordered_candidates(
        self, request: RouterRequest, signals: TaskSignals
    ) -> list[ModelCandidate]:
        """Return eligible candidates ordered by ascending estimated cost.

        Args:
            request: Router request.
            signals: Observed task signals.

        Returns:
            Eligible candidates sorted by (cost asc, quality desc, model name).
        """
        realtime_eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
            and (
                candidate.supports_realtime
                or signals.latency_requirement is LatencyRequirement.BATCH
            )
        ]
        domain_eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ]
        eligible = realtime_eligible or domain_eligible or list(self._model_catalog.values())
        return sorted(
            eligible,
            key=lambda candidate: (
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                -candidate.quality_score,
                candidate.model,
            ),
        )


class EpsilonGreedyStrategy(RoutingStrategy):
    """Explore randomly with probability epsilon; otherwise exploit quality.

    Pure quality-maximizing strategies never sample lower-ranked models, so a
    catalog that is slightly mis-calibrated (or a new model whose prior is
    conservative) never gets live traffic. This strategy borrows the classic
    epsilon-greedy bandit policy: with probability ``epsilon`` it *explores* by
    picking uniformly among domain-eligible candidates, and otherwise *exploits*
    by selecting the highest-``quality_score`` eligible model.

    Both the explore/exploit coin flip and the explore arm are derived from
    stable hashes of ``request_id`` (matching canary/A/B bucketing), so a given
    request always resolves to the same decision for replay and auditability
    while distinct requests still explore at the configured rate.
    """

    strategy_name = RoutingStrategyName.EPSILON_GREEDY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        epsilon: float = 0.1,
    ) -> None:
        """Initialize the epsilon-greedy strategy.

        Args:
            model_catalog: Available model candidates by model name.
            epsilon: Explore probability within ``[0.0, 1.0]``. Defaults to
                ``0.1`` (10% exploration).

        Raises:
            ValueError: If ``epsilon`` is outside the ``[0.0, 1.0]`` range.
        """
        super().__init__(model_catalog)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be within [0.0, 1.0], got {epsilon}")
        self._epsilon = epsilon

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Explore or exploit among domain-eligible candidates."""
        eligible = self._domain_eligible(signals)
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < self._epsilon:
            explore_digest = sha256(f"{request.request_id}:explore".encode()).hexdigest()
            index = int(explore_digest[:8], 16) % len(eligible)
            # Stable order so the explore arm does not depend on dict iteration.
            ordered = sorted(eligible, key=lambda candidate: candidate.model)
            selected_candidate = ordered[index]
            rationale = (
                f"epsilon-greedy explore bucket={bucket:.4f} < epsilon {self._epsilon:.2f}; "
                f"uniform arm {index}/{len(ordered)} -> {selected_candidate.model} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible,
            key=lambda candidate: (candidate.quality_score, candidate.model),
        )
        rationale = (
            f"epsilon-greedy exploit bucket={bucket:.4f} >= epsilon {self._epsilon:.2f}; "
            f"routed to highest-quality eligible model {selected_candidate.model} "
            f"(quality {selected_candidate.quality_score:.2f})"
        )
        return self._decision(selected_candidate.model, rationale)

    def _domain_eligible(self, signals: TaskSignals) -> list[ModelCandidate]:
        """Return domain-eligible candidates, or the full catalog as fallback.

        Args:
            signals: Observed task signals.

        Returns:
            Non-empty list of candidates to choose among.
        """
        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ]
        return eligible or list(self._model_catalog.values())


class TokenBudgetStrategy(RoutingStrategy):
    """Route to the highest-quality model that fits the request token budget.

    Long prompts and large ``max_tokens`` caps can exceed a model's
    ``context_window``, causing provider 400s even when the chosen model would
    otherwise be ideal on quality. Cost- and quality-optimizing strategies ignore
    context capacity entirely. This strategy maximizes quality subject to a hard
    *token* ceiling: it selects the highest-quality domain-eligible candidate
    whose context window can hold the estimated prompt plus completion tokens
    *and* whose window is at least as large as the request's ``token_budget``.

    The effective capacity for a candidate is
    ``min(candidate.context_window, request.token_budget)``. A candidate fits when
    ``prompt_tokens_estimate + max_tokens <=`` that capacity. When no candidate
    fits (for example an oversized prompt against a tight budget), it falls back
    to the largest-context eligible model so the request still routes
    deterministically rather than failing at decide time.
    """

    strategy_name = RoutingStrategyName.TOKEN_BUDGET

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model whose context fits the token budget."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        tokens_needed = signals.prompt_tokens_estimate + request.max_tokens

        def effective_capacity(candidate: ModelCandidate) -> int:
            return min(candidate.context_window, request.token_budget)

        fitting_candidates = [
            candidate
            for candidate in eligible_candidates
            if tokens_needed <= effective_capacity(candidate)
        ]
        if fitting_candidates:
            selected_candidate = max(
                fitting_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    candidate.context_window,
                ),
            )
            rationale = (
                f"token-budget selected highest quality {selected_candidate.quality_score:.2f} "
                f"fitting {tokens_needed} tokens within "
                f"min(context={selected_candidate.context_window}, "
                f"budget={request.token_budget})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (candidate.context_window, candidate.quality_score),
        )
        rationale = (
            f"token-budget found no model fitting {tokens_needed} tokens within "
            f"budget {request.token_budget}; routed to largest-context model "
            f"(context {selected_candidate.context_window})"
        )
        return self._decision(selected_candidate.model, rationale)


class ABRoutingStrategy(RoutingStrategy):
    """Route deterministic request-id buckets between two models."""

    strategy_name = RoutingStrategyName.AB_TEST

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        model_a: str,
        model_b: str,
        model_a_weight: float,
    ) -> None:
        """Initialize A/B strategy.

        Args:
            model_catalog: Available model candidates by model name.
            model_a: First experiment arm.
            model_b: Second experiment arm.
            model_a_weight: Bucket weight for model A, within ``[0.0, 1.0]``.

        Raises:
            ValueError: If an arm is missing from the catalog or the weight is
                outside the ``[0.0, 1.0]`` range.
        """
        super().__init__(model_catalog)
        unknown_arms = [arm for arm in (model_a, model_b) if arm not in model_catalog]
        if unknown_arms:
            raise ValueError(
                f"A/B experiment arms not in model catalog: {', '.join(sorted(unknown_arms))}"
            )
        if not 0.0 <= model_a_weight <= 1.0:
            raise ValueError(f"A/B model_a_weight must be within [0.0, 1.0], got {model_a_weight}")
        self._model_a = model_a
        self._model_b = model_b
        self._model_a_weight = model_a_weight

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose an experiment arm using stable hashing."""
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        selected_model = self._model_a if bucket < self._model_a_weight else self._model_b
        rationale = f"A/B bucket={bucket:.4f} routed to {selected_model}"
        return self._decision(selected_model, rationale)


class GeoRegionStrategy(RoutingStrategy):
    """Prefer domain-eligible models that match the request's geo region.

    Data-residency and latency-affinity deployments often need traffic to stay
    on models whose providers serve a specific region (for example ``eu`` or
    ``cn``). Quality- and cost-optimizing strategies ignore geography entirely
    and can route an EU request to a US-only SKU. This strategy first restricts
    the pool to domain-eligible candidates whose ``supported_regions`` include
    the request ``region`` (or ``global`` when the request omits a region), then
    selects the highest-quality match. When no candidate advertises the
    requested region it falls back to the highest-quality domain-eligible model
    so the request still routes deterministically.
    """

    strategy_name = RoutingStrategyName.GEO_REGION

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model matching the request region."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        requested_region = (request.region or "global").strip().lower()
        region_matches = [
            candidate
            for candidate in eligible_candidates
            if requested_region in {region.lower() for region in candidate.supported_regions}
        ]
        if region_matches:
            selected_candidate = max(
                region_matches,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                ),
            )
            rationale = (
                f"geo-region matched region '{requested_region}' to "
                f"{selected_candidate.model} (quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            ),
        )
        rationale = (
            f"geo-region found no model for region '{requested_region}'; "
            f"fell back to highest-quality eligible model {selected_candidate.model}"
        )
        return self._decision(selected_candidate.model, rationale)


class RegionTierAffinityStrategy(RoutingStrategy):
    """Prefer models matching both request geo region and complexity tier.

    LiteLLM/OpenRouter-style deployments often need *both* data-residency (or
    latency affinity) *and* capability-tier alignment: an EU realtime chat should
    not land on a US-only frontier SKU, and a trivial prompt should not burn a
    frontier budget when an economy regional model exists. This strategy maps
    ``TaskSignals.complexity_score`` onto a target tier via
    :func:`infer_target_tier`, classifies each candidate with
    :func:`infer_model_tier`, and prefers domain-eligible candidates in this
    order:

    1. region **and** tier match
    2. tier match only
    3. region match only
    4. highest-quality domain-eligible (quality fallback)

    Within each pool it picks the highest ``quality_score`` (ties break toward
    lower estimated cost). The request ``region`` defaults to ``global`` when
    omitted. No extra ``NEXUS_*`` knobs are required.
    """

    strategy_name = RoutingStrategyName.REGION_TIER_AFFINITY

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a model preferring region+tier affinity, then quality."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        requested_region = (request.region or "global").strip().lower()
        target_tier = infer_target_tier(signals.complexity_score)

        def matches_region(candidate: ModelCandidate) -> bool:
            return requested_region in {region.lower() for region in candidate.supported_regions}

        def matches_tier(candidate: ModelCandidate) -> bool:
            return infer_model_tier(candidate.model) is target_tier

        def quality_key(candidate: ModelCandidate) -> tuple[float, float]:
            return (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            )

        both_matches = [
            candidate
            for candidate in eligible_candidates
            if matches_region(candidate) and matches_tier(candidate)
        ]
        if both_matches:
            selected_candidate = max(both_matches, key=quality_key)
            rationale = (
                f"region-tier-affinity matched region '{requested_region}' and "
                f"{target_tier.value} tier to {selected_candidate.model} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        tier_matches = [candidate for candidate in eligible_candidates if matches_tier(candidate)]
        if tier_matches:
            selected_candidate = max(tier_matches, key=quality_key)
            rationale = (
                f"region-tier-affinity found no region+tier match for "
                f"'{requested_region}'/{target_tier.value}; preferred "
                f"{target_tier.value} tier model {selected_candidate.model} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        region_matches = [
            candidate for candidate in eligible_candidates if matches_region(candidate)
        ]
        if region_matches:
            selected_candidate = max(region_matches, key=quality_key)
            rationale = (
                f"region-tier-affinity found no {target_tier.value} tier model; "
                f"preferred region '{requested_region}' model "
                f"{selected_candidate.model} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(eligible_candidates, key=quality_key)
        rationale = (
            f"region-tier-affinity found no region or tier match for "
            f"'{requested_region}'/{target_tier.value}; fell back to "
            f"highest-quality eligible model {selected_candidate.model}"
        )
        return self._decision(selected_candidate.model, rationale)


class FamilySpendWindow:
    """Rolling provider-family spend tracker for soft-family-budget routing."""

    def __init__(self, window_seconds: float = 3600.0) -> None:
        """Initialize empty family spend windows.

        Args:
            window_seconds: Rolling window length in seconds.

        Raises:
            ValueError: If the window length is not positive.
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._window_seconds = window_seconds
        self._entries: dict[str, list[tuple[float, float]]] = {}

    def record(self, provider_family: str, amount_usd: float, *, now: float | None = None) -> None:
        """Record spend against a provider family.

        Args:
            provider_family: Provider family name (``openai``, ``anthropic``, …).
            amount_usd: Observed spend in USD.
            now: Optional monotonic timestamp override for tests.
        """
        timestamp = time.monotonic() if now is None else now
        family_entries = self._entries.setdefault(provider_family, [])
        family_entries.append((timestamp, amount_usd))
        self._prune(provider_family, timestamp)

    def family_spend(self, provider_family: str, *, now: float | None = None) -> float:
        """Return rolling spend for a provider family inside the window.

        Args:
            provider_family: Provider family name.
            now: Optional monotonic timestamp override for tests.

        Returns:
            Total USD spent by the family inside the rolling window.
        """
        timestamp = time.monotonic() if now is None else now
        self._prune(provider_family, timestamp)
        return sum(amount for _, amount in self._entries.get(provider_family, []))

    def is_over_budget(
        self, provider_family: str, soft_budget_usd: float, *, now: float | None = None
    ) -> bool:
        """Return whether a family's rolling spend exceeds the soft budget.

        Args:
            provider_family: Provider family name.
            soft_budget_usd: Soft spend ceiling in USD.
            now: Optional monotonic timestamp override for tests.

        Returns:
            True when rolling spend is strictly above the soft budget.
        """
        return self.family_spend(provider_family, now=now) > soft_budget_usd

    def _prune(self, provider_family: str, now: float) -> None:
        cutoff = now - self._window_seconds
        family_entries = self._entries.get(provider_family, [])
        while family_entries and family_entries[0][0] < cutoff:
            family_entries.pop(0)
        if not family_entries:
            self._entries.pop(provider_family, None)


class SoftFamilyBudgetStrategy(RoutingStrategy):
    """Route with rolling soft spend budgets per provider family.

    OpenRouter/LiteLLM-style multi-provider gateways often track spend per
    provider family over a rolling window. Unlike
    :class:`ProviderFamilyCostCeilingStrategy` (a hard per-request ceiling),
    this strategy deprioritizes families whose *observed* rolling spend has
    crossed a soft budget while still admitting them when every family is hot.
    Among families under budget it picks the highest-quality domain-eligible
    model; when every family is over budget it falls back to the cheapest
    candidate from another family so routing stays deterministic.
    """

    strategy_name = RoutingStrategyName.SOFT_FAMILY_BUDGET

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        family_spend_window: FamilySpendWindow,
        soft_family_budget_usd: float = 5.0,
    ) -> None:
        """Initialize soft-family-budget routing.

        Args:
            model_catalog: Available model candidates by model name.
            family_spend_window: Rolling observed spend per provider family.
            soft_family_budget_usd: Soft spend ceiling per family in USD.

        Raises:
            ValueError: If the soft budget is negative.
        """
        super().__init__(model_catalog)
        if soft_family_budget_usd < 0.0:
            raise ValueError(
                f"soft_family_budget_usd must be non-negative, got {soft_family_budget_usd}"
            )
        self._family_spend_window = family_spend_window
        self._soft_family_budget_usd = soft_family_budget_usd

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer highest-quality models from families under the soft budget."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }

        def quality_key(candidate: ModelCandidate) -> tuple[float, float]:
            return (
                candidate.quality_score,
                -costs[candidate.model],
            )

        under_budget = [
            candidate
            for candidate in eligible_candidates
            if not self._family_spend_window.is_over_budget(
                candidate.provider,
                self._soft_family_budget_usd,
            )
        ]
        if under_budget:
            selected_candidate = max(under_budget, key=quality_key)
            family_spend = self._family_spend_window.family_spend(selected_candidate.provider)
            rationale = (
                "soft-family-budget selected "
                f"{selected_candidate.provider} family under "
                f"${self._soft_family_budget_usd:.4f} soft budget "
                f"(rolling spend ${family_spend:.6f}, "
                f"quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        primary_family = max(eligible_candidates, key=quality_key).provider
        other_family_candidates = [
            candidate for candidate in eligible_candidates if candidate.provider != primary_family
        ]
        fallback_pool = other_family_candidates or eligible_candidates
        selected_candidate = min(
            fallback_pool,
            key=lambda candidate: (costs[candidate.model], -candidate.quality_score),
        )
        rationale = (
            "soft-family-budget deprioritized over-budget families "
            f"(including {primary_family}); fell back to cheapest other family "
            f"{selected_candidate.provider}/{selected_candidate.model} "
            f"(est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class StickyRegionFailoverStrategy(RoutingStrategy):
    """Pin sessions to a region pool with ordered failover on ill health.

    Geo-residency deployments often need traffic to stay in a preferred region
    for latency and compliance, but still recover when that region's providers
    are unhealthy. This strategy walks an ordered region preference list
    (request ``region`` first, then ``NEXUS_STICKY_REGION_FAILOVER_PREFERENCES``),
    selects the first region with at least one healthy domain-eligible model, and
    pins the request's ``session_id`` to one model in that pool via consistent
    hashing — the same sticky affinity as :class:`StickySessionStrategy`, but
    scoped to the active region. When the preferred region has no healthy
    candidates the strategy failovers to the next region in the list while
    preserving session stickiness inside whichever region pool is active.
    """

    strategy_name = RoutingStrategyName.STICKY_REGION_FAILOVER

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        region_preferences: list[str] | None = None,
    ) -> None:
        """Initialize sticky-region-failover routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            region_preferences: Ordered region failover list when the request
                omits ``region``. Defaults to ``["eu", "us", "cn", "global"]``.
        """
        super().__init__(model_catalog)
        self._provider_health = provider_health
        self._default_region_preferences = [
            region.strip().lower()
            for region in (region_preferences or ["eu", "us", "cn", "global"])
        ]

    def _region_preferences(self, request: RouterRequest) -> list[str]:
        """Build the ordered region preference list for a request."""
        requested_region = (request.region or "").strip().lower()
        ordered: list[str] = []
        if requested_region:
            ordered.append(requested_region)
        for region in self._default_region_preferences:
            if region not in ordered:
                ordered.append(region)
        return ordered or ["global"]

    def _matches_region(self, candidate: ModelCandidate, region: str) -> bool:
        return region in {
            supported_region.lower() for supported_region in candidate.supported_regions
        }

    def _sticky_pick(
        self,
        request: RouterRequest,
        candidates: list[ModelCandidate],
        region_label: str,
        *,
        healthy_only: bool,
    ) -> RoutingDecision:
        pool = [
            candidate
            for candidate in candidates
            if not healthy_only or self._provider_health.is_available(candidate.provider)
        ] or candidates
        ordered_candidates = sorted(pool, key=lambda candidate: candidate.model)
        digest = sha256(request.session_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % len(ordered_candidates)
        selected_candidate = ordered_candidates[bucket]
        health_note = (
            "healthy providers in region"
            if healthy_only and pool is not candidates
            else "all providers in region"
        )
        rationale = (
            f"sticky-region-failover pinned session '{request.session_id}' to "
            f"{selected_candidate.model} in region '{region_label}' "
            f"({health_note}, bucket {bucket}/{len(ordered_candidates)})"
        )
        return self._decision(selected_candidate.model, rationale)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a sticky model within the first healthy region preference."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        for region in self._region_preferences(request):
            region_candidates = [
                candidate
                for candidate in eligible_candidates
                if self._matches_region(candidate, region)
            ]
            healthy_region_candidates = [
                candidate
                for candidate in region_candidates
                if self._provider_health.is_available(candidate.provider)
            ]
            if healthy_region_candidates:
                return self._sticky_pick(
                    request,
                    region_candidates,
                    region,
                    healthy_only=True,
                )

        healthy_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
        ]
        if healthy_candidates:
            return self._sticky_pick(
                request,
                eligible_candidates,
                "fallback",
                healthy_only=True,
            )
        return self._sticky_pick(
            request,
            eligible_candidates,
            "fallback",
            healthy_only=False,
        )


class CanaryTierBlendStrategy(RoutingStrategy):
    """Blend progressive canary delivery with complexity-tier affinity.

    Progressive canary rollouts often target a specific capability tier (for
    example validating a new frontier SKU without starving mid-tier traffic).
    This strategy combines :class:`CanaryStrategy` bucketing with
    :class:`RegionTierAffinityStrategy`-style tier matching:

    1. On the canary slice, prefer the canary when it matches the complexity
       tier; otherwise still route to the canary when healthy.
    2. Off the canary slice (or when the canary is unhealthy), prefer the
       highest-quality domain-eligible model in the target tier.
    3. When no tier match exists, fall back to the highest-quality eligible
       model.

    Bucketing reuses the stable ``request_id`` hash from canary/A/B routing so
    decisions stay replayable.
    """

    strategy_name = RoutingStrategyName.CANARY_TIER_BLEND

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        stable_model: str,
        canary_model: str,
        canary_weight: float,
    ) -> None:
        """Initialize the canary-tier-blend strategy.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            stable_model: Proven model that serves non-canary traffic.
            canary_model: New model under gradual rollout.
            canary_weight: Fraction of traffic sent to the canary, within
                ``[0.0, 1.0]``.

        Raises:
            ValueError: If a model is missing from the catalog or the weight is
                outside the ``[0.0, 1.0]`` range.
        """
        super().__init__(model_catalog)
        unknown_models = [
            model for model in (stable_model, canary_model) if model not in model_catalog
        ]
        if unknown_models:
            raise ValueError(
                "canary-tier-blend models not in model catalog: "
                f"{', '.join(sorted(unknown_models))}"
            )
        if not 0.0 <= canary_weight <= 1.0:
            raise ValueError(f"canary_weight must be within [0.0, 1.0], got {canary_weight}")
        self._provider_health = provider_health
        self._stable_model = stable_model
        self._canary_model = canary_model
        self._canary_weight = canary_weight

    def _canary_bucket(self, request: RouterRequest) -> float:
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _in_canary_slice(self, request: RouterRequest) -> bool:
        return self._canary_bucket(request) < self._canary_weight

    def _canary_is_healthy(self) -> bool:
        canary_provider = self._model_catalog[self._canary_model].provider
        return self._provider_health.is_available(canary_provider)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Blend canary traffic with tier affinity and quality fallback."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        target_tier = infer_target_tier(signals.complexity_score)
        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }

        def quality_key(candidate: ModelCandidate) -> tuple[float, float]:
            return (
                candidate.quality_score,
                -costs[candidate.model],
            )

        tier_matches = [
            candidate
            for candidate in eligible_candidates
            if infer_model_tier(candidate.model) is target_tier
        ]
        canary_candidate = self._model_catalog[self._canary_model]
        canary_matches_tier = infer_model_tier(self._canary_model) is target_tier
        bucket = self._canary_bucket(request)

        if self._in_canary_slice(request) and self._canary_is_healthy():
            if canary_matches_tier:
                return self._decision(
                    self._canary_model,
                    f"canary-tier-blend bucket={bucket:.4f} < weight {self._canary_weight:.2f}; "
                    f"canary matches {target_tier.value} tier",
                )
            return self._decision(
                self._canary_model,
                f"canary-tier-blend bucket={bucket:.4f} < weight {self._canary_weight:.2f}; "
                f"routed to canary despite {target_tier.value} tier mismatch",
            )

        if not self._in_canary_slice(request):
            stable_note = f"bucket={bucket:.4f} >= weight {self._canary_weight:.2f}; "
        else:
            canary_provider = canary_candidate.provider
            stable_note = f"canary paused: provider {canary_provider} unhealthy; "

        if tier_matches:
            selected_candidate = max(tier_matches, key=quality_key)
            rationale = (
                "canary-tier-blend "
                f"{stable_note}preferred {target_tier.value} tier model "
                f"{selected_candidate.model} (quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(eligible_candidates, key=quality_key)
        rationale = (
            "canary-tier-blend "
            f"{stable_note}found no {target_tier.value} tier model; "
            f"fell back to highest-quality eligible model {selected_candidate.model}"
        )
        return self._decision(selected_candidate.model, rationale)

    def _fallback_chain(self, chosen_model: str) -> list[str]:
        """Anchor the fallback chain on the stable model when canary is chosen."""
        quality_ordered = super()._fallback_chain(chosen_model)
        if chosen_model == self._canary_model and self._stable_model != chosen_model:
            remainder = [model for model in quality_ordered if model != self._stable_model]
            return [self._stable_model, *remainder][:3]
        return quality_ordered


class LatencySloShedStrategy(RoutingStrategy):
    """Shed slow providers when faster alternatives meet the latency SLO.

    LiteLLM/OpenRouter-style gateways often enforce a latency service-level
    objective: when at least one healthy candidate's rolling p95 fits under
    ``NEXUS_LATENCY_SLO_MS``, candidates above the SLO are deprioritized
    (shed) rather than competing on quality. Among under-SLO providers the
    strategy picks the highest-quality eligible model; when every provider is
    over the SLO it falls back to the lowest observed p95 so routing stays
    deterministic instead of failing open.
    """

    strategy_name = RoutingStrategyName.LATENCY_SLO_SHED

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        latency_slo_ms: float,
    ) -> None:
        """Initialize latency-SLO shedding routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            latency_slo_ms: Maximum acceptable provider p95 latency per request,
                in milliseconds.

        Raises:
            ValueError: If the latency SLO is negative.
        """
        super().__init__(model_catalog)
        if latency_slo_ms < 0.0:
            raise ValueError(f"latency_slo_ms must be non-negative, got {latency_slo_ms}")
        self._latency_stats = latency_stats
        self._latency_slo_ms = latency_slo_ms

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer under-SLO candidates and shed slower providers when possible."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        p95_by_model = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        under_slo = [
            candidate
            for candidate in eligible_candidates
            if p95_by_model[candidate.model] <= self._latency_slo_ms
        ]
        if under_slo:
            selected_candidate = max(
                under_slo,
                key=lambda candidate: (
                    candidate.quality_score,
                    -p95_by_model[candidate.model],
                ),
            )
            rationale = (
                "latency-slo-shed selected highest quality "
                f"{selected_candidate.quality_score:.2f} under "
                f"{self._latency_slo_ms:.0f}ms SLO "
                f"(provider p95 {p95_by_model[selected_candidate.model]:.1f}ms; "
                "shed slower alternatives)"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (p95_by_model[candidate.model], -candidate.quality_score),
        )
        rationale = (
            f"latency-slo-shed found no provider under {self._latency_slo_ms:.0f}ms SLO; "
            f"shed fallback to lowest-latency model "
            f"(provider p95 {p95_by_model[selected_candidate.model]:.1f}ms)"
        )
        return self._decision(selected_candidate.model, rationale)


class SuccessStats:
    """Rolling success/failure summary used by SLO-aware routing."""

    def __init__(self) -> None:
        """Initialize empty success observations."""
        self._successes: dict[str, int] = {}
        self._attempts: dict[str, int] = {}

    def observe(self, provider: str, *, success: bool) -> None:
        """Record a provider attempt outcome.

        Args:
            provider: Provider name.
            success: Whether the attempt succeeded.
        """
        self._attempts[provider] = self._attempts.get(provider, 0) + 1
        if success:
            self._successes[provider] = self._successes.get(provider, 0) + 1

    def success_rate(self, provider: str) -> float:
        """Return the rolling success rate for a provider.

        Providers with no observations yet are treated as fully healthy
        (``1.0``) so a cold start still admits every candidate under the SLO.

        Args:
            provider: Provider name.

        Returns:
            Success rate in ``[0.0, 1.0]``.
        """
        attempts = self._attempts.get(provider, 0)
        if attempts == 0:
            return 1.0
        return self._successes.get(provider, 0) / attempts

    def total_successes(self) -> int:
        """Return the sum of recorded successes across all providers.

        Returns:
            Non-negative total success count used by adaptive exploration.
        """
        return sum(self._successes.values())


class AdaptiveExplorationStrategy(RoutingStrategy):
    """Epsilon-greedy with an observation-decaying explore rate.

    Fixed ``epsilon-greedy`` keeps a constant explore budget forever. Production
    LLM gateways often prefer a schedule that explores aggressively while
    ``SuccessStats`` are cold, then tightens toward a small residual explore
    rate as successes accumulate. This strategy uses:

    ``epsilon = min + (base - min) / (1 + total_successes)``

    so a cold start explores at ``base`` (default ``0.2``) and asymptotes toward
    ``min`` (default ``0.02``). Explore/exploit bucketing and uniform explore
    arms reuse the same deterministic ``request_id`` hashes as
    :class:`EpsilonGreedyStrategy` for replayability.
    """

    strategy_name = RoutingStrategyName.ADAPTIVE_EXPLORATION

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        success_stats: SuccessStats,
        base_epsilon: float = 0.2,
        min_epsilon: float = 0.02,
    ) -> None:
        """Initialize adaptive exploration routing.

        Args:
            model_catalog: Available model candidates by model name.
            success_stats: Rolling provider success observations.
            base_epsilon: Cold-start explore probability within ``[0.0, 1.0]``.
            min_epsilon: Floor explore probability within ``[0.0, 1.0]``.

        Raises:
            ValueError: If either epsilon is outside ``[0.0, 1.0]`` or
                ``min_epsilon`` exceeds ``base_epsilon``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= base_epsilon <= 1.0:
            raise ValueError(f"base_epsilon must be within [0.0, 1.0], got {base_epsilon}")
        if not 0.0 <= min_epsilon <= 1.0:
            raise ValueError(f"min_epsilon must be within [0.0, 1.0], got {min_epsilon}")
        if min_epsilon > base_epsilon:
            raise ValueError(
                f"min_epsilon ({min_epsilon}) must be <= base_epsilon ({base_epsilon})"
            )
        self._success_stats = success_stats
        self._base_epsilon = base_epsilon
        self._min_epsilon = min_epsilon

    def current_epsilon(self) -> float:
        """Return the explore probability given current success observations."""
        total_successes = self._success_stats.total_successes()
        return self._min_epsilon + (self._base_epsilon - self._min_epsilon) / (
            1.0 + total_successes
        )

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Explore or exploit with a success-decayed epsilon schedule."""
        eligible = self._domain_eligible(signals)
        epsilon = self.current_epsilon()
        total_successes = self._success_stats.total_successes()
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < epsilon:
            explore_digest = sha256(f"{request.request_id}:explore".encode()).hexdigest()
            ordered = sorted(eligible, key=lambda candidate: candidate.model)
            index = int(explore_digest[:8], 16) % len(ordered)
            selected_candidate = ordered[index]
            rationale = (
                f"adaptive-exploration explore bucket={bucket:.4f} < epsilon "
                f"{epsilon:.4f} (base {self._base_epsilon:.2f} -> min "
                f"{self._min_epsilon:.2f}, successes={total_successes}); "
                f"uniform arm {index}/{len(ordered)} -> {selected_candidate.model} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible,
            key=lambda candidate: (candidate.quality_score, candidate.model),
        )
        rationale = (
            f"adaptive-exploration exploit bucket={bucket:.4f} >= epsilon "
            f"{epsilon:.4f} (base {self._base_epsilon:.2f} -> min "
            f"{self._min_epsilon:.2f}, successes={total_successes}); "
            f"routed to highest-quality eligible model {selected_candidate.model} "
            f"(quality {selected_candidate.quality_score:.2f})"
        )
        return self._decision(selected_candidate.model, rationale)

    def _domain_eligible(self, signals: TaskSignals) -> list[ModelCandidate]:
        """Return domain-eligible candidates, or the full catalog as fallback."""
        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ]
        return eligible or list(self._model_catalog.values())


class SloAwareStrategy(RoutingStrategy):
    """Route to models whose providers meet a rolling availability SLO.

    Latency- and reliability-aware strategies react to p95 delay or open
    circuit breakers, but neither tracks soft degradation: a provider can stay
    below an availability SLO (for example 99%) while its circuit is still
    closed. This strategy consults rolling success stats and selects the
    highest-quality domain-eligible model whose provider success rate meets the
    configured availability SLO. When no candidate meets the SLO it falls back
    to the highest success-rate eligible model so the request still routes.
    """

    strategy_name = RoutingStrategyName.SLO_AWARE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        success_stats: SuccessStats,
        availability_slo: float,
    ) -> None:
        """Initialize the SLO-aware strategy.

        Args:
            model_catalog: Available model candidates by model name.
            success_stats: Rolling provider success observations.
            availability_slo: Minimum acceptable success rate in ``[0.0, 1.0]``.

        Raises:
            ValueError: If the availability SLO is outside ``[0.0, 1.0]``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= availability_slo <= 1.0:
            raise ValueError(f"availability_slo must be within [0.0, 1.0], got {availability_slo}")
        self._success_stats = success_stats
        self._availability_slo = availability_slo

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best-quality model whose provider meets the availability SLO."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        rates = {
            candidate.model: self._success_stats.success_rate(candidate.provider)
            for candidate in eligible_candidates
        }
        within_slo = [
            candidate
            for candidate in eligible_candidates
            if rates[candidate.model] >= self._availability_slo
        ]
        if within_slo:
            selected_candidate = max(
                within_slo,
                key=lambda candidate: (
                    candidate.quality_score,
                    rates[candidate.model],
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                ),
            )
            rationale = (
                f"slo-aware selected highest quality {selected_candidate.quality_score:.2f} "
                f"meeting availability SLO {self._availability_slo:.2%} "
                f"(provider success {rates[selected_candidate.model]:.2%})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                rates[candidate.model],
                candidate.quality_score,
            ),
        )
        rationale = (
            f"slo-aware found no provider meeting availability SLO "
            f"{self._availability_slo:.2%}; routed to highest success-rate model "
            f"(provider success {rates[selected_candidate.model]:.2%})"
        )
        return self._decision(selected_candidate.model, rationale)


class AdaptiveTimeoutStrategy(RoutingStrategy):
    """Route by adapting timeout fit to request urgency and live provider signals.

    ``LatencyBudgetStrategy`` applies one hard SLA regardless of request shape,
    while ``SloAwareStrategy`` considers provider errors without latency. This
    strategy combines both signals for LiteLLM/OpenRouter-style timeout
    adaptation: realtime requests use the configured latency SLA as the base
    timeout budget, batch requests receive a wider budget, and latency pressure
    tightens that budget around the fastest observed providers. Provider error
    signals are folded into a risk-adjusted latency so a provider with recent
    failures needs extra headroom to remain eligible.

    With no latency/error observations, every candidate fits and the highest
    quality domain-eligible model wins. When no candidate fits the adaptive
    budget, the strategy falls back to the lowest risk-adjusted latency so
    routing remains deterministic and the engine's normal fallback chain still
    protects dispatch.
    """

    strategy_name = RoutingStrategyName.ADAPTIVE_TIMEOUT
    _BATCH_BUDGET_MULTIPLIER = 4.0
    _REALTIME_PRESSURE_MULTIPLIER = 1.5
    _BATCH_PRESSURE_MULTIPLIER = 2.0
    _MIN_TIGHTENED_BUDGET_RATIO = 0.5
    _MIN_SUCCESS_RATE = 0.05

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        success_stats: SuccessStats,
        base_timeout_ms: float,
    ) -> None:
        """Initialize the adaptive-timeout strategy.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            success_stats: Rolling provider success observations.
            base_timeout_ms: Realtime adaptive timeout base in milliseconds.

        Raises:
            ValueError: If the base timeout is negative.
        """
        super().__init__(model_catalog)
        if base_timeout_ms < 0.0:
            raise ValueError(f"base_timeout_ms must be non-negative, got {base_timeout_ms}")
        self._latency_stats = latency_stats
        self._success_stats = success_stats
        self._base_timeout_ms = base_timeout_ms

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the highest-quality model fitting the adaptive timeout budget."""
        eligible_candidates = self._eligible_candidates(signals)
        p95_by_model = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        success_by_model = {
            candidate.model: self._success_stats.success_rate(candidate.provider)
            for candidate in eligible_candidates
        }
        risk_latency_by_model = {
            candidate.model: self._risk_adjusted_latency_ms(
                p95_by_model[candidate.model],
                success_by_model[candidate.model],
            )
            for candidate in eligible_candidates
        }
        budget_ms = self._adaptive_budget_ms(signals, p95_by_model.values())
        within_budget = [
            candidate
            for candidate in eligible_candidates
            if risk_latency_by_model[candidate.model] <= budget_ms
        ]

        if within_budget:
            selected_candidate = max(
                within_budget,
                key=lambda candidate: (
                    candidate.quality_score,
                    success_by_model[candidate.model],
                    -risk_latency_by_model[candidate.model],
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                ),
            )
            rationale = (
                "adaptive-timeout selected highest quality "
                f"{selected_candidate.quality_score:.2f} within "
                f"{budget_ms:.0f}ms adaptive timeout budget "
                f"(provider p95 {p95_by_model[selected_candidate.model]:.1f}ms, "
                f"success {success_by_model[selected_candidate.model]:.2%}, "
                f"risk-adjusted {risk_latency_by_model[selected_candidate.model]:.1f}ms)"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (
                risk_latency_by_model[candidate.model],
                -candidate.quality_score,
                -success_by_model[candidate.model],
            ),
        )
        rationale = (
            f"adaptive-timeout found no provider within {budget_ms:.0f}ms adaptive timeout "
            "budget; routed to lowest risk-adjusted latency "
            f"(provider p95 {p95_by_model[selected_candidate.model]:.1f}ms, "
            f"success {success_by_model[selected_candidate.model]:.2%}, "
            f"risk-adjusted {risk_latency_by_model[selected_candidate.model]:.1f}ms)"
        )
        return self._decision(selected_candidate.model, rationale)

    def _eligible_candidates(self, signals: TaskSignals) -> list[ModelCandidate]:
        """Return domain- and urgency-eligible candidates with graceful fallback."""
        domain_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        if signals.latency_requirement is LatencyRequirement.BATCH:
            return domain_candidates

        realtime_candidates = [
            candidate for candidate in domain_candidates if candidate.supports_realtime
        ]
        return realtime_candidates or domain_candidates

    def _adaptive_budget_ms(
        self,
        signals: TaskSignals,
        provider_p95_latencies: Iterable[float],
    ) -> float:
        """Compute a timeout budget from request urgency and observed latency pressure."""
        base_budget_ms = self._base_timeout_ms
        if signals.latency_requirement is LatencyRequirement.BATCH:
            base_budget_ms *= self._BATCH_BUDGET_MULTIPLIER

        observed_latencies = [latency for latency in provider_p95_latencies if latency > 0.0]
        if not observed_latencies or max(observed_latencies) <= base_budget_ms:
            return base_budget_ms

        pressure_multiplier = (
            self._REALTIME_PRESSURE_MULTIPLIER
            if signals.latency_requirement is LatencyRequirement.REALTIME
            else self._BATCH_PRESSURE_MULTIPLIER
        )
        tightened_budget_ms = min(base_budget_ms, min(observed_latencies) * pressure_multiplier)
        minimum_budget_ms = base_budget_ms * self._MIN_TIGHTENED_BUDGET_RATIO
        return max(minimum_budget_ms, tightened_budget_ms)

    def _risk_adjusted_latency_ms(self, p95_latency_ms: float, success_rate: float) -> float:
        """Inflate observed p95 latency when recent provider errors suggest timeout risk."""
        if p95_latency_ms <= 0.0:
            return 0.0
        return p95_latency_ms / max(success_rate, self._MIN_SUCCESS_RATE)


class PromptPrefixCacheStrategy(RoutingStrategy):
    """Route long shared system-prompt prefixes with sticky model affinity.

    OpenRouter/LiteLLM-style prompt caching works best when requests that share
    the same long prompt prefix keep hitting the same provider/model: provider KV
    caches are usually scoped by model deployment, so spreading one prefix across
    providers dilutes cache-hit probability. This strategy hashes the first
    ``min_prefix_chars`` characters of the request's system prompt and uses that
    digest to pick a deterministic domain-eligible candidate. Requests sharing
    that long prefix therefore stick to the same provider/model, while unrelated
    prefixes spread across the candidate pool.

    Requests without a sufficiently long system prompt fall back to
    :class:`CostOptimalStrategy`, because there is no useful prefix-cache signal
    to optimize.
    """

    strategy_name = RoutingStrategyName.PROMPT_PREFIX_CACHE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        quality_floor: float,
        min_prefix_chars: int,
    ) -> None:
        """Initialize prompt-prefix-cache routing.

        Args:
            model_catalog: Available model candidates by model name.
            quality_floor: Quality floor used when no long prefix is present.
            min_prefix_chars: Minimum system prompt prefix length to hash.

        Raises:
            ValueError: If ``min_prefix_chars`` is not positive.
        """
        super().__init__(model_catalog)
        if min_prefix_chars < 1:
            raise ValueError(f"min_prefix_chars must be positive, got {min_prefix_chars}")
        self._cost_optimal = CostOptimalStrategy(model_catalog, quality_floor)
        self._min_prefix_chars = min_prefix_chars

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a sticky model for long system-prompt prefixes."""
        prefix = self._system_prompt_prefix(request)
        if prefix is None:
            fallback_decision = self._cost_optimal.choose(request, signals)
            rationale = (
                "prompt-prefix-cache found no system prompt prefix "
                f">={self._min_prefix_chars} chars; {fallback_decision.rationale}"
            )
            return self._decision(fallback_decision.chosen_model, rationale)

        eligible_candidates = (
            [
                candidate
                for candidate in self._model_catalog.values()
                if signals.domain_tag in candidate.supports_domains
                and (
                    candidate.supports_realtime
                    or signals.latency_requirement is LatencyRequirement.BATCH
                )
            ]
            or [
                candidate
                for candidate in self._model_catalog.values()
                if signals.domain_tag in candidate.supports_domains
            ]
            or list(self._model_catalog.values())
        )

        ordered_candidates = sorted(
            eligible_candidates,
            key=lambda candidate: (candidate.provider, candidate.model),
        )
        digest = sha256(prefix.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % len(ordered_candidates)
        selected_candidate = ordered_candidates[bucket]
        rationale = (
            "prompt-prefix-cache routed prefix "
            f"{digest[:12]} to {selected_candidate.provider}/{selected_candidate.model} "
            f"(bucket {bucket}/{len(ordered_candidates)}, "
            f"min_prefix_chars={self._min_prefix_chars})"
        )
        return self._decision(selected_candidate.model, rationale)

    def _system_prompt_prefix(self, request: RouterRequest) -> str | None:
        """Return a hashable long system-prompt prefix, if present.

        Args:
            request: Router request with OpenAI-compatible messages.

        Returns:
            The first ``min_prefix_chars`` of joined system message content, or
            ``None`` when the request has no sufficiently long system prompt.
        """
        system_prompt = "\n".join(
            message.content for message in request.messages if message.role == "system"
        )
        if len(system_prompt) < self._min_prefix_chars:
            return None
        return system_prompt[: self._min_prefix_chars]


class SemanticCacheStrategy(RoutingStrategy):
    """Route cache hits to the cheapest eligible model; miss falls to cost-optimal.

    Portkey- and LiteLLM-style semantic caches often mark a request as a hit in
    metadata when a prior embedding match can serve the answer. Serving a hit
    through a frontier model wastes spend; the useful signal is already cached.
    On ``metadata.cache_hit`` (truthy), this strategy picks the cheapest
    domain-eligible realtime-capable model (ties break toward higher quality,
    then model name) so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    traffic stays cheap on hits. On a miss it falls through to
    :class:`CostOptimalStrategy` so cold requests still respect the quality
    floor.
    """

    strategy_name = RoutingStrategyName.SEMANTIC_CACHE

    def __init__(self, model_catalog: Mapping[str, ModelCandidate], quality_floor: float) -> None:
        """Initialize semantic-cache strategy.

        Args:
            model_catalog: Available model candidates by model name.
            quality_floor: Quality floor used on cache misses via cost-optimal.
        """
        super().__init__(model_catalog)
        self._cost_optimal = CostOptimalStrategy(model_catalog, quality_floor)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a cheap model on cache hit; otherwise cost-optimal."""
        if self._is_cache_hit(request):
            eligible_candidates = (
                [
                    candidate
                    for candidate in self._model_catalog.values()
                    if signals.domain_tag in candidate.supports_domains
                    and (
                        candidate.supports_realtime
                        or signals.latency_requirement is LatencyRequirement.BATCH
                    )
                ]
                or [
                    candidate
                    for candidate in self._model_catalog.values()
                    if signals.domain_tag in candidate.supports_domains
                ]
                or list(self._model_catalog.values())
            )
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    candidate.estimate_cost(
                        signals.prompt_tokens_estimate,
                        request.max_tokens,
                    ),
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            estimated_cost = selected_candidate.estimate_cost(
                signals.prompt_tokens_estimate,
                request.max_tokens,
            )
            rationale = (
                "semantic-cache hit; preferred cheapest eligible model "
                f"{selected_candidate.model} at ${estimated_cost:.6f}"
            )
            return self._decision(selected_candidate.model, rationale)

        miss_decision = self._cost_optimal.choose(request, signals)
        rationale = f"semantic-cache miss; {miss_decision.rationale}"
        return self._decision(miss_decision.chosen_model, rationale)

    @staticmethod
    def _is_cache_hit(request: RouterRequest) -> bool:
        """Return whether request metadata signals a semantic cache hit.

        Args:
            request: Router request that may carry ``metadata.cache_hit``.

        Returns:
            True when ``cache_hit`` is a truthy value (bool, ``\"true\"``, ``1``).
        """
        raw_value = request.metadata.get("cache_hit")
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return raw_value != 0
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "hit"}
        return False


class FailoverPriorityStrategy(RoutingStrategy):
    """Walk an explicit ordered model list and pick the first healthy candidate.

    LiteLLM-style failover uses an operator-defined preference order rather than
    optimizing for cost or quality. This strategy consults
    ``NEXUS_FAILOVER_PRIORITY`` (an ordered model list spanning GPT-5.5, Claude
    Sonnet 4.6, Gemini 3.x, and Kimi K2 by default) and selects the first
    catalog model whose provider circuit is closed. Unhealthy providers are
    skipped; when every preference is unhealthy it still routes to the first
    in-catalog preference so the request does not fail at decide time. The
    fallback chain preserves the remaining priority order.
    """

    strategy_name = RoutingStrategyName.FAILOVER_PRIORITY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        failover_priority: list[str],
    ) -> None:
        """Initialize failover-priority strategy.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            failover_priority: Ordered preferred model names.

        Raises:
            ValueError: If the priority list is empty or contains no catalog models.
        """
        super().__init__(model_catalog)
        if not failover_priority:
            raise ValueError("failover_priority must contain at least one model")
        resolved = [model for model in failover_priority if model in model_catalog]
        if not resolved:
            raise ValueError(
                "failover_priority models not in catalog: "
                + ", ".join(sorted(set(failover_priority)))
            )
        self._provider_health = provider_health
        self._failover_priority = resolved

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the first healthy model in the configured priority order."""
        del request, signals  # preference order is explicit; signals unused
        healthy = [
            model
            for model in self._failover_priority
            if self._provider_health.is_available(self._model_catalog[model].provider)
        ]
        if healthy:
            selected_model = healthy[0]
            rationale = (
                f"failover-priority selected first healthy preference "
                f"{selected_model} (provider {self._model_catalog[selected_model].provider})"
            )
            return self._decision(selected_model, rationale)

        selected_model = self._failover_priority[0]
        rationale = (
            "failover-priority found no healthy preference; "
            f"routed to first listed model {selected_model}"
        )
        return self._decision(selected_model, rationale)

    def _fallback_chain(self, chosen_model: str) -> list[str]:
        """Preserve remaining priority order after the chosen model.

        Args:
            chosen_model: Primary selected model.

        Returns:
            Ordered fallback model names following the preference list.
        """
        remaining = [model for model in self._failover_priority if model != chosen_model]
        return remaining[:3]


class ShadowTrafficMirrorStrategy(RoutingStrategy):
    """Cost-optimal primary routing with optional shadow-mirror telemetry.

    LiteLLM/OpenRouter-style gateways often dual-run a small traffic slice
    against a secondary provider for latency/quality comparison without changing
    the user-visible primary. This strategy picks the primary model like
    :class:`CostOptimalStrategy` (minimum estimated cost subject to the quality
    floor), then on a deterministic ``request_id`` hash slice annotates the
    rationale with a *shadow mirror* candidate: the best domain-eligible model
    from a different provider than the primary (the second-best alternative
    provider). The decide phase still returns one primary candidate; audit
    trails capture the mirror model for downstream dual-run hooks.
    """

    strategy_name = RoutingStrategyName.SHADOW_TRAFFIC_MIRROR

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        quality_floor: float,
        shadow_traffic_percent: float = 5.0,
    ) -> None:
        """Initialize shadow-traffic-mirror routing.

        Args:
            model_catalog: Available model candidates by model name.
            quality_floor: Minimum acceptable quality score for the primary.
            shadow_traffic_percent: Percentage of traffic whose rationale
                annotates a shadow mirror candidate, within ``[0.0, 100.0]``.

        Raises:
            ValueError: If the shadow percentage is outside ``[0.0, 100.0]``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= shadow_traffic_percent <= 100.0:
            raise ValueError(
                f"shadow_traffic_percent must be within [0.0, 100.0], got {shadow_traffic_percent}"
            )
        self._quality_floor = quality_floor
        self._shadow_traffic_percent = shadow_traffic_percent

    def _shadow_bucket(self, request: RouterRequest) -> float:
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _in_shadow_slice(self, request: RouterRequest) -> bool:
        return self._shadow_bucket(request) < (self._shadow_traffic_percent / 100.0)

    def _feasible_candidates(
        self,
        request: RouterRequest,
        signals: TaskSignals,
    ) -> list[ModelCandidate]:
        feasible = [
            candidate
            for candidate in self._model_catalog.values()
            if candidate.quality_score >= self._quality_floor
            and signals.domain_tag in candidate.supports_domains
            and (
                candidate.supports_realtime
                or signals.latency_requirement is LatencyRequirement.BATCH
            )
        ]
        if feasible:
            return feasible
        return [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

    def _shadow_mirror(
        self,
        primary: ModelCandidate,
        candidates: list[ModelCandidate],
        costs: dict[str, float],
    ) -> ModelCandidate | None:
        alternate_providers = [
            candidate for candidate in candidates if candidate.provider != primary.provider
        ]
        if not alternate_providers:
            return None
        return max(
            alternate_providers,
            key=lambda candidate: (
                candidate.quality_score,
                -costs[candidate.model],
            ),
        )

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pick a cost-optimal primary and optionally annotate a shadow mirror."""
        candidates = self._feasible_candidates(request, signals)
        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in candidates
        }
        primary = min(
            candidates,
            key=lambda candidate: (costs[candidate.model], -candidate.quality_score),
        )
        estimated_cost = costs[primary.model]
        bucket = self._shadow_bucket(request)
        shadow = self._shadow_mirror(primary, candidates, costs)

        if self._in_shadow_slice(request) and shadow is not None:
            rationale = (
                "shadow-traffic-mirror selected primary "
                f"{primary.model} (est ${estimated_cost:.6f}, quality floor "
                f"{self._quality_floor:.2f}); shadow mirror "
                f"{shadow.model} ({shadow.provider}) queued for dual-run "
                f"telemetry (bucket={bucket:.4f} < "
                f"{self._shadow_traffic_percent:.1f}%)"
            )
        else:
            rationale = (
                "shadow-traffic-mirror selected primary "
                f"{primary.model} (est ${estimated_cost:.6f}, quality floor "
                f"{self._quality_floor:.2f})"
            )
            if shadow is not None:
                rationale += (
                    f"; shadow mirror {shadow.model} not annotated "
                    f"(bucket={bucket:.4f} >= {self._shadow_traffic_percent:.1f}%)"
                )

        return self._decision(primary.model, rationale)


class CanaryCostBlendStrategy(RoutingStrategy):
    """Blend cost exploration with healthy-provider cost minimization.

    Unlike :class:`CanaryTierBlendStrategy`, which targets complexity tiers,
    this strategy optimizes for **spend**: on the default path it picks the
    cheapest domain-eligible model among healthy providers; on a deterministic
    ``request_id`` explore slice it steps one cost rung cheaper (the
    next-cheapest healthy alternative) so gateways can sample cheaper SKUs
    without starving the primary cost floor. When no cheaper alternative exists
    the explore slice still returns the cheapest healthy model.
    """

    strategy_name = RoutingStrategyName.CANARY_COST_BLEND

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        canary_cost_blend_percent: float = 10.0,
    ) -> None:
        """Initialize canary-cost-blend routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            canary_cost_blend_percent: Percentage of traffic that explores the
                next-cheaper healthy tier, within ``[0.0, 100.0]``.

        Raises:
            ValueError: If the explore percentage is outside ``[0.0, 100.0]``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= canary_cost_blend_percent <= 100.0:
            raise ValueError(
                "canary_cost_blend_percent must be within [0.0, 100.0], "
                f"got {canary_cost_blend_percent}"
            )
        self._provider_health = provider_health
        self._canary_cost_blend_percent = canary_cost_blend_percent

    def _explore_bucket(self, request: RouterRequest) -> float:
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _in_explore_slice(self, request: RouterRequest) -> bool:
        return self._explore_bucket(request) < (self._canary_cost_blend_percent / 100.0)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pick cheapest healthy model or explore the next-cheaper tier."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        healthy_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
        ] or eligible_candidates
        cost_ordered = sorted(
            healthy_candidates,
            key=lambda candidate: (costs[candidate.model], -candidate.quality_score),
        )
        bucket = self._explore_bucket(request)

        if self._in_explore_slice(request) and len(cost_ordered) > 1:
            selected_candidate = cost_ordered[1]
            rationale = (
                "canary-cost-blend explore slice "
                f"(bucket={bucket:.4f} < {self._canary_cost_blend_percent:.1f}%); "
                f"selected next-cheaper healthy tier {selected_candidate.model} "
                f"(est ${costs[selected_candidate.model]:.6f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = cost_ordered[0]
        explore_note = (
            f"explore slice bucket={bucket:.4f} but only one healthy tier; "
            if self._in_explore_slice(request)
            else f"bucket={bucket:.4f} >= {self._canary_cost_blend_percent:.1f}%; "
        )
        rationale = (
            "canary-cost-blend "
            f"{explore_note}selected cheapest healthy model "
            f"{selected_candidate.model} (est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class CostAnomalyStats:
    """Rolling cost-per-1k-token observations for anomaly shedding."""

    def __init__(self, max_observations: int = 100) -> None:
        """Initialize empty cost anomaly observations."""
        self._observations: list[float] = []
        self._max_observations = max_observations

    def observe(self, cost_per_1k: float) -> None:
        """Record an observed cost-per-1k-tokens sample.

        Args:
            cost_per_1k: Observed blended cost per 1000 tokens in USD.
        """
        self._observations.append(cost_per_1k)
        if len(self._observations) > self._max_observations:
            del self._observations[0]

    def mean(self) -> float:
        """Return the rolling mean cost per 1000 tokens.

        Returns:
            Mean USD per 1k tokens, or ``0.0`` when no observations exist.
        """
        if not self._observations:
            return 0.0
        return sum(self._observations) / len(self._observations)


class TokenCostAnomalyShedStrategy(RoutingStrategy):
    """Shed candidates whose projected token cost spikes above a rolling baseline.

      LiteLLM/OpenRouter-style gateways often see sudden per-token spend spikes
      when a provider SKU reprices or a prompt pattern shifts toward expensive
      output tokens. This strategy tracks a rolling mean cost-per-1k-tokens via
      :class:`CostAnomalyStats`. When the highest-quality eligible candidate's
      projected cost/1k exceeds ``mean * NEXUS_TOKEN_COST_ANOMALY_RATIO`` (default
    ``2.0``), it sheds to the cheapest healthy alternative below that top cost.
      When no cheaper healthy option exists it falls back to pure quality ranking
      so routing stays deterministic instead of failing open.
    """

    strategy_name = RoutingStrategyName.TOKEN_COST_ANOMALY_SHED

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        cost_anomaly_stats: CostAnomalyStats,
        token_cost_anomaly_ratio: float = 2.0,
    ) -> None:
        """Initialize token-cost-anomaly shedding routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            cost_anomaly_stats: Rolling cost-per-1k observations.
            token_cost_anomaly_ratio: Multiplier above the rolling mean that
                triggers shedding (must be positive).

        Raises:
            ValueError: If the anomaly ratio is not positive.
        """
        super().__init__(model_catalog)
        if token_cost_anomaly_ratio <= 0.0:
            raise ValueError(
                f"token_cost_anomaly_ratio must be positive, got {token_cost_anomaly_ratio}"
            )
        self._provider_health = provider_health
        self._cost_anomaly_stats = cost_anomaly_stats
        self._token_cost_anomaly_ratio = token_cost_anomaly_ratio

    def _projected_cost_per_1k(
        self, candidate: ModelCandidate, input_tokens: int, output_tokens: int
    ) -> float:
        total_tokens = input_tokens + output_tokens
        if total_tokens == 0:
            return (candidate.input_cost_per_1k + candidate.output_cost_per_1k) / 2.0
        return (candidate.estimate_cost(input_tokens, output_tokens) / total_tokens) * 1000.0

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer quality unless the top pick's token cost is anomalously high."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs_per_1k = {
            candidate.model: self._projected_cost_per_1k(
                candidate, signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        quality_ordered = sorted(
            eligible_candidates,
            key=lambda candidate: (candidate.quality_score, -costs_per_1k[candidate.model]),
            reverse=True,
        )
        top_candidate = quality_ordered[0]
        rolling_mean = self._cost_anomaly_stats.mean()
        anomaly_threshold = rolling_mean * self._token_cost_anomaly_ratio

        if rolling_mean > 0.0 and costs_per_1k[top_candidate.model] > anomaly_threshold:
            healthy_candidates = [
                candidate
                for candidate in eligible_candidates
                if self._provider_health.is_available(candidate.provider)
            ] or eligible_candidates
            top_cost = costs_per_1k[top_candidate.model]
            cheaper_healthy = [
                candidate
                for candidate in healthy_candidates
                if costs_per_1k[candidate.model] < top_cost
            ]
            if cheaper_healthy:
                selected_candidate = min(
                    cheaper_healthy,
                    key=lambda candidate: (
                        costs_per_1k[candidate.model],
                        -candidate.quality_score,
                    ),
                )
                rationale = (
                    "token-cost-anomaly-shed detected projected cost/1k "
                    f"${top_cost:.6f} above rolling mean ${rolling_mean:.6f} "
                    f"* {self._token_cost_anomaly_ratio:.1f}; shed to cheaper "
                    f"healthy {selected_candidate.model} "
                    f"(cost/1k ${costs_per_1k[selected_candidate.model]:.6f})"
                )
                return self._decision(selected_candidate.model, rationale)

            rationale = (
                "token-cost-anomaly-shed found no cheaper healthy alternative; "
                f"quality fallback to {top_candidate.model} "
                f"(cost/1k ${top_cost:.6f})"
            )
            return self._decision(top_candidate.model, rationale)

        rationale = (
            "token-cost-anomaly-shed selected highest-quality "
            f"{top_candidate.model} (cost/1k ${costs_per_1k[top_candidate.model]:.6f} "
            f"within rolling baseline mean ${rolling_mean:.6f})"
        )
        return self._decision(top_candidate.model, rationale)


class StickyTenantHashStrategy(RoutingStrategy):
    """Pin tenants to a primary model via consistent hashing on tenant identity.

    Multi-tenant gateways often need **per-tenant** routing affinity so billing,
    quotas, and provider prompt caches stay stable across requests from the same
    customer. Unlike :class:`StickySessionStrategy`, which hashes only
    ``session_id`` for conversational turn consistency, this strategy resolves
    a tenant sticky key from ``metadata.tenant_id`` (then ``metadata.user_id``,
    ``metadata.sticky_key``, ``user_id``, and finally ``session_id``) and maps
    that key onto a domain-eligible model. When the sticky primary's provider is
    unhealthy the strategy failovers to the next healthy candidate in the
    deterministic ring so routing stays replayable.
    """

    strategy_name = RoutingStrategyName.STICKY_TENANT_HASH

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
    ) -> None:
        """Initialize sticky-tenant-hash routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
        """
        super().__init__(model_catalog)
        self._provider_health = provider_health

    def _sticky_key(self, request: RouterRequest) -> str:
        """Resolve the tenant sticky key for consistent hashing."""
        metadata = request.metadata
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pin the tenant sticky key to a model with healthy failover."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        ordered_candidates = sorted(eligible_candidates, key=lambda candidate: candidate.model)
        sticky_key = self._sticky_key(request)
        digest = sha256(sticky_key.encode("utf-8")).hexdigest()
        primary_bucket = int(digest[:8], 16) % len(ordered_candidates)

        for offset in range(len(ordered_candidates)):
            bucket = (primary_bucket + offset) % len(ordered_candidates)
            candidate = ordered_candidates[bucket]
            if self._provider_health.is_available(candidate.provider):
                failover_note = (
                    f"failover offset {offset} from primary bucket {primary_bucket}"
                    if offset > 0
                    else f"primary bucket {primary_bucket}"
                )
                rationale = (
                    f"sticky-tenant-hash pinned tenant '{sticky_key}' to "
                    f"{candidate.model} ({failover_note}/"
                    f"{len(ordered_candidates)} healthy ring)"
                )
                return self._decision(candidate.model, rationale)

        fallback_candidate = ordered_candidates[primary_bucket]
        rationale = (
            f"sticky-tenant-hash pinned tenant '{sticky_key}' to "
            f"{fallback_candidate.model} (primary bucket {primary_bucket}; "
            "no healthy providers; sticky fallback)"
        )
        return self._decision(fallback_candidate.model, rationale)


class MultiRegionLatencyHedgeStrategy(RoutingStrategy):
    """Hedge to a faster secondary region when the primary region is hot.

    Multi-region gateways often keep a preferred residency region for GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic but still need an escape
    hatch when that region's rolling p50 latency spikes. This strategy picks the
    highest-quality domain-eligible model in the request's primary region (or
    ``global`` when omitted). When that model's provider p50 exceeds
    ``NEXUS_LATENCY_HEDGE_MS`` it hedges to the lowest-p50 candidate in a
    **secondary** region (any eligible model that does not match the primary
    region). Otherwise it stays on the primary quality preference.
    """

    strategy_name = RoutingStrategyName.MULTI_REGION_LATENCY_HEDGE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        latency_hedge_ms: float,
    ) -> None:
        """Initialize multi-region latency hedge routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            latency_hedge_ms: Primary-region p50 threshold that triggers hedging
                to a secondary region, in milliseconds.

        Raises:
            ValueError: If the hedge threshold is negative.
        """
        super().__init__(model_catalog)
        if latency_hedge_ms < 0.0:
            raise ValueError(f"latency_hedge_ms must be non-negative, got {latency_hedge_ms}")
        self._latency_stats = latency_stats
        self._latency_hedge_ms = latency_hedge_ms

    def _matches_region(self, candidate: ModelCandidate, region: str) -> bool:
        return region in {supported.lower() for supported in candidate.supported_regions}

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Stay on primary quality unless the region is hot enough to hedge."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        primary_region = (request.region or "global").strip().lower()
        primary_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._matches_region(candidate, primary_region)
        ] or eligible_candidates
        secondary_candidates = [
            candidate for candidate in eligible_candidates if candidate not in primary_candidates
        ]

        primary_pick = max(
            primary_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -self._latency_stats.p50(candidate.provider),
            ),
        )
        primary_p50 = self._latency_stats.p50(primary_pick.provider)

        if secondary_candidates and primary_p50 > self._latency_hedge_ms:
            selected_candidate = min(
                secondary_candidates,
                key=lambda candidate: (
                    self._latency_stats.p50(candidate.provider),
                    -candidate.quality_score,
                ),
            )
            rationale = (
                "multi-region-latency-hedge primary region "
                f"'{primary_region}' hot (provider p50 {primary_p50:.1f}ms > "
                f"{self._latency_hedge_ms:.0f}ms); hedged to secondary "
                f"{selected_candidate.model} "
                f"(provider p50 "
                f"{self._latency_stats.p50(selected_candidate.provider):.1f}ms)"
            )
            return self._decision(selected_candidate.model, rationale)

        rationale = (
            "multi-region-latency-hedge stayed on primary region "
            f"'{primary_region}' quality preference {primary_pick.model} "
            f"(provider p50 {primary_p50:.1f}ms <= "
            f"{self._latency_hedge_ms:.0f}ms hedge threshold)"
        )
        return self._decision(primary_pick.model, rationale)


class PromptLengthTierShedStrategy(RoutingStrategy):
    """Shed frontier models when the prompt exceeds a token-length tier gate.

    Long prompts inflate frontier spend (GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2) without always needing frontier quality. When
    ``signals.prompt_tokens_estimate`` exceeds
    ``NEXUS_PROMPT_LENGTH_TIER_TOKENS`` (default ``8000``), this strategy sheds
    frontier-tier candidates and picks the highest-quality mid/economy model
    that still fits the domain. Short prompts keep pure quality ranking so
    premium models remain available for compact hard tasks.
    """

    strategy_name = RoutingStrategyName.PROMPT_LENGTH_TIER_SHED

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        prompt_length_tier_tokens: int = 8000,
    ) -> None:
        """Initialize prompt-length tier shedding.

        Args:
            model_catalog: Available model candidates by model name.
            prompt_length_tier_tokens: Prompt-token threshold that triggers
                frontier shedding (must be >= 1).

        Raises:
            ValueError: If the token threshold is less than 1.
        """
        super().__init__(model_catalog)
        if prompt_length_tier_tokens < 1:
            raise ValueError(
                f"prompt_length_tier_tokens must be >= 1, got {prompt_length_tier_tokens}"
            )
        self._prompt_length_tier_tokens = prompt_length_tier_tokens

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer quality unless the prompt is long enough to shed frontier tiers."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        prompt_tokens = signals.prompt_tokens_estimate
        if prompt_tokens > self._prompt_length_tier_tokens:
            non_frontier = [
                candidate
                for candidate in eligible_candidates
                if infer_model_tier(candidate.model) is not ModelTier.FRONTIER
            ]
            pool = non_frontier or eligible_candidates
            selected_candidate = max(
                pool,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(prompt_tokens, request.max_tokens),
                ),
            )
            shed_note = (
                "shed frontier tiers"
                if non_frontier
                else "no non-frontier alternative; quality fallback"
            )
            rationale = (
                "prompt-length-tier-shed detected prompt_tokens "
                f"{prompt_tokens} above tier gate "
                f"{self._prompt_length_tier_tokens}; {shed_note}; selected "
                f"{selected_candidate.model} "
                f"(tier {infer_model_tier(selected_candidate.model).value})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(prompt_tokens, request.max_tokens),
            ),
        )
        rationale = (
            "prompt-length-tier-shed selected highest-quality "
            f"{selected_candidate.model} (prompt_tokens {prompt_tokens} within "
            f"tier gate {self._prompt_length_tier_tokens})"
        )
        return self._decision(selected_candidate.model, rationale)


class RetryBudgetAwareFailoverStrategy(RoutingStrategy):
    """Prefer healthy providers while a request still has retry budget left.

    Gateway retries are a scarce resource. When ``metadata.retry_remaining``
    (falling back to ``NEXUS_RETRY_BUDGET_DEFAULT``) is still high, this
    strategy picks the highest-quality healthy candidate. When the remaining
    budget drops to ``<= 1``, it failovers to the lowest-latency healthy
    alternative so the last attempt is more likely to succeed quickly.
    Deterministic and safe for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 traffic.
    """

    strategy_name = RoutingStrategyName.RETRY_BUDGET_AWARE_FAILOVER

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        latency_stats: LatencyStats,
        retry_budget_default: int = 3,
    ) -> None:
        """Initialize retry-budget-aware failover routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            latency_stats: Rolling provider latency observations.
            retry_budget_default: Default remaining retries when metadata omits
                ``retry_remaining`` (must be >= 0).

        Raises:
            ValueError: If the default retry budget is negative.
        """
        super().__init__(model_catalog)
        if retry_budget_default < 0:
            raise ValueError(f"retry_budget_default must be >= 0, got {retry_budget_default}")
        self._provider_health = provider_health
        self._latency_stats = latency_stats
        self._retry_budget_default = retry_budget_default

    def _remaining_retries(self, request: RouterRequest) -> int:
        """Resolve remaining retry budget from request metadata."""
        raw = request.metadata.get("retry_remaining", self._retry_budget_default)
        try:
            value = int(str(raw).strip())
        except ValueError:
            return self._retry_budget_default
        return max(0, value)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pick quality while budget remains; failover to low latency near exhaustion."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        healthy = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
        ] or eligible_candidates

        remaining = self._remaining_retries(request)
        if remaining <= 1:
            selected_candidate = min(
                healthy,
                key=lambda candidate: (
                    self._latency_stats.p95(candidate.provider),
                    -candidate.quality_score,
                ),
            )
            rationale = (
                "retry-budget-aware-failover remaining retries "
                f"{remaining}; failover to lowest-latency healthy "
                f"{selected_candidate.model} "
                f"(p95 {self._latency_stats.p95(selected_candidate.provider):.1f}ms)"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            healthy,
            key=lambda candidate: (
                candidate.quality_score,
                -self._latency_stats.p95(candidate.provider),
            ),
        )
        rationale = (
            "retry-budget-aware-failover remaining retries "
            f"{remaining}; selected highest-quality healthy "
            f"{selected_candidate.model}"
        )
        return self._decision(selected_candidate.model, rationale)


class CacheHitStickyWarmPoolStrategy(RoutingStrategy):
    """Pin repeated prompt prefixes onto one warm model for provider cache hits.

    Provider prompt caches warm when successive requests share a long identical
    prefix on the same model. This strategy hashes the leading system/user
    prefix (at least ``NEXUS_CACHE_HIT_STICKY_MIN_CHARS`` characters) onto a
    domain-eligible model and keeps that mapping sticky so GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic reuses a warm pool.
    When the sticky primary provider is unhealthy it failovers to the next
    healthy ring member.
    """

    strategy_name = RoutingStrategyName.CACHE_HIT_STICKY_WARM_POOL

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        cache_hit_sticky_min_chars: int = 64,
    ) -> None:
        """Initialize cache-hit sticky warm-pool routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            cache_hit_sticky_min_chars: Minimum prefix length required before
                sticky hashing activates (must be >= 1).

        Raises:
            ValueError: If the minimum prefix length is less than 1.
        """
        super().__init__(model_catalog)
        if cache_hit_sticky_min_chars < 1:
            raise ValueError(
                f"cache_hit_sticky_min_chars must be >= 1, got {cache_hit_sticky_min_chars}"
            )
        self._provider_health = provider_health
        self._cache_hit_sticky_min_chars = cache_hit_sticky_min_chars

    def _prefix_key(self, request: RouterRequest) -> str:
        """Build the sticky cache key from leading message content."""
        parts: list[str] = []
        for message in request.messages:
            content = message.content.strip()
            if content:
                parts.append(content)
            if sum(len(part) for part in parts) >= self._cache_hit_sticky_min_chars:
                break
        joined = "\n".join(parts)
        if len(joined) < self._cache_hit_sticky_min_chars:
            return request.session_id
        return joined[: max(self._cache_hit_sticky_min_chars * 4, self._cache_hit_sticky_min_chars)]

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pin the prompt prefix to a warm model with healthy failover."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        ordered_candidates = sorted(eligible_candidates, key=lambda candidate: candidate.model)
        sticky_key = self._prefix_key(request)
        digest = sha256(sticky_key.encode("utf-8")).hexdigest()
        primary_bucket = int(digest[:8], 16) % len(ordered_candidates)

        for offset in range(len(ordered_candidates)):
            bucket = (primary_bucket + offset) % len(ordered_candidates)
            candidate = ordered_candidates[bucket]
            if self._provider_health.is_available(candidate.provider):
                failover_note = (
                    f"failover offset {offset} from primary bucket {primary_bucket}"
                    if offset > 0
                    else f"primary bucket {primary_bucket}"
                )
                rationale = (
                    "cache-hit-sticky-warm-pool pinned prefix to "
                    f"{candidate.model} ({failover_note}/"
                    f"{len(ordered_candidates)} warm ring)"
                )
                return self._decision(candidate.model, rationale)

        fallback_candidate = ordered_candidates[primary_bucket]
        rationale = (
            "cache-hit-sticky-warm-pool pinned prefix to "
            f"{fallback_candidate.model} (primary bucket {primary_bucket}; "
            "no healthy providers; sticky fallback)"
        )
        return self._decision(fallback_candidate.model, rationale)


class EmbeddingCacheKeyNamespaceStrategy(RoutingStrategy):
    """Isolate sticky/cache routing under a tenant embedding-cache namespace.

    Shared embedding or semantic-cache layers collide when multiple tenants hash
    into the same key space. This strategy builds a namespaced sticky key
    ``{prefix}:{tenant_or_session}`` (prefix via
    ``NEXUS_EMBEDDING_CACHE_NAMESPACE_PREFIX``, default ``embed``) from
    ``metadata.tenant_id`` (then ``user_id`` / ``sticky_key`` / ``session_id``)
    and consistent-hashes it onto a domain-eligible model with healthy failover
    so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 cache affinity stays
    isolated across tenants.
    """

    strategy_name = RoutingStrategyName.EMBEDDING_CACHE_KEY_NAMESPACE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        namespace_prefix: str = "embed",
    ) -> None:
        """Initialize embedding-cache key namespace routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            namespace_prefix: Prefix prepended to the tenant sticky key before
                hashing (must be non-empty).

        Raises:
            ValueError: If the namespace prefix is empty or whitespace-only.
        """
        super().__init__(model_catalog)
        cleaned_prefix = namespace_prefix.strip()
        if not cleaned_prefix:
            raise ValueError("namespace_prefix must be a non-empty string")
        self._provider_health = provider_health
        self._namespace_prefix = cleaned_prefix

    def _tenant_scope(self, request: RouterRequest) -> str:
        """Resolve the tenant/session scope used inside the namespace."""
        metadata = request.metadata
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def _namespaced_key(self, request: RouterRequest) -> str:
        """Build the embedding-cache namespace sticky key."""
        return f"{self._namespace_prefix}:{self._tenant_scope(request)}"

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pin the namespaced embedding-cache key to a warm model."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        ordered_candidates = sorted(eligible_candidates, key=lambda candidate: candidate.model)
        namespaced_key = self._namespaced_key(request)
        digest = sha256(namespaced_key.encode("utf-8")).hexdigest()
        primary_bucket = int(digest[:8], 16) % len(ordered_candidates)

        for offset in range(len(ordered_candidates)):
            bucket = (primary_bucket + offset) % len(ordered_candidates)
            candidate = ordered_candidates[bucket]
            if self._provider_health.is_available(candidate.provider):
                failover_note = (
                    f"failover offset {offset} from primary bucket {primary_bucket}"
                    if offset > 0
                    else f"primary bucket {primary_bucket}"
                )
                rationale = (
                    "embedding-cache-key-namespace pinned "
                    f"'{namespaced_key}' to {candidate.model} ({failover_note}/"
                    f"{len(ordered_candidates)} namespace ring)"
                )
                return self._decision(candidate.model, rationale)

        fallback_candidate = ordered_candidates[primary_bucket]
        rationale = (
            "embedding-cache-key-namespace pinned "
            f"'{namespaced_key}' to {fallback_candidate.model} "
            f"(primary bucket {primary_bucket}; no healthy providers; "
            "namespaced sticky fallback)"
        )
        return self._decision(fallback_candidate.model, rationale)


class CircuitBreakerHalfOpenProbeStrategy(RoutingStrategy):
    """Prefer healthy providers; allow limited probes into half-open circuits.

    When a provider circuit opens after failures, the recovery window eventually
    allows a probe. Unbounded probe traffic can slam recovering GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 backends (a LiteLLM/Portkey-style
    gap). This strategy prefers fully closed (healthy) providers, and only
    routes to half-open/recovering providers while live probe load stays under
    ``NEXUS_CIRCUIT_HALF_OPEN_PROBE_BUDGET``.
    """

    strategy_name = RoutingStrategyName.CIRCUIT_BREAKER_HALF_OPEN_PROBE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        inflight_stats: InflightStats,
        probe_budget: int = 2,
    ) -> None:
        """Initialize circuit-breaker half-open probe routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            inflight_stats: Live provider in-flight attempt counters used to
                measure concurrent probe load on recovering providers.
            probe_budget: Maximum concurrent in-flight attempts allowed across
                half-open providers before probes are deferred (must be >= 1).

        Raises:
            ValueError: If the probe budget is less than 1.
        """
        super().__init__(model_catalog)
        if probe_budget < 1:
            raise ValueError(f"probe_budget must be >= 1, got {probe_budget}")
        self._provider_health = provider_health
        self._inflight_stats = inflight_stats
        self._probe_budget = probe_budget

    def _is_half_open(self, provider: str) -> bool:
        """Return whether a provider is in the half-open recovery window."""
        half_open = getattr(self._provider_health, "is_half_open", None)
        if callable(half_open):
            return bool(half_open(provider))
        return False

    def _quality_key(
        self, candidate: ModelCandidate, request: RouterRequest, signals: TaskSignals
    ) -> tuple[float, float, str]:
        """Rank candidates by quality, then cheaper estimated cost, then name."""
        return (
            candidate.quality_score,
            -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            candidate.model,
        )

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer healthy providers; probe half-open ones under the budget."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        healthy_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
            and not self._is_half_open(candidate.provider)
        ]
        half_open_candidates = [
            candidate for candidate in eligible_candidates if self._is_half_open(candidate.provider)
        ]

        if healthy_candidates:
            selected_candidate = max(
                healthy_candidates,
                key=lambda candidate: self._quality_key(candidate, request, signals),
            )
            rationale = (
                "circuit-breaker-half-open-probe preferred healthy provider "
                f"{selected_candidate.provider} (quality "
                f"{selected_candidate.quality_score:.2f}; probe budget "
                f"{self._probe_budget})"
            )
            return self._decision(selected_candidate.model, rationale)

        if half_open_candidates:
            half_open_providers = {candidate.provider for candidate in half_open_candidates}
            probe_inflight = sum(
                self._inflight_stats.load_score(provider) for provider in half_open_providers
            )
            selected_candidate = max(
                half_open_candidates,
                key=lambda candidate: self._quality_key(candidate, request, signals),
            )
            if probe_inflight < self._probe_budget:
                rationale = (
                    "circuit-breaker-half-open-probe allowed recovery probe to "
                    f"{selected_candidate.provider} (half-open load "
                    f"{probe_inflight}/{self._probe_budget}; quality "
                    f"{selected_candidate.quality_score:.2f})"
                )
            else:
                rationale = (
                    "circuit-breaker-half-open-probe probe budget exhausted "
                    f"({probe_inflight}/{self._probe_budget}); routed to "
                    f"half-open fallback {selected_candidate.provider} "
                    f"(quality {selected_candidate.quality_score:.2f})"
                )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: self._quality_key(candidate, request, signals),
        )
        rationale = (
            "circuit-breaker-half-open-probe found no healthy or half-open "
            "provider; routed to highest-quality eligible "
            f"{selected_candidate.model}"
        )
        return self._decision(selected_candidate.model, rationale)


class SemanticCacheTtlAffinityStrategy(RoutingStrategy):
    """Stick cacheable requests to providers with remaining semantic-cache TTL.

    Popular semantic-cache layers (GPTCache, Redis semantic caches, LiteLLM
    cache) keep warm embeddings for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 prompts. This strategy prefers providers whose remaining TTL
    (``cache_ttl_remaining:<provider>`` metadata, seconds) is positive and
    within ``NEXUS_SEMANTIC_CACHE_TTL_SECONDS`` when the request is marked
    cacheable.
    """

    strategy_name = RoutingStrategyName.SEMANTIC_CACHE_TTL_AFFINITY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        ttl_seconds: float = 300.0,
    ) -> None:
        """Initialize semantic-cache TTL affinity routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view.
            ttl_seconds: Maximum TTL window considered warm (seconds).

        Raises:
            ValueError: If ttl_seconds is negative.
        """
        super().__init__(model_catalog)
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")
        self._provider_health = provider_health
        self._ttl_seconds = ttl_seconds

    def _ttl_remaining(self, provider: str, request: RouterRequest) -> float:
        """Return remaining cache TTL seconds for a provider."""
        raw = request.metadata.get(f"cache_ttl_remaining:{provider}") or request.metadata.get(
            f"semantic_cache_ttl:{provider}", ""
        )
        try:
            return max(0.0, float(raw)) if str(raw).strip() else 0.0
        except ValueError:
            return 0.0

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer warm semantic-cache providers for cacheable requests."""
        eligible_candidates = (
            [
                candidate
                for candidate in self._model_catalog.values()
                if signals.domain_tag in candidate.supports_domains
                and self._provider_health.is_available(candidate.provider)
            ]
            or [
                candidate
                for candidate in self._model_catalog.values()
                if self._provider_health.is_available(candidate.provider)
            ]
            or list(self._model_catalog.values())
        )

        cacheable = str(request.metadata.get("cacheable", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "cacheable",
        }
        ttls = {
            candidate.provider: self._ttl_remaining(candidate.provider, request)
            for candidate in eligible_candidates
        }
        warm = [
            candidate
            for candidate in eligible_candidates
            if 0.0 < ttls[candidate.provider] <= self._ttl_seconds
        ]
        if cacheable and warm:
            selected = max(
                warm,
                key=lambda c: (
                    ttls[c.provider],
                    c.quality_score,
                    -c.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    c.model,
                ),
            )
            rationale = (
                "semantic-cache-ttl-affinity pinned warm provider "
                f"{selected.provider} (ttl remaining {ttls[selected.provider]:.1f}s / "
                f"{self._ttl_seconds:.1f}s)"
            )
            return self._decision(selected.model, rationale)

        selected = max(
            eligible_candidates,
            key=lambda c: (
                c.quality_score,
                -c.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                c.model,
            ),
        )
        rationale = (
            "semantic-cache-ttl-affinity fallback quality route to "
            f"{selected.model} on {selected.provider} "
            f"(cacheable={cacheable}; warm_providers={len(warm)})"
        )
        return self._decision(selected.model, rationale)


class ProviderSpendTelemetryStrategy(RoutingStrategy):
    """Prefer lower estimated provider spend when soft spend threshold is exceeded.

    Gateway routers (LiteLLM / Portkey style) increasingly expose spend
    telemetry so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic can
    shed away from high-burn providers. This strategy ranks eligible healthy
    providers by estimated spend from request metadata (``provider_spend_usd``
    keyed as ``spend:<provider>``) once any observed spend meets the soft USD
    threshold.
    """

    strategy_name = RoutingStrategyName.PROVIDER_SPEND_TELEMETRY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        soft_spend_usd: float = 10.0,
    ) -> None:
        """Initialize provider spend telemetry routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view.
            soft_spend_usd: Soft spend threshold in USD that activates
                spend-aware preference (must be >= 0).

        Raises:
            ValueError: If soft_spend_usd is negative.
        """
        super().__init__(model_catalog)
        if soft_spend_usd < 0:
            raise ValueError(f"soft_spend_usd must be >= 0, got {soft_spend_usd}")
        self._provider_health = provider_health
        self._soft_spend_usd = soft_spend_usd

    def _spend_for(self, provider: str, request: RouterRequest) -> float:
        """Return estimated spend USD for a provider from request metadata."""
        raw = request.metadata.get(f"spend:{provider}") or request.metadata.get(
            f"provider_spend_usd:{provider}", ""
        )
        try:
            return max(0.0, float(raw)) if str(raw).strip() else 0.0
        except ValueError:
            return 0.0

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a model preferring lower spend when soft threshold is met."""
        eligible_candidates = (
            [
                candidate
                for candidate in self._model_catalog.values()
                if signals.domain_tag in candidate.supports_domains
                and self._provider_health.is_available(candidate.provider)
            ]
            or [
                candidate
                for candidate in self._model_catalog.values()
                if self._provider_health.is_available(candidate.provider)
            ]
            or list(self._model_catalog.values())
        )

        spends = {
            candidate.provider: self._spend_for(candidate.provider, request)
            for candidate in eligible_candidates
        }
        max_spend = max(spends.values()) if spends else 0.0
        if max_spend < self._soft_spend_usd:
            selected = max(
                eligible_candidates,
                key=lambda c: (
                    c.quality_score,
                    -c.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    c.model,
                ),
            )
            rationale = (
                "provider-spend-telemetry under soft spend "
                f"{self._soft_spend_usd:.2f} USD (max observed {max_spend:.2f}); "
                f"selected {selected.model} on {selected.provider}"
            )
            return self._decision(selected.model, rationale)

        selected = min(
            eligible_candidates,
            key=lambda c: (
                spends[c.provider],
                -c.quality_score,
                c.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                c.model,
            ),
        )
        rationale = (
            "provider-spend-telemetry preferred lower-spend provider "
            f"{selected.provider} (spend {spends[selected.provider]:.2f} USD; "
            f"soft threshold {self._soft_spend_usd:.2f})"
        )
        return self._decision(selected.model, rationale)


class CarbonAwarePreferenceStrategy(RoutingStrategy):
    """Prefer providers with lower carbon intensity for eligible models.

    Sustainability-aware gateways increasingly bias GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 traffic toward lower-carbon regions. This strategy
    reads ``carbon_intensity:<provider>`` (gCO2eq/kWh) from request metadata
    (with a regional heuristic fallback) and prefers intensities at or below
    ``NEXUS_CARBON_AWARE_MAX_INTENSITY``.
    """

    strategy_name = RoutingStrategyName.CARBON_AWARE_PREFERENCE

    _REGION_DEFAULTS = {
        "eu": 250.0,
        "us": 380.0,
        "cn": 550.0,
        "global": 420.0,
    }

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        max_intensity: float = 400.0,
    ) -> None:
        """Initialize carbon-aware preference routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view.
            max_intensity: Soft maximum carbon intensity (gCO2eq/kWh).

        Raises:
            ValueError: If max_intensity is negative.
        """
        super().__init__(model_catalog)
        if max_intensity < 0:
            raise ValueError(f"max_intensity must be >= 0, got {max_intensity}")
        self._provider_health = provider_health
        self._max_intensity = max_intensity

    def _intensity_for(self, candidate: ModelCandidate, request: RouterRequest) -> float:
        """Resolve carbon intensity for a candidate from metadata or region."""
        raw = request.metadata.get(f"carbon_intensity:{candidate.provider}")
        if raw is not None and str(raw).strip():
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        region = (
            request.metadata.get("region") or request.metadata.get("preferred_region") or "global"
        ).lower()
        return float(self._REGION_DEFAULTS.get(region, self._REGION_DEFAULTS["global"]))

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer lower-carbon eligible healthy providers under the intensity cap."""
        eligible_candidates = (
            [
                candidate
                for candidate in self._model_catalog.values()
                if signals.domain_tag in candidate.supports_domains
                and self._provider_health.is_available(candidate.provider)
            ]
            or [
                candidate
                for candidate in self._model_catalog.values()
                if self._provider_health.is_available(candidate.provider)
            ]
            or list(self._model_catalog.values())
        )

        intensities = {
            candidate.model: self._intensity_for(candidate, request)
            for candidate in eligible_candidates
        }
        under_cap = [c for c in eligible_candidates if intensities[c.model] <= self._max_intensity]
        pool = under_cap or eligible_candidates
        selected = min(
            pool,
            key=lambda c: (
                intensities[c.model],
                -c.quality_score,
                c.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                c.model,
            ),
        )
        rationale = (
            "carbon-aware-preference selected "
            f"{selected.model} on {selected.provider} "
            f"(intensity {intensities[selected.model]:.1f} gCO2eq/kWh; "
            f"max {self._max_intensity:.1f})"
        )
        return self._decision(selected.model, rationale)


class TenantConcurrencyLeaseStrategy(RoutingStrategy):
    """Prefer providers with remaining per-tenant concurrency lease headroom.

    Global concurrency caps ignore multi-tenant fairness: one noisy tenant can
    saturate a provider for everyone else. This strategy resolves a tenant key
    from ``metadata.tenant_id`` (then ``user_id`` / ``sticky_key`` /
    ``session_id``) and prefers domain-eligible models whose
    tenant-scoped ``InflightStats`` load stays below
    ``NEXUS_TENANT_CONCURRENCY_LEASE``. When every provider is at the lease it
    falls back to the least-loaded tenant/provider pair so GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic keeps moving.
    """

    strategy_name = RoutingStrategyName.TENANT_CONCURRENCY_LEASE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        inflight_stats: InflightStats,
        tenant_concurrency_lease: int = 8,
    ) -> None:
        """Initialize per-tenant concurrency lease routing.

        Args:
            model_catalog: Available model candidates by model name.
            inflight_stats: Live in-flight counters, including tenant-scoped keys.
            tenant_concurrency_lease: Maximum concurrent in-flight attempts per
                tenant/provider pair before that provider is skipped (must be >= 1).

        Raises:
            ValueError: If the lease is less than 1.
        """
        super().__init__(model_catalog)
        if tenant_concurrency_lease < 1:
            raise ValueError(
                f"tenant_concurrency_lease must be >= 1, got {tenant_concurrency_lease}"
            )
        self._inflight_stats = inflight_stats
        self._tenant_concurrency_lease = tenant_concurrency_lease

    def _tenant_key(self, request: RouterRequest) -> str:
        """Resolve the tenant/session key used for lease accounting."""
        metadata = request.metadata
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best model still under the tenant concurrency lease."""
        tenant_key = self._tenant_key(request)
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        under_lease_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._inflight_stats.tenant_load_score(tenant_key, candidate.provider)
            < self._tenant_concurrency_lease
        ]

        if under_lease_candidates:
            selected_candidate = max(
                under_lease_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            load_score = self._inflight_stats.tenant_load_score(
                tenant_key, selected_candidate.provider
            )
            rationale = (
                "tenant-concurrency-lease selected under lease "
                f"{self._tenant_concurrency_lease} for tenant '{tenant_key}'; "
                f"{selected_candidate.provider} load {load_score}/"
                f"{self._tenant_concurrency_lease} with highest eligible quality "
                f"{selected_candidate.quality_score:.2f}"
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    self._inflight_stats.tenant_load_score(tenant_key, candidate.provider),
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            load_score = self._inflight_stats.tenant_load_score(
                tenant_key, selected_candidate.provider
            )
            rationale = (
                "tenant-concurrency-lease found every eligible provider at or above "
                f"lease {self._tenant_concurrency_lease} for tenant '{tenant_key}'; "
                f"routed to least-loaded fallback {selected_candidate.provider} load "
                f"{load_score}/{self._tenant_concurrency_lease}"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                self._inflight_stats.tenant_load_score(tenant_key, candidate.provider)
                >= self._tenant_concurrency_lease,
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class ProviderErrorBudgetShedStrategy(RoutingStrategy):
    """Prefer providers whose rolling error rate stays within budget.

    Large multi-provider fleets need a soft error-budget guard before circuit
    breakers fully open. This strategy filters to domain-eligible healthy
    providers, keeps providers whose ``SuccessStats``-derived error rate is at
    or below ``NEXUS_PROVIDER_ERROR_BUDGET_RATE`` (default ``0.15``), then picks
    the highest-quality model. If every eligible provider is over budget, it
    sheds toward the lowest error rate before quality so GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic avoids degrading backends.
    """

    strategy_name = RoutingStrategyName.PROVIDER_ERROR_BUDGET_SHED

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        success_stats: SuccessStats,
        provider_error_budget_rate: float = 0.15,
    ) -> None:
        """Initialize provider-error-budget-shed routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view.
            success_stats: Rolling provider success observations.
            provider_error_budget_rate: Maximum acceptable rolling error rate
                within ``[0.0, 1.0]`` before a provider is shed.

        Raises:
            ValueError: If the error-budget rate is outside ``[0.0, 1.0]``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= provider_error_budget_rate <= 1.0:
            raise ValueError(
                "provider_error_budget_rate must be within [0.0, 1.0], "
                f"got {provider_error_budget_rate}"
            )
        self._provider_health = provider_health
        self._success_stats = success_stats
        self._provider_error_budget_rate = provider_error_budget_rate

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the highest-quality healthy provider under the error budget."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        healthy_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
        ]
        active_candidates = healthy_candidates or eligible_candidates
        error_rates = {
            candidate.model: self._provider_error_rate(candidate.provider)
            for candidate in active_candidates
        }
        under_budget = [
            candidate
            for candidate in active_candidates
            if error_rates[candidate.model] <= self._provider_error_budget_rate
        ]

        if under_budget:
            selected_candidate = max(
                under_budget,
                key=lambda candidate: (
                    candidate.quality_score,
                    -error_rates[candidate.model],
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "provider-error-budget-shed selected highest-quality provider under "
                f"error budget {self._provider_error_budget_rate:.2%}; "
                f"{selected_candidate.provider} error "
                f"{error_rates[selected_candidate.model]:.2%}, quality "
                f"{selected_candidate.quality_score:.2f}"
            )
            if not healthy_candidates:
                rationale += "; no closed circuits were available"
        else:
            selected_candidate = min(
                active_candidates,
                key=lambda candidate: (
                    error_rates[candidate.model],
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "provider-error-budget-shed found every eligible provider over "
                f"error budget {self._provider_error_budget_rate:.2%}; "
                f"routed to lowest-error provider {selected_candidate.provider} "
                f"({error_rates[selected_candidate.model]:.2%}) with quality "
                f"{selected_candidate.quality_score:.2f}"
            )
            if not healthy_candidates:
                rationale += "; no closed circuits were available"

        fallback_candidates = self._ordered_fallback_candidates(
            selected_candidate.model,
            eligible_candidates,
            request,
            signals,
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )

    def _provider_error_rate(self, provider: str) -> float:
        """Return a provider's rolling error rate from success observations."""
        return max(0.0, min(1.0, 1.0 - self._success_stats.success_rate(provider)))

    def _ordered_fallback_candidates(
        self,
        chosen_model: str,
        eligible_candidates: list[ModelCandidate],
        request: RouterRequest,
        signals: TaskSignals,
    ) -> list[ModelCandidate]:
        """Order fallbacks by health, error-budget status, error, and quality."""
        return sorted(
            [candidate for candidate in eligible_candidates if candidate.model != chosen_model],
            key=lambda candidate: (
                not self._provider_health.is_available(candidate.provider),
                self._provider_error_rate(candidate.provider) > self._provider_error_budget_rate,
                self._provider_error_rate(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )


class RegionLatencyP99ShedStrategy(RoutingStrategy):
    """Shed region providers whose recent p99 latency exceeds a threshold.

    Multi-region fleets often need a softer tail-latency gate than p95 SLO
    shedding: prefer domain-eligible models that match the request region, then
    keep providers whose rolling ``LatencyStats`` p99 stays at or below
    ``NEXUS_REGION_LATENCY_P99_MS``. When every regional candidate is over the
    threshold it falls back to the lowest p99 so GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 traffic still routes deterministically.
    """

    strategy_name = RoutingStrategyName.REGION_LATENCY_P99_SHED

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        region_latency_p99_ms: float = 3000.0,
    ) -> None:
        """Initialize region-latency-p99-shed routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            region_latency_p99_ms: Maximum acceptable provider p99 latency in
                milliseconds for primary selection.

        Raises:
            ValueError: If the p99 threshold is negative.
        """
        super().__init__(model_catalog)
        if region_latency_p99_ms < 0.0:
            raise ValueError(
                f"region_latency_p99_ms must be non-negative, got {region_latency_p99_ms}"
            )
        self._latency_stats = latency_stats
        self._region_latency_p99_ms = region_latency_p99_ms

    def _matches_region(self, candidate: ModelCandidate, region: str) -> bool:
        """Return whether a candidate advertises the requested region."""
        return region in {supported.lower() for supported in candidate.supported_regions}

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer under-p99 regional candidates and shed slower providers."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        requested_region = (request.region or "global").strip().lower()
        region_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._matches_region(candidate, requested_region)
        ] or eligible_candidates

        p99_by_model = {
            candidate.model: self._latency_stats.p99(candidate.provider)
            for candidate in region_candidates
        }
        under_threshold = [
            candidate
            for candidate in region_candidates
            if p99_by_model[candidate.model] <= self._region_latency_p99_ms
        ]

        if under_threshold:
            selected_candidate = max(
                under_threshold,
                key=lambda candidate: (
                    candidate.quality_score,
                    -p99_by_model[candidate.model],
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "region-latency-p99-shed selected highest quality "
                f"{selected_candidate.quality_score:.2f} in region '{requested_region}' "
                f"under {self._region_latency_p99_ms:.0f}ms p99 "
                f"(provider p99 {p99_by_model[selected_candidate.model]:.1f}ms; "
                "shed slower alternatives)"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            region_candidates,
            key=lambda candidate: (
                p99_by_model[candidate.model],
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        rationale = (
            "region-latency-p99-shed found no provider in region "
            f"'{requested_region}' under {self._region_latency_p99_ms:.0f}ms p99; "
            f"shed fallback to lowest-p99 model "
            f"(provider p99 {p99_by_model[selected_candidate.model]:.1f}ms)"
        )
        return self._decision(selected_candidate.model, rationale)


class StickyCanaryCostStrategy(RoutingStrategy):
    """Sticky tenant routing with canary cost-aware blend toward cheaper models.

    Pins tenants via consistent hashing on tenant identity (like
    ``sticky-tenant-hash``), then on a deterministic ``request_id`` explore
    slice blends toward cheaper healthy models. The majority of GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic keeps sticky affinity
    while ``NEXUS_STICKY_CANARY_COST_PERCENT`` samples lower-cost SKUs.
    """

    strategy_name = RoutingStrategyName.STICKY_CANARY_COST

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        sticky_canary_cost_percent: float = 10.0,
    ) -> None:
        """Initialize sticky-canary-cost routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view (circuit breaker).
            sticky_canary_cost_percent: Percentage of traffic that explores a
                cheaper healthy model, within ``[0.0, 100.0]``.

        Raises:
            ValueError: If the explore percentage is outside ``[0.0, 100.0]``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= sticky_canary_cost_percent <= 100.0:
            raise ValueError(
                "sticky_canary_cost_percent must be within [0.0, 100.0], "
                f"got {sticky_canary_cost_percent}"
            )
        self._provider_health = provider_health
        self._sticky_canary_cost_percent = sticky_canary_cost_percent

    def _sticky_key(self, request: RouterRequest) -> str:
        """Resolve the tenant sticky key for consistent hashing."""
        metadata = request.metadata
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def _explore_bucket(self, request: RouterRequest) -> float:
        digest = sha256(request.request_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _in_explore_slice(self, request: RouterRequest) -> bool:
        return self._explore_bucket(request) < (self._sticky_canary_cost_percent / 100.0)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pin sticky tenants, with a cost-aware canary explore slice."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        ordered_candidates = sorted(eligible_candidates, key=lambda candidate: candidate.model)
        sticky_key = self._sticky_key(request)
        digest = sha256(sticky_key.encode("utf-8")).hexdigest()
        primary_bucket = int(digest[:8], 16) % len(ordered_candidates)
        sticky_primary = ordered_candidates[primary_bucket]
        sticky_cost = sticky_primary.estimate_cost(
            signals.prompt_tokens_estimate, request.max_tokens
        )
        bucket = self._explore_bucket(request)

        if self._in_explore_slice(request):
            cheaper_healthy = [
                candidate
                for candidate in eligible_candidates
                if self._provider_health.is_available(candidate.provider)
                and candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens)
                < sticky_cost
            ]
            if cheaper_healthy:
                selected_candidate = min(
                    cheaper_healthy,
                    key=lambda candidate: (
                        candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                        -candidate.quality_score,
                        candidate.model,
                    ),
                )
                selected_cost = selected_candidate.estimate_cost(
                    signals.prompt_tokens_estimate, request.max_tokens
                )
                rationale = (
                    "sticky-canary-cost explore slice "
                    f"(bucket={bucket:.4f} < {self._sticky_canary_cost_percent:.1f}%); "
                    f"blended tenant '{sticky_key}' from sticky {sticky_primary.model} "
                    f"toward cheaper healthy {selected_candidate.model} "
                    f"(est ${selected_cost:.6f} < sticky ${sticky_cost:.6f})"
                )
                return self._decision(selected_candidate.model, rationale)

        for offset in range(len(ordered_candidates)):
            ring_bucket = (primary_bucket + offset) % len(ordered_candidates)
            candidate = ordered_candidates[ring_bucket]
            if self._provider_health.is_available(candidate.provider):
                explore_note = (
                    f"explore slice bucket={bucket:.4f} but no cheaper healthy option; "
                    if self._in_explore_slice(request)
                    else f"bucket={bucket:.4f} >= {self._sticky_canary_cost_percent:.1f}%; "
                )
                failover_note = (
                    f"failover offset {offset} from primary bucket {primary_bucket}"
                    if offset > 0
                    else f"primary bucket {primary_bucket}"
                )
                rationale = (
                    "sticky-canary-cost "
                    f"{explore_note}pinned tenant '{sticky_key}' to "
                    f"{candidate.model} ({failover_note}/"
                    f"{len(ordered_candidates)} healthy ring)"
                )
                return self._decision(candidate.model, rationale)

        rationale = (
            f"sticky-canary-cost pinned tenant '{sticky_key}' to "
            f"{sticky_primary.model} (primary bucket {primary_bucket}; "
            "no healthy providers; sticky fallback)"
        )
        return self._decision(sticky_primary.model, rationale)


class QueueDepthFairnessStrategy(RoutingStrategy):
    """Prefer providers with lower local queue/inflight depth for fair sharing.

    Multi-tenant fleets can starve quieter tenants when traffic concentrates on
    one high-quality provider. This strategy treats live ``InflightStats`` load as
    queue depth: providers at or above ``NEXUS_QUEUE_DEPTH_SOFT_CAP`` are shed
    when shallower alternatives exist. Among under-cap providers it picks highest
    quality; when every provider is deep it falls back to lowest depth so
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic keeps moving.
    """

    strategy_name = RoutingStrategyName.QUEUE_DEPTH_FAIRNESS

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        inflight_stats: InflightStats,
        queue_depth_soft_cap: int = 4,
    ) -> None:
        """Initialize queue-depth-fairness routing.

        Args:
            model_catalog: Available model candidates by model name.
            inflight_stats: Live provider in-flight / queue-depth counters.
            queue_depth_soft_cap: Soft maximum in-flight depth per provider
                before that provider is shed (must be >= 1).

        Raises:
            ValueError: If the soft cap is less than 1.
        """
        super().__init__(model_catalog)
        if queue_depth_soft_cap < 1:
            raise ValueError(f"queue_depth_soft_cap must be >= 1, got {queue_depth_soft_cap}")
        self._inflight_stats = inflight_stats
        self._queue_depth_soft_cap = queue_depth_soft_cap

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best model still under the soft queue-depth cap."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        depths = {
            candidate.model: self._inflight_stats.load_score(candidate.provider)
            for candidate in eligible_candidates
        }
        under_cap = [
            candidate
            for candidate in eligible_candidates
            if depths[candidate.model] < self._queue_depth_soft_cap
        ]

        if under_cap:
            selected_candidate = max(
                under_cap,
                key=lambda candidate: (
                    candidate.quality_score,
                    -depths[candidate.model],
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "queue-depth-fairness selected under soft cap "
                f"{self._queue_depth_soft_cap}; {selected_candidate.provider} depth "
                f"{depths[selected_candidate.model]}/{self._queue_depth_soft_cap} with "
                f"highest eligible quality {selected_candidate.quality_score:.2f}"
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    depths[candidate.model],
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "queue-depth-fairness found every eligible provider at or above soft cap "
                f"{self._queue_depth_soft_cap}; routed to lowest-depth fallback "
                f"{selected_candidate.provider} depth "
                f"{depths[selected_candidate.model]}/{self._queue_depth_soft_cap}"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                depths[candidate.model] >= self._queue_depth_soft_cap,
                depths[candidate.model],
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class ProviderRequestShareStats:
    """Rolling provider selections used by fair-share quota routing."""

    def __init__(self, lookback: int = 100) -> None:
        """Initialize the bounded provider request window.

        Args:
            lookback: Maximum recent routing decisions retained.

        Raises:
            ValueError: If ``lookback`` is less than one.
        """
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self._lookback = lookback
        self._observations: list[str] = []

    @property
    def lookback(self) -> int:
        """Return the configured rolling-window size."""
        return self._lookback

    def observe(self, provider: str) -> None:
        """Record one provider routing decision."""
        self._observations.append(provider)
        if len(self._observations) > self._lookback:
            del self._observations[0]

    def observation_count(self, eligible_providers: Iterable[str]) -> int:
        """Return observations belonging to the currently eligible provider set."""
        eligible = set(eligible_providers)
        return sum(provider in eligible for provider in self._observations)

    def request_share(self, provider: str, eligible_providers: Iterable[str]) -> float:
        """Return a provider's recent share among currently eligible providers."""
        eligible = set(eligible_providers)
        eligible_observations = [
            observed_provider
            for observed_provider in self._observations
            if observed_provider in eligible
        ]
        if not eligible_observations:
            return 0.0
        return eligible_observations.count(provider) / len(eligible_observations)


class ProviderQuotaFairShareStrategy(RoutingStrategy):
    """Shed providers above their equal share of recent eligible traffic.

    The strategy tracks a bounded window of provider selections. Providers below
    the equal fair share (``1 / eligible providers``) form the preferred pool;
    providers at or above that share are shed while an under-share alternative
    exists. Quality and cost retain deterministic tie-breaking within the pool.
    """

    strategy_name = RoutingStrategyName.PROVIDER_QUOTA_FAIR_SHARE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        request_share_stats: ProviderRequestShareStats,
    ) -> None:
        """Initialize provider-quota-fair-share routing."""
        super().__init__(model_catalog)
        self._request_share_stats = request_share_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer providers below equal share and record the selected provider."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        eligible_providers = {candidate.provider for candidate in eligible_candidates}
        fair_share = 1.0 / len(eligible_providers)
        shares = {
            provider: self._request_share_stats.request_share(provider, eligible_providers)
            for provider in eligible_providers
        }
        observation_count = self._request_share_stats.observation_count(eligible_providers)
        under_share = [
            candidate
            for candidate in eligible_candidates
            if shares[candidate.provider] < fair_share
        ]
        preferred_candidates = under_share or eligible_candidates
        selected_candidate = max(
            preferred_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )

        if observation_count == 0:
            rationale = (
                "provider-quota-fair-share cold start; selected highest-quality eligible "
                f"{selected_candidate.model} with equal target {fair_share:.2%} "
                f"over lookback {self._request_share_stats.lookback}"
            )
        elif under_share:
            rationale = (
                "provider-quota-fair-share shed providers at or above equal share "
                f"{fair_share:.2%}; selected under-share provider "
                f"{selected_candidate.provider} at {shares[selected_candidate.provider]:.2%} "
                f"over {observation_count}/{self._request_share_stats.lookback} observations"
            )
        else:
            rationale = (
                "provider-quota-fair-share found eligible provider shares balanced at "
                f"{fair_share:.2%}; selected highest-quality {selected_candidate.model}"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                shares[candidate.provider] >= fair_share,
                shares[candidate.provider],
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        self._request_share_stats.observe(selected_candidate.provider)
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class AdaptiveTimeoutHedgeStrategy(RoutingStrategy):
    """Hedge a slow quality-first choice using a latency-derived threshold.

    Unlike ``adaptive-timeout``, which filters candidates against a timeout
    budget and provider success risk, this strategy first picks the
    highest-quality eligible model. It hedges only when that provider's rolling
    p95 exceeds the fastest observed eligible p95 multiplied by a configurable
    ratio. Unlike ``multi-region-latency-hedge``, it has no region boundary and
    compares rolling p95 rather than a fixed p50 threshold.
    """

    strategy_name = RoutingStrategyName.ADAPTIVE_TIMEOUT_HEDGE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        hedge_ratio: float = 1.5,
    ) -> None:
        """Initialize adaptive-timeout-hedge routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            hedge_ratio: Multiplier applied to the fastest observed eligible p95.

        Raises:
            ValueError: If ``hedge_ratio`` is less than one.
        """
        super().__init__(model_catalog)
        if hedge_ratio < 1.0:
            raise ValueError(f"hedge_ratio must be >= 1.0, got {hedge_ratio}")
        self._latency_stats = latency_stats
        self._hedge_ratio = hedge_ratio

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Keep the quality leader unless adaptive latency pressure triggers a hedge."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        top_choice = max(
            eligible_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        p95_by_model = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        observed_candidates = [
            candidate for candidate in eligible_candidates if p95_by_model[candidate.model] > 0.0
        ]
        top_p95 = p95_by_model[top_choice.model]

        if not observed_candidates or top_p95 <= 0.0:
            rationale = (
                "adaptive-timeout-hedge kept highest-quality "
                f"{top_choice.model}; insufficient positive p95 observations "
                "for an adaptive hedge threshold"
            )
            return self._decision(top_choice.model, rationale)

        fastest_p95 = min(p95_by_model[candidate.model] for candidate in observed_candidates)
        adaptive_threshold = fastest_p95 * self._hedge_ratio
        faster_alternatives = [
            candidate
            for candidate in observed_candidates
            if candidate.provider != top_choice.provider and p95_by_model[candidate.model] < top_p95
        ]

        if top_p95 > adaptive_threshold and faster_alternatives:
            selected_candidate = min(
                faster_alternatives,
                key=lambda candidate: (
                    p95_by_model[candidate.model],
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "adaptive-timeout-hedge hedged from quality leader "
                f"{top_choice.model} at p95 {top_p95:.1f}ms because it exceeded "
                f"adaptive threshold {adaptive_threshold:.1f}ms "
                f"(fastest observed {fastest_p95:.1f}ms x {self._hedge_ratio:.2f}); "
                f"selected faster alternative {selected_candidate.model} at p95 "
                f"{p95_by_model[selected_candidate.model]:.1f}ms"
            )
            return self._decision(selected_candidate.model, rationale)

        rationale = (
            "adaptive-timeout-hedge kept quality leader "
            f"{top_choice.model} at p95 {top_p95:.1f}ms within adaptive threshold "
            f"{adaptive_threshold:.1f}ms (fastest observed {fastest_p95:.1f}ms x "
            f"{self._hedge_ratio:.2f})"
        )
        if not faster_alternatives:
            rationale += "; no faster observed provider alternative"
        return self._decision(top_choice.model, rationale)


@dataclass
class _TenantTokenBucket:
    """Mutable per-tenant request-token bucket."""

    tokens: float
    updated_at: float


class TenantTokenBucketStats:
    """Per-tenant token buckets with a one-second burst horizon."""

    def __init__(self, rate_per_second: float = 5.0) -> None:
        """Initialize tenant request-token budgets.

        Each request consumes one quota token. Bucket capacity is one second of
        configured rate, with a minimum one-request burst.

        Args:
            rate_per_second: Quota tokens refilled per tenant per second.

        Raises:
            ValueError: If the refill rate is not positive.
        """
        if rate_per_second <= 0.0:
            raise ValueError(f"rate_per_second must be positive, got {rate_per_second}")
        self._rate_per_second = rate_per_second
        self._capacity = max(1.0, rate_per_second)
        self._buckets: dict[str, _TenantTokenBucket] = {}

    @property
    def rate_per_second(self) -> float:
        """Return the configured per-tenant quota-token rate."""
        return self._rate_per_second

    @property
    def capacity(self) -> float:
        """Return the one-second burst capacity."""
        return self._capacity

    def available_tokens(self, tenant_key: str) -> float:
        """Return a tenant's available quota tokens after refill."""
        return self._refill(tenant_key)

    def try_consume(self, tenant_key: str) -> bool:
        """Consume one request token when budget remains."""
        available = self._refill(tenant_key)
        if available < 1.0:
            return False
        self._buckets[tenant_key].tokens = available - 1.0
        return True

    def _refill(self, tenant_key: str) -> float:
        """Refill and return one tenant bucket using monotonic time."""
        now = time.monotonic()
        bucket = self._buckets.get(tenant_key)
        if bucket is None:
            bucket = _TenantTokenBucket(tokens=self._capacity, updated_at=now)
            self._buckets[tenant_key] = bucket
            return bucket.tokens

        elapsed_seconds = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(
            self._capacity,
            bucket.tokens + elapsed_seconds * self._rate_per_second,
        )
        bucket.updated_at = now
        return bucket.tokens


class TokenBucketTenantStrategy(RoutingStrategy):
    """Use a per-tenant token bucket to shed over-budget traffic to low cost.

    ``token-bucket-burst`` balances provider-side burst quota. This strategy
    instead keys one request-token bucket by tenant identity. In-budget requests
    keep quality-first selection; requests arriving without a tenant quota token
    are still served but shed to the cheapest eligible model.
    """

    strategy_name = RoutingStrategyName.TOKEN_BUCKET_TENANT

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        tenant_token_bucket_stats: TenantTokenBucketStats,
    ) -> None:
        """Initialize per-tenant token-bucket routing."""
        super().__init__(model_catalog)
        self._tenant_token_bucket_stats = tenant_token_bucket_stats

    @staticmethod
    def _tenant_key(request: RouterRequest) -> str:
        """Resolve tenant identity from metadata, user, then session."""
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = request.metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Keep quality within tenant budget and shed over-budget requests to cost."""
        tenant_key = self._tenant_key(request)
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        within_budget = self._tenant_token_bucket_stats.try_consume(tenant_key)

        if within_budget:
            selected_candidate = max(
                eligible_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            remaining = self._tenant_token_bucket_stats.available_tokens(tenant_key)
            rationale = (
                f"token-bucket-tenant tenant '{tenant_key}' within budget; "
                f"selected highest-quality {selected_candidate.model} with "
                f"{remaining:.2f}/{self._tenant_token_bucket_stats.capacity:.2f} "
                "request tokens remaining"
            )
            fallback_candidates = sorted(
                [
                    candidate
                    for candidate in eligible_candidates
                    if candidate.model != selected_candidate.model
                ],
                key=lambda candidate: (
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            estimated_cost = selected_candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            rationale = (
                f"token-bucket-tenant tenant '{tenant_key}' over "
                f"{self._tenant_token_bucket_stats.rate_per_second:.2f}/s budget; "
                f"shed to cheapest eligible {selected_candidate.model} "
                f"at estimated ${estimated_cost:.6f}"
            )
            fallback_candidates = sorted(
                [
                    candidate
                    for candidate in eligible_candidates
                    if candidate.model != selected_candidate.model
                ],
                key=lambda candidate: (
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    -candidate.quality_score,
                    candidate.model,
                ),
            )

        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class RegionCarbonBlendStrategy(RoutingStrategy):
    """Blend regional carbon intensity preference with latency scoring.

    Sustainability-aware gateways often want a tunable mix of green routing and
    responsiveness. This strategy scores domain-eligible models by blending an
    inverse-normalized carbon intensity (lower gCO2eq/kWh is better) with an
    inverse-normalized rolling provider p95 latency. ``blend_weight`` of ``0``
    is latency-only; ``1`` is carbon-only. Intensity comes from
    ``carbon_intensity:<provider>`` metadata or a regional heuristic, matching
    carbon-aware-preference for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 traffic.
    """

    strategy_name = RoutingStrategyName.REGION_CARBON_BLEND

    _REGION_DEFAULTS = {
        "eu": 250.0,
        "us": 380.0,
        "cn": 550.0,
        "global": 420.0,
    }

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        blend_weight: float = 0.5,
    ) -> None:
        """Initialize region-carbon-blend routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_stats: Rolling provider latency observations.
            blend_weight: Carbon share in ``[0.0, 1.0]`` (``0`` = latency only).

        Raises:
            ValueError: If blend_weight is outside ``[0.0, 1.0]``.
        """
        super().__init__(model_catalog)
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError(f"blend_weight must be within [0.0, 1.0], got {blend_weight}")
        self._latency_stats = latency_stats
        self._blend_weight = blend_weight

    def _intensity_for(self, candidate: ModelCandidate, request: RouterRequest) -> float:
        """Resolve carbon intensity for a candidate from metadata or region."""
        raw = request.metadata.get(f"carbon_intensity:{candidate.provider}")
        if raw is not None and str(raw).strip():
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        region = (
            request.metadata.get("region")
            or request.metadata.get("preferred_region")
            or getattr(request, "region", None)
            or "global"
        )
        region_key = str(region).lower()
        return float(self._REGION_DEFAULTS.get(region_key, self._REGION_DEFAULTS["global"]))

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Select by blended carbon and latency scores."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        intensities = {
            candidate.model: self._intensity_for(candidate, request)
            for candidate in eligible_candidates
        }
        latencies = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        carbon_scores = _inverse_min_max(intensities)
        latency_scores = _inverse_min_max(latencies)
        carbon_weight = self._blend_weight
        latency_weight = 1.0 - carbon_weight

        def blended_score(candidate: ModelCandidate) -> float:
            return (
                carbon_weight * carbon_scores[candidate.model]
                + latency_weight * latency_scores[candidate.model]
            )

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                blended_score(candidate),
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        rationale = (
            "region-carbon-blend selected "
            f"{selected_candidate.model} on {selected_candidate.provider} "
            f"(score {blended_score(selected_candidate):.3f}; "
            f"carbon_weight={carbon_weight:.2f}, "
            f"intensity {intensities[selected_candidate.model]:.1f} gCO2eq/kWh, "
            f"p95 {latencies[selected_candidate.model]:.1f}ms)"
        )
        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                -blended_score(candidate),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class ProviderWeightStats:
    """Per-provider selection weights with exponential decay and slow recovery."""

    def __init__(self, decay_factor: float = 0.5, recover: float = 0.1) -> None:
        """Initialize provider selection weights.

        Args:
            decay_factor: Multiplicative penalty applied after each failure in
                ``(0.0, 1.0]``.
            recover: Additive recovery applied after each success (``>= 0``).

        Raises:
            ValueError: If ``decay_factor`` is outside ``(0.0, 1.0]`` or
                ``recover`` is negative.
        """
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be within (0.0, 1.0], got {decay_factor}")
        if recover < 0.0:
            raise ValueError(f"recover must be >= 0, got {recover}")
        self._decay_factor = decay_factor
        self._recover = recover
        self._weights: dict[str, float] = {}

    @property
    def decay_factor(self) -> float:
        """Return the configured failure decay factor."""
        return self._decay_factor

    @property
    def recover(self) -> float:
        """Return the configured success recovery step."""
        return self._recover

    def weight(self, provider: str) -> float:
        """Return the current selection weight (defaults to ``1.0``)."""
        return self._weights.get(provider, 1.0)

    def observe(self, provider: str, *, success: bool) -> None:
        """Decay on failure or slowly recover toward ``1.0`` on success."""
        current = self.weight(provider)
        if success:
            self._weights[provider] = min(1.0, current + self._recover)
        else:
            self._weights[provider] = current * self._decay_factor


class ProviderWeightDecayStrategy(RoutingStrategy):
    """Select providers using exponentially decaying selection weights.

    Repeated failures multiply a provider's weight by
    ``NEXUS_PROVIDER_WEIGHT_DECAY_FACTOR``; successes add
    ``NEXUS_PROVIDER_WEIGHT_RECOVER`` capped at ``1.0``. Routing multiplies the
    live weight by model quality so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 traffic gradually shifts away from failing backends and recovers
    slowly once they stabilize.
    """

    strategy_name = RoutingStrategyName.PROVIDER_WEIGHT_DECAY

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_weight_stats: ProviderWeightStats,
    ) -> None:
        """Initialize provider-weight-decay routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_weight_stats: Shared per-provider selection weights.
        """
        super().__init__(model_catalog)
        self._provider_weight_stats = provider_weight_stats

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer higher weight times quality among domain-eligible models."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        def score(candidate: ModelCandidate) -> float:
            return self._provider_weight_stats.weight(candidate.provider) * candidate.quality_score

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                score(candidate),
                self._provider_weight_stats.weight(candidate.provider),
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        weight = self._provider_weight_stats.weight(selected_candidate.provider)
        rationale = (
            "provider-weight-decay selected "
            f"{selected_candidate.model} on {selected_candidate.provider} "
            f"(weight {weight:.3f}; score {score(selected_candidate):.3f}; "
            f"decay={self._provider_weight_stats.decay_factor:.2f}, "
            f"recover={self._provider_weight_stats.recover:.2f})"
        )
        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                -score(candidate),
                -self._provider_weight_stats.weight(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class ProviderRetryAfterCooldown:
    """Per-provider Retry-After cooldown map for retry-after-respect routing."""

    def __init__(self, default_seconds: float = 30.0) -> None:
        """Initialize provider cooldown tracking.

        Args:
            default_seconds: Fallback wait when a Retry-After value is absent.

        Raises:
            ValueError: If ``default_seconds`` is negative.
        """
        if default_seconds < 0.0:
            raise ValueError(f"default_seconds must be >= 0, got {default_seconds}")
        self._default_seconds = default_seconds
        self._until: dict[str, float] = {}

    @property
    def default_seconds(self) -> float:
        """Return the configured default Retry-After wait in seconds."""
        return self._default_seconds

    def set_cooldown(
        self,
        provider: str,
        seconds: float | None = None,
        *,
        now: float | None = None,
    ) -> None:
        """Mark a provider unavailable until the Retry-After wait expires."""
        wait = self._default_seconds if seconds is None else max(0.0, float(seconds))
        clock = time.monotonic() if now is None else now
        self._until[provider] = clock + wait

    def clear(self, provider: str) -> None:
        """Clear any active cooldown for a provider."""
        self._until.pop(provider, None)

    def remaining_seconds(self, provider: str, *, now: float | None = None) -> float:
        """Return remaining cooldown seconds (``0`` when ready)."""
        until = self._until.get(provider)
        if until is None:
            return 0.0
        clock = time.monotonic() if now is None else now
        return max(0.0, until - clock)

    def is_cooling_down(self, provider: str, *, now: float | None = None) -> bool:
        """Return whether the provider is still inside a Retry-After wait."""
        return self.remaining_seconds(provider, now=now) > 0.0


class RetryAfterRespectStrategy(RoutingStrategy):
    """Skip providers still honoring a Retry-After cooldown.

    When providers return HTTP 429 with ``Retry-After``, hammering them before
    the wait expires wastes GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    retry budget. This strategy prefers healthy providers whose cooldown has
    expired, then falls back to the next healthy cooling provider (soonest
    expiry) so decide-time remains deterministic.
    """

    strategy_name = RoutingStrategyName.RETRY_AFTER_RESPECT

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        retry_after_cooldown: ProviderRetryAfterCooldown,
    ) -> None:
        """Initialize retry-after-respect routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_health: Live provider health view.
            retry_after_cooldown: Shared provider Retry-After cooldown map.
        """
        super().__init__(model_catalog)
        self._provider_health = provider_health
        self._retry_after_cooldown = retry_after_cooldown

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer healthy ready providers; fall back to next healthy cooling one."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        def quality_key(candidate: ModelCandidate) -> tuple[float, float, str]:
            return (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            )

        healthy_ready = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
            and not self._retry_after_cooldown.is_cooling_down(candidate.provider)
        ]
        healthy_cooling = [
            candidate
            for candidate in eligible_candidates
            if self._provider_health.is_available(candidate.provider)
            and self._retry_after_cooldown.is_cooling_down(candidate.provider)
        ]
        ready = [
            candidate
            for candidate in eligible_candidates
            if not self._retry_after_cooldown.is_cooling_down(candidate.provider)
        ]

        if healthy_ready:
            selected_candidate = max(healthy_ready, key=quality_key)
            rationale = (
                "retry-after-respect selected healthy ready provider "
                f"{selected_candidate.provider} "
                f"(quality {selected_candidate.quality_score:.2f}; "
                f"default wait {self._retry_after_cooldown.default_seconds:.0f}s)"
            )
        elif healthy_cooling:
            selected_candidate = min(
                healthy_cooling,
                key=lambda candidate: (
                    self._retry_after_cooldown.remaining_seconds(candidate.provider),
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            remaining = self._retry_after_cooldown.remaining_seconds(selected_candidate.provider)
            rationale = (
                "retry-after-respect all healthy providers cooling; "
                f"fell back to next healthy {selected_candidate.provider} "
                f"with {remaining:.1f}s remaining"
            )
        elif ready:
            selected_candidate = max(ready, key=quality_key)
            rationale = (
                "retry-after-respect no healthy ready providers; "
                f"selected ready {selected_candidate.provider} "
                f"(quality {selected_candidate.quality_score:.2f})"
            )
        else:
            selected_candidate = max(eligible_candidates, key=quality_key)
            remaining = self._retry_after_cooldown.remaining_seconds(selected_candidate.provider)
            rationale = (
                "retry-after-respect every eligible provider cooling; "
                f"selected {selected_candidate.provider} with {remaining:.1f}s remaining"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                self._retry_after_cooldown.is_cooling_down(candidate.provider),
                not self._provider_health.is_available(candidate.provider),
                self._retry_after_cooldown.remaining_seconds(candidate.provider),
                -candidate.quality_score,
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class LatencySlopeStats:
    """Rolling per-provider latency samples for EWMA slope shedding."""

    def __init__(self, window: int = 10, alpha: float = 0.3) -> None:
        """Initialize empty latency slope observations.

        Args:
            window: Maximum recent samples retained per provider.
            alpha: EWMA smoothing factor in ``(0.0, 1.0]``.

        Raises:
            ValueError: If ``window`` is less than 2 or ``alpha`` is outside
                ``(0.0, 1.0]``.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be within (0.0, 1.0], got {alpha}")
        self._window = window
        self._alpha = alpha
        self._samples: dict[str, list[float]] = {}

    @property
    def window(self) -> int:
        """Return the configured sample window size."""
        return self._window

    def observe(self, provider: str, latency_ms: float) -> None:
        """Record a latency observation for a provider.

        Args:
            provider: Provider name.
            latency_ms: Observed latency in milliseconds.
        """
        samples = self._samples.setdefault(provider, [])
        samples.append(latency_ms)
        if len(samples) > self._window:
            del samples[0]

    def mean_latency(self, provider: str) -> float:
        """Return the mean of recent latency samples (``0.0`` when cold)."""
        samples = self._samples.get(provider, [])
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def ewma_slope(self, provider: str) -> float:
        """Return EWMA slope in ms/step for a provider.

        Builds an EWMA series over the recent window and returns the average
        change per step from the first EWMA value to the latest. Cold starts
        (fewer than two samples) report ``0.0`` so routing does not shed
        before a trend is observable.

        Args:
            provider: Provider name.

        Returns:
            Average EWMA change per sample step in milliseconds.
        """
        samples = self._samples.get(provider, [])
        if len(samples) < 2:
            return 0.0
        ewma = samples[0]
        series = [ewma]
        for sample in samples[1:]:
            ewma = self._alpha * sample + (1.0 - self._alpha) * ewma
            series.append(ewma)
        return (series[-1] - series[0]) / float(len(series) - 1)


class LatencySlopeShedStrategy(RoutingStrategy):
    """Shed providers whose EWMA latency slope is rising above a threshold.

    LiteLLM/OpenRouter-style gateways often react when a provider's recent
    latency trend is climbing, not only when absolute p95 is already hot.
    This strategy tracks a short per-provider sample window, computes the
    EWMA slope (ms per sample step), and when the quality leader's slope
    exceeds ``NEXUS_LATENCY_SLOPE_THRESHOLD_MS`` it sheds to a lower-latency /
    cheaper healthy alternative. Cold starts (flat slope) keep quality-first
    selection for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.
    """

    strategy_name = RoutingStrategyName.LATENCY_SLOPE_SHED

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_slope_stats: LatencySlopeStats,
        provider_health: ProviderHealth,
        latency_slope_threshold_ms: float = 25.0,
    ) -> None:
        """Initialize latency-slope-shed routing.

        Args:
            model_catalog: Available model candidates by model name.
            latency_slope_stats: Shared per-provider EWMA slope window.
            provider_health: Live provider health view (circuit breaker).
            latency_slope_threshold_ms: Maximum acceptable EWMA slope in
                milliseconds per sample step before shedding.

        Raises:
            ValueError: If the slope threshold is negative.
        """
        super().__init__(model_catalog)
        if latency_slope_threshold_ms < 0.0:
            raise ValueError(
                f"latency_slope_threshold_ms must be non-negative, got {latency_slope_threshold_ms}"
            )
        self._latency_slope_stats = latency_slope_stats
        self._provider_health = provider_health
        self._latency_slope_threshold_ms = latency_slope_threshold_ms

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer quality unless the leader's EWMA slope requires shedding."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        slopes = {
            candidate.model: self._latency_slope_stats.ewma_slope(candidate.provider)
            for candidate in eligible_candidates
        }
        means = {
            candidate.model: self._latency_slope_stats.mean_latency(candidate.provider)
            for candidate in eligible_candidates
        }

        primary = max(
            eligible_candidates,
            key=lambda candidate: (
                candidate.quality_score,
                -costs[candidate.model],
                candidate.model,
            ),
        )
        primary_slope = slopes[primary.model]
        if primary_slope <= self._latency_slope_threshold_ms:
            rationale = (
                "latency-slope-shed kept highest quality "
                f"{primary.quality_score:.2f} "
                f"(EWMA slope {primary_slope:.2f}ms/step <= "
                f"{self._latency_slope_threshold_ms:.2f}ms threshold)"
            )
            return self._decision(primary.model, rationale)

        stable = [
            candidate
            for candidate in eligible_candidates
            if slopes[candidate.model] <= self._latency_slope_threshold_ms
        ]
        healthy_stable = [
            candidate
            for candidate in stable
            if self._provider_health.is_available(candidate.provider)
        ]
        shed_pool = (
            healthy_stable
            or [
                candidate
                for candidate in eligible_candidates
                if self._provider_health.is_available(candidate.provider)
            ]
            or stable
            or eligible_candidates
        )

        selected_candidate = min(
            shed_pool,
            key=lambda candidate: (
                means[candidate.model],
                costs[candidate.model],
                -candidate.quality_score,
                candidate.model,
            ),
        )
        rationale = (
            "latency-slope-shed shed rising "
            f"{primary.provider} (EWMA slope {primary_slope:.2f}ms/step > "
            f"{self._latency_slope_threshold_ms:.2f}ms) to lower-latency/cheaper "
            f"{selected_candidate.model} "
            f"(mean {means[selected_candidate.model]:.1f}ms, "
            f"est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class ProviderHourlySpendWindow:
    """Rolling hourly estimated-spend tracker per provider."""

    def __init__(self, window_seconds: float = 3600.0) -> None:
        """Initialize empty provider hourly spend windows.

        Args:
            window_seconds: Rolling window length in seconds (default one hour).

        Raises:
            ValueError: If the window length is not positive.
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._window_seconds = window_seconds
        self._entries: dict[str, list[tuple[float, float]]] = {}

    @property
    def window_seconds(self) -> float:
        """Return the configured rolling window length in seconds."""
        return self._window_seconds

    def record(self, provider: str, amount_usd: float, *, now: float | None = None) -> None:
        """Record estimated spend against a provider.

        Args:
            provider: Provider name.
            amount_usd: Estimated spend in USD.
            now: Optional monotonic timestamp override for tests.
        """
        timestamp = time.monotonic() if now is None else now
        entries = self._entries.setdefault(provider, [])
        entries.append((timestamp, amount_usd))
        self._prune(provider, timestamp)

    def provider_spend(self, provider: str, *, now: float | None = None) -> float:
        """Return rolling estimated spend for a provider inside the window."""
        timestamp = time.monotonic() if now is None else now
        self._prune(provider, timestamp)
        return sum(amount for _, amount in self._entries.get(provider, []))

    def is_over_ceiling(
        self, provider: str, ceiling_usd: float, *, now: float | None = None
    ) -> bool:
        """Return whether provider rolling spend exceeds the hourly ceiling."""
        return self.provider_spend(provider, now=now) > ceiling_usd

    def _prune(self, provider: str, now: float) -> None:
        cutoff = now - self._window_seconds
        entries = self._entries.get(provider, [])
        while entries and entries[0][0] < cutoff:
            entries.pop(0)
        if not entries:
            self._entries.pop(provider, None)


class ProviderHourlyCostCeilingStrategy(RoutingStrategy):
    """Skip providers whose rolling hourly estimated spend exceeds a ceiling.

    Distinct from :class:`ProviderFamilyCostCeilingStrategy` (a hard *per-request*
    family ceiling) and :class:`SoftFamilyBudgetStrategy` (soft deprioritization
    of observed family spend). This strategy tracks rolling *hourly estimated*
    spend per provider and hard-skips providers already over
    ``NEXUS_PROVIDER_HOURLY_COST_CEILING_USD``. Among under-ceiling providers it
    picks highest quality; when every provider is over ceiling it falls back to
    the cheapest eligible model so GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 traffic still routes deterministically.
    """

    strategy_name = RoutingStrategyName.PROVIDER_HOURLY_COST_CEILING

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_hourly_spend_window: ProviderHourlySpendWindow,
        provider_hourly_cost_ceiling_usd: float = 5.0,
    ) -> None:
        """Initialize provider-hourly-cost-ceiling routing.

        Args:
            model_catalog: Available model candidates by model name.
            provider_hourly_spend_window: Rolling estimated spend per provider.
            provider_hourly_cost_ceiling_usd: Hard hourly estimated-spend
                ceiling per provider in USD.

        Raises:
            ValueError: If the ceiling is negative.
        """
        super().__init__(model_catalog)
        if provider_hourly_cost_ceiling_usd < 0.0:
            raise ValueError(
                "provider_hourly_cost_ceiling_usd must be non-negative, "
                f"got {provider_hourly_cost_ceiling_usd}"
            )
        self._spend_window = provider_hourly_spend_window
        self._ceiling_usd = provider_hourly_cost_ceiling_usd

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer highest quality among providers under the hourly ceiling."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible_candidates
        }
        under_ceiling = [
            candidate
            for candidate in eligible_candidates
            if not self._spend_window.is_over_ceiling(candidate.provider, self._ceiling_usd)
        ]
        if under_ceiling:
            selected_candidate = max(
                under_ceiling,
                key=lambda candidate: (
                    candidate.quality_score,
                    -costs[candidate.model],
                    candidate.model,
                ),
            )
            spend = self._spend_window.provider_spend(selected_candidate.provider)
            rationale = (
                "provider-hourly-cost-ceiling selected "
                f"{selected_candidate.provider}/{selected_candidate.model} "
                f"quality {selected_candidate.quality_score:.2f} under "
                f"${self._ceiling_usd:.4f}/hr ceiling "
                f"(rolling est ${spend:.6f})"
            )
            return self._decision(selected_candidate.model, rationale)

        selected_candidate = min(
            eligible_candidates,
            key=lambda candidate: (
                costs[candidate.model],
                -candidate.quality_score,
                candidate.model,
            ),
        )
        rationale = (
            "provider-hourly-cost-ceiling every provider over "
            f"${self._ceiling_usd:.4f}/hr ceiling; fell back to cheapest "
            f"{selected_candidate.provider}/{selected_candidate.model} "
            f"(est ${costs[selected_candidate.model]:.6f})"
        )
        return self._decision(selected_candidate.model, rationale)


class QualityWeightedStickyStrategy(RoutingStrategy):
    """Sticky-session hashing with bucket weights proportional to quality.

    Like :class:`StickySessionStrategy`, this pins every request sharing a
    ``session_id`` to one domain-eligible model via a stable hash. Unlike
    uniform sticky hashing (and unlike :class:`StickyTenantHashStrategy`, which
    hashes tenant identity onto an equal ring), each candidate receives a hash
    ring share proportional to ``quality_score`` so higher-quality models such
    as Claude Sonnet 4.6 / GPT-5.5 absorb a larger sticky share while cheaper
    Gemini 3.x / Kimi K2 arms still receive some sessions.
    """

    strategy_name = RoutingStrategyName.QUALITY_WEIGHTED_STICKY

    _WEIGHT_SCALE = 100

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Pin the session onto a quality-weighted sticky hash ring."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())

        ordered_candidates = sorted(eligible_candidates, key=lambda candidate: candidate.model)
        weights = [
            max(1, int(round(candidate.quality_score * self._WEIGHT_SCALE)))
            for candidate in ordered_candidates
        ]
        total_weight = sum(weights)
        digest = sha256(request.session_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % total_weight

        cumulative = 0
        selected_index = 0
        for index, weight in enumerate(weights):
            cumulative += weight
            if bucket < cumulative:
                selected_index = index
                break

        selected_candidate = ordered_candidates[selected_index]
        rationale = (
            f"quality-weighted-sticky pinned session '{request.session_id}' to "
            f"{selected_candidate.model} "
            f"(quality weight {weights[selected_index]}/{total_weight}; "
            f"bucket {bucket})"
        )
        return self._decision(selected_candidate.model, rationale)


class TokenRpmWindow:
    """Rolling estimated prompt-token window per provider."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        """Initialize empty provider token windows.

        Args:
            window_seconds: Rolling window length in seconds.

        Raises:
            ValueError: If the window length is not positive.
        """
        if window_seconds <= 0.0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._window_seconds = window_seconds
        self._entries: dict[str, list[tuple[float, int]]] = {}

    @property
    def window_seconds(self) -> float:
        """Return the configured rolling window length."""
        return self._window_seconds

    def record(self, provider: str, prompt_tokens: int, *, now: float | None = None) -> None:
        """Record estimated prompt tokens for a completed provider request."""
        if prompt_tokens < 0:
            raise ValueError(f"prompt_tokens must be non-negative, got {prompt_tokens}")
        timestamp = time.monotonic() if now is None else now
        entries = self._entries.setdefault(provider, [])
        entries.append((timestamp, prompt_tokens))
        self._prune(provider, timestamp)

    def provider_tokens(self, provider: str, *, now: float | None = None) -> int:
        """Return prompt tokens observed for a provider inside the window."""
        timestamp = time.monotonic() if now is None else now
        self._prune(provider, timestamp)
        return sum(tokens for _, tokens in self._entries.get(provider, []))

    def would_exceed(
        self,
        provider: str,
        prompt_tokens: int,
        ceiling: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Return whether another request would exceed the provider ceiling."""
        return self.provider_tokens(provider, now=now) + prompt_tokens > ceiling

    def _prune(self, provider: str, now: float) -> None:
        cutoff = now - self._window_seconds
        entries = self._entries.get(provider, [])
        while entries and entries[0][0] < cutoff:
            entries.pop(0)
        if not entries:
            self._entries.pop(provider, None)


class TokenRpmCeilingStrategy(RoutingStrategy):
    """Shed providers that would exceed a rolling prompt-token RPM ceiling.

    The strategy keeps the highest-quality domain-eligible model while its
    provider's last 60 seconds of estimated prompt tokens plus the current
    request fit under ``NEXUS_TOKEN_RPM_CEILING``. When the quality leader would
    cross the ceiling, traffic sheds to the next eligible provider so GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 workloads can avoid provider token
    throttles.
    """

    strategy_name = RoutingStrategyName.TOKEN_RPM_CEILING

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        token_rpm_window: TokenRpmWindow,
        token_rpm_ceiling: int = 100_000,
    ) -> None:
        """Initialize token-rpm-ceiling routing."""
        super().__init__(model_catalog)
        if token_rpm_ceiling < 1:
            raise ValueError(f"token_rpm_ceiling must be >= 1, got {token_rpm_ceiling}")
        self._token_rpm_window = token_rpm_window
        self._token_rpm_ceiling = token_rpm_ceiling

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the best provider whose projected token RPM fits the ceiling."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        prompt_tokens = signals.prompt_tokens_estimate
        projected_tokens = {
            candidate.model: self._token_rpm_window.provider_tokens(candidate.provider)
            + prompt_tokens
            for candidate in eligible_candidates
        }
        under_ceiling = [
            candidate
            for candidate in eligible_candidates
            if not self._token_rpm_window.would_exceed(
                candidate.provider,
                prompt_tokens,
                self._token_rpm_ceiling,
            )
        ]

        if under_ceiling:
            selected_candidate = max(
                under_ceiling,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "token-rpm-ceiling selected "
                f"{selected_candidate.provider}/{selected_candidate.model} with projected "
                f"{projected_tokens[selected_candidate.model]}/{self._token_rpm_ceiling} "
                "prompt tokens in the rolling 60s window"
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    projected_tokens[candidate.model],
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "token-rpm-ceiling found every eligible provider over projected ceiling "
                f"{self._token_rpm_ceiling}; fell back to least-loaded "
                f"{selected_candidate.provider}/{selected_candidate.model} at "
                f"{projected_tokens[selected_candidate.model]} projected prompt tokens"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                projected_tokens[candidate.model] > self._token_rpm_ceiling,
                -candidate.quality_score,
                projected_tokens[candidate.model],
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class ProviderCircuitProbeStrategy(RoutingStrategy):
    """Route around an open quality leader and budget its recovery probes.

    Unlike :class:`CircuitBreakerHalfOpenProbeStrategy`, which prefers any
    closed provider before considering recovery traffic, this strategy follows
    the highest-quality provider's circuit state directly. A closed leader wins,
    an open leader immediately sheds to the best healthy alternate, and a
    half-open leader receives at most ``NEXUS_PROVIDER_CIRCUIT_PROBE_BUDGET``
    probe decisions before traffic falls back. This keeps GPT-5.5 / Claude
    Sonnet 4.6 / Gemini 3.x / Kimi K2 recovery controlled without abandoning
    the preferred quality arm.
    """

    strategy_name = RoutingStrategyName.PROVIDER_CIRCUIT_PROBE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        probe_budget: int = 1,
    ) -> None:
        """Initialize preferred-provider circuit probe routing."""
        super().__init__(model_catalog)
        if probe_budget < 1:
            raise ValueError(f"probe_budget must be >= 1, got {probe_budget}")
        self._provider_health = provider_health
        self._probe_budget = probe_budget
        self._probe_counts: dict[str, int] = {}

    def _is_half_open(self, provider: str) -> bool:
        """Return whether a provider is in its recovery probe window."""
        half_open = getattr(self._provider_health, "is_half_open", None)
        return bool(half_open(provider)) if callable(half_open) else False

    def _quality_key(
        self,
        candidate: ModelCandidate,
        request: RouterRequest,
        signals: TaskSignals,
    ) -> tuple[float, float, str]:
        return (
            candidate.quality_score,
            -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
            candidate.model,
        )

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the quality leader, a bounded probe, or a healthy alternate."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        leader = max(
            eligible_candidates,
            key=lambda candidate: self._quality_key(candidate, request, signals),
        )
        leader_half_open = self._is_half_open(leader.provider)
        leader_available = self._provider_health.is_available(leader.provider)
        healthy_alternates = [
            candidate
            for candidate in eligible_candidates
            if candidate.provider != leader.provider
            and self._provider_health.is_available(candidate.provider)
            and not self._is_half_open(candidate.provider)
        ]

        if leader_available and not leader_half_open:
            self._probe_counts.pop(leader.provider, None)
            selected_candidate = leader
            rationale = (
                "provider-circuit-probe kept closed quality leader "
                f"{leader.provider}/{leader.model} at quality {leader.quality_score:.2f}"
            )
        elif leader_half_open:
            probes_used = self._probe_counts.get(leader.provider, 0)
            if probes_used < self._probe_budget:
                self._probe_counts[leader.provider] = probes_used + 1
                selected_candidate = leader
                rationale = (
                    "provider-circuit-probe allowed half-open quality-leader probe "
                    f"{probes_used + 1}/{self._probe_budget} to "
                    f"{leader.provider}/{leader.model}"
                )
            elif healthy_alternates:
                selected_candidate = max(
                    healthy_alternates,
                    key=lambda candidate: self._quality_key(candidate, request, signals),
                )
                rationale = (
                    "provider-circuit-probe exhausted half-open probe budget "
                    f"{probes_used}/{self._probe_budget} for {leader.provider}; "
                    f"fell back to healthy {selected_candidate.provider}/"
                    f"{selected_candidate.model}"
                )
            else:
                selected_candidate = max(
                    [candidate for candidate in eligible_candidates if candidate != leader]
                    or [leader],
                    key=lambda candidate: self._quality_key(candidate, request, signals),
                )
                rationale = (
                    "provider-circuit-probe exhausted half-open leader budget with no "
                    f"healthy alternate; chose deterministic non-leader "
                    f"{selected_candidate.provider}/{selected_candidate.model}"
                )
        elif healthy_alternates:
            selected_candidate = max(
                healthy_alternates,
                key=lambda candidate: self._quality_key(candidate, request, signals),
            )
            rationale = (
                "provider-circuit-probe quality leader "
                f"{leader.provider}/{leader.model} circuit open; actively selected "
                f"healthy alternate {selected_candidate.provider}/"
                f"{selected_candidate.model}"
            )
        else:
            selected_candidate = max(
                [candidate for candidate in eligible_candidates if candidate != leader] or [leader],
                key=lambda candidate: self._quality_key(candidate, request, signals),
            )
            rationale = (
                "provider-circuit-probe quality leader circuit open with no healthy "
                f"alternate; chose deterministic fallback {selected_candidate.provider}/"
                f"{selected_candidate.model}"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                not (
                    self._provider_health.is_available(candidate.provider)
                    and not self._is_half_open(candidate.provider)
                ),
                self._is_half_open(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class CarbonLatencyBlendStrategy(RoutingStrategy):
    """Blend provider-region carbon intensity with rolling provider latency.

    This strategy extends the carbon metadata pattern with provider-and-region
    keys and separate carbon/latency weights. It scores each domain-eligible
    candidate as ``w_carbon * normalized_inverse_carbon + w_latency *
    normalized_inverse_latency``. Unlike ``region-carbon-blend``, the two
    weights are independent rather than complementary, so operators can
    increase or suppress either signal for GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 traffic.
    """

    strategy_name = RoutingStrategyName.CARBON_LATENCY_BLEND

    _PROVIDER_REGION_DEFAULTS: dict[str, dict[str, float]] = {
        "openai": {"eu": 260.0, "us": 380.0, "cn": 540.0, "global": 420.0},
        "anthropic": {"eu": 180.0, "us": 350.0, "cn": 520.0, "global": 360.0},
        "google": {"eu": 220.0, "us": 400.0, "cn": 500.0, "global": 390.0},
        "moonshot": {"eu": 300.0, "us": 430.0, "cn": 550.0, "global": 500.0},
    }
    _REGION_DEFAULTS = {
        "eu": 250.0,
        "us": 380.0,
        "cn": 550.0,
        "global": 420.0,
    }

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        carbon_weight: float = 0.5,
        latency_weight: float = 0.5,
    ) -> None:
        """Initialize carbon-latency-blend routing."""
        super().__init__(model_catalog)
        if carbon_weight < 0.0:
            raise ValueError(f"carbon_weight must be non-negative, got {carbon_weight}")
        if latency_weight < 0.0:
            raise ValueError(f"latency_weight must be non-negative, got {latency_weight}")
        self._latency_stats = latency_stats
        self._carbon_weight = carbon_weight
        self._latency_weight = latency_weight

    @staticmethod
    def _region_for(request: RouterRequest) -> str:
        region = (
            request.metadata.get("region")
            or request.metadata.get("preferred_region")
            or request.region
            or "global"
        )
        return str(region).lower()

    def _intensity_for(self, candidate: ModelCandidate, request: RouterRequest) -> float:
        """Resolve provider-region carbon intensity from metadata or defaults."""
        region = self._region_for(request)
        for key in (
            f"carbon_intensity:{candidate.provider}:{region}",
            f"carbon_intensity:{candidate.provider}",
        ):
            raw = request.metadata.get(key)
            if raw is not None and str(raw).strip():
                try:
                    return max(0.0, float(raw))
                except ValueError:
                    continue
        provider_defaults = self._PROVIDER_REGION_DEFAULTS.get(candidate.provider)
        if provider_defaults is not None:
            return float(provider_defaults.get(region, provider_defaults.get("global", 420.0)))
        return float(self._REGION_DEFAULTS.get(region, self._REGION_DEFAULTS["global"]))

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Select the highest independently weighted carbon/latency score."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        intensities = {
            candidate.model: self._intensity_for(candidate, request)
            for candidate in eligible_candidates
        }
        latencies = {
            candidate.model: self._latency_stats.p95(candidate.provider)
            for candidate in eligible_candidates
        }
        carbon_scores = _inverse_min_max(intensities)
        latency_scores = _inverse_min_max(latencies)

        def blended_score(candidate: ModelCandidate) -> float:
            return (
                self._carbon_weight * carbon_scores[candidate.model]
                + self._latency_weight * latency_scores[candidate.model]
            )

        selected_candidate = max(
            eligible_candidates,
            key=lambda candidate: (
                blended_score(candidate),
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        rationale = (
            "carbon-latency-blend selected "
            f"{selected_candidate.provider}/{selected_candidate.model} at score "
            f"{blended_score(selected_candidate):.3f} "
            f"(w_carbon={self._carbon_weight:.2f}, "
            f"w_latency={self._latency_weight:.2f}, intensity "
            f"{intensities[selected_candidate.model]:.1f} gCO2eq/kWh, p95 "
            f"{latencies[selected_candidate.model]:.1f}ms, region "
            f"'{self._region_for(request)}')"
        )
        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                -blended_score(candidate),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class AdaptiveConcurrencyCapStrategy(RoutingStrategy):
    """Cap in-flight requests per provider with health-derived dynamic limits.

    Unlike :class:`ConcurrencyCapStrategy`, which applies a static
    ``NEXUS_CONCURRENCY_CAP`` to every provider equally, this strategy scales
    each provider's effective cap by rolling success rate and inverse p95
    latency. Unhealthy or slow providers receive tighter caps so GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic drains toward healthier
    backends under load.
    """

    strategy_name = RoutingStrategyName.ADAPTIVE_CONCURRENCY_CAP

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        inflight_stats: InflightStats,
        success_stats: SuccessStats,
        latency_stats: LatencyStats,
        base_cap: int = 8,
        min_cap: int = 1,
        latency_reference_ms: float = 2000.0,
    ) -> None:
        """Initialize adaptive-concurrency-cap routing."""
        super().__init__(model_catalog)
        if base_cap < 1:
            raise ValueError(f"base_cap must be >= 1, got {base_cap}")
        if min_cap < 1:
            raise ValueError(f"min_cap must be >= 1, got {min_cap}")
        if min_cap > base_cap:
            raise ValueError(f"min_cap ({min_cap}) must be <= base_cap ({base_cap})")
        if latency_reference_ms <= 0.0:
            raise ValueError(f"latency_reference_ms must be positive, got {latency_reference_ms}")
        self._inflight_stats = inflight_stats
        self._success_stats = success_stats
        self._latency_stats = latency_stats
        self._base_cap = base_cap
        self._min_cap = min_cap
        self._latency_reference_ms = latency_reference_ms

    def _effective_cap(self, provider: str) -> int:
        success_rate = self._success_stats.success_rate(provider)
        p95 = self._latency_stats.p95(provider)
        latency_factor = 1.0 if p95 <= 0.0 else min(1.0, self._latency_reference_ms / p95)
        health_factor = success_rate * latency_factor
        scaled = int(self._base_cap * health_factor)
        return max(self._min_cap, min(self._base_cap, scaled))

    def _health_score(self, provider: str) -> float:
        success_rate = self._success_stats.success_rate(provider)
        p95 = self._latency_stats.p95(provider)
        latency_factor = 1.0 if p95 <= 0.0 else min(1.0, self._latency_reference_ms / p95)
        return success_rate * latency_factor

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose the healthiest eligible provider below its adaptive cap."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        under_cap_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._inflight_stats.load_score(candidate.provider)
            < self._effective_cap(candidate.provider)
        ]

        if under_cap_candidates:
            selected_candidate = max(
                under_cap_candidates,
                key=lambda candidate: (
                    self._health_score(candidate.provider),
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            load_score = self._inflight_stats.load_score(selected_candidate.provider)
            effective_cap = self._effective_cap(selected_candidate.provider)
            rationale = (
                "adaptive-concurrency-cap selected below adaptive cap "
                f"{effective_cap}/{self._base_cap}; "
                f"{selected_candidate.provider} load {load_score}/{effective_cap} "
                f"health {self._health_score(selected_candidate.provider):.2f}"
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    self._inflight_stats.load_score(candidate.provider)
                    / max(self._effective_cap(candidate.provider), 1),
                    -self._health_score(candidate.provider),
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            load_score = self._inflight_stats.load_score(selected_candidate.provider)
            effective_cap = self._effective_cap(selected_candidate.provider)
            rationale = (
                "adaptive-concurrency-cap found every eligible provider at or above "
                f"adaptive cap; routed to least-saturated fallback "
                f"{selected_candidate.provider} load {load_score}/{effective_cap} "
                f"health {self._health_score(selected_candidate.provider):.2f}"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                self._inflight_stats.load_score(candidate.provider)
                >= self._effective_cap(candidate.provider),
                -self._health_score(candidate.provider),
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class ProviderTokenFairShareStrategy(RoutingStrategy):
    """Fair-share token budget across providers by remaining window quota.

    Tracks estimated prompt tokens per provider in a rolling 60-second window
    and selects providers with the most remaining fair-share headroom. Unlike
    :class:`ProviderQuotaFairShareStrategy`, which balances request counts,
    this strategy balances token volume and uses request-id weighted
    round-robin among providers tied on remaining quota for GPT-5.5 / Claude
    Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.
    """

    strategy_name = RoutingStrategyName.PROVIDER_TOKEN_FAIR_SHARE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        token_rpm_window: TokenRpmWindow,
        token_fair_share_ceiling: int = 100_000,
    ) -> None:
        """Initialize provider-token-fair-share routing."""
        super().__init__(model_catalog)
        if token_fair_share_ceiling < 1:
            raise ValueError(
                f"token_fair_share_ceiling must be >= 1, got {token_fair_share_ceiling}"
            )
        self._token_rpm_window = token_rpm_window
        self._token_fair_share_ceiling = token_fair_share_ceiling

    def _remaining_quota(self, provider: str, prompt_tokens: int) -> int:
        used = self._token_rpm_window.provider_tokens(provider)
        return max(0, self._token_fair_share_ceiling - used - prompt_tokens)

    def _weighted_provider(
        self,
        request: RouterRequest,
        providers: Iterable[str],
        remaining: Mapping[str, int],
    ) -> str:
        weighted = [
            (provider, remaining[provider]) for provider in providers if remaining[provider] > 0
        ]
        if not weighted:
            return min(providers, key=lambda provider: (remaining[provider], provider))
        total_weight = sum(weight for _, weight in weighted)
        bucket = int(sha256(request.request_id.encode("utf-8")).hexdigest()[:8], 16) % total_weight
        cumulative = 0
        for provider, weight in sorted(weighted, key=lambda item: item[0]):
            cumulative += weight
            if bucket < cumulative:
                return provider
        return weighted[-1][0]

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Prefer providers with the most remaining token fair-share quota."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        prompt_tokens = signals.prompt_tokens_estimate
        eligible_providers = sorted({candidate.provider for candidate in eligible_candidates})
        remaining = {
            provider: self._remaining_quota(provider, prompt_tokens)
            for provider in eligible_providers
        }
        used = {
            provider: self._token_rpm_window.provider_tokens(provider)
            for provider in eligible_providers
        }
        positive_remaining = [
            provider for provider in eligible_providers if remaining[provider] > 0
        ]

        if positive_remaining:
            max_remaining = max(remaining[provider] for provider in positive_remaining)
            tied_providers = [
                provider for provider in positive_remaining if remaining[provider] == max_remaining
            ]
            cold_start = all(used[provider] == 0 for provider in eligible_providers)
            if cold_start or len(tied_providers) == 1:
                selected_provider = max(
                    tied_providers,
                    key=lambda provider: max(
                        candidate.quality_score
                        for candidate in eligible_candidates
                        if candidate.provider == provider
                    ),
                )
                selection_note = (
                    f"highest remaining quota {max_remaining}/{self._token_fair_share_ceiling}"
                )
            else:
                selected_provider = self._weighted_provider(request, tied_providers, remaining)
                selection_note = (
                    "round-robin weighted tie on remaining quota "
                    f"{max_remaining}/{self._token_fair_share_ceiling}"
                )
            provider_candidates = [
                candidate
                for candidate in eligible_candidates
                if candidate.provider == selected_provider
            ]
            selected_candidate = max(
                provider_candidates,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "provider-token-fair-share selected "
                f"{selected_candidate.provider}/{selected_candidate.model} with "
                f"{selection_note}; used {used[selected_provider]}/"
                f"{self._token_fair_share_ceiling} prompt tokens in rolling window"
            )
        else:
            selected_candidate = min(
                eligible_candidates,
                key=lambda candidate: (
                    used[candidate.provider],
                    -candidate.quality_score,
                    candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "provider-token-fair-share found every eligible provider over fair-share ceiling "
                f"{self._token_fair_share_ceiling}; fell back to least-loaded "
                f"{selected_candidate.provider}/{selected_candidate.model} at "
                f"{used[selected_candidate.provider]} used prompt tokens"
            )

        fallback_candidates = sorted(
            [
                candidate
                for candidate in eligible_candidates
                if candidate.model != selected_candidate.model
            ],
            key=lambda candidate: (
                remaining[candidate.provider] <= 0,
                -remaining[candidate.provider],
                used[candidate.provider],
                -candidate.quality_score,
                candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected_candidate.model,
            provider=selected_candidate.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallback_candidates[:3]],
        )


class RegionFailoverHysteresisStats:
    """Tracks region recovery streaks and session failover state."""

    def __init__(self) -> None:
        self._recovery_streak: dict[str, int] = {}
        self._active_failover: dict[str, str] = {}

    def record_success(self, region: str) -> None:
        normalized = region.strip().lower()
        self._recovery_streak[normalized] = self._recovery_streak.get(normalized, 0) + 1

    def record_failure(self, region: str) -> None:
        self._recovery_streak[region.strip().lower()] = 0

    def recovery_streak(self, region: str) -> int:
        return self._recovery_streak.get(region.strip().lower(), 0)

    def active_failover_region(self, session_id: str) -> str | None:
        return self._active_failover.get(session_id)

    def set_failover(self, session_id: str, region: str) -> None:
        self._active_failover[session_id] = region.strip().lower()

    def clear_failover(self, session_id: str) -> None:
        self._active_failover.pop(session_id, None)


class RegionFailoverHysteresisStrategy(RoutingStrategy):
    """Region preference with hysteresis before flapping back to preferred region.

    Like :class:`StickyRegionFailoverStrategy`, this strategy walks an ordered
    region preference list and pins ``session_id`` inside the active region
    pool. After failovers, it will not return to the preferred region until
    that region accumulates ``NEXUS_REGION_FAILOVER_HYSTERESIS_SUCCESSES``
    consecutive successes, preventing GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 traffic from flapping during intermittent recovery blips.
    """

    strategy_name = RoutingStrategyName.REGION_FAILOVER_HYSTERESIS

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        hysteresis_stats: RegionFailoverHysteresisStats,
        region_preferences: list[str] | None = None,
        hysteresis_successes: int = 3,
    ) -> None:
        """Initialize region-failover-hysteresis routing."""
        super().__init__(model_catalog)
        if hysteresis_successes < 1:
            raise ValueError(f"hysteresis_successes must be >= 1, got {hysteresis_successes}")
        self._provider_health = provider_health
        self._hysteresis_stats = hysteresis_stats
        self._hysteresis_successes = hysteresis_successes
        self._default_region_preferences = [
            region.strip().lower()
            for region in (region_preferences or ["eu", "us", "cn", "global"])
        ]

    def _region_preferences(self, request: RouterRequest) -> list[str]:
        requested_region = (request.region or "").strip().lower()
        ordered: list[str] = []
        if requested_region:
            ordered.append(requested_region)
        for region in self._default_region_preferences:
            if region not in ordered:
                ordered.append(region)
        return ordered or ["global"]

    def _matches_region(self, candidate: ModelCandidate, region: str) -> bool:
        return region in {
            supported_region.lower() for supported_region in candidate.supported_regions
        }

    def _healthy_region_candidates(
        self,
        eligible_candidates: list[ModelCandidate],
        region: str,
    ) -> list[ModelCandidate]:
        region_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._matches_region(candidate, region)
        ]
        healthy = [
            candidate
            for candidate in region_candidates
            if self._provider_health.is_available(candidate.provider)
        ]
        return healthy or region_candidates

    def _sticky_pick(
        self,
        request: RouterRequest,
        candidates: list[ModelCandidate],
        region_label: str,
        *,
        note: str,
    ) -> RoutingDecision:
        ordered_candidates = sorted(candidates, key=lambda candidate: candidate.model)
        digest = sha256(request.session_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % len(ordered_candidates)
        selected_candidate = ordered_candidates[bucket]
        rationale = (
            "region-failover-hysteresis pinned session "
            f"'{request.session_id}' to {selected_candidate.model} in region "
            f"'{region_label}' ({note}, bucket {bucket}/{len(ordered_candidates)})"
        )
        return self._decision(selected_candidate.model, rationale)

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a sticky model with hysteresis-gated preferred-region return."""
        eligible_candidates = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        region_preferences = self._region_preferences(request)
        preferred_region = region_preferences[0]
        active_failover = self._hysteresis_stats.active_failover_region(request.session_id)

        if active_failover is not None:
            preferred_healthy = self._healthy_region_candidates(
                eligible_candidates,
                preferred_region,
            )
            preferred_available = any(
                self._provider_health.is_available(candidate.provider)
                for candidate in preferred_healthy
            )
            if (
                preferred_available
                and self._hysteresis_stats.recovery_streak(preferred_region)
                >= self._hysteresis_successes
            ):
                self._hysteresis_stats.clear_failover(request.session_id)
                return self._sticky_pick(
                    request,
                    preferred_healthy,
                    preferred_region,
                    note=(
                        f"preferred region recovered with "
                        f"{self._hysteresis_stats.recovery_streak(preferred_region)}/"
                        f"{self._hysteresis_successes} consecutive successes"
                    ),
                )

            failover_candidates = self._healthy_region_candidates(
                eligible_candidates,
                active_failover,
            )
            if any(
                self._provider_health.is_available(candidate.provider)
                for candidate in failover_candidates
            ):
                return self._sticky_pick(
                    request,
                    failover_candidates,
                    active_failover,
                    note=(
                        "holding failover until preferred region reaches hysteresis "
                        f"{self._hysteresis_stats.recovery_streak(preferred_region)}/"
                        f"{self._hysteresis_successes} successes"
                    ),
                )

        for region in region_preferences:
            healthy_candidates = self._healthy_region_candidates(eligible_candidates, region)
            if not any(
                self._provider_health.is_available(candidate.provider)
                for candidate in healthy_candidates
            ):
                continue
            if region != preferred_region:
                self._hysteresis_stats.set_failover(request.session_id, region)
                note = f"failover from preferred '{preferred_region}'"
            else:
                self._hysteresis_stats.clear_failover(request.session_id)
                note = "preferred region healthy"
            return self._sticky_pick(
                request,
                healthy_candidates,
                region,
                note=note,
            )

        return self._sticky_pick(
            request,
            eligible_candidates,
            "fallback",
            note="no healthy regional pool; using fallback",
        )


class TenantBudgetExceededError(RuntimeError):
    """Raised when tenant-budget-cascade reaches its hard spend ceiling."""


class TenantBudgetCascadeStats:
    """In-memory rolling spend windows keyed by tenant identity."""

    def __init__(self, window_seconds: float = 3600.0) -> None:
        """Initialize empty tenant spend windows."""
        if window_seconds <= 0.0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._window_seconds = window_seconds
        self._observations: dict[str, list[tuple[float, float]]] = {}

    def record_spend(self, tenant_key: str, amount: float) -> None:
        """Record non-negative observed spend for one tenant."""
        if amount < 0.0:
            raise ValueError(f"amount must be non-negative, got {amount}")
        now = time.monotonic()
        observations = self._trim(tenant_key, now)
        observations.append((now, amount))

    def spend(self, tenant_key: str) -> float:
        """Return current rolling spend for one tenant."""
        return sum(amount for _, amount in self._trim(tenant_key, time.monotonic()))

    def _trim(self, tenant_key: str, now: float) -> list[tuple[float, float]]:
        cutoff = now - self._window_seconds
        observations = [
            observation
            for observation in self._observations.get(tenant_key, [])
            if observation[0] >= cutoff
        ]
        self._observations[tenant_key] = observations
        return observations


class TenantBudgetCascadeStrategy(RoutingStrategy):
    """Cascade tenants toward cheaper providers as rolling spend approaches a cap.

    Requests start quality-first while their projected spend fits the soft
    threshold. Once that headroom is exhausted, routing sheds to the cheapest
    candidate that remains below the hard ceiling. The strategy fails closed
    before a request would cross the hard ceiling, protecting GPT-5.5 /
    Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 tenant budgets.
    """

    strategy_name = RoutingStrategyName.TENANT_BUDGET_CASCADE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        tenant_budget_stats: TenantBudgetCascadeStats,
        soft_budget: float = 10.0,
        hard_budget: float = 12.5,
    ) -> None:
        """Initialize tenant-budget-cascade routing."""
        super().__init__(model_catalog)
        if soft_budget < 0.0:
            raise ValueError(f"soft_budget must be non-negative, got {soft_budget}")
        if hard_budget <= soft_budget:
            raise ValueError(
                f"hard_budget ({hard_budget}) must be greater than soft_budget ({soft_budget})"
            )
        self._tenant_budget_stats = tenant_budget_stats
        self._soft_budget = soft_budget
        self._hard_budget = hard_budget

    @staticmethod
    def _tenant_key(request: RouterRequest) -> str:
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = request.metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a budget-safe cascade rung or fail closed at the hard ceiling."""
        tenant_key = self._tenant_key(request)
        spend = self._tenant_budget_stats.spend(tenant_key)
        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate, request.max_tokens
            )
            for candidate in eligible
        }
        hard_safe = [
            candidate
            for candidate in eligible
            if spend + costs[candidate.model] <= self._hard_budget
        ]
        if not hard_safe:
            cheapest_cost = min(costs.values())
            raise TenantBudgetExceededError(
                "tenant-budget-cascade hard ceiling reached for tenant "
                f"'{tenant_key}': rolling spend {spend:.6f} + cheapest projected "
                f"{cheapest_cost:.6f} exceeds {self._hard_budget:.6f}; fail closed"
            )

        soft_safe = [
            candidate
            for candidate in hard_safe
            if spend + costs[candidate.model] <= self._soft_budget
        ]
        if soft_safe:
            selected = max(
                soft_safe,
                key=lambda candidate: (
                    candidate.quality_score,
                    -costs[candidate.model],
                    candidate.model,
                ),
            )
            rationale = (
                f"tenant-budget-cascade tenant '{tenant_key}' rolling spend "
                f"{spend:.6f} under soft {self._soft_budget:.6f}; selected highest-quality "
                f"soft-safe {selected.model} at projected {costs[selected.model]:.6f}"
            )
        else:
            selected = min(
                hard_safe,
                key=lambda candidate: (
                    costs[candidate.model],
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            rationale = (
                f"tenant-budget-cascade tenant '{tenant_key}' near hard ceiling "
                f"{self._hard_budget:.6f} after soft {self._soft_budget:.6f}; shed to "
                f"cheapest hard-safe provider {selected.provider} model {selected.model} "
                f"at projected {costs[selected.model]:.6f}"
            )

        fallbacks = sorted(
            (candidate for candidate in hard_safe if candidate.model != selected.model),
            key=lambda candidate: (
                costs[candidate.model],
                -candidate.quality_score,
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected.model,
            provider=selected.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallbacks[:3]],
        )


class ProviderErrorBudgetResetStats:
    """Per-provider error windows that clear after a fixed reset interval."""

    def __init__(self, reset_seconds: float = 60.0) -> None:
        """Initialize empty provider error windows."""
        if reset_seconds <= 0.0:
            raise ValueError(f"reset_seconds must be positive, got {reset_seconds}")
        self._reset_seconds = reset_seconds
        self._windows: dict[str, tuple[float, int, int]] = {}

    @property
    def reset_seconds(self) -> float:
        """Return the configured reset interval."""
        return self._reset_seconds

    def observe(self, provider: str, *, success: bool, now: float | None = None) -> None:
        """Record a provider result, resetting an expired window first."""
        observed_at = time.monotonic() if now is None else now
        window = self._active_window(provider, observed_at)
        if window is None:
            started_at, attempts, errors = observed_at, 0, 0
        else:
            started_at, attempts, errors = window
        self._windows[provider] = (
            started_at,
            attempts + 1,
            errors + (0 if success else 1),
        )

    def error_rate(self, provider: str, *, now: float | None = None) -> float:
        """Return the active error rate, or zero after the timer resets."""
        observed_at = time.monotonic() if now is None else now
        window = self._active_window(provider, observed_at)
        if window is None:
            return 0.0
        _, attempts, errors = window
        return errors / attempts

    def seconds_until_reset(self, provider: str, *, now: float | None = None) -> float:
        """Return seconds until the provider's active window resets."""
        observed_at = time.monotonic() if now is None else now
        window = self._active_window(provider, observed_at)
        if window is None:
            return 0.0
        started_at, _, _ = window
        return max(0.0, self._reset_seconds - (observed_at - started_at))

    def _active_window(
        self,
        provider: str,
        now: float,
    ) -> tuple[float, int, int] | None:
        window = self._windows.get(provider)
        if window is None:
            return None
        if now - window[0] >= self._reset_seconds:
            self._windows.pop(provider, None)
            return None
        return window


class ProviderErrorBudgetResetStrategy(RoutingStrategy):
    """Temporarily shed providers whose timed-window error budget is exhausted.

    Unlike ``provider-error-budget-shed``, whose shared cumulative success stats
    do not reset on a strategy timer, this strategy explicitly clears each
    provider's error window after ``NEXUS_PROVIDER_ERROR_BUDGET_RESET_SECONDS``.
    Recovered GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 providers are
    then restored automatically.
    """

    strategy_name = RoutingStrategyName.PROVIDER_ERROR_BUDGET_RESET

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        error_budget_stats: ProviderErrorBudgetResetStats,
        error_budget_fraction: float = 0.15,
    ) -> None:
        """Initialize provider-error-budget-reset routing."""
        super().__init__(model_catalog)
        if not 0.0 <= error_budget_fraction <= 1.0:
            raise ValueError(
                f"error_budget_fraction must be within [0.0, 1.0], got {error_budget_fraction}"
            )
        self._provider_health = provider_health
        self._error_budget_stats = error_budget_stats
        self._error_budget_fraction = error_budget_fraction

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Choose a healthy provider whose active error window is within budget."""
        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        healthy = [
            candidate
            for candidate in eligible
            if self._provider_health.is_available(candidate.provider)
        ] or eligible
        rates = {
            candidate.provider: self._error_budget_stats.error_rate(candidate.provider)
            for candidate in healthy
        }
        within_budget = [
            candidate
            for candidate in healthy
            if rates[candidate.provider] <= self._error_budget_fraction
        ]

        if within_budget:
            selected = max(
                within_budget,
                key=lambda candidate: (
                    candidate.quality_score,
                    -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                    candidate.model,
                ),
            )
            rationale = (
                "provider-error-budget-reset selected highest-quality provider within "
                f"{self._error_budget_fraction:.2%} timed error budget; "
                f"{selected.provider} error {rates[selected.provider]:.2%}, reset every "
                f"{self._error_budget_stats.reset_seconds:.1f}s"
            )
        else:
            selected = min(
                healthy,
                key=lambda candidate: (
                    self._error_budget_stats.seconds_until_reset(candidate.provider),
                    rates[candidate.provider],
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            rationale = (
                "provider-error-budget-reset found every healthy provider temporarily "
                f"shed above {self._error_budget_fraction:.2%}; emergency fallback "
                f"{selected.provider} resets in "
                f"{self._error_budget_stats.seconds_until_reset(selected.provider):.1f}s"
            )

        fallbacks = sorted(
            (candidate for candidate in healthy if candidate.model != selected.model),
            key=lambda candidate: (
                rates[candidate.provider] > self._error_budget_fraction,
                rates[candidate.provider],
                -candidate.quality_score,
                candidate.model,
            ),
        )
        return RoutingDecision(
            chosen_model=selected.model,
            provider=selected.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallbacks[:3]],
        )


class StickyRegionWarmupStats:
    """In-memory request counts and pinned regions for sticky sessions."""

    def __init__(self) -> None:
        """Initialize empty session warmup state."""
        self._request_counts: dict[str, int] = {}
        self._pinned_regions: dict[str, str] = {}

    def advance(self, session_id: str) -> int:
        """Increment and return a session's request count."""
        request_count = self._request_counts.get(session_id, 0) + 1
        self._request_counts[session_id] = request_count
        return request_count

    def request_count(self, session_id: str) -> int:
        """Return the number of observed routing decisions for a session."""
        return self._request_counts.get(session_id, 0)

    def pinned_region(self, session_id: str) -> str | None:
        """Return a session's post-warmup pinned region, if assigned."""
        return self._pinned_regions.get(session_id)

    def pin(self, session_id: str, region: str) -> None:
        """Pin a session to a normalized non-empty region."""
        normalized = region.strip().lower()
        if not normalized:
            raise ValueError("region must not be empty")
        self._pinned_regions[session_id] = normalized


class StickyRegionWarmupStrategy(RoutingStrategy):
    """Warm new sessions in one region before pinning them elsewhere.

    The first ``NEXUS_STICKY_REGION_WARMUP_REQUESTS`` decisions use a shared
    warmup region so provider caches and capacity can stabilize. Later requests
    pin the session to its requested or hash-selected region, preventing
    cold-start region flaps for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2 traffic.
    """

    strategy_name = RoutingStrategyName.STICKY_REGION_WARMUP

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        provider_health: ProviderHealth,
        warmup_stats: StickyRegionWarmupStats,
        region_preferences: list[str] | None = None,
        warmup_request_count: int = 3,
    ) -> None:
        """Initialize sticky-region-warmup routing."""
        super().__init__(model_catalog)
        if warmup_request_count < 1:
            raise ValueError(f"warmup_request_count must be >= 1, got {warmup_request_count}")
        self._provider_health = provider_health
        self._warmup_stats = warmup_stats
        self._region_preferences = [
            region.strip().lower()
            for region in (region_preferences or ["eu", "us", "cn", "global"])
            if region.strip()
        ]
        if not self._region_preferences:
            raise ValueError("region_preferences must contain at least one region")
        self._warmup_request_count = warmup_request_count

    @staticmethod
    def _supports_region(candidate: ModelCandidate, region: str) -> bool:
        return region in {
            supported_region.strip().lower() for supported_region in candidate.supported_regions
        }

    def _warmup_region(self, request: RouterRequest) -> str:
        configured = request.metadata.get("warmup_region")
        if configured is not None and str(configured).strip():
            return str(configured).strip().lower()
        return self._region_preferences[0]

    def _post_warmup_region(self, request: RouterRequest, warmup_region: str) -> str:
        pinned = self._warmup_stats.pinned_region(request.session_id)
        if pinned is not None:
            return pinned
        requested = (request.region or "").strip().lower()
        if requested:
            return requested
        alternatives = [
            region for region in self._region_preferences if region != warmup_region
        ] or [warmup_region]
        digest = sha256(request.session_id.encode("utf-8")).hexdigest()
        return alternatives[int(digest[:8], 16) % len(alternatives)]

    def _pick(
        self,
        request: RouterRequest,
        eligible: list[ModelCandidate],
        region: str,
        rationale: str,
    ) -> RoutingDecision:
        region_candidates = [
            candidate for candidate in eligible if self._supports_region(candidate, region)
        ]
        healthy_region = [
            candidate
            for candidate in region_candidates
            if self._provider_health.is_available(candidate.provider)
        ]
        pool = healthy_region
        active_region = region
        if not pool:
            pool = [
                candidate
                for candidate in eligible
                if self._provider_health.is_available(candidate.provider)
            ] or eligible
            active_region = "fallback"
            rationale += f"; region '{region}' unavailable, using healthy fallback pool"

        ordered = sorted(pool, key=lambda candidate: candidate.model)
        digest = sha256(f"{request.session_id}:{active_region}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % len(ordered)
        selected = ordered[bucket]
        fallbacks = [candidate.model for candidate in ordered if candidate.model != selected.model][
            :3
        ]
        return RoutingDecision(
            chosen_model=selected.model,
            provider=selected.provider,
            routing_strategy=self.strategy_name,
            rationale=f"{rationale}; sticky bucket {bucket}/{len(ordered)} -> {selected.model}",
            fallback_chain=fallbacks,
        )

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Route warmup requests, then pin the session to a stable region."""
        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        request_count = self._warmup_stats.advance(request.session_id)
        warmup_region = self._warmup_region(request)

        if request_count <= self._warmup_request_count:
            return self._pick(
                request,
                eligible,
                warmup_region,
                "sticky-region-warmup "
                f"warmup request {request_count}/{self._warmup_request_count} "
                f"for session '{request.session_id}' in region '{warmup_region}'",
            )

        target_region = self._post_warmup_region(request, warmup_region)
        if self._warmup_stats.pinned_region(request.session_id) is None:
            self._warmup_stats.pin(request.session_id, target_region)
        return self._pick(
            request,
            eligible,
            target_region,
            "sticky-region-warmup completed "
            f"{self._warmup_request_count} warmup requests; pinned session "
            f"'{request.session_id}' to region '{target_region}'",
        )


class TenantQuotaBurstExceededError(RuntimeError):
    """Raised when a tenant has consumed its hard rolling request quota."""


class TenantQuotaBurstStats:
    """Rolling request timestamps used to enforce independent tenant quotas."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        """Initialize empty tenant quota windows."""
        if window_seconds <= 0.0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    @property
    def window_seconds(self) -> float:
        """Return the rolling quota window length."""
        return self._window_seconds

    def record(self, tenant_key: str, *, now: float | None = None) -> None:
        """Record one admitted tenant request."""
        observed_at = time.monotonic() if now is None else now
        requests = self._requests.setdefault(tenant_key, [])
        requests.append(observed_at)
        self._trim(tenant_key, observed_at)

    def usage(self, tenant_key: str, *, now: float | None = None) -> int:
        """Return admitted requests inside the active tenant window."""
        observed_at = time.monotonic() if now is None else now
        return len(self._trim(tenant_key, observed_at))

    def _trim(self, tenant_key: str, now: float) -> list[float]:
        cutoff = now - self._window_seconds
        requests = [
            observed_at
            for observed_at in self._requests.get(tenant_key, [])
            if observed_at >= cutoff
        ]
        if requests:
            self._requests[tenant_key] = requests
        else:
            self._requests.pop(tenant_key, None)
        return requests


class TenantQuotaBurstStrategy(RoutingStrategy):
    """Allow bounded tenant bursts, then shed load at a hard ceiling.

    Each tenant receives a steady request allowance and a short rolling burst
    band. Requests below the soft quota stay quality-first. Burst requests above
    the soft quota use the cheapest domain-compatible fallback, and requests
    beyond the hard quota fail closed before dispatch.
    """

    strategy_name = RoutingStrategyName.TENANT_QUOTA_BURST

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        quota_stats: TenantQuotaBurstStats,
        soft_quota: int = 60,
        hard_quota: int = 75,
    ) -> None:
        """Initialize tenant-quota-burst routing."""
        super().__init__(model_catalog)
        if soft_quota < 1:
            raise ValueError(f"soft_quota must be >= 1, got {soft_quota}")
        if hard_quota <= soft_quota:
            raise ValueError(
                f"hard_quota ({hard_quota}) must be greater than soft_quota ({soft_quota})"
            )
        self._quota_stats = quota_stats
        self._soft_quota = soft_quota
        self._hard_quota = hard_quota

    @staticmethod
    def _tenant_key(request: RouterRequest) -> str:
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = request.metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Admit a steady or burst request, or reject beyond the hard quota."""
        tenant_key = self._tenant_key(request)
        current_usage = self._quota_stats.usage(tenant_key)
        if current_usage >= self._hard_quota:
            raise TenantQuotaBurstExceededError(
                "tenant-quota-burst hard ceiling reached for tenant "
                f"'{tenant_key}': {current_usage}/{self._hard_quota} requests in "
                f"{self._quota_stats.window_seconds:.1f}s; shed before dispatch"
            )

        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        costs = {
            candidate.model: candidate.estimate_cost(
                signals.prompt_tokens_estimate,
                request.max_tokens,
            )
            for candidate in eligible
        }
        self._quota_stats.record(tenant_key)
        admitted_usage = current_usage + 1

        if current_usage < self._soft_quota:
            selected = max(
                eligible,
                key=lambda candidate: (
                    candidate.quality_score,
                    -costs[candidate.model],
                    candidate.model,
                ),
            )
            fallbacks = sorted(
                (candidate for candidate in eligible if candidate.model != selected.model),
                key=lambda candidate: (
                    -candidate.quality_score,
                    costs[candidate.model],
                    candidate.model,
                ),
            )
            rationale = (
                f"tenant-quota-burst tenant '{tenant_key}' steady usage "
                f"{admitted_usage}/{self._soft_quota}; selected quality-first "
                f"{selected.model}"
            )
        else:
            selected = min(
                eligible,
                key=lambda candidate: (
                    costs[candidate.model],
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            fallbacks = sorted(
                (candidate for candidate in eligible if candidate.model != selected.model),
                key=lambda candidate: (
                    costs[candidate.model],
                    -candidate.quality_score,
                    candidate.model,
                ),
            )
            rationale = (
                f"tenant-quota-burst tenant '{tenant_key}' admitted burst request "
                f"{admitted_usage}/{self._hard_quota} above soft quota "
                f"{self._soft_quota}; shed to cheapest fallback {selected.model}"
            )

        return RoutingDecision(
            chosen_model=selected.model,
            provider=selected.provider,
            routing_strategy=self.strategy_name,
            rationale=rationale,
            fallback_chain=[candidate.model for candidate in fallbacks[:3]],
        )


class ProviderTailLatencyHedgeStrategy(RoutingStrategy):
    """Hedge a quality-first request only when its provider p95 is too slow.

    Unlike the existing region hedge, which reacts to primary-region p50, this
    strategy watches provider tail latency across every domain-compatible model.
    It keeps the quality leader until that provider's p95 crosses a fixed
    threshold, then selects the fastest observed healthy provider alternative.
    """

    strategy_name = RoutingStrategyName.PROVIDER_TAIL_LATENCY_HEDGE

    def __init__(
        self,
        model_catalog: Mapping[str, ModelCandidate],
        latency_stats: LatencyStats,
        provider_health: ProviderHealth,
        tail_latency_threshold_ms: float = 1500.0,
    ) -> None:
        """Initialize provider-tail-latency-hedge routing."""
        super().__init__(model_catalog)
        if tail_latency_threshold_ms < 0.0:
            raise ValueError(
                f"tail_latency_threshold_ms must be non-negative, got {tail_latency_threshold_ms}"
            )
        self._latency_stats = latency_stats
        self._provider_health = provider_health
        self._tail_latency_threshold_ms = tail_latency_threshold_ms

    def choose(self, request: RouterRequest, signals: TaskSignals) -> RoutingDecision:
        """Keep the quality leader unless observed provider p95 triggers a hedge."""
        eligible = [
            candidate
            for candidate in self._model_catalog.values()
            if signals.domain_tag in candidate.supports_domains
        ] or list(self._model_catalog.values())
        healthy = [
            candidate
            for candidate in eligible
            if self._provider_health.is_available(candidate.provider)
        ] or eligible
        primary = max(
            healthy,
            key=lambda candidate: (
                candidate.quality_score,
                -candidate.estimate_cost(signals.prompt_tokens_estimate, request.max_tokens),
                candidate.model,
            ),
        )
        primary_p95 = self._latency_stats.p95(primary.provider)
        observed_alternatives = [
            candidate
            for candidate in healthy
            if candidate.provider != primary.provider
            and self._latency_stats.p95(candidate.provider) > 0.0
        ]

        if primary_p95 > self._tail_latency_threshold_ms and observed_alternatives:
            selected = min(
                observed_alternatives,
                key=lambda candidate: (
                    self._latency_stats.p95(candidate.provider),
                    -candidate.quality_score,
                    candidate.estimate_cost(
                        signals.prompt_tokens_estimate,
                        request.max_tokens,
                    ),
                    candidate.model,
                ),
            )
            rationale = (
                "provider-tail-latency-hedge quality leader "
                f"{primary.model} tail p95 {primary_p95:.1f}ms exceeded "
                f"{self._tail_latency_threshold_ms:.1f}ms; hedged across providers "
                f"to {selected.model} at p95 "
                f"{self._latency_stats.p95(selected.provider):.1f}ms"
            )
            return self._decision(selected.model, rationale)

        reason = (
            "no observed healthy provider alternative"
            if not observed_alternatives
            else "tail p95 remained within threshold"
        )
        rationale = (
            "provider-tail-latency-hedge stayed on quality leader "
            f"{primary.model}: provider p95 {primary_p95:.1f}ms, threshold "
            f"{self._tail_latency_threshold_ms:.1f}ms ({reason})"
        )
        return self._decision(primary.model, rationale)


def build_strategies(
    model_catalog: Mapping[str, ModelCandidate],
    latency_stats: LatencyStats,
    inflight_stats: InflightStats,
    quality_floor: float,
    ab_model_a: str,
    ab_model_b: str,
    ab_model_a_weight: float,
    provider_health: ProviderHealth,
    blend_quality_weight: float,
    blend_cost_weight: float,
    blend_latency_weight: float,
    request_cost_ceiling_usd: float,
    canary_stable_model: str,
    canary_model: str,
    canary_weight: float,
    latency_sla_ms: float,
    min_prefix_chars: int = 64,
    epsilon: float = 0.1,
    availability_slo: float | SuccessStats = 0.99,
    success_stats: SuccessStats | list[str] | None = None,
    failover_priority: list[str] | float | None = None,
    health_blend_success_weight: float = 0.35,
    health_blend_latency_weight: float = 0.25,
    health_blend_quality_weight: float = 0.25,
    health_blend_cost_weight: float = 0.15,
    concurrency_cap: int = 8,
    rate_limit_stats: RateLimitStats | None = None,
    token_bucket_stats: TokenBucketStats | None = None,
    hcl_health_weight: float = 0.4,
    hcl_cost_weight: float = 0.3,
    hcl_latency_weight: float = 0.3,
    tier_request_stats: TierRequestStats | None = None,
    tier_frontier_rpm: int = 30,
    tier_mid_rpm: int = 60,
    tier_economy_rpm: int = 120,
    provider_family_cost_ceiling_usd: float = 0.05,
    adaptive_exploration_base: float = 0.2,
    adaptive_exploration_min: float = 0.02,
    family_spend_window: FamilySpendWindow | None = None,
    soft_family_budget_usd: float = 5.0,
    sticky_region_failover_preferences: list[str] | None = None,
    latency_slo_ms: float = 2000.0,
    shadow_traffic_percent: float = 5.0,
    canary_cost_blend_percent: float = 10.0,
    cost_anomaly_stats: CostAnomalyStats | None = None,
    token_cost_anomaly_ratio: float = 2.0,
    latency_hedge_ms: float = 500.0,
    prompt_length_tier_tokens: int = 8000,
    retry_budget_default: int = 3,
    cache_hit_sticky_min_chars: int = 64,
    embedding_cache_namespace_prefix: str = "embed",
    circuit_half_open_probe_budget: int = 2,
    semantic_cache_ttl_seconds: float = 300.0,
    provider_spend_soft_usd: float = 10.0,
    carbon_aware_max_intensity: float = 400.0,
    tenant_concurrency_lease: int = 8,
    provider_error_budget_rate: float = 0.15,
    region_latency_p99_ms: float = 3000.0,
    sticky_canary_cost_percent: float = 10.0,
    queue_depth_soft_cap: int = 4,
    provider_quota_lookback: int = 100,
    adaptive_timeout_hedge_ratio: float = 1.5,
    token_bucket_tenant_rate: float = 5.0,
    region_carbon_blend_weight: float = 0.5,
    provider_weight_decay_factor: float = 0.5,
    provider_weight_recover: float = 0.1,
    provider_weight_stats: ProviderWeightStats | None = None,
    retry_after_default_seconds: float = 30.0,
    retry_after_cooldown: ProviderRetryAfterCooldown | None = None,
    latency_slope_window: int = 10,
    latency_slope_threshold_ms: float = 25.0,
    latency_slope_stats: LatencySlopeStats | None = None,
    provider_hourly_cost_ceiling_usd: float = 5.0,
    provider_hourly_spend_window: ProviderHourlySpendWindow | None = None,
    token_rpm_ceiling: int = 100_000,
    token_rpm_window: TokenRpmWindow | None = None,
    provider_circuit_probe_budget: int = 1,
    carbon_latency_carbon_weight: float = 0.5,
    carbon_latency_latency_weight: float = 0.5,
    adaptive_concurrency_base_cap: int = 8,
    adaptive_concurrency_min_cap: int = 1,
    adaptive_concurrency_latency_ms: float = 2000.0,
    provider_token_fair_share_ceiling: int = 100_000,
    region_failover_hysteresis_successes: int = 3,
    region_failover_hysteresis_stats: RegionFailoverHysteresisStats | None = None,
    tenant_budget_cascade_soft: float = 10.0,
    tenant_budget_cascade_hard: float = 12.5,
    tenant_budget_cascade_stats: TenantBudgetCascadeStats | None = None,
    provider_error_budget_reset_stats: ProviderErrorBudgetResetStats | None = None,
    sticky_region_warmup_stats: StickyRegionWarmupStats | None = None,
    sticky_region_warmup_requests: int = 3,
    provider_error_budget_reset_seconds: float = 60.0,
    provider_error_budget_reset_fraction: float = 0.15,
    tenant_quota_burst_stats: TenantQuotaBurstStats | None = None,
    tenant_quota_burst_soft: int = 60,
    tenant_quota_burst_hard: int = 75,
    tenant_quota_burst_window_seconds: float = 60.0,
    provider_tail_latency_hedge_ms: float = 1500.0,
) -> dict[RoutingStrategyName, RoutingStrategy]:
    """Build all built-in routing strategies.

    Args:
        model_catalog: Available model candidates by model name.
        latency_stats: Rolling provider latency observations.
        inflight_stats: Live provider in-flight attempt counters.
        quality_floor: Cost optimizer quality floor.
        ab_model_a: First A/B model arm.
        ab_model_b: Second A/B model arm.
        ab_model_a_weight: Bucket weight for model A.
        provider_health: Live provider health view for reliability routing.
        blend_quality_weight: Weighted-blend quality component weight.
        blend_cost_weight: Weighted-blend cost component weight.
        blend_latency_weight: Weighted-blend latency component weight.
        request_cost_ceiling_usd: Budget-aware per-request cost ceiling in USD.
        canary_stable_model: Canary strategy stable (majority) model.
        canary_model: Canary strategy model under gradual rollout.
        canary_weight: Fraction of traffic routed to the canary model.
        latency_sla_ms: Latency-budget per-request provider p95 SLA in
            milliseconds; also the adaptive-timeout realtime base budget.
        min_prefix_chars: Minimum system-prompt prefix length for
            prompt-prefix-cache sticky hashing.
        epsilon: Epsilon-greedy explore probability within ``[0.0, 1.0]``.
        availability_slo: SLO-aware minimum provider success rate within
            ``[0.0, 1.0]``.
        success_stats: Optional rolling provider success observations for
            SLO-aware, adaptive-timeout, and adaptive-exploration routing.
            When omitted a fresh empty stats window is used.
        failover_priority: Ordered model preference list for failover-priority
            routing. When omitted, uses the first four catalog models by
            insertion order as a deterministic default.
        health_blend_success_weight: Provider-health blend success-rate weight.
        health_blend_latency_weight: Provider-health blend inverse-latency
            weight.
        health_blend_quality_weight: Provider-health blend quality weight.
        health_blend_cost_weight: Provider-health blend inverse-cost weight.
        concurrency_cap: Per-provider live in-flight cap for concurrency-cap
            routing.
        rate_limit_stats: Optional rolling provider rate-limit observations for
            soft-rate-limit routing. When omitted a fresh empty window is used.
        token_bucket_stats: Optional shared per-provider token buckets for
            token-bucket-burst routing. When omitted a fresh bucket state is
            created with default capacity ``10`` and refill ``1.0``/s.
        hcl_health_weight: Health/cost/latency blend success-rate weight.
        hcl_cost_weight: Health/cost/latency blend inverse-cost weight.
        hcl_latency_weight: Health/cost/latency blend inverse-latency weight.
        tier_request_stats: Optional shared per-provider rolling request
            timestamps for model-tier-rate-limit routing. When omitted a fresh
            empty window is used.
        tier_frontier_rpm: RPM ceiling for frontier-tier models.
        tier_mid_rpm: RPM ceiling for mid-tier models.
        tier_economy_rpm: RPM ceiling for economy-tier models.
        provider_family_cost_ceiling_usd: Default per-provider-family cost
            ceiling in USD for provider-family-cost-ceiling routing.
        adaptive_exploration_base: Cold-start explore rate for
            adaptive-exploration within ``[0.0, 1.0]``.
        adaptive_exploration_min: Floor explore rate for adaptive-exploration
            within ``[0.0, 1.0]``.
        family_spend_window: Optional rolling provider-family spend tracker for
            soft-family-budget routing. When omitted a fresh empty window is
            used with the default one-hour horizon.
        soft_family_budget_usd: Soft rolling spend ceiling per provider family in
            USD for soft-family-budget routing.
        sticky_region_failover_preferences: Ordered region failover list for
            sticky-region-failover routing when a request omits ``region``.
        latency_slo_ms: Maximum acceptable provider p95 latency per request in
            milliseconds for latency-slo-shed routing.
        cost_anomaly_stats: Optional rolling cost-per-1k observations for
            token-cost-anomaly-shed routing. When omitted a fresh empty window
            is used.
        token_cost_anomaly_ratio: Multiplier above the rolling mean cost/1k that
            triggers shedding for token-cost-anomaly-shed routing.
        latency_hedge_ms: Primary-region provider p50 threshold in milliseconds
            for multi-region-latency-hedge routing.
        prompt_length_tier_tokens: Prompt-token threshold that triggers
            frontier shedding for prompt-length-tier-shed routing.
        retry_budget_default: Default remaining retries when a request omits
            metadata.retry_remaining for retry-budget-aware-failover routing.
        cache_hit_sticky_min_chars: Minimum prompt-prefix length for
            cache-hit-sticky-warm-pool sticky hashing.
        embedding_cache_namespace_prefix: Prefix for embedding-cache-key-namespace
            sticky hashing (isolates tenant/session cache keys).
        circuit_half_open_probe_budget: Maximum concurrent in-flight attempts
            across half-open providers for circuit-breaker-half-open-probe
            routing.
        provider_error_budget_rate: Maximum provider rolling error rate for
            provider-error-budget-shed routing.
        region_latency_p99_ms: Maximum acceptable provider p99 latency in
            milliseconds for region-latency-p99-shed routing.
        sticky_canary_cost_percent: Percentage of traffic that explores a cheaper
            healthy model for sticky-canary-cost routing.
        queue_depth_soft_cap: Soft maximum in-flight queue depth per provider for
            queue-depth-fairness routing.
        provider_quota_lookback: Recent provider selection count retained for
            provider-quota-fair-share routing.
        adaptive_timeout_hedge_ratio: Multiplier applied to the fastest observed
            eligible p95 for adaptive-timeout-hedge routing.

        region_carbon_blend_weight: Carbon share in ``[0.0, 1.0]`` for
            region-carbon-blend routing (``0`` = latency only, ``1`` = carbon only).
        provider_weight_decay_factor: Multiplicative failure penalty in
            ``(0.0, 1.0]`` for provider-weight-decay routing.
        provider_weight_recover: Additive success recovery step for
            provider-weight-decay routing.
        provider_weight_stats: Optional shared provider selection weights for
            provider-weight-decay routing. When omitted a fresh map is created.
        retry_after_default_seconds: Default Retry-After wait in seconds for
            retry-after-respect routing when a provider omits the header.
        retry_after_cooldown: Optional shared provider cooldown map for
            retry-after-respect routing. When omitted a fresh map is created.
        latency_slope_window: Recent latency sample window size for
            latency-slope-shed routing.
        latency_slope_threshold_ms: Maximum acceptable EWMA slope in
            milliseconds per sample step for latency-slope-shed routing.
        latency_slope_stats: Optional shared EWMA slope window for
            latency-slope-shed routing. When omitted a fresh window is created.

        provider_hourly_cost_ceiling_usd: Hard rolling hourly estimated-spend
            ceiling per provider in USD for provider-hourly-cost-ceiling
            routing.
        provider_hourly_spend_window: Optional shared rolling hourly estimated
            spend tracker for provider-hourly-cost-ceiling routing. When
            omitted a fresh one-hour window is created.
        token_rpm_ceiling: Rolling 60-second prompt-token ceiling per provider
            for token-rpm-ceiling routing.
        token_rpm_window: Optional shared rolling provider prompt-token tracker.
            When omitted a fresh 60-second window is created.

    Returns:
        Routing strategies keyed by strategy name.
    """
    if isinstance(availability_slo, SuccessStats):
        # Keep the factory compatible with the pre-prompt-prefix positional
        # layout used by existing adaptive-timeout callers/tests.
        legacy_success_stats = availability_slo
        legacy_failover_priority = success_stats
        legacy_health_blend_success_weight = failover_priority
        legacy_health_blend_latency_weight = health_blend_success_weight
        legacy_health_blend_quality_weight = health_blend_latency_weight
        legacy_health_blend_cost_weight = health_blend_quality_weight
        min_prefix_chars, epsilon, availability_slo = 64, float(min_prefix_chars), epsilon
        success_stats = legacy_success_stats
        failover_priority = (
            legacy_failover_priority if isinstance(legacy_failover_priority, list) else None
        )
        health_blend_success_weight = (
            float(legacy_health_blend_success_weight)
            if isinstance(legacy_health_blend_success_weight, (int, float))
            else health_blend_success_weight
        )
        health_blend_latency_weight = float(legacy_health_blend_latency_weight)
        health_blend_quality_weight = float(legacy_health_blend_quality_weight)
        health_blend_cost_weight = float(legacy_health_blend_cost_weight)

    # Narrow SuccessStats | list[str] | None after legacy positional remapping.
    resolved_success_stats: SuccessStats = (
        success_stats if isinstance(success_stats, SuccessStats) else SuccessStats()
    )
    resolved_rate_limit_stats = rate_limit_stats or RateLimitStats()
    resolved_token_bucket_stats = token_bucket_stats or TokenBucketStats(10, 1.0)
    resolved_tier_request_stats = tier_request_stats or TierRequestStats()
    resolved_failover_priority = (
        failover_priority if isinstance(failover_priority, list) else list(model_catalog.keys())[:4]
    )
    resolved_family_spend_window = family_spend_window or FamilySpendWindow()
    resolved_sticky_region_preferences = sticky_region_failover_preferences or [
        "eu",
        "us",
        "cn",
        "global",
    ]
    resolved_cost_anomaly_stats = cost_anomaly_stats or CostAnomalyStats()
    provider_request_share_stats = ProviderRequestShareStats(provider_quota_lookback)
    tenant_token_bucket_stats = TenantTokenBucketStats(token_bucket_tenant_rate)
    resolved_provider_weight_stats = provider_weight_stats or ProviderWeightStats(
        provider_weight_decay_factor,
        provider_weight_recover,
    )
    resolved_retry_after_cooldown = retry_after_cooldown or ProviderRetryAfterCooldown(
        retry_after_default_seconds
    )
    resolved_latency_slope_stats = latency_slope_stats or LatencySlopeStats(latency_slope_window)
    resolved_provider_hourly_spend_window = (
        provider_hourly_spend_window or ProviderHourlySpendWindow()
    )
    resolved_token_rpm_window = token_rpm_window or TokenRpmWindow()
    resolved_region_failover_hysteresis_stats = (
        region_failover_hysteresis_stats or RegionFailoverHysteresisStats()
    )
    resolved_tenant_budget_cascade_stats = tenant_budget_cascade_stats or TenantBudgetCascadeStats()
    resolved_provider_error_budget_reset_stats = (
        provider_error_budget_reset_stats
        or ProviderErrorBudgetResetStats(provider_error_budget_reset_seconds)
    )
    resolved_sticky_region_warmup_stats = sticky_region_warmup_stats or StickyRegionWarmupStats()
    resolved_tenant_quota_burst_stats = tenant_quota_burst_stats or TenantQuotaBurstStats(
        tenant_quota_burst_window_seconds
    )
    return {
        RoutingStrategyName.RULE_BASED: RuleBasedStrategy(model_catalog),
        RoutingStrategyName.CLASSIFIER: ClassifierStrategy(model_catalog),
        RoutingStrategyName.COST_OPTIMAL: CostOptimalStrategy(model_catalog, quality_floor),
        RoutingStrategyName.LATENCY_AWARE: LatencyAwareStrategy(model_catalog, latency_stats),
        RoutingStrategyName.COST_LATENCY_PARETO: CostLatencyParetoStrategy(
            model_catalog, latency_stats
        ),
        RoutingStrategyName.LEAST_BUSY: LeastBusyStrategy(model_catalog, inflight_stats),
        RoutingStrategyName.CONCURRENCY_CAP: ConcurrencyCapStrategy(
            model_catalog,
            inflight_stats,
            concurrency_cap,
        ),
        RoutingStrategyName.RELIABILITY_AWARE: ReliabilityAwareStrategy(
            model_catalog, provider_health
        ),
        RoutingStrategyName.WEIGHTED_BLEND: WeightedBlendStrategy(
            model_catalog,
            latency_stats,
            blend_quality_weight,
            blend_cost_weight,
            blend_latency_weight,
        ),
        RoutingStrategyName.BUDGET_AWARE: BudgetAwareStrategy(
            model_catalog,
            request_cost_ceiling_usd,
        ),
        RoutingStrategyName.PROVIDER_FAMILY_COST_CEILING: ProviderFamilyCostCeilingStrategy(
            model_catalog,
            provider_family_cost_ceiling_usd,
        ),
        RoutingStrategyName.STICKY_SESSION: StickySessionStrategy(model_catalog),
        RoutingStrategyName.LATENCY_BUDGET: LatencyBudgetStrategy(
            model_catalog,
            latency_stats,
            latency_sla_ms,
        ),
        RoutingStrategyName.ADAPTIVE_TIMEOUT: AdaptiveTimeoutStrategy(
            model_catalog,
            latency_stats,
            resolved_success_stats,
            latency_sla_ms,
        ),
        RoutingStrategyName.VALUE: ValueStrategy(model_catalog),
        RoutingStrategyName.COMPLEXITY_TIER: ComplexityTierStrategy(model_catalog),
        RoutingStrategyName.ROUND_ROBIN: RoundRobinStrategy(model_catalog),
        RoutingStrategyName.CASCADE: CascadeStrategy(model_catalog),
        RoutingStrategyName.EPSILON_GREEDY: EpsilonGreedyStrategy(model_catalog, epsilon),
        RoutingStrategyName.ADAPTIVE_EXPLORATION: AdaptiveExplorationStrategy(
            model_catalog,
            resolved_success_stats,
            adaptive_exploration_base,
            adaptive_exploration_min,
        ),
        RoutingStrategyName.TOKEN_BUDGET: TokenBudgetStrategy(model_catalog),
        RoutingStrategyName.GEO_REGION: GeoRegionStrategy(model_catalog),
        RoutingStrategyName.REGION_TIER_AFFINITY: RegionTierAffinityStrategy(model_catalog),
        RoutingStrategyName.SOFT_FAMILY_BUDGET: SoftFamilyBudgetStrategy(
            model_catalog,
            resolved_family_spend_window,
            soft_family_budget_usd,
        ),
        RoutingStrategyName.STICKY_REGION_FAILOVER: StickyRegionFailoverStrategy(
            model_catalog,
            provider_health,
            resolved_sticky_region_preferences,
        ),
        RoutingStrategyName.SLO_AWARE: SloAwareStrategy(
            model_catalog,
            resolved_success_stats,
            availability_slo,
        ),
        RoutingStrategyName.PROMPT_PREFIX_CACHE: PromptPrefixCacheStrategy(
            model_catalog,
            quality_floor,
            min_prefix_chars,
        ),
        RoutingStrategyName.SOFT_RATE_LIMIT: SoftRateLimitStrategy(
            model_catalog,
            provider_health,
            resolved_rate_limit_stats,
        ),
        RoutingStrategyName.TOKEN_BUCKET_BURST: TokenBucketBurstStrategy(
            model_catalog,
            resolved_token_bucket_stats,
        ),
        RoutingStrategyName.MODEL_TIER_RATE_LIMIT: ModelTierRateLimitStrategy(
            model_catalog,
            resolved_tier_request_stats,
            tier_frontier_rpm,
            tier_mid_rpm,
            tier_economy_rpm,
        ),
        RoutingStrategyName.SEMANTIC_CACHE: SemanticCacheStrategy(model_catalog, quality_floor),
        RoutingStrategyName.FAILOVER_PRIORITY: FailoverPriorityStrategy(
            model_catalog,
            provider_health,
            resolved_failover_priority,
        ),
        RoutingStrategyName.PROVIDER_HEALTH_SCORE_BLEND: ProviderHealthScoreBlendStrategy(
            model_catalog,
            provider_health,
            resolved_success_stats,
            latency_stats,
            health_blend_success_weight,
            health_blend_latency_weight,
            health_blend_quality_weight,
            health_blend_cost_weight,
        ),
        RoutingStrategyName.HEALTH_COST_LATENCY: HealthCostLatencyStrategy(
            model_catalog,
            resolved_success_stats,
            latency_stats,
            hcl_health_weight,
            hcl_cost_weight,
            hcl_latency_weight,
        ),
        RoutingStrategyName.CANARY: CanaryStrategy(
            model_catalog,
            provider_health,
            canary_stable_model,
            canary_model,
            canary_weight,
        ),
        RoutingStrategyName.CANARY_TIER_BLEND: CanaryTierBlendStrategy(
            model_catalog,
            provider_health,
            canary_stable_model,
            canary_model,
            canary_weight,
        ),
        RoutingStrategyName.LATENCY_SLO_SHED: LatencySloShedStrategy(
            model_catalog,
            latency_stats,
            latency_slo_ms,
        ),
        RoutingStrategyName.SHADOW_TRAFFIC_MIRROR: ShadowTrafficMirrorStrategy(
            model_catalog=model_catalog,
            quality_floor=quality_floor,
            shadow_traffic_percent=shadow_traffic_percent,
        ),
        RoutingStrategyName.CANARY_COST_BLEND: CanaryCostBlendStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            canary_cost_blend_percent=canary_cost_blend_percent,
        ),
        RoutingStrategyName.TOKEN_COST_ANOMALY_SHED: TokenCostAnomalyShedStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            cost_anomaly_stats=resolved_cost_anomaly_stats,
            token_cost_anomaly_ratio=token_cost_anomaly_ratio,
        ),
        RoutingStrategyName.STICKY_TENANT_HASH: StickyTenantHashStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
        ),
        RoutingStrategyName.MULTI_REGION_LATENCY_HEDGE: MultiRegionLatencyHedgeStrategy(
            model_catalog=model_catalog,
            latency_stats=latency_stats,
            latency_hedge_ms=latency_hedge_ms,
        ),
        RoutingStrategyName.PROMPT_LENGTH_TIER_SHED: PromptLengthTierShedStrategy(
            model_catalog=model_catalog,
            prompt_length_tier_tokens=prompt_length_tier_tokens,
        ),
        RoutingStrategyName.RETRY_BUDGET_AWARE_FAILOVER: RetryBudgetAwareFailoverStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            latency_stats=latency_stats,
            retry_budget_default=retry_budget_default,
        ),
        RoutingStrategyName.CACHE_HIT_STICKY_WARM_POOL: CacheHitStickyWarmPoolStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            cache_hit_sticky_min_chars=cache_hit_sticky_min_chars,
        ),
        RoutingStrategyName.EMBEDDING_CACHE_KEY_NAMESPACE: EmbeddingCacheKeyNamespaceStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            namespace_prefix=embedding_cache_namespace_prefix,
        ),
        RoutingStrategyName.CIRCUIT_BREAKER_HALF_OPEN_PROBE: CircuitBreakerHalfOpenProbeStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            inflight_stats=inflight_stats,
            probe_budget=circuit_half_open_probe_budget,
        ),
        RoutingStrategyName.SEMANTIC_CACHE_TTL_AFFINITY: SemanticCacheTtlAffinityStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            ttl_seconds=semantic_cache_ttl_seconds,
        ),
        RoutingStrategyName.PROVIDER_SPEND_TELEMETRY: ProviderSpendTelemetryStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            soft_spend_usd=provider_spend_soft_usd,
        ),
        RoutingStrategyName.CARBON_AWARE_PREFERENCE: CarbonAwarePreferenceStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            max_intensity=carbon_aware_max_intensity,
        ),
        RoutingStrategyName.TENANT_CONCURRENCY_LEASE: TenantConcurrencyLeaseStrategy(
            model_catalog=model_catalog,
            inflight_stats=inflight_stats,
            tenant_concurrency_lease=tenant_concurrency_lease,
        ),
        RoutingStrategyName.PROVIDER_ERROR_BUDGET_SHED: ProviderErrorBudgetShedStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            success_stats=resolved_success_stats,
            provider_error_budget_rate=provider_error_budget_rate,
        ),
        RoutingStrategyName.REGION_LATENCY_P99_SHED: RegionLatencyP99ShedStrategy(
            model_catalog=model_catalog,
            latency_stats=latency_stats,
            region_latency_p99_ms=region_latency_p99_ms,
        ),
        RoutingStrategyName.STICKY_CANARY_COST: StickyCanaryCostStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            sticky_canary_cost_percent=sticky_canary_cost_percent,
        ),
        RoutingStrategyName.QUEUE_DEPTH_FAIRNESS: QueueDepthFairnessStrategy(
            model_catalog=model_catalog,
            inflight_stats=inflight_stats,
            queue_depth_soft_cap=queue_depth_soft_cap,
        ),
        RoutingStrategyName.PROVIDER_QUOTA_FAIR_SHARE: ProviderQuotaFairShareStrategy(
            model_catalog=model_catalog,
            request_share_stats=provider_request_share_stats,
        ),
        RoutingStrategyName.ADAPTIVE_TIMEOUT_HEDGE: AdaptiveTimeoutHedgeStrategy(
            model_catalog=model_catalog,
            latency_stats=latency_stats,
            hedge_ratio=adaptive_timeout_hedge_ratio,
        ),
        RoutingStrategyName.TOKEN_BUCKET_TENANT: TokenBucketTenantStrategy(
            model_catalog,
            tenant_token_bucket_stats,
        ),
        RoutingStrategyName.REGION_CARBON_BLEND: RegionCarbonBlendStrategy(
            model_catalog=model_catalog,
            latency_stats=latency_stats,
            blend_weight=region_carbon_blend_weight,
        ),
        RoutingStrategyName.PROVIDER_WEIGHT_DECAY: ProviderWeightDecayStrategy(
            model_catalog=model_catalog,
            provider_weight_stats=resolved_provider_weight_stats,
        ),
        RoutingStrategyName.RETRY_AFTER_RESPECT: RetryAfterRespectStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            retry_after_cooldown=resolved_retry_after_cooldown,
        ),
        RoutingStrategyName.LATENCY_SLOPE_SHED: LatencySlopeShedStrategy(
            model_catalog=model_catalog,
            latency_slope_stats=resolved_latency_slope_stats,
            provider_health=provider_health,
            latency_slope_threshold_ms=latency_slope_threshold_ms,
        ),
        RoutingStrategyName.PROVIDER_HOURLY_COST_CEILING: ProviderHourlyCostCeilingStrategy(
            model_catalog=model_catalog,
            provider_hourly_spend_window=resolved_provider_hourly_spend_window,
            provider_hourly_cost_ceiling_usd=provider_hourly_cost_ceiling_usd,
        ),
        RoutingStrategyName.QUALITY_WEIGHTED_STICKY: QualityWeightedStickyStrategy(
            model_catalog,
        ),
        RoutingStrategyName.TOKEN_RPM_CEILING: TokenRpmCeilingStrategy(
            model_catalog=model_catalog,
            token_rpm_window=resolved_token_rpm_window,
            token_rpm_ceiling=token_rpm_ceiling,
        ),
        RoutingStrategyName.PROVIDER_CIRCUIT_PROBE: ProviderCircuitProbeStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            probe_budget=provider_circuit_probe_budget,
        ),
        RoutingStrategyName.CARBON_LATENCY_BLEND: CarbonLatencyBlendStrategy(
            model_catalog=model_catalog,
            latency_stats=latency_stats,
            carbon_weight=carbon_latency_carbon_weight,
            latency_weight=carbon_latency_latency_weight,
        ),
        RoutingStrategyName.ADAPTIVE_CONCURRENCY_CAP: AdaptiveConcurrencyCapStrategy(
            model_catalog=model_catalog,
            inflight_stats=inflight_stats,
            success_stats=resolved_success_stats,
            latency_stats=latency_stats,
            base_cap=adaptive_concurrency_base_cap,
            min_cap=adaptive_concurrency_min_cap,
            latency_reference_ms=adaptive_concurrency_latency_ms,
        ),
        RoutingStrategyName.PROVIDER_TOKEN_FAIR_SHARE: ProviderTokenFairShareStrategy(
            model_catalog=model_catalog,
            token_rpm_window=resolved_token_rpm_window,
            token_fair_share_ceiling=provider_token_fair_share_ceiling,
        ),
        RoutingStrategyName.REGION_FAILOVER_HYSTERESIS: RegionFailoverHysteresisStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            hysteresis_stats=resolved_region_failover_hysteresis_stats,
            region_preferences=resolved_sticky_region_preferences,
            hysteresis_successes=region_failover_hysteresis_successes,
        ),
        RoutingStrategyName.TENANT_BUDGET_CASCADE: TenantBudgetCascadeStrategy(
            model_catalog=model_catalog,
            tenant_budget_stats=resolved_tenant_budget_cascade_stats,
            soft_budget=tenant_budget_cascade_soft,
            hard_budget=tenant_budget_cascade_hard,
        ),
        RoutingStrategyName.PROVIDER_ERROR_BUDGET_RESET: ProviderErrorBudgetResetStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            error_budget_stats=resolved_provider_error_budget_reset_stats,
            error_budget_fraction=provider_error_budget_reset_fraction,
        ),
        RoutingStrategyName.STICKY_REGION_WARMUP: StickyRegionWarmupStrategy(
            model_catalog=model_catalog,
            provider_health=provider_health,
            warmup_stats=resolved_sticky_region_warmup_stats,
            region_preferences=resolved_sticky_region_preferences,
            warmup_request_count=sticky_region_warmup_requests,
        ),
        RoutingStrategyName.TENANT_QUOTA_BURST: TenantQuotaBurstStrategy(
            model_catalog=model_catalog,
            quota_stats=resolved_tenant_quota_burst_stats,
            soft_quota=tenant_quota_burst_soft,
            hard_quota=tenant_quota_burst_hard,
        ),
        RoutingStrategyName.PROVIDER_TAIL_LATENCY_HEDGE: ProviderTailLatencyHedgeStrategy(
            model_catalog=model_catalog,
            latency_stats=latency_stats,
            provider_health=provider_health,
            tail_latency_threshold_ms=provider_tail_latency_hedge_ms,
        ),
        RoutingStrategyName.AB_TEST: ABRoutingStrategy(
            model_catalog,
            ab_model_a,
            ab_model_b,
            ab_model_a_weight,
        ),
    }
