"""Shared request, response, and decision schemas."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DomainTag(StrEnum):
    """Supported task domains extracted during observation."""

    CODE = "code"
    MEDICAL = "medical"
    LEGAL = "legal"
    GENERAL = "general"


class LatencyRequirement(StrEnum):
    """Latency profile requested or inferred for a prompt."""

    REALTIME = "realtime"
    BATCH = "batch"


class RoutingStrategyName(StrEnum):
    """Names of pluggable routing strategies."""

    RULE_BASED = "rule-based"
    CLASSIFIER = "classifier"
    COST_OPTIMAL = "cost-optimal"
    LATENCY_AWARE = "latency-aware"
    RELIABILITY_AWARE = "reliability-aware"
    WEIGHTED_BLEND = "weighted-blend"
    BUDGET_AWARE = "budget-aware"
    STICKY_SESSION = "sticky-session"
    VALUE = "value"
    CANARY = "canary"
    LATENCY_BUDGET = "latency-budget"
    ADAPTIVE_TIMEOUT = "adaptive-timeout"
    COMPLEXITY_TIER = "complexity-tier"
    ROUND_ROBIN = "round-robin"
    CASCADE = "cascade"
    EPSILON_GREEDY = "epsilon-greedy"
    ADAPTIVE_EXPLORATION = "adaptive-exploration"
    TOKEN_BUDGET = "token-budget"  # noqa: S105  # strategy name, not a secret
    GEO_REGION = "geo-region"
    SLO_AWARE = "slo-aware"
    SEMANTIC_CACHE = "semantic-cache"
    LEAST_BUSY = "least-busy"
    PROMPT_PREFIX_CACHE = "prompt-prefix-cache"
    CONCURRENCY_CAP = "concurrency-cap"
    SOFT_RATE_LIMIT = "soft-rate-limit"
    FAILOVER_PRIORITY = "failover-priority"
    PROVIDER_HEALTH_SCORE_BLEND = "provider-health-score-blend"
    HEALTH_COST_LATENCY = "health-cost-latency"
    COST_LATENCY_PARETO = "cost-latency-pareto"
    TOKEN_BUCKET_BURST = "token-bucket-burst"  # noqa: S105  # strategy name, not a secret
    MODEL_TIER_RATE_LIMIT = "model-tier-rate-limit"
    PROVIDER_FAMILY_COST_CEILING = "provider-family-cost-ceiling"
    REGION_TIER_AFFINITY = "region-tier-affinity"
    SOFT_FAMILY_BUDGET = "soft-family-budget"
    STICKY_REGION_FAILOVER = "sticky-region-failover"
    CANARY_TIER_BLEND = "canary-tier-blend"
    LATENCY_SLO_SHED = "latency-slo-shed"
    SHADOW_TRAFFIC_MIRROR = "shadow-traffic-mirror"
    CANARY_COST_BLEND = "canary-cost-blend"
    TOKEN_COST_ANOMALY_SHED = "token-cost-anomaly-shed"  # noqa: S105  # strategy name, not a secret
    STICKY_TENANT_HASH = "sticky-tenant-hash"
    MULTI_REGION_LATENCY_HEDGE = "multi-region-latency-hedge"
    PROMPT_LENGTH_TIER_SHED = "prompt-length-tier-shed"
    RETRY_BUDGET_AWARE_FAILOVER = "retry-budget-aware-failover"
    CACHE_HIT_STICKY_WARM_POOL = "cache-hit-sticky-warm-pool"
    EMBEDDING_CACHE_KEY_NAMESPACE = "embedding-cache-key-namespace"
    CIRCUIT_BREAKER_HALF_OPEN_PROBE = "circuit-breaker-half-open-probe"
    SEMANTIC_CACHE_TTL_AFFINITY = "semantic-cache-ttl-affinity"
    PROVIDER_SPEND_TELEMETRY = "provider-spend-telemetry"
    CARBON_AWARE_PREFERENCE = "carbon-aware-preference"
    TENANT_CONCURRENCY_LEASE = "tenant-concurrency-lease"
    PROVIDER_ERROR_BUDGET_SHED = "provider-error-budget-shed"
    REGION_LATENCY_P99_SHED = "region-latency-p99-shed"
    STICKY_CANARY_COST = "sticky-canary-cost"
    QUEUE_DEPTH_FAIRNESS = "queue-depth-fairness"
    PROVIDER_QUOTA_FAIR_SHARE = "provider-quota-fair-share"
    ADAPTIVE_TIMEOUT_HEDGE = "adaptive-timeout-hedge"
    TOKEN_BUCKET_TENANT = "token-bucket-tenant"  # noqa: S105  # strategy name
    REGION_CARBON_BLEND = "region-carbon-blend"
    PROVIDER_WEIGHT_DECAY = "provider-weight-decay"
    RETRY_AFTER_RESPECT = "retry-after-respect"
    LATENCY_SLOPE_SHED = "latency-slope-shed"
    PROVIDER_HOURLY_COST_CEILING = "provider-hourly-cost-ceiling"
    QUALITY_WEIGHTED_STICKY = "quality-weighted-sticky"
    TOKEN_RPM_CEILING = "token-rpm-ceiling"  # noqa: S105  # strategy name
    PROVIDER_CIRCUIT_PROBE = "provider-circuit-probe"
    CARBON_LATENCY_BLEND = "carbon-latency-blend"
    ADAPTIVE_CONCURRENCY_CAP = "adaptive-concurrency-cap"
    PROVIDER_TOKEN_FAIR_SHARE = "provider-token-fair-share"  # noqa: S105  # strategy name
    REGION_FAILOVER_HYSTERESIS = "region-failover-hysteresis"
    TENANT_BUDGET_CASCADE = "tenant-budget-cascade"
    PROVIDER_ERROR_BUDGET_RESET = "provider-error-budget-reset"
    STICKY_REGION_WARMUP = "sticky-region-warmup"
    TENANT_QUOTA_BURST = "tenant-quota-burst"
    PROVIDER_TAIL_LATENCY_HEDGE = "provider-tail-latency-hedge"
    STICKY_SESSION_MIGRATE = "sticky-session-migrate"
    PROVIDER_COLD_START_BIAS = "provider-cold-start-bias"
    TENANT_FAIR_QUEUE = "tenant-fair-queue"
    STICKY_REGION_DRAIN = "sticky-region-drain"
    PROVIDER_CANARY_SHADOW_SPLIT = "provider-canary-shadow-split"
    STICKY_MODEL_PIN_EXPIRE = "sticky-model-pin-expire"
    TENANT_PRIORITY_LANES = "tenant-priority-lanes"
    AB_TEST = "ab"


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class RouterRequest(BaseModel):
    """Internal request shape consumed by the routing engine."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    messages: list[ChatMessage]
    api_key_id: str = "anonymous"
    user_id: str = "anonymous"
    session_id: str = "default"
    requested_model: str | None = None
    strategy: RoutingStrategyName | None = None
    token_budget: int = Field(default=4096, ge=1)
    region: str | None = None
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Return the concatenated prompt text used by classifiers."""
        return "\n".join(message.content for message in self.messages)


class TaskSignals(BaseModel):
    """Signals extracted from a request during the observe phase."""

    complexity_score: float = Field(ge=0.0, le=1.0)
    domain_tag: DomainTag
    latency_requirement: LatencyRequirement
    token_budget: int = Field(ge=1)
    prompt_tokens_estimate: int = Field(ge=0)


class ModelCandidate(BaseModel):
    """A routable model and its operating characteristics."""

    model: str
    provider: str
    quality_score: float = Field(ge=0.0, le=1.0)
    input_cost_per_1k: float = Field(ge=0.0)
    output_cost_per_1k: float = Field(ge=0.0)
    supports_domains: set[DomainTag]
    supports_realtime: bool = True
    context_window: int = Field(default=128_000, ge=1)
    supported_regions: set[str] = Field(default_factory=lambda: {"global"})

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate request cost in USD for this candidate.

        Args:
            input_tokens: Estimated input tokens.
            output_tokens: Estimated output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (input_tokens / 1000.0) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k
        return input_cost + output_cost


class RoutingDecision(BaseModel):
    """Output of the decide phase."""

    chosen_model: str
    provider: str
    routing_strategy: RoutingStrategyName
    rationale: str
    fallback_chain: list[str] = Field(default_factory=list)


class ProviderResponse(BaseModel):
    """Provider-normalized completion response."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = Field(ge=0.0)


class RouterResponse(BaseModel):
    """Unified completion response returned by Nexus LLM Router."""

    content: str
    model_used: str
    routing_strategy: RoutingStrategyName
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    rationale: str
    request_id: str


class AuditRecord(BaseModel):
    """Durable audit record for a routing decision."""

    request_id: str
    chosen_model: str
    provider: str
    routing_strategy: RoutingStrategyName
    rationale: str
    latency_ms: float
    token_cost: float
    input_tokens: int
    output_tokens: int
    state: str
