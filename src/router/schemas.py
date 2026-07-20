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
    COMPLEXITY_TIER = "complexity-tier"
    ROUND_ROBIN = "round-robin"
    CASCADE = "cascade"
    EPSILON_GREEDY = "epsilon-greedy"
    TOKEN_BUDGET = "token-budget"  # noqa: S105  # strategy name, not a secret
    GEO_REGION = "geo-region"
    SEMANTIC_CACHE = "semantic-cache"
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
