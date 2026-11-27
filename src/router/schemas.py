"""Shared request, response, and decision schemas."""

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)


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
    REQUEST_CLASS_QOS = "request-class-qos"
    DEADLINE_AWARE_PICK = "deadline-aware-pick"
    PROVIDER_SUCCESS_FLOOR = "provider-success-floor"
    MODEL_CAPABILITY_GATE = "model-capability-gate"
    PROVIDER_WARMUP_WEIGHT = "provider-warmup-weight"
    TENANT_SOFT_ISOLATION = "tenant-soft-isolation"
    STRUCTURED_OUTPUT_PREFER = "structured-output-prefer"
    PROVIDER_EXCLUSION = "provider-exclusion"
    PROMPT_INJECTION_RISK_SHED = "prompt-injection-risk-shed"
    THINKING_MODEL_PREFER = "thinking-model-prefer"
    TOOL_CALLING_PREFER = "tool-calling-prefer"
    MULTIMODAL_INPUT_PREFER = "multimodal-input-prefer"
    STREAMING_PREFER = "streaming-prefer"
    LONG_CONTEXT_PREFER = "long-context-prefer"
    AUDIO_INPUT_PREFER = "audio-input-prefer"
    WEB_SEARCH_PREFER = "web-search-prefer"
    COMPUTER_USE_PREFER = "computer-use-prefer"
    IMAGE_GEN_PREFER = "image-gen-prefer"
    CODE_EXECUTION_PREFER = "code-execution-prefer"
    VIDEO_INPUT_PREFER = "video-input-prefer"
    PDF_INPUT_PREFER = "pdf-input-prefer"
    BATCH_API_PREFER = "batch-api-prefer"
    REASONING_EFFORT_PREFER = "reasoning-effort-prefer"
    TRANSCRIPTION_PREFER = "transcription-prefer"
    MODERATION_PREFER = "moderation-prefer"
    PREDICTED_OUTPUTS_PREFER = "predicted-outputs-prefer"
    PROMPT_CACHING_PREFER = "prompt-caching-prefer"
    FINE_TUNE_PREFER = "fine-tune-prefer"
    EMBEDDINGS_PREFER = "embeddings-prefer"
    RESPONSES_API_PREFER = "responses-api-prefer"
    FILE_SEARCH_PREFER = "file-search-prefer"
    VECTOR_STORE_PREFER = "vector-store-prefer"
    REALTIME_API_PREFER = "realtime-api-prefer"
    MCP_PREFER = "mcp-prefer"
    LOGPROBS_PREFER = "logprobs-prefer"
    BACKGROUND_MODE_PREFER = "background-mode-prefer"
    PARALLEL_TOOL_PREFER = "parallel-tool-prefer"
    DEEP_RESEARCH_PREFER = "deep-research-prefer"
    AGENTS_API_PREFER = "agents-api-prefer"
    WEBHOOK_PREFER = "webhook-prefer"
    MEMORY_TOOL_PREFER = "memory-tool-prefer"
    FEEDBACK_AWARE = "feedback-aware"
    AB_TEST = "ab"


TOOL_CAPABILITY = "tools"
"""Capability name (``/v1/models`` ``nexus.capabilities``) of models that can call functions."""

TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"
"""Function names every provider accepts (OpenAI's documented rule)."""

MAX_TOOLS = 128
"""Most tools one request may offer (OpenAI's documented limit)."""

MIN_TOOL_OUTPUT_TOKENS = 16
"""Smallest ``max_tokens`` of a request that uses tools.

The OpenAI Responses API, which serves OpenAI tool calls, rejects a smaller
``max_output_tokens``; requiring it for every provider keeps the client's limit
and the budget estimate exact whichever model is routed.
"""


class ToolCallFunction(BaseModel):
    """Function name and JSON-encoded arguments of one tool call."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str


class ToolCall(BaseModel):
    """OpenAI ``tool_calls[]`` entry: a function call the model asks the client to run."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class FunctionDefinition(BaseModel):
    """OpenAI ``tools[].function``: name, description, and JSON Schema parameters.

    ``parameters`` must be an object schema (``"type": "object"``): OpenAI,
    Anthropic (``input_schema``), and Gemini all require one, so a schema only
    some providers accept would make routing outcomes depend on the provider.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

    @field_validator("parameters")
    @classmethod
    def _require_object_schema(cls, parameters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject parameter schemas whose ``type`` is not ``"object"``."""
        if parameters is not None and parameters.get("type") != "object":
            raise ValueError('must be a JSON Schema object with "type": "object"')
        return parameters

    def parameters_schema(self) -> dict[str, Any]:
        """Return the JSON Schema of the arguments (an empty object schema when omitted)."""
        if self.parameters is None:
            return {"type": "object", "properties": {}}
        return self.parameters


class ToolDefinition(BaseModel):
    """OpenAI function tool offered to the model."""

    model_config = ConfigDict(frozen=True)

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    """Function named by a forced ``tool_choice``."""

    model_config = ConfigDict(frozen=True)

    name: str


class NamedToolChoice(BaseModel):
    """``{"type": "function", "function": {"name": ...}}``: the model must call this function."""

    model_config = ConfigDict(frozen=True)

    type: Literal["function"] = "function"
    function: ToolChoiceFunction


ToolChoice = Literal["none", "auto", "required"] | NamedToolChoice


class ToolConfig(BaseModel):
    """Validated function-calling settings of one request.

    ``tools`` holds 1 to ``MAX_TOOLS`` uniquely named functions, and a named
    ``tool_choice`` must name one of them. ``tool_choice`` and
    ``parallel_tool_calls`` stay ``None`` when the client omitted them, so
    adapters send provider defaults.
    """

    model_config = ConfigDict(frozen=True)

    tools: tuple[ToolDefinition, ...] = Field(min_length=1, max_length=MAX_TOOLS)
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None

    @model_validator(mode="after")
    def _check_names(self) -> "ToolConfig":
        """Reject duplicate function names and a ``tool_choice`` naming no declared tool."""
        names = self.tool_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate function names in tools: {', '.join(duplicates)}")
        forced = self.forced_function
        if forced is not None and forced not in names:
            raise ValueError(f"tool_choice names function '{forced}', which tools does not declare")
        return self

    @property
    def forced_function(self) -> str | None:
        """Return the function a named ``tool_choice`` forces, if any."""
        if isinstance(self.tool_choice, NamedToolChoice):
            return self.tool_choice.function.name
        return None

    @property
    def tool_names(self) -> list[str]:
        """Return the declared function names in request order."""
        return [tool.function.name for tool in self.tools]

    def canonical(self) -> dict[str, Any]:
        """Return a JSON-serializable form for cache keys and token estimates."""
        return {
            "tools": [tool.model_dump(exclude_none=True) for tool in self.tools],
            "tool_choice": (
                self.tool_choice.model_dump()
                if isinstance(self.tool_choice, NamedToolChoice)
                else self.tool_choice
            ),
            "parallel_tool_calls": self.parallel_tool_calls,
        }


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message.

    An assistant message may carry ``tool_calls``; its ``content`` may then be
    ``null`` (or omitted), which is stored as an empty string. A ``tool``
    message answers one of those calls through ``tool_call_id``.
    """

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_tool_fields(cls, data: object) -> object:
        """Accept ``content: null`` next to ``tool_calls``; treat ``tool_calls: []`` as absent."""
        if not isinstance(data, dict):
            return data
        if data.get("tool_calls") == []:
            data = {**data, "tool_calls": None}
        calls_tools = data.get("role") == "assistant" and bool(data.get("tool_calls"))
        if calls_tools and data.get("content") is None:
            data = {**data, "content": ""}
        return data


class KeyLimits(BaseModel):
    """Limits of the gateway API key (virtual key) a request was admitted with.

    The API layer attaches them; the engine routes only to ``allowed_models``
    (``None`` permits every catalog model) and skips candidates whose estimated
    cost exceeds ``remaining_budget_usd`` (``None`` means no key budget).
    """

    model_config = ConfigDict(frozen=True)

    allowed_models: frozenset[str] | None = None
    remaining_budget_usd: float | None = None


class RouterRequest(BaseModel):
    """Internal request shape consumed by the routing engine."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    messages: list[ChatMessage]
    api_key_id: str = "anonymous"
    key_limits: KeyLimits | None = None
    tenant_id: str | None = None
    user_id: str = "anonymous"
    session_id: str = "default"
    requested_model: str | None = None
    strategy: RoutingStrategyName | None = None
    token_budget: int = Field(default=4096, ge=1)
    region: str | None = None
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_config: ToolConfig | None = None

    @property
    def prompt_text(self) -> str:
        """Return the concatenated prompt text used by classifiers."""
        return "\n".join(message.content for message in self.messages)

    @property
    def uses_tools(self) -> bool:
        """Return whether the request offers tools or replays tool calls and results.

        Such a request can only be served by a model with the ``tools`` capability.
        """
        return self.tool_config is not None or any(
            message.tool_calls or message.role == "tool" for message in self.messages
        )

    @property
    def token_estimate_text(self) -> str:
        """Return the text the prompt-token estimate counts.

        Providers bill tool definitions and the assistant's earlier tool calls
        as prompt tokens, so they are appended to ``prompt_text``. Requests
        without tools estimate exactly ``prompt_text``.
        """
        parts = [self.prompt_text]
        if self.tool_config is not None:
            parts.append(json.dumps(self.tool_config.canonical(), sort_keys=True))
        parts.extend(
            f"{call.function.name}{call.function.arguments}"
            for message in self.messages
            for call in message.tool_calls or ()
        )
        return "\n".join(parts)


class TaskSignals(BaseModel):
    """Signals extracted from a request during the observe phase."""

    complexity_score: float = Field(ge=0.0, le=1.0)
    domain_tag: DomainTag
    latency_requirement: LatencyRequirement
    token_budget: int = Field(ge=1)
    prompt_tokens_estimate: int = Field(ge=0)


class LongPromptPricing(BaseModel):
    """Whole-request rates a provider bills once a prompt exceeds a token threshold.

    GPT-6 models bill 2x input and 1.5x output above 272K input tokens, and Claude
    Haiku 5.5 bills $0.50/$2.50 per 1M above 100K prompt tokens. In both cases the
    higher rates apply to the entire request, not only to the tokens past the
    threshold.
    """

    model_config = ConfigDict(frozen=True)

    threshold_tokens: int = Field(ge=1)
    input_cost_per_1k: float = Field(ge=0.0)
    output_cost_per_1k: float = Field(ge=0.0)

    def rates_for(self, input_tokens: int, base_rates: tuple[float, float]) -> tuple[float, float]:
        """Return the per-1K ``(input, output)`` rates billed for ``input_tokens``.

        Args:
            input_tokens: Prompt tokens of the request.
            base_rates: Standard per-1K ``(input, output)`` rates.

        Returns:
            The long-prompt rates above the threshold, otherwise ``base_rates``.
        """
        if input_tokens > self.threshold_tokens:
            return self.input_cost_per_1k, self.output_cost_per_1k
        return base_rates


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
    max_input_tokens: int | None = Field(default=None, ge=1)
    long_prompt_pricing: LongPromptPricing | None = None
    supported_regions: set[str] = Field(default_factory=lambda: {"global"})

    @property
    def input_token_limit(self) -> int:
        """Return the largest prompt this model accepts.

        Some models cap input below their total window (GPT-6 accepts at most
        922K input tokens of its 1.05M window); others accept prompts up to the
        full ``context_window``.
        """
        if self.max_input_tokens is None:
            return self.context_window
        return min(self.context_window, self.max_input_tokens)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate request cost in USD for this candidate.

        Long prompts are billed at ``long_prompt_pricing`` rates for the whole
        request when the candidate defines a prompt-length pricing tier.

        Args:
            input_tokens: Estimated input tokens.
            output_tokens: Estimated output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_rate, output_rate = self.input_cost_per_1k, self.output_cost_per_1k
        if self.long_prompt_pricing is not None:
            input_rate, output_rate = self.long_prompt_pricing.rates_for(
                input_tokens, (input_rate, output_rate)
            )
        input_cost = (input_tokens / 1000.0) * input_rate
        output_cost = (output_tokens / 1000.0) * output_rate
        return input_cost + output_cost


class RoutingDecision(BaseModel):
    """Output of the decide phase."""

    chosen_model: str
    provider: str
    routing_strategy: RoutingStrategyName
    rationale: str
    fallback_chain: list[str] = Field(default_factory=list)


FinishReason = Literal["stop", "length", "tool_calls"]
"""Normalized stop reason: ``length`` when output hit the token cap, ``tool_calls``
when the model stopped to call tools, otherwise ``stop``."""


class ProviderResponse(BaseModel):
    """Provider-normalized completion response."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = Field(ge=0.0)
    finish_reason: FinishReason = "stop"
    tool_calls: list[ToolCall] = Field(default_factory=list)


class RouterResponse(BaseModel):
    """Unified completion response returned by Nexus LLM Router.

    ``tool_calls`` is excluded from serialization: the API renders it in
    ``choices[0].message`` and the ``router`` block does not repeat arguments.
    """

    content: str
    model_used: str
    routing_strategy: RoutingStrategyName
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    rationale: str
    request_id: str
    finish_reason: FinishReason = "stop"
    tool_calls: list[ToolCall] = Field(default_factory=list, exclude=True)


class AuditRecord(BaseModel):
    """Durable audit record for a routing decision.

    ``api_key_id`` is the caller's non-reversible key fingerprint and
    ``tenant_id`` its optional ``X-Tenant-Id``; together they scope feedback on
    the request. ``domain`` is the analyzer's domain tag. For tool-calling
    requests, ``tool_count`` is the number of tools offered and
    ``tool_call_names`` the functions the model called; tool arguments and
    tool results are never recorded.
    """

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
    api_key_id: str | None = None
    tenant_id: str | None = None
    domain: DomainTag | None = None
    tool_count: int | None = None
    tool_call_names: list[str] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_tool_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep audit lines of tool-free requests unchanged by dropping unset tool fields."""
        data: dict[str, Any] = handler(self)
        for key in ("tool_count", "tool_call_names"):
            if data.get(key) is None:
                data.pop(key, None)
        return data
