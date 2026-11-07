"""FastAPI request and response schemas."""

from time import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from router.schemas import ChatMessage, RouterResponse, RoutingStrategyName

DEFAULT_MAX_TOKENS = 512
_METADATA_MAX_PAIRS = 16
_METADATA_MAX_KEY_CHARS = 64
_METADATA_MAX_VALUE_CHARS = 512


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float = Field(default=1.0)
    stream: bool = False
    user: str | None = None


class Usage(BaseModel):
    """OpenAI-compatible token usage payload."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    """OpenAI-compatible assistant message payload."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response with router metadata."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    router: RouterResponse

    @classmethod
    def from_router_response(cls, router_response: RouterResponse) -> "ChatCompletionResponse":
        """Build an OpenAI-compatible response from a router response.

        Args:
            router_response: Router response.

        Returns:
            OpenAI-compatible completion response.
        """
        return cls(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time()),
            model=router_response.model_used,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=router_response.content),
                    finish_reason=router_response.finish_reason,
                ),
            ],
            usage=Usage(
                prompt_tokens=router_response.input_tokens,
                completion_tokens=router_response.output_tokens,
                total_tokens=router_response.input_tokens + router_response.output_tokens,
            ),
            router=router_response,
        )


class SpendRecordRequest(BaseModel):
    """Request body for recording a spend event via the API."""

    request_id: str
    tenant: str = "anonymous"
    provider: str
    model: str
    cost_usd: float = Field(ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class SpendSummaryResponse(BaseModel):
    """Aggregated spend summary payload."""

    total_cost_usd: float
    request_count: int
    by_tenant: dict[str, float]
    by_provider: dict[str, float]
    by_model: dict[str, float]


class ResponseCreateRequest(BaseModel):
    """OpenAI Responses API ``POST /v1/responses`` request (supported subset).

    Unknown fields are ignored, mirroring ``ChatCompletionRequest``. Known
    Responses features that nexus cannot honor (stored state, tools, prompt
    templates, structured outputs) are declared here only so the endpoint can
    reject them explicitly instead of silently dropping them.
    """

    model: str | None = None
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=16)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool | None = False
    user: str | None = None
    safety_identifier: str | None = Field(default=None, max_length=128)
    metadata: dict[str, str] | None = None
    parallel_tool_calls: bool | None = None
    store: bool | None = None
    background: bool | None = None
    previous_response_id: str | None = None
    conversation: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    prompt: dict[str, Any] | None = None
    text: dict[str, Any] | None = None

    @field_validator("metadata")
    @classmethod
    def _limit_metadata(cls, metadata: dict[str, str] | None) -> dict[str, str] | None:
        """Enforce the OpenAI metadata limits (16 pairs, 64-char keys, 512-char values)."""
        if metadata is None:
            return None
        if len(metadata) > _METADATA_MAX_PAIRS:
            raise ValueError(f"metadata supports at most {_METADATA_MAX_PAIRS} key-value pairs")
        for key, value in metadata.items():
            if len(key) > _METADATA_MAX_KEY_CHARS:
                raise ValueError(f"metadata keys must be at most {_METADATA_MAX_KEY_CHARS} chars")
            if len(value) > _METADATA_MAX_VALUE_CHARS:
                raise ValueError(
                    f"metadata values must be at most {_METADATA_MAX_VALUE_CHARS} chars"
                )
        return metadata


class ResponseOutputText(BaseModel):
    """``output_text`` content part of an assistant output message."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """Assistant ``message`` output item."""

    id: str
    type: Literal["message"] = "message"
    status: Literal["in_progress", "completed", "incomplete"]
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText] = Field(default_factory=list)


class ResponseInputTokensDetails(BaseModel):
    """Input token breakdown."""

    cached_tokens: int = 0


class ResponseOutputTokensDetails(BaseModel):
    """Output token breakdown."""

    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Responses API token usage."""

    input_tokens: int
    input_tokens_details: ResponseInputTokensDetails = Field(
        default_factory=ResponseInputTokensDetails
    )
    output_tokens: int
    output_tokens_details: ResponseOutputTokensDetails = Field(
        default_factory=ResponseOutputTokensDetails
    )
    total_tokens: int


class ResponseErrorInfo(BaseModel):
    """Error attached to a ``failed`` response."""

    code: str
    message: str
    type: str | None = None
    param: str | None = None


class NexusRoutingMetadata(BaseModel):
    """Nexus routing decision attached to a Responses ``response`` or Anthropic ``message``."""

    routing_strategy: RoutingStrategyName
    rationale: str
    cost_usd: float
    latency_ms: float
    request_id: str

    @classmethod
    def from_router_response(cls, router_response: RouterResponse) -> "NexusRoutingMetadata":
        """Return the routing block for a routed completion.

        Args:
            router_response: Router response.

        Returns:
            Strategy, rationale, cost, latency, and request id of the route.
        """
        return cls(
            routing_strategy=router_response.routing_strategy,
            rationale=router_response.rationale,
            cost_usd=router_response.cost_usd,
            latency_ms=router_response.latency_ms,
            request_id=router_response.request_id,
        )


class ResponseObject(BaseModel):
    """OpenAI Responses API ``response`` object plus a namespaced ``nexus`` field."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    completed_at: int | None = None
    error: ResponseErrorInfo | None = None
    incomplete_details: dict[str, str] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    model: str
    output: list[ResponseOutputMessage] = Field(default_factory=list)
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    background: bool = False
    store: bool = False
    temperature: float = 1.0
    text: dict[str, Any] = Field(default_factory=lambda: {"format": {"type": "text"}})
    tool_choice: str = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_p: float | None = None
    truncation: Literal["disabled"] = "disabled"
    usage: ResponseUsage | None = None
    user: str | None = None
    safety_identifier: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    nexus: NexusRoutingMetadata | None = None


class ModelLongPromptPricing(BaseModel):
    """Whole-request rates billed once a prompt exceeds ``threshold_tokens``."""

    threshold_tokens: int
    input_per_1m_tokens: float
    output_per_1m_tokens: float


class ModelPricing(BaseModel):
    """List prices in USD per 1M tokens."""

    input_per_1m_tokens: float
    output_per_1m_tokens: float
    long_prompt: ModelLongPromptPricing | None = None


class ModelHealth(BaseModel):
    """Live provider health for a catalog model."""

    api_key_configured: bool
    routable: bool


class NexusModelInfo(BaseModel):
    """Nexus routing metadata for one catalog model."""

    provider: str
    context_window: int
    max_input_tokens: int | None
    pricing: ModelPricing
    capabilities: list[str]
    domains: list[str]
    regions: list[str]
    supports_realtime: bool
    quality_score: float
    health: ModelHealth


class ModelObject(BaseModel):
    """OpenAI ``model`` object with a namespaced ``nexus`` metadata block."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    nexus: NexusModelInfo


class ModelList(BaseModel):
    """OpenAI ``list`` response for ``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class AnthropicMetadata(BaseModel):
    """Anthropic ``metadata``; ``user_id`` is the end user nexus budgets for."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, max_length=512)


class AnthropicInputMessage(BaseModel):
    """One Anthropic ``messages`` entry; ``content`` is checked by the translation layer."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: Any


class MessageCountTokensRequest(BaseModel):
    """Anthropic ``POST /v1/messages/count_tokens`` request (supported subset).

    Unknown fields are rejected, as the Anthropic API does ("Extra inputs are
    not permitted"). Known features nexus cannot honor (tools, thinking,
    structured outputs) are declared only so the endpoint can reject them with
    a precise message instead of silently dropping them.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[AnthropicInputMessage]
    system: Any = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None
    speed: Literal["standard", "fast"] | None = None


class MessageCreateRequest(MessageCountTokensRequest):
    """Anthropic ``POST /v1/messages`` request (documented text-only subset)."""

    max_tokens: StrictInt = Field(ge=1)
    stream: StrictBool = False
    stop_sequences: list[str] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: StrictInt | None = Field(default=None, ge=0)
    metadata: AnthropicMetadata | None = None
    service_tier: Literal["auto", "standard_only"] | None = None
    inference_geo: str | None = None
    container: Any = None
    diagnostics: dict[str, Any] | None = None


class AnthropicTextBlock(BaseModel):
    """``text`` content block of an Anthropic ``message``."""

    type: Literal["text"] = "text"
    text: str


class AnthropicUsage(BaseModel):
    """Anthropic token usage (provider-reported counts)."""

    input_tokens: int
    output_tokens: int


AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence"]


class AnthropicMessage(BaseModel):
    """Anthropic ``message`` object plus a namespaced ``nexus`` routing block."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicTextBlock] = Field(default_factory=list)
    stop_reason: AnthropicStopReason | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
    nexus: NexusRoutingMetadata | None = None


class AnthropicTokenCount(BaseModel):
    """Anthropic ``POST /v1/messages/count_tokens`` response."""

    input_tokens: int
