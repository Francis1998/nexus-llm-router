"""FastAPI application for Nexus LLM Router."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from time import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from adapters.registry import AdapterRegistry, build_adapter_registry
from api.errors import ROUTING_ERRORS, GatewayError, conversation_error, routing_error
from api.models import model_object
from api.responses import (
    RESPONSES_PATH,
    complete_response,
    draft_response,
    new_response_id,
    reject_unsupported,
    stream_response_events,
    to_router_messages,
    validation_error,
)
from api.schemas import (
    DEFAULT_MAX_TOKENS,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelHealth,
    ModelList,
    ModelObject,
    ResponseCreateRequest,
    ResponseObject,
    SpendRecordRequest,
    SpendSummaryResponse,
)
from cache.response_cache import ResponseCache
from observability.logging import configure_logging
from observability.metrics import (
    metrics_response,
    response_cache_hits_total,
    response_cache_misses_total,
)
from observability.tracing import configure_tracing
from router.config import RouterSettings, load_settings
from router.engine import NexusRouter, RoutingFailedError
from router.schemas import ChatMessage, RouterRequest, RouterResponse, RoutingStrategyName
from safety.budget import BudgetExceededError
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import (
    IdempotencyKeyInUseError,
    IdempotencyRecord,
    IdempotencyReservation,
    IdempotencyStore,
)
from safety.key_fingerprint import fingerprint_api_key
from safety.rate_limiter import RateLimitExceededError
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter, TenantRateLimitExceededError

configure_logging()
app = FastAPI(title="Nexus LLM Router", version="0.1.0")
configure_tracing(app, "nexus-llm-router")


@lru_cache(maxsize=1)
def get_settings() -> RouterSettings:
    """Return cached runtime settings.

    Returns:
        Router settings.
    """
    return load_settings()


@lru_cache(maxsize=1)
def get_adapter_registry() -> AdapterRegistry:
    """Return cached provider adapter registry.

    Returns:
        Adapter registry.
    """
    return build_adapter_registry(get_settings().provider_settings)


@lru_cache(maxsize=1)
def get_router() -> NexusRouter:
    """Return cached router engine.

    Returns:
        Nexus router.
    """
    return NexusRouter(get_settings(), get_adapter_registry())


@lru_cache(maxsize=1)
def get_spend_ledger() -> SpendLedger:
    """Return the durable spend ledger.

    Returns:
        Spend ledger instance.
    """
    return SpendLedger(get_settings().spend_ledger_path)


@lru_cache(maxsize=1)
def get_response_cache() -> ResponseCache:
    """Return the process-local exact-match response cache.

    Returns:
        Response cache instance.
    """
    settings = get_settings()
    return ResponseCache(
        ttl_seconds=settings.response_cache_ttl_seconds,
        enabled=settings.response_cache_enabled,
    )


@lru_cache(maxsize=1)
def get_idempotency_store() -> IdempotencyStore:
    """Return the durable idempotency key store.

    Returns:
        Idempotency store instance.
    """
    settings = get_settings()
    return IdempotencyStore(
        settings.idempotency_path,
        ttl_seconds=settings.idempotency_ttl_seconds,
    )


@lru_cache(maxsize=1)
def get_tenant_rate_limiter() -> TenantRateLimiter:
    """Return the hard per-tenant rate limiter.

    Returns:
        Tenant rate limiter instance.
    """
    settings = get_settings()
    return TenantRateLimiter(
        capacity=settings.tenant_rate_limit_capacity,
        refill_per_second=settings.tenant_rate_limit_refill_per_second,
    )


@lru_cache(maxsize=1)
def get_gateway_guards() -> GatewayGuardService:
    """Return the composed gateway guard service.

    Returns:
        Gateway guard service wiring idempotency + tenant rate limits.
    """
    return GatewayGuardService(
        idempotency_store=get_idempotency_store(),
        tenant_rate_limiter=get_tenant_rate_limiter(),
        idempotency_lease_seconds=get_settings().idempotency_lease_seconds,
    )


@app.get("/health")
async def health() -> dict[str, object]:
    """Return application and provider health.

    Returns:
        Health payload.
    """
    provider_health: dict[str, bool] = {}
    for provider, adapter in get_adapter_registry().items():
        provider_health[provider] = await adapter.health_check()
    return {"status": "ok", "providers": provider_health}


@app.get("/metrics")
def metrics() -> object:
    """Return Prometheus metrics.

    Returns:
        Prometheus exposition response.
    """
    return metrics_response()


@dataclass(frozen=True, slots=True)
class _Caller:
    """Identity resolved for one gateway request."""

    api_key_id: str
    end_user: str | None
    tenant_id: str | None

    @property
    def principal(self) -> str:
        """Budget, spend, and response-cache subject (end user or key fingerprint)."""
        return self.end_user or self.api_key_id

    @property
    def idempotency_tenant(self) -> str:
        """Scope for ``Idempotency-Key`` replays."""
        return self.tenant_id or self.principal


def _parse_strategy(header: str | None) -> RoutingStrategyName | None:
    """Parse the optional ``X-Router-Strategy`` header."""
    if not header:
        return None
    try:
        return RoutingStrategyName(header)
    except ValueError as exception:
        raise GatewayError(
            status.HTTP_400_BAD_REQUEST,
            f"unsupported routing strategy: {header}",
            code="invalid_value",
            param="X-Router-Strategy",
        ) from exception


def _admit_caller(request: Request, tenant_id: str | None, end_user: str | None) -> _Caller:
    """Fingerprint the API key and enforce the optional hard tenant rate limit."""
    # Never use the raw Authorization header as an identifier: it would leak the
    # secret into error details, the spend ledger, and idempotency/cache tenants.
    caller = _Caller(
        api_key_id=fingerprint_api_key(request.headers.get("authorization")),
        end_user=end_user,
        tenant_id=tenant_id,
    )
    # Hard tenant limit only when X-Tenant-Id is present (backward compatible).
    if tenant_id:
        try:
            get_gateway_guards().assert_tenant_allowed(tenant_id)
        except TenantRateLimitExceededError as exception:
            raise GatewayError(
                status.HTTP_429_TOO_MANY_REQUESTS,
                str(exception),
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            ) from exception
    return caller


def _responses_idempotency_scope(idempotency_key: str) -> str:
    """Scope a ``/v1/responses`` Idempotency-Key away from chat-completion keys.

    Chat keys stay raw so records stored before this endpoint existed still
    replay. HTTP header values cannot contain line breaks, so a key ending in
    ``"\\n/v1/responses"`` can never equal any client-supplied chat key.
    """
    return f"{idempotency_key}\n{RESPONSES_PATH}"


async def _complete_cached(
    router_request: RouterRequest, *, temperature: float, tenant: str
) -> RouterResponse:
    """Serve from the exact-match response cache or route and cache the result."""
    cache = get_response_cache()
    cache_key = ResponseCache.make_key(
        model=router_request.requested_model,
        messages=router_request.messages,
        temperature=temperature,
        tenant=tenant,
        max_tokens=router_request.max_tokens,
        strategy=(router_request.strategy or get_settings().default_strategy).value,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        response_cache_hits_total.inc()
        if not isinstance(cached, RouterResponse):
            raise GatewayError(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "invalid cached router response type",
                error_type="server_error",
            )
        return cached
    response_cache_misses_total.inc()
    try:
        router_response = await get_router().complete(router_request)
    except ROUTING_ERRORS as exception:
        raise routing_error(exception) from exception
    cache.set(cache_key, router_response)
    return router_response


async def _provider_model_health(router: NexusRouter) -> dict[str, ModelHealth]:
    """Return live health for every provider in the router catalog."""
    adapters = dict(get_adapter_registry().items())
    health: dict[str, ModelHealth] = {}
    for provider in {candidate.provider for candidate in router.model_catalog.values()}:
        adapter = adapters.get(provider)
        health[provider] = ModelHealth(
            api_key_configured=adapter is not None and await adapter.health_check(),
            routable=router.provider_routable(provider),
        )
    return health


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List routable catalog models in the OpenAI format, with nexus metadata.

    Returns:
        OpenAI ``list`` of ``model`` objects in catalog order.
    """
    router = get_router()
    health = await _provider_model_health(router)
    return ModelList(
        data=[
            model_object(candidate, health[candidate.provider])
            for candidate in router.model_catalog.values()
        ]
    )


@app.get("/v1/models/{model_id}", response_model=None)
async def retrieve_model(model_id: str) -> ModelObject | JSONResponse:
    """Return one catalog model, or an OpenAI-style 404 for unknown ids.

    Args:
        model_id: Model identifier such as ``gpt-6-astra``.

    Returns:
        The ``model`` object, or a ``model_not_found`` error envelope.
    """
    router = get_router()
    candidate = router.model_catalog.get(model_id)
    if candidate is None:
        error = GatewayError(
            404, f"The model '{model_id}' does not exist", param="model", code="model_not_found"
        )
        return JSONResponse(status_code=404, content=error.to_openai_body())
    health = await _provider_model_health(router)
    return model_object(candidate, health[candidate.provider])


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    x_router_strategy: Annotated[str | None, Header(alias="X-Router-Strategy")] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Route an OpenAI-compatible chat completion request.

    Args:
        payload: OpenAI-compatible request payload.
        request: FastAPI request object.
        x_router_strategy: Optional routing strategy header.
        idempotency_key: Optional durable idempotency key for safe retries.
        x_tenant_id: Optional tenant id for hard per-tenant rate limiting.

    Returns:
        OpenAI-compatible chat completion response.

    Raises:
        HTTPException: If routing fails or safety controls reject the request.
    """
    if (invalid := conversation_error(payload.messages, "messages")) is not None:
        return JSONResponse(status_code=invalid.status_code, content=invalid.to_openai_body())
    try:
        strategy = _parse_strategy(x_router_strategy)
        caller = _admit_caller(request, x_tenant_id, payload.user)
    except GatewayError as exception:
        raise HTTPException(exception.status_code, detail=exception.message) from exception
    guards = get_gateway_guards()
    reservation: IdempotencyReservation | None = None
    if idempotency_key:
        existing: IdempotencyRecord | None
        if payload.stream:
            # Streams are never stored, so they only replay a completed response.
            existing = guards.check_idempotency(idempotency_key, caller.idempotency_tenant)
        else:
            try:
                claim = guards.reserve_idempotency(idempotency_key, caller.idempotency_tenant)
            except IdempotencyKeyInUseError as exception:
                raise HTTPException(409, detail=str(exception)) from exception
            if isinstance(claim, IdempotencyReservation):
                reservation, existing = claim, None
            else:
                existing = claim
        if existing is not None:
            return JSONResponse(
                content=json.loads(existing.response_json),
                status_code=existing.status_code,
            )
    try:
        router_request = RouterRequest(
            request_id=uuid4().hex,
            messages=payload.messages,
            api_key_id=caller.api_key_id,
            user_id=caller.principal,
            session_id=payload.user or "default",
            requested_model=payload.model,
            strategy=strategy,
            max_tokens=payload.max_tokens,
            stream=payload.stream,
        )
        if payload.stream:
            return _stream_chat_completion(router_request)

        try:
            router_response = await _complete_cached(
                router_request, temperature=payload.temperature, tenant=caller.principal
            )
        except GatewayError as exception:
            raise HTTPException(exception.status_code, detail=exception.message) from exception
        completion = ChatCompletionResponse.from_router_response(router_response)
        if reservation is not None:
            guards.complete_idempotency(reservation, completion.model_dump_json())
    except BaseException:
        if reservation is not None:
            guards.release_idempotency(reservation)
        raise
    return completion


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exception: RequestValidationError
) -> JSONResponse:
    """Return OpenAI-style 400s for ``/v1/responses``; keep FastAPI 422s elsewhere.

    Args:
        request: FastAPI request object.
        exception: Request validation failure.

    Returns:
        JSON error response.
    """
    if request.url.path != RESPONSES_PATH:
        return await request_validation_exception_handler(request, exception)
    error = validation_error(exception.errors())
    return JSONResponse(status_code=error.status_code, content=error.to_openai_body())


@app.post(RESPONSES_PATH, response_model=None)
async def create_response(
    payload: ResponseCreateRequest,
    request: Request,
    x_router_strategy: Annotated[str | None, Header(alias="X-Router-Strategy")] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> JSONResponse | StreamingResponse:
    """Route an OpenAI Responses API request (documented text-only subset).

    Args:
        payload: Responses API request body.
        request: FastAPI request object.
        x_router_strategy: Optional routing strategy header.
        idempotency_key: Optional idempotency key (non-streaming requests only).
        x_tenant_id: Optional tenant id for hard per-tenant rate limiting.

    Returns:
        A ``response`` object, an SSE event stream, or an OpenAI-style error.
    """
    try:
        return await _create_response(
            payload, request, x_router_strategy, idempotency_key, x_tenant_id
        )
    except GatewayError as exception:
        return JSONResponse(status_code=exception.status_code, content=exception.to_openai_body())


async def _create_response(
    payload: ResponseCreateRequest,
    request: Request,
    x_router_strategy: str | None,
    idempotency_key: str | None,
    x_tenant_id: str | None,
) -> JSONResponse | StreamingResponse:
    """Validate, route, and render one ``POST /v1/responses`` request."""
    reject_unsupported(payload)
    messages = to_router_messages(payload)
    strategy = _parse_strategy(x_router_strategy)
    end_user = payload.safety_identifier or payload.user
    caller = _admit_caller(request, x_tenant_id, end_user)
    guards = get_gateway_guards()
    replay_key = _responses_idempotency_scope(idempotency_key) if idempotency_key else None
    reservation: IdempotencyReservation | None = None
    if replay_key and not payload.stream:
        try:
            claim = guards.reserve_idempotency(replay_key, caller.idempotency_tenant)
        except IdempotencyKeyInUseError as exception:
            raise GatewayError(409, str(exception), code="idempotency_key_in_use") from exception
        if isinstance(claim, IdempotencyRecord):
            return JSONResponse(
                content=json.loads(claim.response_json),
                status_code=claim.status_code,
            )
        reservation = claim
    try:
        return await _route_response(
            payload, messages, strategy, caller, end_user, guards, reservation
        )
    except BaseException:
        if reservation is not None:
            guards.release_idempotency(reservation)
        raise


async def _route_response(
    payload: ResponseCreateRequest,
    messages: list[ChatMessage],
    strategy: RoutingStrategyName | None,
    caller: _Caller,
    end_user: str | None,
    guards: GatewayGuardService,
    reservation: IdempotencyReservation | None,
) -> JSONResponse | StreamingResponse:
    """Route an admitted ``/v1/responses`` request and store its idempotent result."""
    router_request = RouterRequest(
        request_id=uuid4().hex,
        messages=messages,
        api_key_id=caller.api_key_id,
        user_id=caller.principal,
        session_id=end_user or "default",
        requested_model=payload.model,
        strategy=strategy,
        max_tokens=payload.max_output_tokens or DEFAULT_MAX_TOKENS,
        stream=bool(payload.stream),
    )
    draft = draft_response(
        payload,
        response_id=new_response_id(),
        created_at=int(time()),
        model=payload.model or "nexus-routed",
    )
    if payload.stream:
        return await _stream_response(router_request, draft)

    router_response = await _complete_cached(
        router_request,
        temperature=draft.temperature,
        tenant=caller.principal,
    )
    body = complete_response(draft, router_response, f"msg_{uuid4().hex}").model_dump(mode="json")
    if reservation is not None:
        guards.complete_idempotency(reservation, json.dumps(body))
    return JSONResponse(content=body)


async def _stream_response(
    router_request: RouterRequest, draft: ResponseObject
) -> StreamingResponse:
    """Start a routed stream and render it as Responses API SSE events.

    The stream is pulled until its first non-empty chunk before the HTTP
    response starts, so routing failures before any output return a normal
    HTTP error.
    """
    completed: list[RouterResponse] = []
    chunks = get_router().complete_stream(router_request, on_complete=completed.append)
    first_chunk: tuple[str, str] | None = None
    try:
        # Empty chunks are not output: keep pulling until text arrives or the
        # stream ends, remembering the model for an empty but successful answer.
        while (chunk := await anext(chunks, None)) is not None:
            first_chunk = chunk
            if chunk[1]:
                break
    except ROUTING_ERRORS as exception:
        raise routing_error(exception) from exception
    return StreamingResponse(
        stream_response_events(draft, first_chunk, chunks, completed),
        media_type="text/event-stream",
    )


def _stream_chat_completion(router_request: RouterRequest) -> StreamingResponse:
    """Return an OpenAI-compatible SSE stream for a routed completion."""

    async def event_publisher() -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())
        model_name = router_request.requested_model or "nexus-routed"
        completed: list[RouterResponse] = []
        try:
            first = True
            stream = get_router().complete_stream(router_request, on_complete=completed.append)
            async for model_name, chunk in stream:
                delta: dict[str, str] = {"content": chunk}
                if first:
                    delta = {"role": "assistant", "content": chunk}
                    first = False
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": completed[-1].finish_reason if completed else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
        except RateLimitExceededError as exception:
            error = {"error": {"message": str(exception), "type": "rate_limit_error"}}
            yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"
        except BudgetExceededError as exception:
            error = {"error": {"message": str(exception), "type": "budget_exceeded"}}
            yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"
        except RoutingFailedError as exception:
            error_type = routing_error(exception).error_type
            error = {"error": {"message": str(exception), "type": error_type}}
            yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_publisher(), media_type="text/event-stream")


@app.get("/v1/spend", response_model=SpendSummaryResponse)
def get_spend(
    tenant: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> SpendSummaryResponse:
    """Return aggregated spend from the durable ledger.

    Args:
        tenant: Optional tenant filter.
        provider: Optional provider filter.
        model: Optional model filter.

    Returns:
        Spend summary payload.
    """
    summary = get_spend_ledger().summary(tenant=tenant, provider=provider, model=model)
    return SpendSummaryResponse(
        total_cost_usd=summary.total_cost_usd,
        request_count=summary.request_count,
        by_tenant=summary.by_tenant,
        by_provider=summary.by_provider,
        by_model=summary.by_model,
    )


@app.post("/v1/spend", response_model=SpendSummaryResponse)
def post_spend(payload: SpendRecordRequest) -> SpendSummaryResponse:
    """Record a spend event (portfolio / demo ingestion) and return summary.

    Args:
        payload: Spend event to persist.

    Returns:
        Updated spend summary for the event tenant.
    """
    ledger = get_spend_ledger()
    ledger.record(
        request_id=payload.request_id,
        tenant=payload.tenant,
        provider=payload.provider,
        model=payload.model,
        cost_usd=payload.cost_usd,
        input_tokens=payload.input_tokens,
        output_tokens=payload.output_tokens,
    )
    summary = ledger.summary(tenant=payload.tenant)
    return SpendSummaryResponse(
        total_cost_usd=summary.total_cost_usd,
        request_count=summary.request_count,
        by_tenant=summary.by_tenant,
        by_provider=summary.by_provider,
        by_model=summary.by_model,
    )
