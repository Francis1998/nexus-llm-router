"""FastAPI application for Nexus LLM Router."""

import json
from collections.abc import AsyncIterator
from functools import lru_cache
from time import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from adapters.registry import AdapterRegistry, build_adapter_registry
from api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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
from router.schemas import RouterRequest, RouterResponse, RoutingStrategyName
from safety.budget import BudgetExceededError
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
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
    try:
        strategy = RoutingStrategyName(x_router_strategy) if x_router_strategy else None
    except ValueError as exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported routing strategy: {x_router_strategy}",
        ) from exception
    api_key_id = request.headers.get("authorization", "anonymous")
    guards = get_gateway_guards()
    # Hard tenant limit only when X-Tenant-Id is present (backward compatible).
    if x_tenant_id:
        try:
            guards.assert_tenant_allowed(x_tenant_id)
        except TenantRateLimitExceededError as exception:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exception),
            ) from exception
    idempotency_tenant = x_tenant_id or payload.user or api_key_id
    if idempotency_key:
        existing = guards.check_idempotency(idempotency_key, idempotency_tenant)
        if existing is not None:
            return JSONResponse(
                content=json.loads(existing.response_json),
                status_code=existing.status_code,
            )
    router_request = RouterRequest(
        request_id=uuid4().hex,
        messages=payload.messages,
        api_key_id=api_key_id,
        user_id=payload.user or api_key_id,
        session_id=payload.user or "default",
        requested_model=payload.model,
        strategy=strategy,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
    )
    if payload.stream:
        return _stream_chat_completion(router_request)

    tenant = payload.user or api_key_id
    cache = get_response_cache()
    cache_key = ResponseCache.make_key(
        model=payload.model,
        messages=payload.messages,
        temperature=payload.temperature,
        tenant=tenant,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        response_cache_hits_total.inc()
        if not isinstance(cached, RouterResponse):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="invalid cached router response type",
            )
        completion = ChatCompletionResponse.from_router_response(cached)
        if idempotency_key:
            guards.remember_response(
                idempotency_key,
                idempotency_tenant,
                completion.model_dump_json(),
                status_code=200,
            )
        return completion
    response_cache_misses_total.inc()

    try:
        router_response = await get_router().complete(router_request)
    except RateLimitExceededError as exception:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exception)
        ) from exception
    except BudgetExceededError as exception:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=str(exception)
        ) from exception
    except RoutingFailedError as exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exception)
        ) from exception
    cache.set(cache_key, router_response)
    completion = ChatCompletionResponse.from_router_response(router_response)
    if idempotency_key:
        guards.remember_response(
            idempotency_key,
            idempotency_tenant,
            completion.model_dump_json(),
            status_code=200,
        )
    return completion


def _stream_chat_completion(router_request: RouterRequest) -> StreamingResponse:
    """Return an OpenAI-compatible SSE stream for a routed completion."""

    async def event_publisher() -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid4().hex}"
        created = int(time())
        model_name = router_request.requested_model or "nexus-routed"
        try:
            first = True
            async for model_name, chunk in get_router().complete_stream(router_request):
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
                        "finish_reason": "stop",
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
            error = {"error": {"message": str(exception), "type": "routing_failed"}}
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
