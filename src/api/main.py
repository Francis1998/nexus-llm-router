"""FastAPI application for Nexus LLM Router."""

import json
from collections.abc import AsyncIterator
from functools import lru_cache
from time import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from adapters.registry import AdapterRegistry, build_adapter_registry
from api.schemas import ChatCompletionRequest, ChatCompletionResponse
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
from safety.rate_limiter import RateLimitExceededError

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
) -> ChatCompletionResponse | StreamingResponse:
    """Route an OpenAI-compatible chat completion request.

    Args:
        payload: OpenAI-compatible request payload.
        request: FastAPI request object.
        x_router_strategy: Optional routing strategy header.

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
        return ChatCompletionResponse.from_router_response(cached)
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
    return ChatCompletionResponse.from_router_response(router_response)


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
