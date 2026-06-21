"""FastAPI application for Nexus LLM Router."""

from functools import lru_cache
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, status

from adapters.registry import AdapterRegistry, build_adapter_registry
from api.schemas import ChatCompletionRequest, ChatCompletionResponse
from observability.logging import configure_logging
from observability.metrics import metrics_response
from observability.tracing import configure_tracing
from router.config import RouterSettings, load_settings
from router.engine import NexusRouter, RoutingFailedError
from router.schemas import RouterRequest, RoutingStrategyName
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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    x_router_strategy: Annotated[str | None, Header(alias="X-Router-Strategy")] = None,
) -> ChatCompletionResponse:
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
    if payload.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="streaming responses are not supported by the API endpoint yet",
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
    return ChatCompletionResponse.from_router_response(router_response)
