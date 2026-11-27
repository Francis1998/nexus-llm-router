"""FastAPI application for Nexus LLM Router."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from time import time
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from adapters.registry import AdapterRegistry, build_adapter_registry
from api import anthropic_messages
from api import feedback as feedback_api
from api import keys as keys_api
from api.anthropic_messages import COUNT_TOKENS_PATH, MESSAGES_PATH
from api.errors import (
    ROUTING_ERRORS,
    GatewayError,
    anthropic_error_body,
    conversation_error,
    routing_error,
)
from api.feedback import FEEDBACK_PATH, FEEDBACK_SUMMARY_PATH
from api.keys import (
    KEY_INFO_PATH,
    KEY_PATH,
    KEY_REVOKE_PATH,
    KEY_ROTATE_PATH,
    KEYS_PATH,
    KeyAdmin,
    KeyCreateRequest,
    KeyGate,
    KeyUpdateRequest,
)
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
    AnthropicTokenCount,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FeedbackCreateRequest,
    MessageCountTokensRequest,
    MessageCreateRequest,
    ModelHealth,
    ModelList,
    ModelObject,
    ResponseCreateRequest,
    ResponseObject,
    SpendRecordRequest,
    SpendSummaryResponse,
)
from cache.response_cache import ResponseCache
from observability.logging import configure_logging, get_logger
from observability.metrics import (
    BoundedLabelValues,
    metrics_response,
    response_cache_hits_total,
    response_cache_misses_total,
    router_feedback_total,
)
from observability.tracing import configure_tracing
from router.analyzer import estimate_prompt_tokens
from router.config import RouterSettings, load_settings
from router.engine import NexusRouter, RoutingFailedError
from router.feedback import METRIC_PATTERN, FeedbackKindConflictError, FeedbackScope, FeedbackStore
from router.schemas import (
    ChatMessage,
    KeyLimits,
    RouterRequest,
    RouterResponse,
    RoutingStrategyName,
)
from safety.budget import BudgetExceededError
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import (
    IdempotencyKeyInUseError,
    IdempotencyRecord,
    IdempotencyReservation,
    IdempotencyStore,
)
from safety.key_fingerprint import (
    ConflictingCredentialsError,
    fingerprint_api_key,
    fingerprint_request_credentials,
)
from safety.rate_limiter import RateLimitExceededError
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter, TenantRateLimitExceededError
from safety.virtual_keys import KeyRateLimiter, VirtualKeyStore


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate settings at startup so a misconfigured gateway fails before serving."""
    settings = get_settings()
    if settings.auth_mode == "virtual_keys":
        get_virtual_key_store()
        if settings.key_hash_pepper is None:
            _logger.warning(
                "virtual_keys_unpeppered",
                hint="set NEXUS_KEY_HASH_PEPPER so a copy of the key store cannot verify keys",
            )
    yield


configure_logging()
app = FastAPI(title="Nexus LLM Router", version="0.1.0", lifespan=_lifespan)
configure_tracing(app, "nexus-llm-router")
_logger = get_logger(__name__)
# Feedback metric names are client-chosen; cap the Prometheus label values.
_FEEDBACK_METRIC_LABELS = BoundedLabelValues(limit=32)


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


def get_feedback_store() -> FeedbackStore:
    """Return the request-feedback store the router indexes completed requests in.

    Returns:
        The router's feedback store.
    """
    return get_router().feedback_store


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
def get_virtual_key_store() -> VirtualKeyStore:
    """Return the durable virtual-key store.

    Returns:
        Virtual key store at ``NEXUS_VIRTUAL_KEYS_PATH``.
    """
    settings = get_settings()
    pepper = settings.key_hash_pepper
    return VirtualKeyStore(
        settings.virtual_keys_path,
        pepper=None if pepper is None else pepper.get_secret_value(),
    )


@lru_cache(maxsize=1)
def get_key_rate_limiter() -> KeyRateLimiter:
    """Return the per-virtual-key RPM limiter.

    Returns:
        Key rate limiter instance.
    """
    return KeyRateLimiter()


def get_key_gate() -> KeyGate:
    """Return the gate that authenticates credentials against the virtual-key store.

    The gate reads the router's own spend ledger, which also holds the virtual-key
    spend whose write failed until it is settled.

    Returns:
        Key gate over the current settings, store, ledger, and catalog.
    """
    router = get_router()
    return KeyGate(
        settings=get_settings(),
        store=get_virtual_key_store(),
        ledger=router.spend_ledger,
        catalog=router.model_catalog,
        rate_limiter=get_key_rate_limiter(),
    )


def get_key_admin() -> KeyAdmin:
    """Return the key admin service behind ``/v1/keys``.

    Returns:
        Key admin service.
    """
    return KeyAdmin(
        store=get_virtual_key_store(),
        ledger=get_spend_ledger(),
        catalog=get_router().model_catalog,
    )


def _virtual_keys_enforced() -> bool:
    """Return whether model endpoints admit only the master key and issued keys."""
    return get_settings().auth_mode == "virtual_keys"


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
    key_limits: KeyLimits | None = None

    @property
    def principal(self) -> str:
        """Budget and spend subject (end user or key fingerprint)."""
        return self.end_user or self.api_key_id

    @property
    def cache_namespace(self) -> str:
        """Response-cache namespace: API key fingerprint, ``X-Tenant-Id``, and principal.

        A cache hit replays the original ``request_id`` and routing decision, so
        it must stay with the caller allowed to rate that request (and, for
        ``feedback-aware``, the caller whose feedback chose the model). A virtual
        key's model allowlist is part of it, so narrowing the allowlist never
        replays an answer from a model the key may no longer use. JSON keeps the
        fields apart whatever the client-chosen tenant and end-user ids contain.
        """
        limits = self.key_limits
        allowed = None if limits is None or limits.allowed_models is None else limits.allowed_models
        return json.dumps(
            [
                self.api_key_id,
                self.tenant_id or "",
                self.principal,
                None if allowed is None else sorted(allowed),
            ]
        )

    @property
    def idempotency_tenant(self) -> str:
        """Scope for ``Idempotency-Key`` replays: key fingerprint, ``X-Tenant-Id``, end user.

        A replay returns the stored response verbatim, so it must only reach the
        caller that made the original request. ``X-Tenant-Id`` and end-user ids are
        client-chosen and can repeat across API keys, so they narrow the key's scope
        but never replace it. JSON keeps every caller's scope distinct, whatever the
        tenant and end-user ids contain.
        """
        return json.dumps([self.api_key_id, self.tenant_id or "", self.end_user or ""])


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


def _admit_caller(
    request: Request,
    tenant_id: str | None,
    end_user: str | None,
    *,
    x_api_key: bool = False,
    model: str | None = None,
    metered: bool = False,
) -> _Caller:
    """Identify the API key and enforce the optional hard tenant rate limit.

    In ``open`` mode the key is only fingerprinted. In ``virtual_keys`` mode it
    must be the master key or an active virtual key (sent as ``Authorization``
    or ``x-api-key``), whose id identifies the caller and whose limits travel
    with the request (see ``KeyGate.admit``).

    Args:
        request: FastAPI request object.
        tenant_id: ``X-Tenant-Id`` header value.
        end_user: End-user id from the body, if any.
        x_api_key: Whether ``open`` mode also fingerprints ``x-api-key``
            (``/v1/messages*``, ``/v1/feedback*``).
        model: Model the request names (checked against the key's allowlist).
        metered: Whether the request may spend (budget and RPM limits apply).

    Returns:
        The admitted caller.
    """
    key_limits: KeyLimits | None = None
    if _virtual_keys_enforced():
        access = get_key_gate().admit(
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
            model=model,
            tenant_id=tenant_id,
            metered=metered,
        )
        api_key_id, key_limits = access.api_key_id, access.limits
        tenant_id = access.tenant(tenant_id)
    elif x_api_key:
        api_key_id = _anthropic_api_key_id(request)
    else:
        # Never use the raw Authorization header as an identifier: it would leak the
        # secret into error details, the spend ledger, and idempotency/cache tenants.
        api_key_id = fingerprint_api_key(request.headers.get("authorization"))
    caller = _Caller(
        api_key_id=api_key_id, end_user=end_user, tenant_id=tenant_id, key_limits=key_limits
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


def _key_access(request: Request, *, model: str | None = None) -> keys_api.KeyAccess | None:
    """Authenticate a request that never spends, in ``virtual_keys`` mode only.

    A tenant-bound key still refuses an ``X-Tenant-Id`` naming another tenant;
    no budget, RPM, or tenant rate limit applies. Returns ``None`` in ``open`` mode.
    """
    if not _virtual_keys_enforced():
        return None
    return get_key_gate().admit(
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
        model=model,
        tenant_id=request.headers.get("x-tenant-id"),
    )


def _anthropic_api_key_id(request: Request) -> str:
    """Fingerprint ``x-api-key`` and/or ``Authorization`` for ``/v1/messages*``.

    Both headers carrying the same key is fine; different keys are a 401 so a
    request is never billed to an ambiguous identity.
    """
    try:
        return fingerprint_request_credentials(
            request.headers.get("authorization"), request.headers.get("x-api-key")
        )
    except ConflictingCredentialsError as exception:
        raise GatewayError(
            status.HTTP_401_UNAUTHORIZED, str(exception), error_type="authentication_error"
        ) from exception


_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def _idempotency_scope(idempotency_key: str, path: str) -> str:
    """Return the stored key for an ``Idempotency-Key`` sent to one endpoint.

    The path keeps endpoints apart, so a Responses key never replays a chat body.
    The leading line break separates these keys from every key earlier builds
    stored (the raw header on chat, ``"<key>\\n<path>"`` elsewhere): HTTP header
    values cannot contain line breaks, so neither those keys nor any client-sent
    key can start with one. Records that earlier builds stored under client-chosen
    tenants (``X-Tenant-Id`` or end-user ids, which could spell another caller's
    scope) therefore never replay, even when old and new instances share a store.
    """
    return f"\n{path}\n{idempotency_key}"


def _claim_idempotency(
    guards: GatewayGuardService, scoped_key: str, caller: _Caller
) -> IdempotencyRecord | IdempotencyReservation:
    """Return the stored response to replay, or reserve the key for this request."""
    try:
        return guards.reserve_idempotency(scoped_key, caller.idempotency_tenant)
    except IdempotencyKeyInUseError as exception:
        raise GatewayError(409, str(exception), code="idempotency_key_in_use") from exception


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


@app.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList | JSONResponse:
    """List routable catalog models in the OpenAI format, with nexus metadata.

    In ``virtual_keys`` mode the caller needs a valid key, and a key with a
    model allowlist sees only the models it may use.

    Args:
        request: FastAPI request object.

    Returns:
        OpenAI ``list`` of ``model`` objects in catalog order, or an OpenAI-style error.
    """
    try:
        access = _key_access(request)
    except GatewayError as error:
        return _openai_error(error)
    allowed = None if access is None else access.allowed_models
    router = get_router()
    health = await _provider_model_health(router)
    return ModelList(
        data=[
            model_object(candidate, health[candidate.provider])
            for candidate in router.model_catalog.values()
            if allowed is None or candidate.model in allowed
        ]
    )


@app.get("/v1/models/{model_id}", response_model=None)
async def retrieve_model(model_id: str, request: Request) -> ModelObject | JSONResponse:
    """Return one catalog model, or an OpenAI-style 404 for unknown ids.

    In ``virtual_keys`` mode a model outside the calling key's allowlist is a
    404 as well.

    Args:
        model_id: Model identifier such as ``gpt-6.2-astra``.
        request: FastAPI request object.

    Returns:
        The ``model`` object, or a ``model_not_found`` error envelope.
    """
    try:
        access = _key_access(request)
    except GatewayError as denied:
        return _openai_error(denied)
    allowed = None if access is None else access.allowed_models
    router = get_router()
    candidate = router.model_catalog.get(model_id)
    if candidate is None or (allowed is not None and model_id not in allowed):
        message = f"The model '{model_id}' does not exist"
        if candidate is not None:
            message += " or you do not have access to it"
        return _openai_error(GatewayError(404, message, param="model", code="model_not_found"))
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
        caller = _admit_caller(
            request, x_tenant_id, payload.user, model=payload.model, metered=True
        )
    except GatewayError as exception:
        raise HTTPException(exception.status_code, detail=exception.message) from exception
    guards = get_gateway_guards()
    reservation: IdempotencyReservation | None = None
    if idempotency_key:
        scoped_key = _idempotency_scope(idempotency_key, _CHAT_COMPLETIONS_PATH)
        existing: IdempotencyRecord | None
        if payload.stream:
            # Streams are never stored, so they only replay a completed response.
            existing = guards.check_idempotency(scoped_key, caller.idempotency_tenant)
        else:
            try:
                claim = guards.reserve_idempotency(scoped_key, caller.idempotency_tenant)
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
            key_limits=caller.key_limits,
            tenant_id=caller.tenant_id,
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
                router_request, temperature=payload.temperature, tenant=caller.cache_namespace
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
    """Return API-style errors for the Responses, Messages, feedback, and key APIs.

    ``/v1/responses`` and ``/v1/keys*`` get an OpenAI-envelope 400,
    ``/v1/messages*`` the Anthropic one, and ``/v1/feedback*`` an
    OpenAI-envelope 422; every other route keeps FastAPI's 422s.

    Args:
        request: FastAPI request object.
        exception: Request validation failure.

    Returns:
        JSON error response.
    """
    path = request.url.path
    if anthropic_messages.is_messages_path(path):
        error = anthropic_messages.validation_error(exception.errors())
        return JSONResponse(status_code=error.status_code, content=error.to_anthropic_body())
    if feedback_api.is_feedback_path(path):
        error = feedback_api.validation_error(exception.errors())
        return JSONResponse(status_code=error.status_code, content=error.to_openai_body())
    if keys_api.is_keys_path(path):
        if path != KEY_INFO_PATH:
            # A body that is not JSON fails before route dependencies run, so
            # authenticate here too: unauthenticated callers only ever see a 401 / 403.
            try:
                _require_master_key(request)
            except GatewayError as denied:
                return _openai_error(denied)
        return _openai_error(keys_api.validation_error(exception.errors()))
    if path != RESPONSES_PATH:
        return await request_validation_exception_handler(request, exception)
    error = validation_error(exception.errors())
    return JSONResponse(status_code=error.status_code, content=error.to_openai_body())


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exception: GatewayError) -> JSONResponse:
    """Render a ``GatewayError`` raised by a route dependency (the key checks).

    Args:
        request: FastAPI request object.
        exception: The error.

    Returns:
        An Anthropic envelope below ``/v1/messages``, an OpenAI envelope elsewhere.
    """
    if anthropic_messages.is_messages_path(request.url.path):
        return JSONResponse(
            status_code=exception.status_code, content=exception.to_anthropic_body()
        )
    return _openai_error(exception)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exception: StarletteHTTPException) -> Response:
    """Use the Anthropic error envelope below ``/v1/messages``; keep FastAPI's elsewhere.

    Args:
        request: FastAPI request object.
        exception: HTTP error such as an unknown path (404) or method (405).

    Returns:
        Error response.
    """
    if not anthropic_messages.is_messages_path(request.url.path):
        return await http_exception_handler(request, exception)
    return JSONResponse(
        status_code=exception.status_code,
        content=anthropic_error_body(exception.status_code, str(exception.detail)),
        headers=exception.headers,
    )


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
    caller = _admit_caller(request, x_tenant_id, end_user, model=payload.model, metered=True)
    guards = get_gateway_guards()
    reservation: IdempotencyReservation | None = None
    if idempotency_key and not payload.stream:
        claim = _claim_idempotency(
            guards, _idempotency_scope(idempotency_key, RESPONSES_PATH), caller
        )
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
        key_limits=caller.key_limits,
        tenant_id=caller.tenant_id,
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
        tenant=caller.cache_namespace,
    )
    body = complete_response(draft, router_response, f"msg_{uuid4().hex}").model_dump(mode="json")
    if reservation is not None:
        guards.complete_idempotency(reservation, json.dumps(body))
    return JSONResponse(content=body)


@dataclass(frozen=True, slots=True)
class _RoutedStream:
    """A routed stream whose output was pulled up to its first non-empty chunk."""

    first_chunk: tuple[str, str] | None
    chunks: AsyncGenerator[tuple[str, str], None]
    completed: list[RouterResponse]


async def _start_stream(router_request: RouterRequest) -> _RoutedStream:
    """Start a routed stream and pull it until its first non-empty chunk.

    Routing failures before any output (rate limits, every provider failed)
    therefore surface as normal HTTP errors instead of a 200 event stream.
    ``completed`` receives the final ``RouterResponse`` (usage, cost, stop
    reason) once the stream finished; spend is recorded by the router exactly
    as for ``/v1/chat/completions`` streams.
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
    return _RoutedStream(first_chunk=first_chunk, chunks=chunks, completed=completed)


async def _stream_response(
    router_request: RouterRequest, draft: ResponseObject
) -> StreamingResponse:
    """Start a routed stream and render it as Responses API SSE events."""
    stream = await _start_stream(router_request)
    return StreamingResponse(
        stream_response_events(draft, stream.first_chunk, stream.chunks, stream.completed),
        media_type="text/event-stream",
    )


@app.post(MESSAGES_PATH, response_model=None)
async def create_message(
    payload: MessageCreateRequest,
    request: Request,
    x_router_strategy: Annotated[str | None, Header(alias="X-Router-Strategy")] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> JSONResponse | StreamingResponse:
    """Route an Anthropic Messages API request (documented text-only subset).

    Accepts ``x-api-key`` or ``Authorization`` credentials; ``anthropic-version``
    and ``anthropic-beta`` headers are accepted and ignored.

    Args:
        payload: Messages API request body.
        request: FastAPI request object.
        x_router_strategy: Optional routing strategy header.
        idempotency_key: Optional idempotency key (non-streaming requests only).
        x_tenant_id: Optional tenant id for hard per-tenant rate limiting.

    Returns:
        A ``message`` object, an SSE event stream, or an Anthropic-style error.
    """
    try:
        return await _create_message(
            payload, request, x_router_strategy, idempotency_key, x_tenant_id
        )
    except GatewayError as exception:
        return JSONResponse(
            status_code=exception.status_code, content=exception.to_anthropic_body()
        )


async def _create_message(
    payload: MessageCreateRequest,
    request: Request,
    x_router_strategy: str | None,
    idempotency_key: str | None,
    x_tenant_id: str | None,
) -> JSONResponse | StreamingResponse:
    """Validate, route, and render one ``POST /v1/messages`` request."""
    anthropic_messages.reject_unsupported(payload)
    messages = anthropic_messages.to_router_messages(payload)
    strategy = _parse_strategy(x_router_strategy)
    end_user = payload.metadata.user_id if payload.metadata else None
    caller = _admit_caller(
        request, x_tenant_id, end_user, x_api_key=True, model=payload.model, metered=True
    )
    guards = get_gateway_guards()
    reservation: IdempotencyReservation | None = None
    if idempotency_key and not payload.stream:
        claim = _claim_idempotency(
            guards, _idempotency_scope(idempotency_key, MESSAGES_PATH), caller
        )
        if isinstance(claim, IdempotencyRecord):
            return JSONResponse(
                content=json.loads(claim.response_json),
                status_code=claim.status_code,
            )
        reservation = claim
    try:
        return await _route_message(payload, messages, strategy, caller, guards, reservation)
    except BaseException:
        if reservation is not None:
            guards.release_idempotency(reservation)
        raise


async def _route_message(
    payload: MessageCreateRequest,
    messages: list[ChatMessage],
    strategy: RoutingStrategyName | None,
    caller: _Caller,
    guards: GatewayGuardService,
    reservation: IdempotencyReservation | None,
) -> JSONResponse | StreamingResponse:
    """Route an admitted ``/v1/messages`` request and store its idempotent result."""
    router_request = RouterRequest(
        request_id=uuid4().hex,
        messages=messages,
        api_key_id=caller.api_key_id,
        key_limits=caller.key_limits,
        tenant_id=caller.tenant_id,
        user_id=caller.principal,
        session_id=caller.end_user or "default",
        requested_model=payload.model,
        strategy=strategy,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
    )
    message_id = anthropic_messages.new_message_id()
    stop_sequences = payload.stop_sequences or []
    if payload.stream:
        stream = await _start_stream(router_request)
        return StreamingResponse(
            anthropic_messages.stream_message_events(
                message_id=message_id,
                model=payload.model,
                input_tokens=estimate_prompt_tokens(router_request.prompt_text),
                first_chunk=stream.first_chunk,
                chunks=stream.chunks,
                completed=stream.completed,
                stop_sequences=stop_sequences,
            ),
            media_type="text/event-stream",
        )

    # Sampling parameters are never forwarded upstream; temperature only keys the cache.
    router_response = await _complete_cached(
        router_request,
        temperature=1.0 if payload.temperature is None else payload.temperature,
        tenant=caller.cache_namespace,
    )
    message = anthropic_messages.complete_message(message_id, router_response, stop_sequences)
    body = message.model_dump(mode="json")
    if reservation is not None:
        guards.complete_idempotency(reservation, json.dumps(body))
    return JSONResponse(content=body)


@app.post(COUNT_TOKENS_PATH, response_model=None)
async def count_message_tokens(
    payload: MessageCountTokensRequest, request: Request
) -> JSONResponse:
    """Return nexus's input-token estimate for an Anthropic Messages payload.

    The count is the router's own prompt estimate (about four characters per
    token), the number routing strategies and the budget guardrail use. No
    provider is called and no rate limit, budget, or spend is touched.

    Args:
        payload: ``count_tokens`` request body (``model``, ``messages``, ``system``).
        request: FastAPI request object.

    Returns:
        ``{"input_tokens": N}`` or an Anthropic-style error.
    """
    try:
        anthropic_messages.reject_unsupported(payload)
        messages = anthropic_messages.to_router_messages(payload)
        if _key_access(request, model=payload.model) is None:
            _anthropic_api_key_id(request)
    except GatewayError as exception:
        return JSONResponse(
            status_code=exception.status_code, content=exception.to_anthropic_body()
        )
    prompt_text = RouterRequest(request_id=uuid4().hex, messages=messages).prompt_text
    count = AnthropicTokenCount(input_tokens=estimate_prompt_tokens(prompt_text))
    return JSONResponse(content=count.model_dump())


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
    request: Request,
    tenant: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> SpendSummaryResponse | JSONResponse:
    """Return aggregated spend from the durable ledger.

    In ``virtual_keys`` mode the caller needs a valid key: the master key sees
    every request, a virtual key only the requests made with it.

    Args:
        request: FastAPI request object.
        tenant: Optional tenant filter.
        provider: Optional provider filter.
        model: Optional model filter.

    Returns:
        Spend summary payload, or an OpenAI-style error.
    """
    try:
        access = _key_access(request)
    except GatewayError as error:
        return _openai_error(error)
    api_key_id = None if access is None or access.is_master else access.api_key_id
    summary = get_spend_ledger().summary(
        tenant=tenant, provider=provider, model=model, api_key_id=api_key_id
    )
    return SpendSummaryResponse(
        total_cost_usd=summary.total_cost_usd,
        request_count=summary.request_count,
        by_tenant=summary.by_tenant,
        by_provider=summary.by_provider,
        by_model=summary.by_model,
    )


def _require_master_key(request: Request) -> None:
    """Admit only the master key; raise the 401 / 403 ``GatewayError`` otherwise."""
    get_key_gate().authorize_admin(
        request.headers.get("authorization"), request.headers.get("x-api-key")
    )


def _require_master_key_when_enforced(request: Request) -> None:
    """Admit only the master key in ``virtual_keys`` mode; admit everyone in ``open`` mode."""
    if _virtual_keys_enforced():
        _require_master_key(request)


@app.post(
    "/v1/spend",
    response_model=SpendSummaryResponse,
    dependencies=[Depends(_require_master_key_when_enforced)],
)
def post_spend(payload: SpendRecordRequest) -> SpendSummaryResponse:
    """Record a spend event (portfolio / demo ingestion) and return summary.

    In ``virtual_keys`` mode only the master key may write ledger rows.

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


def _openai_error(error: GatewayError) -> JSONResponse:
    """Render a gateway error as an OpenAI-style JSON response."""
    return JSONResponse(status_code=error.status_code, content=error.to_openai_body())


def _feedback_scope(request: Request, x_tenant_id: str | None) -> FeedbackScope:
    """Admit a feedback caller and return the scope its requests belong to.

    The key may arrive as ``Authorization`` or ``x-api-key``, so feedback works
    for requests made through ``/v1/chat/completions``, ``/v1/responses``, and
    ``/v1/messages`` alike.
    """
    caller = _admit_caller(request, x_tenant_id, None, x_api_key=True)
    return FeedbackScope(api_key_id=caller.api_key_id, tenant_id=caller.tenant_id)


@app.post(FEEDBACK_PATH, response_model=None)
def create_feedback(
    payload: FeedbackCreateRequest,
    request: Request,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> JSONResponse:
    """Attach feedback to a completed request, replacing the caller's earlier value.

    One value is kept per ``(request_id, metric, source)``. Only the API key
    and ``X-Tenant-Id`` that made the request may rate it; anything else gets
    the same 404 as an unknown id. The comment is stored and returned, never
    logged.

    Args:
        payload: Feedback body.
        request: FastAPI request object.
        x_tenant_id: Optional tenant id; must match the original request's.

    Returns:
        The stored ``feedback`` object, or an OpenAI-style error.
    """
    try:
        scope = _feedback_scope(request, x_tenant_id)
        write = get_feedback_store().record(
            scope,
            request_id=payload.request_id,
            metric=payload.metric,
            value=payload.value,
            source=payload.source,
            comment=payload.comment,
        )
    except FeedbackKindConflictError as conflict:
        return _openai_error(feedback_api.kind_conflict(conflict))
    except GatewayError as error:
        return _openai_error(error)
    if write is None:
        _logger.info("feedback_rejected", request_id=payload.request_id, reason="request_not_found")
        return _openai_error(feedback_api.request_not_found(payload.request_id))
    record = write.record
    metric_label = (
        record.metric
        if record.metric == get_settings().feedback_metric
        else _FEEDBACK_METRIC_LABELS.label(record.metric)
    )
    router_feedback_total.labels(metric_label, record.model).inc()
    _logger.info(
        "feedback_recorded",
        request_id=record.request_id,
        feedback_id=record.feedback_id,
        metric=record.metric,
        source=record.source,
        value=record.value,
        model=record.model,
        provider=record.provider,
        strategy=record.strategy,
        domain=record.domain,
        created=write.created,
    )
    return JSONResponse(content=feedback_api.feedback_object(record).model_dump(mode="json"))


@app.get(FEEDBACK_SUMMARY_PATH, response_model=None)
def feedback_summary(
    request: Request,
    metric: Annotated[str, Query(pattern=METRIC_PATTERN)] = "quality",
    group_by: Literal["model", "strategy", "domain", "provider"] = "model",
    since: datetime | None = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> JSONResponse:
    """Summarize the caller's feedback for one metric, per model, strategy, domain, or provider.

    Boolean metrics report positives and a 95% Wilson score interval; numeric
    metrics report the mean with a 95% normal-approximation interval. Only the
    caller's own feedback (API key fingerprint plus ``X-Tenant-Id``) is counted.

    Args:
        request: FastAPI request object.
        metric: Metric name.
        group_by: Snapshot field to group by.
        since: Optional ISO 8601 timestamp (naive values are UTC); only feedback
            written at or after it is counted.
        x_tenant_id: Optional tenant id.

    Returns:
        The ``feedback.summary`` object, or an OpenAI-style error.
    """
    try:
        scope = _feedback_scope(request, x_tenant_id)
    except GatewayError as error:
        return _openai_error(error)
    try:
        since_utc = feedback_api.as_utc(since)
    except OverflowError:
        return _openai_error(feedback_api.since_out_of_range())
    summary = get_feedback_store().summary(
        scope,
        metric=metric,
        group_by=group_by,
        since=since_utc.timestamp() if since_utc is not None else None,
    )
    body = feedback_api.summary_response(summary, since_utc).model_dump(mode="json")
    return JSONResponse(content=body)


@app.post(KEYS_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def create_key(payload: KeyCreateRequest) -> JSONResponse:
    """Issue a virtual key (master key only); the response is the only time its secret is shown.

    Args:
        payload: Key fields: alias, tenant, models, budget, window, RPM limit,
            expiry, metadata, and whether it starts blocked.

    Returns:
        The new ``virtual_key`` object with its secret ``key``, or an OpenAI-style error.
    """
    try:
        created = get_key_admin().create(payload)
    except GatewayError as error:
        return _openai_error(error)
    _logger.info("virtual_key_created", key_id=created.id, key_prefix=created.key_prefix)
    return JSONResponse(content=created.model_dump(mode="json"))


@app.get(KEYS_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def list_keys(
    alias: str | None = None,
    tenant: str | None = None,
    blocked: bool | None = None,
    expired: bool | None = None,
    revoked: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=keys_api.MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JSONResponse:
    """List virtual keys, newest first, without secrets (master key only).

    Revoked keys stay listed (``status: revoked``) unless ``revoked=false``.

    Args:
        alias: Only keys with this alias.
        tenant: Only keys bound to this tenant.
        blocked: Only blocked (``true``) or unblocked (``false``) keys.
        expired: Only keys past (``true``) or before (``false``) their expiry.
        revoked: Only revoked (``true``) or unrevoked (``false``) keys.
        limit: Page size (1-100).
        offset: Number of matching keys to skip.

    Returns:
        An OpenAI-style ``list`` page with ``total`` and ``has_more``.
    """
    page = get_key_admin().list_keys(
        alias=alias,
        tenant=tenant,
        blocked=blocked,
        expired=expired,
        revoked=revoked,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(content=page.model_dump(mode="json"))


@app.get(KEY_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def retrieve_key(key_id: str) -> JSONResponse:
    """Return one virtual key with its current spend (master key only).

    Args:
        key_id: Key id such as ``vk_...``.

    Returns:
        The ``virtual_key`` object, or an OpenAI-style 404.
    """
    try:
        key = get_key_admin().get(key_id)
    except GatewayError as error:
        return _openai_error(error)
    return JSONResponse(content=key.model_dump(mode="json"))


@app.patch(KEY_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def update_key(key_id: str, payload: KeyUpdateRequest) -> JSONResponse:
    """Change a key's limits (master key only); ``null`` clears a limit, omitted fields stay.

    Args:
        key_id: Key id.
        payload: Fields to change.

    Returns:
        The updated ``virtual_key`` object, or an OpenAI-style error.
    """
    try:
        key = get_key_admin().update(key_id, payload)
    except GatewayError as error:
        return _openai_error(error)
    _logger.info("virtual_key_updated", key_id=key.id, fields=sorted(payload.model_fields_set))
    return JSONResponse(content=key.model_dump(mode="json"))


def _revoke_key(key_id: str) -> JSONResponse:
    try:
        key = get_key_admin().revoke(key_id)
    except GatewayError as error:
        return _openai_error(error)
    _logger.info("virtual_key_revoked", key_id=key.id)
    return JSONResponse(content=key.model_dump(mode="json"))


@app.delete(KEY_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def delete_key(key_id: str) -> JSONResponse:
    """Revoke a key for good (master key only); it stays listed as ``revoked`` for audit.

    Args:
        key_id: Key id.

    Returns:
        The revoked ``virtual_key`` object, or an OpenAI-style 404.
    """
    return _revoke_key(key_id)


@app.post(KEY_REVOKE_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def revoke_key(key_id: str) -> JSONResponse:
    """Revoke a key (same as ``DELETE /v1/keys/{key_id}``; master key only).

    Args:
        key_id: Key id.

    Returns:
        The revoked ``virtual_key`` object, or an OpenAI-style 404.
    """
    return _revoke_key(key_id)


@app.post(KEY_ROTATE_PATH, response_model=None, dependencies=[Depends(_require_master_key)])
def rotate_key(key_id: str) -> JSONResponse:
    """Replace a key's secret (master key only); the old secret stops working at once.

    The key id, limits, and spend are kept.

    Args:
        key_id: Key id.

    Returns:
        The ``virtual_key`` object with its new secret ``key``, or an OpenAI-style error.
    """
    try:
        rotated = get_key_admin().rotate(key_id)
    except GatewayError as error:
        return _openai_error(error)
    _logger.info("virtual_key_rotated", key_id=rotated.id, key_prefix=rotated.key_prefix)
    return JSONResponse(content=rotated.model_dump(mode="json"))


@app.get(KEY_INFO_PATH, response_model=None)
def key_info(request: Request) -> JSONResponse:
    """Describe the calling virtual key: limits, models, window spend, and reset time.

    Args:
        request: FastAPI request object (the key arrives as ``Authorization`` or
            ``x-api-key``).

    Returns:
        The caller's ``virtual_key`` object (never its secret), or an OpenAI-style error.
    """
    try:
        key = get_key_gate().describe(
            request.headers.get("authorization"), request.headers.get("x-api-key")
        )
    except GatewayError as error:
        return _openai_error(error)
    return JSONResponse(content=key.model_dump(mode="json"))
