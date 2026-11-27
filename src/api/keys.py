"""Virtual-key admission and the key admin API (``/v1/keys``, ``/v1/key/info``).

``KeyGate`` authenticates gateway credentials: in ``virtual_keys`` mode every
model endpoint admits only the master key (``NEXUS_MASTER_KEY``) and active
keys from :class:`safety.virtual_keys.VirtualKeyStore`, and turns each key's
model allowlist, budget window, and RPM limit into the ``KeyLimits`` the router
enforces. ``KeyAdmin`` implements the master-key-only admin endpoints. Errors
are ``GatewayError``s that each endpoint renders in its own envelope.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.errors import GatewayError
from router.config import RouterSettings
from router.schemas import KeyLimits, ModelCandidate
from safety.key_fingerprint import ConflictingCredentialsError, presented_credential
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimitExceededError
from safety.virtual_keys import (
    MAX_RPM_LIMIT,
    InvalidVirtualKeyError,
    KeyRateLimiter,
    KeyStatus,
    VirtualKey,
    VirtualKeyNotFoundError,
    VirtualKeyRevokedError,
    VirtualKeyStore,
    budget_window,
)

KEYS_PATH = "/v1/keys"
KEY_PATH = "/v1/keys/{key_id}"
KEY_REVOKE_PATH = "/v1/keys/{key_id}/revoke"
KEY_ROTATE_PATH = "/v1/keys/{key_id}/rotate"
KEY_INFO_PATH = "/v1/key/info"
MASTER_KEY_ID = "master"
MAX_PAGE_SIZE = 100
_AUTH_ERROR = "authentication_error"
_PERMISSION_ERROR = "permission_error"
_INVALID_KEY_CODES = {
    "unknown": "invalid_api_key",
    "revoked": "api_key_revoked",
    "expired": "api_key_expired",
    "blocked": "api_key_blocked",
}


def is_keys_path(path: str) -> bool:
    """Return whether ``path`` belongs to the key admin API or ``/v1/key/info``.

    Args:
        path: Request URL path.

    Returns:
        ``True`` for ``/v1/keys``, ``/v1/keys/...``, and ``/v1/key/info``.
    """
    return path in (KEYS_PATH, KEY_INFO_PATH) or path.startswith(KEYS_PATH + "/")


class _KeyFields(BaseModel):
    """Settable key fields shared by create and update; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    alias: str | None = Field(default=None, max_length=64)
    tenant: str | None = Field(default=None, max_length=128)
    models: list[str] | None = Field(default=None, max_length=256)
    max_budget_usd: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    budget_duration: str | None = Field(default=None, max_length=8)
    rpm_limit: int | None = Field(default=None, ge=1, le=MAX_RPM_LIMIT)
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None


class KeyCreateRequest(_KeyFields):
    """``POST /v1/keys`` body: every field is optional."""

    blocked: bool = False


class KeyUpdateRequest(_KeyFields):
    """``PATCH /v1/keys/{key_id}`` body; ``null`` clears a limit, omitted fields stay."""

    blocked: bool | None = None


class KeySpend(BaseModel):
    """A key's spend in its current budget window and over its lifetime."""

    window_start: int | None
    resets_at: int | None
    spend_usd: float
    remaining_usd: float | None
    total_spend_usd: float
    request_count: int


class KeyObject(BaseModel):
    """A virtual key as the API shows it: never the secret, only its prefix."""

    id: str
    object: Literal["virtual_key"] = "virtual_key"
    key_prefix: str
    status: KeyStatus
    alias: str | None
    tenant: str | None
    models: list[str]
    max_budget_usd: float | None
    budget_duration: str | None
    rpm_limit: int | None
    expires_at: int | None
    blocked: bool
    metadata: dict[str, Any]
    created_at: int
    updated_at: int
    revoked_at: int | None
    spend: KeySpend


class KeyWithSecret(KeyObject):
    """Create and rotate responses: the only place the secret ``key`` appears."""

    key: str


class KeyList(BaseModel):
    """``GET /v1/keys`` page."""

    object: Literal["list"] = "list"
    data: list[KeyObject]
    total: int
    limit: int
    offset: int
    has_more: bool


def missing_credential() -> GatewayError:
    """Return the 401 for a request without an API key."""
    return GatewayError(
        401,
        "Missing API key: send it as 'Authorization: Bearer <key>' or 'x-api-key: <key>'.",
        error_type=_AUTH_ERROR,
        code="invalid_api_key",
    )


def invalid_credential(error: InvalidVirtualKeyError) -> GatewayError:
    """Return the 401 for an unknown, revoked, expired, or blocked key (never echoing it)."""
    messages = {
        "unknown": "Invalid API key: this gateway did not issue it.",
        "revoked": "This API key has been revoked.",
        "expired": f"This API key has expired ({error}).",
        "blocked": "This API key is blocked; ask the gateway administrator to unblock it.",
    }
    return GatewayError(
        401,
        messages[error.reason],
        error_type=_AUTH_ERROR,
        code=_INVALID_KEY_CODES[error.reason],
    )


def permission_denied(message: str) -> GatewayError:
    """Return a 403 ``permission_denied`` (a valid credential that may not do this)."""
    return GatewayError(403, message, error_type=_PERMISSION_ERROR, code="permission_denied")


def model_not_allowed(model: str, allowed: frozenset[str]) -> GatewayError:
    """Return the 403 for a catalog model outside the calling key's allowlist."""
    return GatewayError(
        403,
        f"This API key may not use model '{model}'. Allowed models: {', '.join(sorted(allowed))}.",
        error_type=_PERMISSION_ERROR,
        code="model_not_allowed",
        param="model",
    )


def _invalid_value(message: str, param: str | None) -> GatewayError:
    return GatewayError(400, message, code="invalid_value", param=param)


def validation_error(errors: Sequence[Any]) -> GatewayError:
    """Convert FastAPI request-validation errors on the key endpoints into an OpenAI 400.

    Args:
        errors: ``RequestValidationError.errors()`` entries.

    Returns:
        Gateway error naming the first invalid body field or query parameter.
    """
    first = errors[0] if errors and isinstance(errors[0], Mapping) else {}
    error_type = first.get("type")
    if error_type == "json_invalid":
        return GatewayError(400, "We could not parse the JSON body of your request.")
    location = [
        str(part)
        for part in first.get("loc", ())
        if part not in ("body", "query", "path", "header")
    ]
    param = location[0] if location else None
    detail = str(first.get("msg", "invalid value"))
    if param is None:
        return _invalid_value(f"Invalid request: {detail}.", None)
    if error_type == "missing":
        return GatewayError(
            400,
            f"Missing required parameter: '{param}'.",
            code="missing_required_parameter",
            param=param,
        )
    if error_type == "extra_forbidden":
        return GatewayError(
            400,
            f"Unrecognized request argument supplied: {param}",
            code="unknown_parameter",
            param=param,
        )
    return _invalid_value(f"Invalid value for '{param}': {detail}.", param)


def _timestamp(value: float | None) -> int | None:
    return None if value is None else int(value)


def _iso(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def key_spend(key: VirtualKey, ledger: SpendLedger, now: float) -> KeySpend:
    """Return ``key``'s spend in its current budget window and lifetime, from the ledger.

    Args:
        key: Virtual key.
        ledger: Spend ledger the router records completions in.
        now: Unix timestamp.

    Returns:
        The spend view; ``remaining_usd`` is ``None`` without a budget.
    """
    window = budget_window(key.budget_duration, now)
    total, count = ledger.key_spend(key.key_id)
    spend = ledger.key_spend(key.key_id, since=window[0])[0] if window else total
    remaining = max(key.max_budget_usd - spend, 0.0) if key.has_budget else None
    return KeySpend(
        window_start=_timestamp(window[0]) if window else None,
        resets_at=_timestamp(window[1]) if window else None,
        spend_usd=spend,
        remaining_usd=remaining,
        total_spend_usd=total,
        request_count=count,
    )


def key_object(key: VirtualKey, spend: KeySpend, now: float) -> KeyObject:
    """Render a key without its secret.

    Args:
        key: Virtual key.
        spend: The key's spend view.
        now: Unix timestamp used for ``status``.

    Returns:
        The ``virtual_key`` object.
    """
    return KeyObject(
        id=key.key_id,
        key_prefix=key.key_prefix,
        status=key.status(now),
        alias=key.alias,
        tenant=key.tenant or None,
        models=sorted(key.allowed_models),
        max_budget_usd=key.max_budget_usd if key.has_budget else None,
        budget_duration=key.budget_duration,
        rpm_limit=key.rpm_limit,
        expires_at=_timestamp(key.expires_at),
        blocked=not key.enabled,
        metadata=dict(key.metadata),
        created_at=int(key.created_at),
        updated_at=int(key.updated_at),
        revoked_at=_timestamp(key.revoked_at),
        spend=spend,
    )


@dataclass(frozen=True, slots=True)
class KeyAccess:
    """An admitted credential: the master key (``key is None``) or a virtual key."""

    api_key_id: str
    key: VirtualKey | None = None
    limits: KeyLimits | None = None

    @property
    def is_master(self) -> bool:
        """Return whether the caller used the master key."""
        return self.key is None

    @property
    def allowed_models(self) -> frozenset[str] | None:
        """Return the key's model allowlist (``None`` permits every model)."""
        return None if self.limits is None else self.limits.allowed_models

    def tenant(self, requested: str | None) -> str | None:
        """Return the request's tenant: a tenant-bound key's tenant, else ``X-Tenant-Id``.

        Args:
            requested: ``X-Tenant-Id`` header value, if any.

        Returns:
            The tenant id to attribute the request to.

        Raises:
            GatewayError: 403 ``tenant_mismatch`` when the header names another tenant.
        """
        return _resolve_tenant(self.key, requested)


def _resolve_tenant(key: VirtualKey | None, requested: str | None) -> str | None:
    """Return a tenant-bound key's tenant, else ``requested``; 403 when they differ."""
    bound = key.tenant if key is not None else ""
    if not bound:
        return requested
    if requested and requested != bound:
        raise GatewayError(
            403,
            "X-Tenant-Id does not match the tenant this API key is bound to.",
            error_type=_PERMISSION_ERROR,
            code="tenant_mismatch",
            param="X-Tenant-Id",
        )
    return bound


class KeyGate:
    """Authenticate gateway credentials and derive each virtual key's routing limits."""

    def __init__(
        self,
        *,
        settings: RouterSettings,
        store: VirtualKeyStore,
        ledger: SpendLedger,
        catalog: Mapping[str, ModelCandidate],
        rate_limiter: KeyRateLimiter,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the gate.

        Args:
            settings: Router settings (auth mode and master key).
            store: Virtual key store.
            ledger: Spend ledger holding each key's spend.
            catalog: Routable model catalog.
            rate_limiter: Per-key RPM limiter.
            clock: Unix-time source.
        """
        self._settings = settings
        self._store = store
        self._ledger = ledger
        self._catalog = catalog
        self._rate_limiter = rate_limiter
        self._clock = clock

    @staticmethod
    def _credential(authorization: str | None, x_api_key: str | None) -> str:
        """Return the presented key or raise the 401 for a missing or ambiguous one."""
        try:
            credential = presented_credential(authorization, x_api_key)
        except ConflictingCredentialsError as error:
            raise GatewayError(
                401, str(error), error_type=_AUTH_ERROR, code="invalid_api_key"
            ) from error
        if credential is None:
            raise missing_credential()
        return credential

    def _is_master(self, credential: str) -> bool:
        master = self._settings.master_key
        if master is None:
            return False
        # Comparing fixed-length digests keeps the comparison constant-time and
        # independent of the master key's length.
        presented = hashlib.sha256(credential.encode("utf-8")).digest()
        expected = hashlib.sha256(master.get_secret_value().encode("utf-8")).digest()
        return hmac.compare_digest(presented, expected)

    def _authenticate(self, credential: str) -> VirtualKey:
        try:
            return self._store.authenticate(credential)
        except InvalidVirtualKeyError as error:
            raise invalid_credential(error) from error

    def admit(
        self,
        authorization: str | None,
        x_api_key: str | None = None,
        *,
        model: str | None = None,
        tenant_id: str | None = None,
        metered: bool = False,
    ) -> KeyAccess:
        """Admit a model-endpoint request in ``virtual_keys`` mode.

        Checks run in this order: credential (401), ``X-Tenant-Id`` against a
        tenant-bound key (403), the requested model against the key's allowlist
        (403), the allowlist against the catalog (403), then for ``metered``
        endpoints (the ones that call a provider) the budget window (402) and
        the RPM limit (429), so refused requests never use up the RPM limit. A
        ``model`` that is not a catalog
        model (such as ``auto``) stays a routing hint, as in ``open`` mode: the
        router then picks among the key's allowed models.

        Args:
            authorization: ``Authorization`` header value.
            x_api_key: ``x-api-key`` header value.
            model: Model the request names, if any.
            tenant_id: ``X-Tenant-Id`` header value, if any.
            metered: Whether the request may spend (budget and RPM apply).

        Returns:
            The admitted identity and its routing limits.

        Raises:
            GatewayError: When the request is not admitted.
        """
        credential = self._credential(authorization, x_api_key)
        if self._is_master(credential):
            return KeyAccess(api_key_id=MASTER_KEY_ID)
        key = self._authenticate(credential)
        _resolve_tenant(key, tenant_id)
        allowed = key.allowed_models or None
        if allowed is not None:
            if model and model in self._catalog and model not in allowed:
                raise model_not_allowed(model, allowed)
            if not any(candidate in allowed for candidate in self._catalog):
                raise GatewayError(
                    403,
                    "None of the models this API key may use is in the gateway's model catalog.",
                    error_type=_PERMISSION_ERROR,
                    code="no_eligible_model",
                )
        remaining = self._remaining_budget(key) if metered else None
        if metered and key.rpm_limit is not None:
            try:
                self._rate_limiter.assert_allowed(key.key_id, key.rpm_limit)
            except TenantRateLimitExceededError as error:
                raise GatewayError(
                    429,
                    f"Rate limit reached for API key {key.key_id}: {key.rpm_limit} "
                    f"request{'' if key.rpm_limit == 1 else 's'} per minute.",
                    error_type="rate_limit_error",
                    code="rate_limit_exceeded",
                ) from error
        limits = KeyLimits(allowed_models=allowed, remaining_budget_usd=remaining)
        return KeyAccess(api_key_id=key.key_id, key=key, limits=limits)

    def _remaining_budget(self, key: VirtualKey) -> float | None:
        """Return the key's budget left in its window; 402 when none is left."""
        if not key.has_budget:
            return None
        now = self._clock()
        window = budget_window(key.budget_duration, now)
        spent = self._ledger.key_spend(key.key_id, since=window[0] if window else None)[0]
        remaining = key.max_budget_usd - spent
        if remaining <= 0:
            period = f"in the budget window that resets at {_iso(window[1])}" if window else ""
            raise GatewayError(
                402,
                f"API key budget exhausted: ${spent:.6f} of ${key.max_budget_usd:.6f} spent "
                f"{period or 'over the key lifetime'}.",
                error_type="budget_exceeded",
                code="budget_exceeded",
            )
        return remaining

    def authorize_admin(self, authorization: str | None, x_api_key: str | None = None) -> None:
        """Allow only the master key; 401 for a missing or invalid key, 403 for a virtual key.

        Args:
            authorization: ``Authorization`` header value.
            x_api_key: ``x-api-key`` header value.

        Raises:
            GatewayError: When the caller is not the administrator.
        """
        if self._settings.master_key is None:
            raise permission_denied(
                "The key admin API is disabled: set NEXUS_MASTER_KEY to enable it."
            )
        credential = self._credential(authorization, x_api_key)
        if self._is_master(credential):
            return
        self._authenticate(credential)
        raise permission_denied(
            "This endpoint requires the master key; virtual keys cannot manage keys."
        )

    def describe(self, authorization: str | None, x_api_key: str | None) -> KeyObject:
        """Return the calling virtual key (``GET /v1/key/info``).

        Args:
            authorization: ``Authorization`` header value.
            x_api_key: ``x-api-key`` header value.

        Returns:
            The caller's key with its spend, limits, and reset time.

        Raises:
            GatewayError: 401 for a missing or invalid key, 400 for the master key.
        """
        credential = self._credential(authorization, x_api_key)
        if self._is_master(credential):
            raise GatewayError(
                400,
                "GET /v1/key/info describes the virtual key that calls it; the master key is "
                "not a virtual key (use GET /v1/keys/{key_id}).",
                code="not_a_virtual_key",
            )
        key = self._authenticate(credential)
        now = self._clock()
        return key_object(key, key_spend(key, self._ledger, now), now)


# The store's ``ValueError`` messages start with the field name; map it to the API parameter.
_PARAMS = {
    "alias": "alias",
    "tenant": "tenant",
    "allowed_models": "models",
    "each": "models",
    "max_budget_usd": "max_budget_usd",
    "budget_duration": "budget_duration",
    "rpm_limit": "rpm_limit",
    "expires_at": "expires_at",
    "metadata": "metadata",
    "enabled": "blocked",
}


class KeyAdmin:
    """Master-key-only operations behind ``/v1/keys`` (the caller is already authorized)."""

    def __init__(
        self,
        *,
        store: VirtualKeyStore,
        ledger: SpendLedger,
        catalog: Mapping[str, ModelCandidate],
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the admin service.

        Args:
            store: Virtual key store.
            ledger: Spend ledger holding each key's spend.
            catalog: Routable model catalog (allowlists may only name its models).
            clock: Unix-time source.
        """
        self._store = store
        self._ledger = ledger
        self._catalog = catalog
        self._clock = clock

    def _view(self, key: VirtualKey) -> KeyObject:
        now = self._clock()
        return key_object(key, key_spend(key, self._ledger, now), now)

    def _with_secret(self, raw_key: str, key: VirtualKey) -> KeyWithSecret:
        return KeyWithSecret(**self._view(key).model_dump(), key=raw_key)

    def _check_catalog(self, models: Sequence[str] | None) -> None:
        unknown = sorted(set(models or ()) - set(self._catalog))
        if unknown:
            raise _invalid_value(
                f"Unknown model(s) in 'models': {', '.join(unknown)}. "
                "GET /v1/models lists this gateway's catalog.",
                "models",
            )

    @staticmethod
    def _value_error(error: ValueError) -> GatewayError:
        field = str(error).split(" ", 1)[0]
        return _invalid_value(f"{error}.", _PARAMS.get(field))

    @staticmethod
    def _missing_key(error: VirtualKeyNotFoundError) -> GatewayError:
        return GatewayError(404, str(error), code="key_not_found", param="key_id")

    def create(self, payload: KeyCreateRequest) -> KeyWithSecret:
        """Issue a key; the response is the only time its secret is shown.

        Args:
            payload: Validated request body.

        Returns:
            The new key with its secret.

        Raises:
            GatewayError: 400 for invalid fields.
        """
        self._check_catalog(payload.models)
        expires_at = payload.expires_at
        if expires_at is not None:
            moment = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=UTC)
            if moment.timestamp() <= self._clock():
                raise _invalid_value("expires_at must be in the future.", "expires_at")
        try:
            raw_key, key = self._store.create(
                tenant=payload.tenant,
                max_budget_usd=payload.max_budget_usd,
                allowed_models=payload.models,
                alias=payload.alias,
                budget_duration=payload.budget_duration,
                rpm_limit=payload.rpm_limit,
                expires_at=expires_at,
                metadata=payload.metadata,
                enabled=not payload.blocked,
            )
        except ValueError as error:
            raise self._value_error(error) from error
        return self._with_secret(raw_key, key)

    def list_keys(
        self,
        *,
        alias: str | None,
        tenant: str | None,
        blocked: bool | None,
        expired: bool | None,
        revoked: bool | None,
        limit: int,
        offset: int,
    ) -> KeyList:
        """Return one page of keys (revoked keys included unless filtered out).

        Args:
            alias: Only keys with this alias.
            tenant: Only keys bound to this tenant.
            blocked: Only blocked or unblocked keys.
            expired: Only expired or unexpired keys.
            revoked: Only revoked or unrevoked keys.
            limit: Page size (1-100).
            offset: Keys to skip.

        Returns:
            The page, newest key first.
        """
        keys, total = self._store.list_keys(
            alias=alias,
            tenant=tenant,
            blocked=blocked,
            expired=expired,
            revoked=revoked,
            limit=limit,
            offset=offset,
        )
        return KeyList(
            data=[self._view(key) for key in keys],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(keys) < total,
        )

    def get(self, key_id: str) -> KeyObject:
        """Return one key (revoked keys included).

        Args:
            key_id: Key id.

        Returns:
            The key without its secret.

        Raises:
            GatewayError: 404 for an unknown id.
        """
        try:
            return self._view(self._store.get(key_id))
        except VirtualKeyNotFoundError as error:
            raise self._missing_key(error) from error

    def update(self, key_id: str, payload: KeyUpdateRequest) -> KeyObject:
        """Apply the fields present in ``payload``; ``null`` clears a limit.

        Args:
            key_id: Key id.
            payload: Validated request body.

        Returns:
            The updated key.

        Raises:
            GatewayError: 400 for invalid fields, 404 for an unknown id, 409 for a
                revoked key.
        """
        changes: dict[str, object] = {}
        for name in payload.model_fields_set:
            value = getattr(payload, name)
            if name == "blocked":
                if value is None:
                    raise _invalid_value("blocked must be true or false.", "blocked")
                changes["enabled"] = not value
            else:
                changes["allowed_models" if name == "models" else name] = value
        if "models" in payload.model_fields_set:
            self._check_catalog(payload.models)
        try:
            key = self._store.update(key_id, changes) if changes else self._store.get(key_id)
        except ValueError as error:
            raise self._value_error(error) from error
        except VirtualKeyNotFoundError as error:
            raise self._missing_key(error) from error
        except VirtualKeyRevokedError as error:
            raise GatewayError(409, str(error), code="key_revoked") from error
        return self._view(key)

    def revoke(self, key_id: str) -> KeyObject:
        """Revoke a key; it keeps appearing in listings as ``revoked``.

        Args:
            key_id: Key id.

        Returns:
            The revoked key.

        Raises:
            GatewayError: 404 for an unknown id.
        """
        try:
            return self._view(self._store.revoke(key_id))
        except VirtualKeyNotFoundError as error:
            raise self._missing_key(error) from error

    def rotate(self, key_id: str) -> KeyWithSecret:
        """Replace a key's secret, keeping its id, limits, and spend.

        Args:
            key_id: Key id.

        Returns:
            The key with its new secret.

        Raises:
            GatewayError: 404 for an unknown id, 409 for a revoked key.
        """
        try:
            raw_key, key = self._store.rotate(key_id)
        except VirtualKeyNotFoundError as error:
            raise self._missing_key(error) from error
        except VirtualKeyRevokedError as error:
            raise GatewayError(409, str(error), code="key_revoked") from error
        return self._with_secret(raw_key, key)
