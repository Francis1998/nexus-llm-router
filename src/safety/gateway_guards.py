"""Composable gateway guards for chat-completion ingress."""

from __future__ import annotations

from safety.idempotency import IdempotencyRecord, IdempotencyReservation, IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter


class GatewayGuardService:
    """Compose idempotency replay and hard tenant rate limiting.

    Closes the LiteLLM / Portkey gateway-guard gap for GPT-6 /
    Claude 5.5 / Gemini 4 / Kimi K3.1 chat completions: optional
    ``Idempotency-Key`` replay plus optional ``X-Tenant-Id`` hard rate
    limits before provider dispatch. When headers are absent, callers
    should skip the corresponding methods so behaviour stays backward
    compatible.
    """

    def __init__(
        self,
        idempotency_store: IdempotencyStore,
        tenant_rate_limiter: TenantRateLimiter,
        *,
        idempotency_lease_seconds: float = 300.0,
    ) -> None:
        """Wire the underlying store and limiter.

        Args:
            idempotency_store: Durable ``Idempotency-Key`` store.
            tenant_rate_limiter: Hard per-tenant token-bucket limiter.
            idempotency_lease_seconds: How long an in-flight reservation blocks
                duplicates before it expires (covers crashed requests).
        """
        self._idempotency_store = idempotency_store
        self._tenant_rate_limiter = tenant_rate_limiter
        self._idempotency_lease_seconds = idempotency_lease_seconds

    @property
    def idempotency_store(self) -> IdempotencyStore:
        """Return the composed idempotency store."""
        return self._idempotency_store

    @property
    def tenant_rate_limiter(self) -> TenantRateLimiter:
        """Return the composed tenant rate limiter."""
        return self._tenant_rate_limiter

    def check_idempotency(self, key: str, tenant: str) -> IdempotencyRecord | None:
        """Return a stored response for ``(key, tenant)``, or ``None``.

        Args:
            key: Client ``Idempotency-Key`` value (must be non-empty).
            tenant: Tenant scope for the key.

        Returns:
            Matching ``IdempotencyRecord`` when present and unexpired.
        """
        return self._idempotency_store.get(key, tenant)

    def reserve_idempotency(
        self, key: str, tenant: str
    ) -> IdempotencyRecord | IdempotencyReservation:
        """Return a stored response to replay, or claim the key for this request.

        A returned reservation must be passed to :meth:`complete_idempotency`
        on success or :meth:`release_idempotency` on failure.

        Args:
            key: Client ``Idempotency-Key`` value (must be non-empty).
            tenant: Tenant scope for the key.

        Returns:
            The completed record to replay, or this request's reservation.

        Raises:
            IdempotencyKeyInUseError: While another request holds the key.
        """
        return self._idempotency_store.reserve(
            key, tenant, lease_seconds=self._idempotency_lease_seconds
        )

    def complete_idempotency(
        self, reservation: IdempotencyReservation, response_json: str, status_code: int = 200
    ) -> bool:
        """Store the response for this request's reservation.

        Args:
            reservation: Reservation from :meth:`reserve_idempotency`.
            response_json: Serialized response body to replay.
            status_code: HTTP status to replay (default ``200``).

        Returns:
            ``True`` when stored; ``False`` if the reservation was lost to a newer
            request after its lease expired.
        """
        return self._idempotency_store.complete(reservation, response_json, status_code)

    def release_idempotency(self, reservation: IdempotencyReservation) -> None:
        """Drop this request's pending reservation so the key can be retried.

        Args:
            reservation: Reservation from :meth:`reserve_idempotency`.
        """
        self._idempotency_store.release(reservation)

    def remember_response(
        self,
        key: str,
        tenant: str,
        response_json: str,
        status_code: int = 200,
    ) -> bool:
        """Persist a response for later idempotent replay.

        Args:
            key: Client ``Idempotency-Key`` value.
            tenant: Tenant scope for the key.
            response_json: Serialized response body to replay.
            status_code: HTTP status to replay (default ``200``).

        Returns:
            ``True`` when stored; ``False`` if the key already existed.
        """
        return self._idempotency_store.put(key, tenant, response_json, status_code)

    def assert_tenant_allowed(self, tenant_id: str, tokens: int = 1) -> None:
        """Consume rate-limit tokens or raise when the tenant is over budget.

        Args:
            tenant_id: Tenant / customer identifier from ``X-Tenant-Id``.
            tokens: Tokens to consume (default ``1``).

        Raises:
            TenantRateLimitExceededError: When the bucket is exhausted.
        """
        self._tenant_rate_limiter.assert_allowed(tenant_id, tokens=tokens)
