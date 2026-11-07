# Gateway Guards Guide

Composable ingress guards for GPT-6 / Claude 5.5 / Gemini 4 /
Kimi K3.1 chat completions. Wires durable `Idempotency-Key` replay together with
hard `X-Tenant-Id` rate limiting before provider dispatch.

![Gateway guards demo](../../assets/demo/gateway-guards.gif)

## Why

LiteLLM and Portkey expose gateway middleware that combines idempotent retries
and per-tenant ceilings. Nexus already shipped `IdempotencyStore` and
`TenantRateLimiter` as library modules; `GatewayGuardService` composes them and
`api.main` exposes `@lru_cache` getters so `/v1/chat/completions` can apply
them lightly without breaking clients that omit the headers.

## How it works

1. `get_idempotency_store()` / `get_tenant_rate_limiter()` / `get_gateway_guards()`
   follow the same `@lru_cache` pattern as `get_response_cache` / `get_spend_ledger`.
2. When `X-Tenant-Id` is present, `assert_tenant_allowed` hard-rejects excess
   traffic with HTTP 429 (`TenantRateLimitExceededError`).
3. When `Idempotency-Key` is present on a non-streaming request,
   `reserve_idempotency` atomically claims the key, or returns the stored JSON
   body to replay. While the original is in flight, a duplicate gets HTTP 409.
   The returned `IdempotencyReservation` carries an ownership token. After a
   successful completion, `complete_idempotency` stores the serialized response
   in place of the reservation; after a failure, `release_idempotency` frees
   the key. Both only act while the token still owns the key, so a request that
   outlived its lease cannot release or overwrite a newer request's
   reservation. Streaming requests only replay (`check_idempotency`), because
   streams are never stored.
4. When either header is absent, behaviour matches the pre-guard API
   (fully backward compatible).

## Configuration

```bash
NEXUS_IDEMPOTENCY_PATH=migrations/idempotency.sqlite3
NEXUS_IDEMPOTENCY_TTL_SECONDS=86400
NEXUS_IDEMPOTENCY_LEASE_SECONDS=300
NEXUS_TENANT_RATE_LIMIT_CAPACITY=60
NEXUS_TENANT_RATE_LIMIT_REFILL_PER_SECOND=1
```

## Example

```python
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyReservation, IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

guards = GatewayGuardService(
    IdempotencyStore("migrations/idempotency.sqlite3"),
    TenantRateLimiter(capacity=60, refill_per_second=1.0),
)
guards.assert_tenant_allowed("acme")
claim = guards.reserve_idempotency("retry-1", "acme")  # 409 upstream if in flight
if isinstance(claim, IdempotencyReservation):
    try:
        response_json = '{"id":"chatcmpl-1"}'  # dispatch to GPT-6 / Claude 5.5 / ...
        guards.complete_idempotency(claim, response_json)
    except Exception:
        guards.release_idempotency(claim)
        raise
print(guards.reserve_idempotency("retry-1", "acme"))  # completed record: replay it
```

Curl with both headers:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retry-1' \
  -H 'X-Tenant-Id: acme' \
  -d '{"messages":[{"role":"user","content":"hello"}],"model":"gpt-5.5"}'
```

## Gap vs LiteLLM / Portkey

| Capability | LiteLLM / Portkey | Nexus |
| --- | --- | --- |
| Composed gateway guards | Yes | `GatewayGuardService` |
| Idempotency-Key replay | Yes | `reserve_idempotency` / `complete_idempotency` / `release_idempotency` (409 for in-flight duplicates) |
| Hard per-tenant rate limit | Yes | `assert_tenant_allowed` via `X-Tenant-Id` |
| Optional headers (compat) | Yes | No-op when headers absent |
| FastAPI wiring | Middleware | `@lru_cache` getters + light endpoint hooks |
