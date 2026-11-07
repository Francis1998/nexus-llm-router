# Idempotency Store Guide

Durable `Idempotency-Key` storage for GPT-6 / Claude 5.5 /
Gemini 4 / Kimi K3.1 chat completions. Safe client retries replay the original
response instead of double-dispatching (and double-billing) the provider.

![Idempotency store demo](../../assets/demo/idempotency-store.gif)

## Why

Gateways that accept `Idempotency-Key` must remember the first successful
response for a `(key, tenant)` pair. LiteLLM and Portkey ship durable
idempotency stores; Nexus now provides the same building block via
`safety.idempotency.IdempotencyStore` (SQLite-backed, TTL eviction).

## How it works

1. `reserve(key, tenant, lease_seconds=...)` atomically claims a key before
   routing. It uses `INSERT OR IGNORE` on the `(key, tenant)` primary key, so
   exactly one concurrent caller wins, even across processes sharing the file.
   It returns an `IdempotencyReservation` (with a random ownership token) to
   the winner, returns the completed record to replay, or raises
   `IdempotencyKeyInUseError` while another request holds the key. Pending
   reservations expire after `lease_seconds`.
2. `complete(reservation, response_json, status_code)` stores the response for
   a reservation the caller still owns. If the lease expired, the response is
   stored only when the key is free, never over a newer request's claim.
   `release(reservation)` drops the caller's pending reservation after a
   failure and leaves anyone else's untouched. `put(key, tenant, ...)` stores a
   response without a reservation when the key is new, and returns `False`
   otherwise.
3. `get(key, tenant)` returns an `IdempotencyRecord` (or `None` on miss,
   expiry, or a still-pending reservation) so middleware can replay
   `status_code` + `response_json`.
4. Expired rows are evicted on `reserve` / `put` / `get` and via `evict_expired()`.
5. Use `":memory:"` in unit tests (persistent connection, like
   `VirtualKeyStore`). Prefer a filesystem path in production.

Wire the store in front of `/v1/chat/completions` when clients send an
`Idempotency-Key` header. Module + tests are sufficient for v1; deep API
rewiring is optional.

## Configuration

```bash
NEXUS_IDEMPOTENCY_PATH=migrations/idempotency.sqlite3
NEXUS_IDEMPOTENCY_TTL_SECONDS=86400
```

## Example

```python
from safety.idempotency import IdempotencyStore

store = IdempotencyStore("migrations/idempotency.sqlite3", ttl_seconds=86400)
if store.put("retry-1", "acme", '{"id":"chatcmpl-1","model":"gpt-5.5"}', 200):
    # first request — proceed to provider
    ...
else:
    record = store.get("retry-1", "acme")
    # replay record.status_code + record.response_json
```

## Gap vs LiteLLM / Portkey

| Capability | LiteLLM / Portkey | Nexus |
| --- | --- | --- |
| Durable idempotency keys | Yes | SQLite `IdempotencyStore` |
| Tenant-scoped keys | Yes | Primary key `(key, tenant)` |
| TTL eviction | Yes | `ttl_seconds` + `evict_expired` |
| Chat-completion retry replay | Yes | `response_json` + `status_code` |
