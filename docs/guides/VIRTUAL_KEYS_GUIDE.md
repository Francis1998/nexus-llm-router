# Virtual Keys Guide

LiteLLM-style virtual API keys for multi-tenant GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 gateways. Issues hashed keys with a per-tenant USD budget
and optional model allowlist — without storing the raw secret.

## Why

Shared provider credentials leak spend attribution. LiteLLM virtual keys give
each tenant a distinct key with budget and model gates; Nexus now provides the
same building block via `safety.VirtualKeyStore` (SQLite-backed).

## How it works

1. `create(tenant=..., max_budget_usd=..., allowed_models=...)` returns
   `(raw_key, VirtualKey)`. The raw key is shown once; only the SHA-256 hash is
   persisted.
2. `authenticate(raw_key)` resolves the hash to a durable record (or raises
   `VirtualKeyError` when unknown/disabled).
3. `assert_model_allowed` rejects models outside a non-empty allowlist. An
   empty allowlist permits every model.
4. `assert_budget(spent_usd, estimated_cost_usd)` rejects when
   `spent + estimated > max_budget_usd`.

Wire the store in front of `/v1/chat/completions` (or alongside
`SpendLedger`) when you need LiteLLM-parity virtual-key gates. Module + tests
are sufficient for v1; deep API rewiring is optional.

## Configuration

```bash
NEXUS_VIRTUAL_KEYS_PATH=migrations/virtual-keys.sqlite3
```

Use `":memory:"` in unit tests. Default path is in-memory until the API layer
opts into a durable file.

## Example

```python
from safety.virtual_keys import VirtualKeyStore

store = VirtualKeyStore("migrations/virtual-keys.sqlite3")
raw_key, key = store.create(
    tenant="acme",
    max_budget_usd=25.0,
    allowed_models=["gpt-5.5", "claude-sonnet-4-6"],
)
authenticated = store.authenticate(raw_key)
store.assert_model_allowed(authenticated, "gpt-5.5")
store.assert_budget(authenticated, spent_usd=1.2, estimated_cost_usd=0.05)
```

## Gap vs LiteLLM / Portkey

| Capability | LiteLLM virtual keys | Nexus |
| --- | --- | --- |
| Hashed key issuance | Yes | Yes (`secrets.token_urlsafe` + SHA-256) |
| Per-key USD budget | Yes | `max_budget_usd` + `assert_budget` |
| Model allowlist | Yes | `allowed_models` (empty = all) |
| Durable store | Yes | SQLite (`VirtualKeyStore`) |
| Spend aggregation | `/spend` | Pair with `SpendLedger` / `GET /v1/spend` |
