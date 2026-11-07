# TenantConcurrencySlotGuard Guide

![TenantConcurrencySlotGuard](../../assets/demo/tenant-concurrency-slots.gif)

Per-tenant in-flight concurrency slots with `ok` / `near` / `full` bands
(optional hard gate). Never rejects unless `hard_gate=True`.

Closes the Portkey / LiteLLM / OpenRouter per-tenant concurrency gap for
**GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `TenantRateLimiter` and `TenantSpendQuotaGuard`.

## Usage

See `tests/test_tenant_concurrency.py`.
