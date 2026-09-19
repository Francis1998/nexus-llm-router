# TenantConcurrencySlotGuard Guide

![TenantConcurrencySlotGuard](../../assets/demo/tenant-concurrency-slots.gif)

Per-tenant in-flight concurrency slots with `ok` / `near` / `full` bands
(optional hard gate). Never rejects unless `hard_gate=True`.

Closes the Portkey / LiteLLM / OpenRouter per-tenant concurrency gap for
**GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `TenantRateLimiter` and `TenantSpendQuotaGuard`.

## Usage

See `tests/test_tenant_concurrency.py`.
