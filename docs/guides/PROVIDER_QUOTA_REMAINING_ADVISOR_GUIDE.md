# ProviderQuotaRemainingAdvisor Guide

![ProviderQuotaRemainingAdvisor](../../assets/demo/provider-quota-remaining.gif)

Classify provider quota remaining into `ok` / `low` / `exhausted` bands.
Advisory only — never rejects traffic.

Closes the OpenRouter / LiteLLM / Helicone remaining-quota gap for
**GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `ProviderErrorBudgetShed` and `TenantSpendQuotaGuard`.

## Usage

See `tests/test_provider_quota_remaining.py`.
