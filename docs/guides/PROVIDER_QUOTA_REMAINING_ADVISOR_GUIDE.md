# ProviderQuotaRemainingAdvisor Guide

![ProviderQuotaRemainingAdvisor](../../assets/demo/provider-quota-remaining.gif)

Classify provider quota remaining into `ok` / `low` / `exhausted` bands.
Advisory only — never rejects traffic.

Closes the OpenRouter / LiteLLM / Helicone remaining-quota gap for
**GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ProviderErrorBudgetShed` and `TenantSpendQuotaGuard`.

## Usage

See `tests/test_provider_quota_remaining.py`.
