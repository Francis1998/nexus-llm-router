# ProviderAuthTokenExpiryAdvisor Guide

![ProviderAuthTokenExpiryAdvisor flow](../../assets/demo/provider-auth-token-expiry-advisor.gif)

Offline HITL advisor. Never network I/O.

Gap vs OpenAI/Anthropic/Gemini provider auth-token expiry advisors. Distinct from ``ProviderQuotaRemainingAdvisor`` and ``VirtualKeysAdvisor``.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.provider_auth_token_expiry import ProviderAuthTokenExpiryAdvisor

advice = ProviderAuthTokenExpiryAdvisor().advise(request_id="r1", minutes_to_expiry=120.0)
print(advice.band)
```

## Safety

Advisory only. Humans decide routing actions.
