# Provider Warm Pool Advisor Guide

![Provider warm pool](../../assets/demo/provider-warm-pool.gif)

Advise cold/warming/hot bands from warm replica counts vs target. Never
performs network I/O. Closes the OpenRouter / LiteLLM / Portkey warm-pool gap.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ProviderHealthGuard` and `RequestHedgingAdvisor`.

## Usage

```python
from safety.provider_warm_pool import ProviderWarmPoolAdvisor

advice = ProviderWarmPoolAdvisor().advise("openai", warm_replicas=1, target_replicas=2)
assert advice.band == "warming"
```
