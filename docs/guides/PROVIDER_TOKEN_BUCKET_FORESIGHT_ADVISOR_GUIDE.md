# ProviderTokenBucketForesightAdvisor Guide

![ProviderTokenBucketForesightAdvisor](../../assets/demo/provider-token-bucket-foresight.gif)

Offline advisory token-bucket foresight bands. Closes OpenRouter / LiteLLM /
Portkey foresight gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from tenant rate limiters and `CostForecastAdvisor`.

## Usage

```python
from safety.provider_token_bucket_foresight import ProviderTokenBucketForesightAdvisor

advice = ProviderTokenBucketForesightAdvisor().advise(
    provider="openai", tokens_remaining=1100.0, forecast_demand=1000.0
)
print(advice.band, advice.ratio)
```
