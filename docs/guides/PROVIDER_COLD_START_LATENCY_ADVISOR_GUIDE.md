# ProviderColdStartLatencyAdvisor Guide

![ProviderColdStartLatencyAdvisor](../../assets/demo/provider-cold-start-latency.gif)

Offline advisory bands for provider cold-start latency. Never network I/O.
Closes OpenRouter / LiteLLM / Portkey cold-start routing gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `FirstTokenLatencySloAdvisor` and `ProviderWarmPoolAdvisor`.

## Usage

```python
from safety.provider_cold_start_latency import ProviderColdStartLatencyAdvisor

advice = ProviderColdStartLatencyAdvisor().advise(
    provider="openai",
    cold_start_ms=800.0,
)
print(advice.band)
```
