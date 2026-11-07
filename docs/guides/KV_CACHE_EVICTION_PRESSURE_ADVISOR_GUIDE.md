# KvCacheEvictionPressureAdvisor Guide

![KvCacheEvictionPressureAdvisor HITL flow](../../assets/demo/kv-cache-eviction-pressure-advisor.gif)

Offline HITL advisor. Never auto-acts. Closes gaps vs vLLM/TensorRT-LLM/SGLang KV-cache eviction pressure advisors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `KvCacheHitRateAdvisor and PrefixCacheThrashAdvisor`.

## Usage

```python
from safety.kv_cache_eviction_pressure import KvCacheEvictionPressureAdvisor

advice = KvCacheEvictionPressureAdvisor().advise(request_id="r1", eviction_rate=0.125)
print(advice.band)
```

## Safety

No HTTP. Humans decide. See `SAFETY.md`.
