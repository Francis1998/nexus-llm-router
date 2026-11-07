# KvCacheHitRateAdvisor Guide

![KvCacheHitRateAdvisor](../../assets/demo/kv-cache-hit-rate.gif)

Offline advisory KV-cache hit-rate bands. Closes vLLM / TensorRT-LLM /
OpenRouter cache-hit gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `PromptCacheHitAdvisor` and `PrefillTtftRatioAdvisor`.

## Usage

```python
from safety.kv_cache_hit_rate import KvCacheHitRateAdvisor

advice = KvCacheHitRateAdvisor().advise(request_id="r1", hit_rate=0.6, target_rate=0.8)
print(advice.band, advice.ratio)
```
