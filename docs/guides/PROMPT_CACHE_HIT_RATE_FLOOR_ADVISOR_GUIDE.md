# PromptCacheHitRateFloorAdvisor Guide

![PromptCacheHitRateFloorAdvisor](../../assets/demo/prompt-cache-hit-rate-floor-advisor.gif)

Offline HITL advisor. Never network I/O.
Gap vs Anthropic/OpenAI prompt-cache hit-rate floor monitors. Distinct from `PromptCacheHitAdvisor` and `PrefixCacheThrashAdvisor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.prompt_cache_hit_rate_floor import PromptCacheHitRateFloorAdvisor

advice = PromptCacheHitRateFloorAdvisor().advise(request_id="r1", hit_rate=0.45)
print(advice.band)
```
