# PromptCacheKeyCollisionAdvisor Guide

![PromptCacheKeyCollisionAdvisor HITL flow](../../assets/demo/prompt-cache-key-collision-advisor.gif)

Offline HITL advisor. Never auto-acts. Closes gaps vs vLLM/OpenRouter/LiteLLM prompt-cache key collision controls.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `PromptCacheHitRateAdvisor and PrefixCacheThrashAdvisor`.

## Usage

```python
from safety.prompt_cache_key_collision import PromptCacheKeyCollisionAdvisor

advice = PromptCacheKeyCollisionAdvisor().advise(
    request_id="r1", collision_rate=0.060000000000000005
)
print(advice.band)
```

## Safety

No HTTP. Humans decide. See `SAFETY.md`.
