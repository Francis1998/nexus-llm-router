# PromptCacheHitCollapseBandAdvisor Guide

![PromptCacheHitCollapseBandAdvisor](../../assets/demo/prompt-cache-hit-collapse-band-advisor.gif)

Offline `collapse_ratio` band advisor.
Gap vs vLLM/SGLang/TGI prompt-cache hit-collapse monitors. Distinct from `PromptCacheHitRateAdvisor` / `PromptCacheHitRateFloorAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.prompt_cache_hit_collapse_band import PromptCacheHitCollapseBandAdvisor

advice = PromptCacheHitCollapseBandAdvisor().advise(
    request_id="req-1",
    collapse_ratio=0.35,
    soft_limit=0.2,
    hard_limit=0.5,
)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
