# PrefixCacheEvictionStormAdvisor Guide

![PrefixCacheEvictionStormAdvisor](../../assets/demo/prefix-cache-eviction-storm-advisor.gif)

Offline HITL `eviction_storm_ratio` bands.
Gap vs vLLM/SGLang/TGI prefix-cache eviction-storm monitors. Distinct from `PrefixCachePoisonAdvisor` / `PrefixCacheThrashAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.prefix_cache_eviction_storm import PrefixCacheEvictionStormAdvisor

advice = PrefixCacheEvictionStormAdvisor().advise(
    request_id="r1",
    eviction_storm_ratio=0.3,
)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
