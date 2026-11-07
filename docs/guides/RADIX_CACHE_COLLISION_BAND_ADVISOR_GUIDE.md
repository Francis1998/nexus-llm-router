# RadixCacheCollisionBandAdvisor Guide

![RadixCacheCollisionBandAdvisor](../../assets/demo/radix-cache-collision-band-advisor.gif)

Offline HITL `collision_ratio` bands.
Gap vs vLLM/SGLang/TGI radix-cache collision monitors. Distinct from `PrefixCacheEvictionStormAdvisor` / `KvBlockStealStormBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.radix_cache_collision_band import RadixCacheCollisionBandAdvisor

advice = RadixCacheCollisionBandAdvisor().advise(request_id="r1", collision_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
