# KvCacheOffloadThrashBandAdvisor Guide

![KvCacheOffloadThrashBandAdvisor](../../assets/demo/kv-cache-offload-thrash-band-advisor.gif)

Offline HITL `thrash_ratio` band advisor.
Gap vs vLLM/SGLang/TGI KV-cache offload thrash monitors. Distinct from `KvCacheEvictionPressureAdvisor` / `PrefixCacheThrashAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.kv_cache_offload_thrash_band import KvCacheOffloadThrashBandAdvisor

advice = KvCacheOffloadThrashBandAdvisor().advise(request_id="req-1", thrash_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
