# PrefixCacheEvictionResidualBandAdvisor Guide

![PrefixCacheEvictionResidualBandAdvisor](../../assets/demo/prefix-cache-eviction-residual-band-advisor.gif)

Offline band advisor for `eviction_residual_ratio`.
Gap vs vLLM/SGLang/TGI prefix-cache eviction residual monitors. Distinct from `RadixCacheCollisionBandAdvisor` / `KvCacheOffloadThrashBandAdvisor`.

Works with **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.prefix_cache_eviction_residual_band import PrefixCacheEvictionResidualBandAdvisor

advice = PrefixCacheEvictionResidualBandAdvisor().advise(
    request_id="req-1", eviction_residual_ratio=0.15
)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
