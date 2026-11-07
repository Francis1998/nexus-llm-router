# PrefixCachePoisonAdvisor Guide

![PrefixCachePoisonAdvisor](../../assets/demo/prefix-cache-poison-advisor.gif)

Offline advisor. Never network I/O. Gap vs vLLM/SGLang/TGI prefix-cache poison monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `PrefixCacheThrashAdvisor` / `KvCacheHitRateAdvisor`.

## Usage

```python
from safety.prefix_cache_poison import PrefixCachePoisonAdvisor

advice = PrefixCachePoisonAdvisor().advise(request_id="r1", poison_hit_rate=0.0)
print(advice.band)
```

## Safety

Advisory bands only. Humans decide.
