# KvBlockStealStormBandAdvisor Guide

![KvBlockStealStormBandAdvisor](../../assets/demo/kv-block-steal-storm-band-advisor.gif)

Offline HITL `steal_ratio` bands.
Gap vs vLLM/SGLang/TGI KV block steal-storm monitors. Distinct from `KvBlockStealAdvisor` / `PrefixCacheEvictionStormAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.kv_block_steal_storm_band import KvBlockStealStormBandAdvisor

advice = KvBlockStealStormBandAdvisor().advise(request_id="r1", steal_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
