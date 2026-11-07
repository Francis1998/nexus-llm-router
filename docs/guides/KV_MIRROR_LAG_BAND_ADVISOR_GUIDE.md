# KvMirrorLagBandAdvisor Guide

![KvMirrorLagBandAdvisor](../../assets/demo/kv-mirror-lag-band-advisor.gif)

Offline band advisor for `lag_ratio`.
Gap vs vLLM/SGLang/TGI KV-mirror lag monitors. Distinct from `DisaggKvTransferStallBandAdvisor` / `KvCacheOffloadThrashBandAdvisor`.

Works with **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.kv_mirror_lag_band import KvMirrorLagBandAdvisor

advice = KvMirrorLagBandAdvisor().advise(request_id="req-1", lag_ratio=0.15)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
