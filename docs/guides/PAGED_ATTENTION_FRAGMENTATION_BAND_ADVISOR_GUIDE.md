# PagedAttentionFragmentationBandAdvisor Guide

![PagedAttentionFragmentationBandAdvisor](../../assets/demo/paged-attention-fragmentation-band-advisor.gif)

Offline HITL helper for `fragmentation_ratio` bands.
Gap vs vLLM/SGLang/TGI paged-attention fragmentation monitors. Distinct from `KvBlockStealStormBandAdvisor` / `ChunkedPrefillBubbleBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.paged_attention_fragmentation_band import PagedAttentionFragmentationBandAdvisor

advice = PagedAttentionFragmentationBandAdvisor().advise(request_id="r1", fragmentation_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
