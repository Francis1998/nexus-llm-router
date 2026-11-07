# FlashAttnWorkspaceThrashBandAdvisor Guide

![FlashAttnWorkspaceThrashBandAdvisor](../../assets/demo/flash-attn-workspace-thrash-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI FlashAttention workspace thrash monitors. Distinct from `PagedAttentionFragmentationBandAdvisor` / `GpuMemoryFragmentationBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.flash_attn_workspace_thrash_band import FlashAttnWorkspaceThrashBandAdvisor

advice = FlashAttnWorkspaceThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
