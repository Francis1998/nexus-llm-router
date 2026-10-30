# PagedAttentionBlockLeakBandAdvisor Guide

![PagedAttentionBlockLeakBandAdvisor](../../assets/demo/paged-attention-block-leak-band-advisor.gif)

Safety band advisor for `leak_ratio`.
Gap vs vLLM/SGLang/TGI paged-attention block-leak monitors. Distinct from `PagedAttentionFragmentationBandAdvisor` / `KvBlockStealStormBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.paged_attention_block_leak_band import PagedAttentionBlockLeakBandAdvisor

advice = PagedAttentionBlockLeakBandAdvisor().advise(request_id="req-1", leak_ratio=0.15)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
