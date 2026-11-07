# ChunkedPrefillBubbleBandAdvisor Guide

![ChunkedPrefillBubbleBandAdvisor](../../assets/demo/chunked-prefill-bubble-band-advisor.gif)

Offline HITL `bubble_ratio` bands.
Gap vs vLLM/SGLang/TGI chunked-prefill bubble monitors. Distinct from `ContinuousBatchStarveBandAdvisor` / `DecodeSlotFragmentationBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.chunked_prefill_bubble_band import ChunkedPrefillBubbleBandAdvisor

advice = ChunkedPrefillBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
