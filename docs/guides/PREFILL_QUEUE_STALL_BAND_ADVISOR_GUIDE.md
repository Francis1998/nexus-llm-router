# PrefillQueueStallBandAdvisor Guide

![PrefillQueueStallBandAdvisor](../../assets/demo/prefill-queue-stall-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI prefill-queue stall monitors. Distinct from `DisaggPrefillDecodeSkewBandAdvisor` / `DisaggScheduleBubbleBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.prefill_queue_stall_band import PrefillQueueStallBandAdvisor

advice = PrefillQueueStallBandAdvisor().advise(request_id="r1", stall_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
