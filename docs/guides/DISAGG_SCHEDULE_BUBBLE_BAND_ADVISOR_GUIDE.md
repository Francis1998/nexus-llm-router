# DisaggScheduleBubbleBandAdvisor Guide

![DisaggScheduleBubbleBandAdvisor](../../assets/demo/disagg-schedule-bubble-band-advisor.gif)

Offline `bubble_ratio` band advisor.
Gap vs vLLM/SGLang/TGI disaggregated schedule-bubble monitors. Distinct from `DisaggPrefillDecodeSkewBandAdvisor` / `ChunkedPrefillBubbleBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.disagg_schedule_bubble_band import DisaggScheduleBubbleBandAdvisor

advice = DisaggScheduleBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
