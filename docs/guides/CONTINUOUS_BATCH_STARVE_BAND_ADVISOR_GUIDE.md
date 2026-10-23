# ContinuousBatchStarveBandAdvisor Guide

![ContinuousBatchStarveBandAdvisor](../../assets/demo/continuous-batch-starve-band-advisor.gif)

Offline HITL `starve_ratio` bands.
Gap vs vLLM/SGLang/TGI continuous-batch starve monitors. Distinct from `ContinuousBatchPreemptionChurnAdvisor` / `ContinuousBatchingTailLatencyBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.continuous_batch_starve_band import ContinuousBatchStarveBandAdvisor

advice = ContinuousBatchStarveBandAdvisor().advise(
    request_id="r1",
    starve_ratio=0.3,
)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
