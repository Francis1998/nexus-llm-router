# SchedulerQueueSkewBandAdvisor Guide

![SchedulerQueueSkewBandAdvisor](../../assets/demo/scheduler-queue-skew-band-advisor.gif)

Offline `skew_ratio` band advisor.
Gap vs vLLM/SGLang/TGI scheduler queue-skew monitors. Distinct from `PrefillQueueStallBandAdvisor` / `ContinuousBatchPadWasteBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.scheduler_queue_skew_band import SchedulerQueueSkewBandAdvisor

advice = SchedulerQueueSkewBandAdvisor().advise(
    request_id="req-1",
    skew_ratio=0.35,
    soft_limit=0.2,
    hard_limit=0.5,
)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
