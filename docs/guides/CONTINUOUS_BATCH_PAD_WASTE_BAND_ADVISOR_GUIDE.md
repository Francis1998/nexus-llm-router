# ContinuousBatchPadWasteBandAdvisor Guide

![ContinuousBatchPadWasteBandAdvisor](../../assets/demo/continuous-batch-pad-waste-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI continuous-batch pad-waste monitors. Distinct from `ContinuousBatchTokenBubbleBandAdvisor` / `ContinuousBatchingTailLatencyBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.continuous_batch_pad_waste_band import ContinuousBatchPadWasteBandAdvisor

advice = ContinuousBatchPadWasteBandAdvisor().advise(request_id="r1", pad_waste_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
