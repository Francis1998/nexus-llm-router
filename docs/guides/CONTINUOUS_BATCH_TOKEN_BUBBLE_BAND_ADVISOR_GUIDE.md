# ContinuousBatchTokenBubbleBandAdvisor Guide

![ContinuousBatchTokenBubbleBandAdvisor](../../assets/demo/continuous-batch-token-bubble-band-advisor.gif)

Offline HITL `bubble_ratio` bands.
Gap vs vLLM/SGLang/TGI continuous-batch token-bubble monitors. Distinct from `ChunkedPrefillBubbleBandAdvisor` / `DecodePreemptionPingPongBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.continuous_batch_token_bubble_band import ContinuousBatchTokenBubbleBandAdvisor

advice = ContinuousBatchTokenBubbleBandAdvisor().advise(request_id="r1", bubble_ratio=0.4)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
