# DecodePreemptionPingPongBandAdvisor Guide

![DecodePreemptionPingPongBandAdvisor](../../assets/demo/decode-preemption-ping-pong-band-advisor.gif)

Offline HITL `ping_pong_ratio` bands.
Gap vs vLLM/SGLang/TGI decode preemption ping-pong monitors. Distinct from `ContinuousBatchPreemptionChurnAdvisor` / `ContinuousBatchStarveBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.decode_preemption_ping_pong_band import DecodePreemptionPingPongBandAdvisor

advice = DecodePreemptionPingPongBandAdvisor().advise(request_id="r1", ping_pong_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
