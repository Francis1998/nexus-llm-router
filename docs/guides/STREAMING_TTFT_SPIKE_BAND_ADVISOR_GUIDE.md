# StreamingTtftSpikeBandAdvisor Guide

![StreamingTtftSpikeBandAdvisor](../../assets/demo/streaming-ttft-spike-band-advisor.gif)

Offline `spike_ratio` band advisor.
Gap vs vLLM/SGLang/TGI streaming TTFT spike monitors. Distinct from `PrefillTtftRatioAdvisor` / `FirstTokenLatencySloAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.streaming_ttft_spike_band import StreamingTtftSpikeBandAdvisor

advice = StreamingTtftSpikeBandAdvisor().advise(
    request_id="req-1",
    spike_ratio=0.35,
    soft_limit=0.2,
    hard_limit=0.5,
)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
