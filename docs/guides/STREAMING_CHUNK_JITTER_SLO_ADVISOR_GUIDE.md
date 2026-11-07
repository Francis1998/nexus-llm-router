# StreamingChunkJitterSloAdvisor Guide

![StreamingChunkJitterSloAdvisor](../../assets/demo/streaming-chunk-jitter.gif)

Closes the vLLM/OpenRouter/LiteLLM streaming chunk-jitter SLO gap.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `FirstTokenLatencySloAdvisor` and `StreamingBackpressureAdvisor`.

## Usage

```python
from safety.streaming_chunk_jitter import StreamingChunkJitterSloAdvisor

advice = StreamingChunkJitterSloAdvisor().advise(request_id="r1", jitter_ms=40.0, slo_ms=50.0)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
