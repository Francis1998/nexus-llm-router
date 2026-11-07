# AttentionSinkDriftAdvisor Guide

![AttentionSinkDriftAdvisor](../../assets/demo/attention-sink-drift-advisor.gif)

Offline HITL advisor for `sink_drift_ratio` bands. Gap vs vLLM/SGLang attention-sink drift monitors. Distinct from `KvCacheQuantErrorBandAdvisor` / `PrefillQueueStarvationAdvisor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.attention_sink_drift import AttentionSinkDriftAdvisor

advice = AttentionSinkDriftAdvisor().advise(request_id="r1", sink_drift_ratio=0.1)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Humans decide.
