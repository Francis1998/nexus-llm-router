# AttentionSinkResidualBandAdvisor Guide

![AttentionSinkResidualBandAdvisor](../../assets/demo/attention-sink-residual-band-advisor.gif)

Offline band advisor for `sink_residual_ratio`.
Gap vs vLLM/SGLang/TGI attention-sink residual monitors. Distinct from `RopeScaleMismatchBandAdvisor` / `KvCacheQuantErrorBandAdvisor`.

Works with **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.attention_sink_residual_band import AttentionSinkResidualBandAdvisor

advice = AttentionSinkResidualBandAdvisor().advise(request_id="req-1", sink_residual_ratio=0.15)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
