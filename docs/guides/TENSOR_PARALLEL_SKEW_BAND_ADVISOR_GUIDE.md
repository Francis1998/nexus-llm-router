# TensorParallelSkewBandAdvisor Guide

![TensorParallelSkewBandAdvisor](../../assets/demo/tensor-parallel-skew-band-advisor.gif)

Offline HITL `skew_ratio` bands.
Gap vs vLLM/SGLang/TGI tensor-parallel skew monitors. Distinct from `DisaggPrefillDecodeSkewBandAdvisor` / `TokenizerVocabSkewBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.tensor_parallel_skew_band import TensorParallelSkewBandAdvisor

advice = TensorParallelSkewBandAdvisor().advise(request_id="r1", skew_ratio=0.4)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
