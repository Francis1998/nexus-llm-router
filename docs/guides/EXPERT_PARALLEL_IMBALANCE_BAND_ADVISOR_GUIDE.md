# ExpertParallelImbalanceBandAdvisor Guide

![ExpertParallelImbalanceBandAdvisor](../../assets/demo/expert-parallel-imbalance-band-advisor.gif)

Offline HITL `imbalance_ratio` band advisor.
Gap vs vLLM/SGLang/TGI expert-parallel (MoE) imbalance monitors. Distinct from `TensorParallelSkewBandAdvisor` / `ContinuousBatchTokenBubbleBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.expert_parallel_imbalance_band import ExpertParallelImbalanceBandAdvisor

advice = ExpertParallelImbalanceBandAdvisor().advise(request_id="req-1", imbalance_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
