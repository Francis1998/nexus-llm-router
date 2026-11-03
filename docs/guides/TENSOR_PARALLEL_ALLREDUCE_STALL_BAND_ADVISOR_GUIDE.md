# TensorParallelAllreduceStallBandAdvisor Guide

![TensorParallelAllreduceStallBandAdvisor](../../assets/demo/tensor-parallel-allreduce-stall-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI tensor-parallel allreduce stall monitors. Distinct from `TensorParallelSkewBandAdvisor` / `ExpertParallelImbalanceBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.tensor_parallel_allreduce_stall_band import TensorParallelAllreduceStallBandAdvisor

advice = TensorParallelAllreduceStallBandAdvisor().advise(request_id="r1", stall_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
