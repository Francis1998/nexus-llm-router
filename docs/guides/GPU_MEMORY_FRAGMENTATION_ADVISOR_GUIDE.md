# GpuMemoryFragmentationBandAdvisor Guide

![GpuMemoryFragmentationBandAdvisor HITL flow](../../assets/demo/gpu-memory-fragmentation-advisor.gif)

Offline HITL advisor. Never network I/O. Closes gaps vs vLLM/TGI/TensorRT-LLM GPU memory fragmentation monitors.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `KvCacheEvictionPressureAdvisor` / `BatchCoalescePressureAdvisor`.

## Usage

```python
from safety.gpu_memory_fragmentation import GpuMemoryFragmentationBandAdvisor

advice = GpuMemoryFragmentationBandAdvisor().advise(request_id="r1", frag_ratio=0.15)
print(advice.band)
```
