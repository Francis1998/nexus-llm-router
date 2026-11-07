# GpuMemoryFragmentationBandAdvisor Guide

![GpuMemoryFragmentationBandAdvisor HITL flow](../../assets/demo/gpu-memory-fragmentation-advisor.gif)

Offline HITL advisor. Never network I/O. Closes gaps vs vLLM/TGI/TensorRT-LLM GPU memory fragmentation monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `KvCacheEvictionPressureAdvisor` / `BatchCoalescePressureAdvisor`.

## Usage

```python
from safety.gpu_memory_fragmentation import GpuMemoryFragmentationBandAdvisor

advice = GpuMemoryFragmentationBandAdvisor().advise(request_id="r1", frag_ratio=0.15)
print(advice.band)
```
