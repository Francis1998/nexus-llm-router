# FlashinferKernelFallbackBandAdvisor Guide

![FlashinferKernelFallbackBandAdvisor](../../assets/demo/flashinfer-kernel-fallback-band-advisor.gif)

Safety band advisor for `fallback_ratio`.
Gap vs vLLM/SGLang/TGI FlashInfer kernel-fallback monitors. Distinct from `KvCacheQuantErrorBandAdvisor` / `TensorParallelSkewBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.flashinfer_kernel_fallback_band import FlashinferKernelFallbackBandAdvisor

advice = FlashinferKernelFallbackBandAdvisor().advise(request_id="req-1", fallback_ratio=0.15)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
