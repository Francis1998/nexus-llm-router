# NvlinkPeerCopyStallBandAdvisor Guide

![NvlinkPeerCopyStallBandAdvisor](../../assets/demo/nvlink-peer-copy-stall-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI NVLink peer-copy stall monitors. Distinct from `TensorParallelAllreduceStallBandAdvisor` / `TensorParallelSkewBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.nvlink_peer_copy_stall_band import NvlinkPeerCopyStallBandAdvisor

advice = NvlinkPeerCopyStallBandAdvisor().advise(request_id="req-1", stall_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
