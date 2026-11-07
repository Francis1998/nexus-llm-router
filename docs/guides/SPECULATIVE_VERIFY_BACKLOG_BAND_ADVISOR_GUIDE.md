# SpeculativeVerifyBacklogBandAdvisor Guide

![SpeculativeVerifyBacklogBandAdvisor](../../assets/demo/speculative-verify-backlog-band-advisor.gif)

Offline HITL `backlog_ratio` band advisor.
Gap vs vLLM/SGLang/TGI speculative verify backlog monitors. Distinct from `SpeculativeDecodeQueueStallAdvisor` / `SpeculativeDraftStarveBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.speculative_verify_backlog_band import SpeculativeVerifyBacklogBandAdvisor

advice = SpeculativeVerifyBacklogBandAdvisor().advise(request_id="req-1", backlog_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
