# SpeculativeDecodeQueueStallAdvisor Guide

![SpeculativeDecodeQueueStallAdvisor HITL flow](../../assets/demo/speculative-decode-queue-stall-advisor.gif)

Offline HITL advisor. Never network I/O. Closes gaps vs vLLM/TGI/OpenAI-compatible speculative-decode queue-stall monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SpeculativeDecodeRejectionStormAdvisor` and `SpeculativeDecodeAbortAdvisor`.

## Usage

```python
from safety.speculative_decode_queue_stall import SpeculativeDecodeQueueStallAdvisor

advice = SpeculativeDecodeQueueStallAdvisor().advise(request_id="r1", stall_ratio=0.15)
print(advice.band)
```
