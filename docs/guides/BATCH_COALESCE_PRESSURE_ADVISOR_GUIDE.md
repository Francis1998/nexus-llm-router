# BatchCoalescePressureAdvisor Guide

![BatchCoalescePressureAdvisor](../../assets/demo/batch-coalesce-pressure-advisor.gif)

Offline HITL advisor. Never network I/O.
Gap vs vLLM/TGI/OpenAI-compatible batch-coalesce pressure monitors. Distinct from `EmbeddingBatchSkewAdvisor` and `TokenBucketBurstStrategy`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.batch_coalesce_pressure import BatchCoalescePressureAdvisor

advice = BatchCoalescePressureAdvisor().advise(request_id="r1", coalesce_pressure=0.45)
print(advice.band)
```
