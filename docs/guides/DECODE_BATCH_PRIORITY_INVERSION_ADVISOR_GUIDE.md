# DecodeBatchPriorityInversionAdvisor Guide

![DecodeBatchPriorityInversionAdvisor](../../assets/demo/decode-batch-priority-inversion-advisor.gif)

Offline HITL advisor for `inversion_ratio` bands. Gap vs vLLM/SGLang/TGI decode-batch priority-inversion monitors. Distinct from `PrefillQueueStarvationAdvisor` / `KvBlockStealAdvisor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.decode_batch_priority_inversion import DecodeBatchPriorityInversionAdvisor

advice = DecodeBatchPriorityInversionAdvisor().advise(request_id="r1", inversion_ratio=0.1)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Humans decide.
