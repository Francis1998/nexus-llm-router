# BatchPaddingWasteAdvisor Guide

![BatchPaddingWasteAdvisor](../../assets/demo/batch-padding-waste-advisor.gif)

Offline HITL advisor for padding_waste_ratio. Gap vs vLLM/TGI/SGLang batch-padding waste monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `BatchCoalescePressureAdvisor` / `EmbeddingBatchSkewAdvisor`.

## Usage

```python
from safety.batch_padding_waste import BatchPaddingWasteAdvisor

advice = BatchPaddingWasteAdvisor().advise(request_id="r1", padding_waste_ratio=0.0)
print(advice.band)
```

## Safety

No network I/O. Advisory bands only.
