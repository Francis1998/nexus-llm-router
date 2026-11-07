# AdapterWeightDriftAdvisor Guide

![AdapterWeightDriftAdvisor](../../assets/demo/adapter-weight-drift-advisor.gif)

Offline HITL advisor for weight_drift_score. Gap vs vLLM/TGI LoRA adapter-weight drift monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `LoraAdapterSwapCostAdvisor` / `EmbeddingDimMismatchAdvisor`.

## Usage

```python
from safety.adapter_weight_drift import AdapterWeightDriftAdvisor

advice = AdapterWeightDriftAdvisor().advise(request_id="r1", weight_drift_score=0.0)
print(advice.band)
```

## Safety

No network I/O. Advisory bands only.
