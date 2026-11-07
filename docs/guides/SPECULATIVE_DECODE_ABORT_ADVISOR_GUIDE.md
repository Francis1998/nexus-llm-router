# SpeculativeDecodeAbortAdvisor Guide

![SpeculativeDecodeAbortAdvisor](../../assets/demo/speculative-decode-abort.gif)

Closes the vLLM/TensorRT-LLM/OpenRouter speculative-decode abort-rate gap.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SpeculativeDecodeBudgetAdvisor` and `OutputTokenForecastAdvisor`.

## Usage

```python
from safety.speculative_decode_abort import SpeculativeDecodeAbortAdvisor

advice = SpeculativeDecodeAbortAdvisor().advise(request_id="r1", abort_rate=0.05, budget_rate=0.1)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
