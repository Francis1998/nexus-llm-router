# SpeculativeDecodeAcceptanceBandAdvisor Guide

![SpeculativeDecodeAcceptanceBandAdvisor](../../assets/demo/speculative-decode-acceptance-band.gif)

Closes vLLM / TensorRT-LLM / OpenRouter speculative-decode acceptance-rate bands gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SpeculativeDecodeAbortAdvisor` and `SpeculativeDecodeBudgetAdvisor`.

## Usage

```python
from safety.speculative_decode_acceptance import SpeculativeDecodeAcceptanceBandAdvisor

advice = SpeculativeDecodeAcceptanceBandAdvisor().advise(request_id="r1", acceptance_rate=0.55)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
