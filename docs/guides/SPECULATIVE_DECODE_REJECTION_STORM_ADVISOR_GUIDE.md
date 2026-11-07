# SpeculativeDecodeRejectionStormAdvisor Guide

![SpeculativeDecodeRejectionStormAdvisor flow](../../assets/demo/speculative-decode-rejection-storm-advisor.gif)

Offline HITL advisor. Never network I/O.

Gap vs vLLM/SGLang/TensorRT-LLM speculative-decode rejection-storm advisors. Distinct from ``SpeculativeDecodeAcceptanceBandAdvisor`` and ``SpeculativeDecodeAbortAdvisor``.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.speculative_decode_rejection_storm import SpeculativeDecodeRejectionStormAdvisor

advice = SpeculativeDecodeRejectionStormAdvisor().advise(request_id="r1", rejection_rate=0.1)
print(advice.band)
```

## Safety

Advisory only. Humans decide routing actions.
