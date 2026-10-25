# SpeculativeAcceptanceEntropyBandAdvisor Guide

![SpeculativeAcceptanceEntropyBandAdvisor](../../assets/demo/speculative-acceptance-entropy-band-advisor.gif)

Offline HITL `entropy_ratio` bands.
Gap vs vLLM/SGLang/TGI speculative acceptance-entropy monitors. Distinct from `SpeculativeDecodeAcceptanceBandAdvisor` / `SpeculativeDecodeRejectionHysteresisAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.speculative_acceptance_entropy_band import SpeculativeAcceptanceEntropyBandAdvisor

advice = SpeculativeAcceptanceEntropyBandAdvisor().advise(request_id="r1", entropy_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
