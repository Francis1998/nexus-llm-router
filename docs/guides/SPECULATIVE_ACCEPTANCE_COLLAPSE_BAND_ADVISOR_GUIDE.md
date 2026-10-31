# SpeculativeAcceptanceCollapseBandAdvisor Guide

![SpeculativeAcceptanceCollapseBandAdvisor](../../assets/demo/speculative-acceptance-collapse-band-advisor.gif)

Offline HITL `collapse_ratio` bands.
Gap vs vLLM/SGLang/TGI speculative acceptance-collapse monitors. Distinct from `SpeculativeAcceptanceEntropyBandAdvisor` / `SpeculativeDraftStarveBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.speculative_acceptance_collapse_band import SpeculativeAcceptanceCollapseBandAdvisor

advice = SpeculativeAcceptanceCollapseBandAdvisor().advise(request_id="req-1", collapse_ratio=0.15)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
