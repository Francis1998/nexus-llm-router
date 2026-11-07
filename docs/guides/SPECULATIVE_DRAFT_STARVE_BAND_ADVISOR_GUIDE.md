# SpeculativeDraftStarveBandAdvisor Guide

![SpeculativeDraftStarveBandAdvisor](../../assets/demo/speculative-draft-starve-band-advisor.gif)

Offline HITL helper for `starve_ratio` bands.
Gap vs vLLM/SGLang/TGI speculative draft-starve monitors. Distinct from `SpeculativeTreeDepthBlowupAdvisor` / `SpeculativeAcceptanceEntropyBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.speculative_draft_starve_band import SpeculativeDraftStarveBandAdvisor

advice = SpeculativeDraftStarveBandAdvisor().advise(request_id="r1", starve_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
