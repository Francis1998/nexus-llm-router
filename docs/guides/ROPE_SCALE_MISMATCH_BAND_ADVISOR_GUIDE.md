# RopeScaleMismatchBandAdvisor Guide

![RopeScaleMismatchBandAdvisor](../../assets/demo/rope-scale-mismatch-band-advisor.gif)

Offline HITL advisor for `rope_mismatch_ratio` bands.
Gap vs vLLM/SGLang/TGI RoPE-scale mismatch monitors. Distinct from `TokenizerBpeMergeStormAdvisor` / `ContextWindowFragmentationAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.rope_scale_mismatch_band import RopeScaleMismatchBandAdvisor

advice = RopeScaleMismatchBandAdvisor().advise(request_id="r1", rope_mismatch_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
