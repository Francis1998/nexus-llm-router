# SlidingWindowAttentionBleedBandAdvisor Guide

![SlidingWindowAttentionBleedBandAdvisor](../../assets/demo/sliding-window-attention-bleed-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI sliding-window attention bleed monitors. Distinct from `AttentionSinkResidualBandAdvisor` / `RopeScaleMismatchBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.sliding_window_attention_bleed_band import SlidingWindowAttentionBleedBandAdvisor

advice = SlidingWindowAttentionBleedBandAdvisor().advise(request_id="r1", bleed_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
