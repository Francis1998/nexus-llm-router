# SlidingWindowAttentionBleedBandAdvisor Guide

![SlidingWindowAttentionBleedBandAdvisor](../../assets/demo/sliding-window-attention-bleed-band-advisor.gif)

Offline helper.
Gap vs vLLM/SGLang/TGI sliding-window attention bleed monitors. Distinct from `AttentionSinkResidualBandAdvisor` / `RopeScaleMismatchBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.sliding_window_attention_bleed_band import SlidingWindowAttentionBleedBandAdvisor

advice = SlidingWindowAttentionBleedBandAdvisor().advise(request_id="r1", bleed_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
