# MoeExpertDropoutStormAdvisor Guide

![MoeExpertDropoutStormAdvisor](../../assets/demo/moe-expert-dropout-storm-advisor.gif)

Offline HITL helper for `dropout_storm_ratio` bands.
Gap vs vLLM/SGLang/TGI MoE expert-dropout storm monitors. Distinct from `MoeExpertLoadImbalanceAdvisor` / `MoeRoutingCollapseAdvisor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.moe_expert_dropout_storm import MoeExpertDropoutStormAdvisor

advice = MoeExpertDropoutStormAdvisor().advise(request_id="r1", dropout_storm_ratio=0.35)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
