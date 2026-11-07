# ExpertRoutingEntropyBandAdvisor Guide

![ExpertRoutingEntropyBandAdvisor](../../assets/demo/expert-routing-entropy-band-advisor.gif)

Offline `entropy_ratio` band advisor.
Gap vs vLLM/SGLang/TGI MoE expert-routing entropy monitors. Distinct from `MoeExpertHotspotBandAdvisor` / `ExpertParallelImbalanceBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.expert_routing_entropy_band import ExpertRoutingEntropyBandAdvisor

advice = ExpertRoutingEntropyBandAdvisor().advise(request_id="r1", entropy_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
