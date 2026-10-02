# ProviderWeightDecaySchedulerAdvisor Guide

![ProviderWeightDecaySchedulerAdvisor](../../assets/demo/provider-weight-decay-scheduler.gif)

Closes OpenRouter / LiteLLM / Portkey provider weight-decay scheduler bands gaps.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `ProviderExplorationEpsilonAdvisor` and `ProviderHealthHysteresisAdvisor`.

## Usage

```python
from safety.provider_weight_decay_sched import ProviderWeightDecaySchedulerAdvisor

advice = ProviderWeightDecaySchedulerAdvisor().advise(request_id="r1", decay_step=0.1)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
