# ProviderHealthHysteresisAdvisor Guide

![ProviderHealthHysteresisAdvisor](../../assets/demo/provider-health-hysteresis.gif)

Closes the OpenRouter/LiteLLM/Portkey provider health hysteresis gap.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ProviderHealthScoreboard` and `ProviderErrorBudgetShed`.

## Usage

```python
from safety.provider_health_hysteresis import ProviderHealthHysteresisAdvisor

advice = ProviderHealthHysteresisAdvisor().advise(
    provider_id="p1",
    error_rate=0.01,
    enter_threshold=0.1,
    exit_threshold=0.05,
    currently_unhealthy=False,
)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
