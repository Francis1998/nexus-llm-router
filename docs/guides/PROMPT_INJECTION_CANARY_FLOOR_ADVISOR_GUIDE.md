# PromptInjectionCanaryFloorAdvisor Guide

![PromptInjectionCanaryFloorAdvisor](../../assets/demo/prompt-injection-canary-floor-advisor.gif)

Offline HITL advisor for canary_hit_rate. Gap vs LiteLLM/vLLM/Guardrails prompt-injection canary-floor monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `PromptInjectionGateway` / `ProviderCanaryRolloutGuard`.

## Usage

```python
from safety.prompt_injection_canary_floor import PromptInjectionCanaryFloorAdvisor

advice = PromptInjectionCanaryFloorAdvisor().advise(request_id="r1", canary_hit_rate=0.0)
print(advice.band)
```

## Safety

No network I/O. Advisory bands only.
