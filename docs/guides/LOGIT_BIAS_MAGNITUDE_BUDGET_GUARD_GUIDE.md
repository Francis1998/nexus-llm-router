# LogitBiasMagnitudeBudgetGuard Guide

![LogitBiasMagnitudeBudgetGuard HITL flow](../../assets/demo/logit-bias-magnitude-budget-guard.gif)

Offline HITL advisor. Never auto-acts. Closes gaps vs OpenAI/vLLM/LiteLLM logit-bias magnitude budget guards.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from ``GrammarConstrainedDecodeBudgetAdvisor`` and ``OutputTokenCeilingGuard``.

## Usage

```python
from safety.logit_bias_magnitude_budget import LogitBiasMagnitudeBudgetGuard

advice = LogitBiasMagnitudeBudgetGuard().advise(request_id="r1", l1_magnitude=16.0)
print(advice.band)
```

## Safety

No HTTP. Humans decide. See `SAFETY.md`.
