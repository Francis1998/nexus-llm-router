# ReasoningEffortCostBandAdvisor Guide

![ReasoningEffortCostBandAdvisor HITL flow](../../assets/demo/reasoning-effort-cost-band-advisor.gif)

Offline HITL advisor. Never auto-acts. Closes gaps vs OpenAI/Anthropic/Gemini reasoning-effort cost-band advisors.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from ``ReasoningTokenBudgetAdvisor`` and ``GrammarConstrainedDecodeBudgetAdvisor``.

## Usage

```python
from safety.reasoning_effort_cost_band import ReasoningEffortCostBandAdvisor

advice = ReasoningEffortCostBandAdvisor().advise(request_id="r1", effort_cost_usd=0.15)
print(advice.band)
```

## Safety

No HTTP. Humans decide. See `SAFETY.md`.
