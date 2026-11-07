# GrammarConstrainedDecodeBudgetAdvisor Guide

![GrammarConstrainedDecodeBudgetAdvisor HITL flow](../../assets/demo/grammar-constrained-decode-budget-advisor.gif)

Offline HITL advisor. Never auto-acts. Closes gaps vs vLLM/Outlines/XGrammar grammar-constrained decode budget advisors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from ``StructuredOutputRepairAdvisor`` and ``OutputTokenCeilingGuard``.

## Usage

```python
from safety.grammar_constrained_decode_budget import GrammarConstrainedDecodeBudgetAdvisor

advice = GrammarConstrainedDecodeBudgetAdvisor().advise(request_id="r1", constraint_tokens=400.0)
print(advice.band)
```

## Safety

No HTTP. Humans decide. See `SAFETY.md`.
