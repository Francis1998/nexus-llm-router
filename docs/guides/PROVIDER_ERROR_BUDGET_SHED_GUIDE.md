# Provider-Error-Budget-Shed Routing Guide

![Provider error budget shed flow](../../assets/provider-error-budget-shed.gif)

Use the `provider-error-budget-shed` strategy when provider failures should
steer traffic away from degrading backends before hard circuit breakers fully
open.

## When to use it

- GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic needs
  quality-first routing while avoiding providers burning error budget.
- You already record provider outcomes in shared `SuccessStats`.
- Operators want a soft reliability threshold
  (`NEXUS_PROVIDER_ERROR_BUDGET_RATE`, default `0.15`) instead of only binary
  circuit-breaker state.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Prefer candidates whose provider circuit is closed; if every circuit is open,
   keep routing deterministically across the eligible set.
3. Compute provider error rate as `1 - SuccessStats.success_rate(provider)`.
   Providers with no observations are treated as 0% error for cold starts.
4. Among providers under the error budget, select the highest-quality model.
5. If every provider is over budget, select the lowest error rate, then highest
   quality.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-error-budget-shed
export NEXUS_PROVIDER_ERROR_BUDGET_RATE=0.15
```

Or per request:

```http
X-Router-Strategy: provider-error-budget-shed
```

## Tuning notes

- Lower the budget (for example `0.05`) when providers should be shed after a
  small error spike.
- Raise the budget (for example `0.25`) when transient model/provider errors are
  common and preserving top quality matters more.
- Pair with `slo-aware` when you want a strict success-rate floor, or
  `provider-health-score-blend` when latency/cost should also participate in
  scoring.

## Suggested repo metadata

- **Description:** Multi-provider LLM router with pluggable strategies, safety
  guardrails, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 catalog.
- **Topics:** `llm-router`, `litellm`, `openai`, `anthropic`, `gemini`, `routing`, `python`
