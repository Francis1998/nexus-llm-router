# Provider-Error-Budget-Reset Routing Guide

Use `provider-error-budget-reset` when an unhealthy provider should be shed for
a bounded interval, then automatically restored for GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.

![Provider error budget reset demo](../../assets/provider-error-budget-reset.gif)

## When to use it

- Provider errors should trigger temporary rather than cumulative shedding.
- A fixed timer should admit a recovered provider without restarting the router.
- Circuit breakers are too coarse for the desired error-rate threshold.

## How it works

1. `ProviderErrorBudgetResetStats` tracks attempts and errors independently for
   each provider.
2. Providers whose active error rate exceeds
   `NEXUS_PROVIDER_ERROR_BUDGET_RESET_FRACTION` are removed from the primary pool.
3. The highest-quality healthy provider still within budget wins.
4. After `NEXUS_PROVIDER_ERROR_BUDGET_RESET_SECONDS`, that provider's window
   clears and it becomes eligible again.
5. If every healthy provider is temporarily shed, the provider whose reset is
   due soonest is used as a deterministic emergency fallback.

Unlike `provider-error-budget-shed`, this strategy owns an explicit timed reset
window rather than reading cumulative `SuccessStats`.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-error-budget-reset
export NEXUS_PROVIDER_ERROR_BUDGET_RESET_FRACTION=0.15
export NEXUS_PROVIDER_ERROR_BUDGET_RESET_SECONDS=60.0
```

Or per request:

```http
X-Router-Strategy: provider-error-budget-reset
```
