# Provider-Weight-Decay Routing Guide

Use `provider-weight-decay` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic should gradually leave failing providers and recover slowly
once they stabilize.

![Demo](../../assets/provider-weight-decay.gif)

## When to use it

- Hard circuit breakers are too coarse, but recent failures should still matter.
- You want exponential down-weighting after errors with slow recovery on success.
- Quality should remain part of the score so healthy high-quality models stay
  preferred when weights are equal.

## How it works

1. Filter the catalog to domain-eligible models.
2. Read each provider's selection weight from shared `ProviderWeightStats`
   (cold start weight is `1.0`).
3. Score candidates as `weight * quality_score`.
4. Select the highest score; weight, quality, and cost break remaining ties.
5. The engine multiplies weight by `NEXUS_PROVIDER_WEIGHT_DECAY_FACTOR` after
   failures and adds `NEXUS_PROVIDER_WEIGHT_RECOVER` after successes (capped at
   `1.0`).

Defaults: decay factor `0.5`, recover step `0.1`.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-weight-decay
export NEXUS_PROVIDER_WEIGHT_DECAY_FACTOR=0.5
export NEXUS_PROVIDER_WEIGHT_RECOVER=0.1
```

Or per request:

```http
X-Router-Strategy: provider-weight-decay
```

## Tuning notes

- Lower the decay factor to shed failing providers faster.
- Raise the recover step to re-admit recovered providers more quickly.
- Weights are local to each router process unless shared externally.
