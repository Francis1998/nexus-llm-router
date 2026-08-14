# Token-RPM-Ceiling Routing Guide

Use `token-rpm-ceiling` when GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic must stay below provider prompt-token-per-minute quotas.

![Token RPM ceiling demo](../../assets/token-rpm-ceiling.gif)

## When to use it

- Provider quotas are expressed as tokens per minute rather than requests.
- Large prompts can exhaust quota even when request RPM remains low.
- Traffic should shed to the next quality-eligible provider before a request
  crosses the quota.

## How it works

1. Track estimated prompt tokens from completed requests per provider in a
   rolling 60-second `TokenRpmWindow`.
2. Add the current request's `prompt_tokens_estimate` to each provider total.
3. Exclude providers whose projected total would exceed
   `NEXUS_TOKEN_RPM_CEILING`.
4. Select the highest-quality remaining domain-eligible model, using estimated
   cost as a tie-break.
5. If every provider would exceed the ceiling, use the least-loaded provider so
   routing remains deterministic.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=token-rpm-ceiling
export NEXUS_TOKEN_RPM_CEILING=100000
```

Or per request:

```http
X-Router-Strategy: token-rpm-ceiling
```

The window is process-local. Multi-replica deployments should replace it with
shared quota telemetry when a provider applies one aggregate token limit.
