# Provider-Canary-Shadow-Split Routing Guide

Use `provider-canary-shadow-split` when production traffic should stay on a
preferred provider while a deterministic tenant/request slice identifies a
secondary provider candidate for comparison. It supports controlled GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 evaluations without changing the
user-visible primary response.

## When to use it

- A provider migration needs side-by-side quality or latency measurements.
- Tenant cohorts must remain stable across repeated requests.
- Shadow traffic must skip unhealthy providers and preserve the primary route.

## How it works

1. Filter the catalog to domain-eligible, healthy candidates.
2. Select the highest-quality candidate on
   `NEXUS_PROVIDER_CANARY_PRIMARY_PROVIDER`.
3. If that provider is unavailable, use the highest-quality healthy candidate.
4. Hash `metadata.tenant_id`, a sticky/user identity, or `request_id` into a
   deterministic percentage bucket.
5. Inside `NEXUS_PROVIDER_CANARY_SHADOW_PERCENT`, identify the highest-quality
   candidate on a different provider and annotate it for downstream comparison.
6. Record process-local primary, shadow-provider, and provider-pair counts in
   `CanaryShadowSplitStats`.

The decide phase still returns one primary model. The rationale and first
fallback identify the shadow candidate for a gateway integration that performs
dual-run dispatch. When no healthy different-provider candidate exists, the
primary request remains available without a shadow.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-canary-shadow-split
export NEXUS_PROVIDER_CANARY_PRIMARY_PROVIDER=openai
export NEXUS_PROVIDER_CANARY_SHADOW_PERCENT=5.0
```

Or select it per request:

```http
X-Router-Strategy: provider-canary-shadow-split
```
