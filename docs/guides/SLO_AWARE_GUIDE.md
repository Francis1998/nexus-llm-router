# SLO-Aware Routing Guide

Use the `slo-aware` strategy when you want quality-first routing that still respects a rolling provider availability SLO (error budget).

## When to use it

- A provider is soft-degraded (elevated error rate) but its circuit breaker has not opened yet.
- You want the highest-quality domain-eligible model among GPT-5.5, Claude Sonnet 4.6, Gemini 2.5, and Kimi K2 whose provider still meets the SLO.
- Cold-start traffic should still admit every provider until observations accrue.

## How it works

1. Filter to domain-eligible catalog candidates.
2. Read each provider's rolling success rate from `SuccessStats` (no observations → `1.0`).
3. Keep candidates whose success rate is `>= NEXUS_AVAILABILITY_SLO` (default `0.99`).
4. Pick the highest `quality_score` among those that meet the SLO.
5. If none meet the SLO, fall back to the highest success-rate eligible model.

The router records success on completed provider calls and failure on provider exceptions, so the window tracks live traffic.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=slo-aware
export NEXUS_AVAILABILITY_SLO=0.99
```

Or per request:

```http
X-Router-Strategy: slo-aware
```

## Demo

See the offline routing walkthrough in [`assets/demo.gif`](../../assets/demo.gif).
