# Provider-Circuit-Probe Routing Guide

Use `provider-circuit-probe` when the preferred quality provider should receive
a tightly bounded recovery probe while GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic continues on healthy alternates.

![Provider circuit probe demo](../../assets/provider-circuit-probe.gif)

## When to use it

- The highest-quality provider should remain the explicit recovery target.
- An open preferred circuit must immediately shift traffic to a healthy peer.
- A half-open preferred circuit may receive only a small number of probe
  decisions before fallback.

## How it works

1. Rank domain-eligible candidates by quality, cost, then model name.
2. Keep the preferred quality leader while its provider circuit is closed.
3. If that circuit is open, actively select the highest-quality healthy
   alternate.
4. If it is half-open, allow at most
   `NEXUS_PROVIDER_CIRCUIT_PROBE_BUDGET` probes to the recovering leader.
5. Once the budget is exhausted, use the highest-quality healthy alternate.

This differs from `circuit-breaker-half-open-probe`, which prefers closed
providers globally and budgets concurrent half-open load. `provider-circuit-probe`
tracks the preferred quality leader specifically and bounds its probe decisions.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=provider-circuit-probe
export NEXUS_PROVIDER_CIRCUIT_PROBE_BUDGET=1
```

Or per request:

```http
X-Router-Strategy: provider-circuit-probe
```
