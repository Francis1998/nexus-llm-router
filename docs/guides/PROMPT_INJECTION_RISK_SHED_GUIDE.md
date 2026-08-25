# Prompt-Injection-Risk-Shed Routing Guide

Use `prompt-injection-risk-shed` to demote high prompt-injection-risk
traffic onto lower-cost spare capacity instead of frontier models for
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 — without ever
rejecting the request.

![Prompt injection risk shed demo](../../assets/prompt-injection-risk-shed.gif)

## When to use it

- LLM gateways that already score prompt-injection risk upstream
  (Helicone / Portkey style) and want soft shedding rather than hard
  blocks.
- Fleets that prefer to spare frontier capacity for low-risk traffic
  while still serving suspicious prompts on cheaper models.
- Deployments that can attach a `[0.0, 1.0]` risk score as request
  metadata.

## How it works

1. Read `metadata.prompt_injection_risk` as a float in `[0.0, 1.0]`.
   Missing or non-numeric values default to `0.0`. Out-of-range values
   are clamped.
2. While risk stays below `NEXUS_PROMPT_INJECTION_RISK_THRESHOLD`
   (default `0.7`), route quality-first among healthy domain-eligible
   candidates.
3. Once risk is at or above the threshold, route to the lowest-cost
   healthy domain-compatible model — spare capacity — instead of
   rejecting the request.
4. Provider circuit health is always respected. The strategy never
   rejects; it only demotes routing quality.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=prompt-injection-risk-shed
export NEXUS_PROMPT_INJECTION_RISK_THRESHOLD=0.7
```

Or select it per request:

```http
X-Router-Strategy: prompt-injection-risk-shed
```

```json
{
  "metadata": {
    "prompt_injection_risk": 0.85
  }
}
```

A request reporting `0.85` against the default threshold of `0.7` is shed
to the lowest-cost healthy model for this request.
