# Provider Spend Telemetry Routing Guide

Prefer lower estimated provider spend once a soft USD threshold is crossed — a gap vs popular gateways (LiteLLM, Portkey) that already expose spend telemetry for **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, and **Kimi K2** fleets.

![Demo](../../assets/provider-spend-telemetry.gif)

## Usage

```http
X-Routing-Strategy: provider-spend-telemetry
```

Pass spend hints in request metadata as `spend:<provider>` (USD). Configure the soft threshold with `NEXUS_PROVIDER_SPEND_SOFT_USD` (default `10.0`).

## Behavior

- Under threshold: quality-first among healthy providers
- At/over threshold: lowest spend, then quality, then estimated token cost
