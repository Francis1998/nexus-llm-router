# Carbon-Aware Preference Routing Guide

Prefer lower carbon intensity providers for **GPT-6 Astra**, **Claude Opus 5.5**, **Gemini 4 Argon**, and **Kimi K3.1 Pro** traffic — inspired by sustainability-aware routing in modern LLM gateways.

![Demo](../../assets/carbon-aware-preference.gif)

## Usage

```http
X-Router-Strategy: carbon-aware-preference
```

Metadata: `carbon_intensity:<provider>` (gCO2eq/kWh) or `region` heuristic (`eu`/`us`/`cn`/`global`). Cap via `NEXUS_CARBON_AWARE_MAX_INTENSITY` (default `400`).

> **HTTP note:** strategies read `RouterRequest.metadata`. `POST /v1/chat/completions`,
> `/v1/responses`, and `/v1/messages` do not forward request-body `metadata` to routing
> yet, so over HTTP this strategy takes its no-signal path. Set `metadata` on
> `RouterRequest` when embedding `NexusRouter`, as the demo does
> (`scripts/render_strategy_demos.py`).
