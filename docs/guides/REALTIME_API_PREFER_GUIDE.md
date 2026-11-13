# realtime-api-prefer Guide

![demo](../../assets/demo/realtime-api-prefer.gif)

Capability-aware routing for GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro.

## Why

OpenAI Realtime API WebSocket streaming for low-latency audio/text conversations.

## Signals

Metadata keys: `requires_realtime`, `realtime_api`, `websocket_streaming`; allowlist `realtime_api_models`.

> **HTTP note:** strategies read `RouterRequest.metadata`. `POST /v1/chat/completions`,
> `/v1/responses`, and `/v1/messages` do not forward request-body `metadata` to routing
> yet, so over HTTP this strategy takes its no-signal path. Set `metadata` on
> `RouterRequest` when embedding `NexusRouter`, as the demo does
> (`scripts/render_strategy_demos.py`).
