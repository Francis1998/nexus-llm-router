# logprobs-prefer Guide

![demo](../../assets/demo/logprobs-prefer.gif)

Capability-aware routing for GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro.

## Why

OpenAI logprobs for confidence scoring, calibration, and eval pipelines.

## Signals

Metadata keys: `requires_logprobs`, `logprobs`, `top_logprobs`; allowlist `logprobs_models`.

> **HTTP note:** strategies read `RouterRequest.metadata`. `POST /v1/chat/completions`,
> `/v1/responses`, and `/v1/messages` do not forward request-body `metadata` to routing
> yet, so over HTTP this strategy takes its no-signal path. Set `metadata` on
> `RouterRequest` when embedding `NexusRouter`, as the demo does
> (`scripts/render_strategy_demos.py`).
