# background-mode-prefer Guide

![demo](../../assets/demo/background-mode-prefer.gif)

Capability-aware routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why

OpenAI Background mode / Agents API long-running jobs in LiteLLM/Portkey capability routers.

## Signals

Metadata keys: `requires_background_mode`, `background_mode`, `background_agent`; allowlist `background_mode_models`.

## Capability

Built-in map includes `background_mode` on frontier models where applicable. Override with `metadata.model_capabilities` or `background_mode_models`.

## Behavior

When the signal is truthy, rank healthy domain-eligible candidates by `background_mode` support, then quality, then cost. Without a signal, stay quality-first.
