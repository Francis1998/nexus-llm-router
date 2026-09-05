# agents-api-prefer Guide

![demo](../../assets/demo/agents-api-prefer.gif)

Capability-aware routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why

OpenAI Agents API / Assistants-style agent loops and Anthropic/Gemini multi-step agent orchestration gaps.

## Signals

Metadata keys: `requires_agents_api`, `agents_api`, `openai_agents`; allowlist `agents_api_models`.

## Capability

Built-in map includes `agents_api` on frontier models where applicable. Override with `metadata.model_capabilities` or `agents_api_models`.

## Behavior

When the signal is truthy, rank healthy domain-eligible candidates by `agents_api` support, then quality, then cost. Without a signal, stay quality-first.
