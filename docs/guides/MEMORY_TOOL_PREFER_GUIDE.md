# memory-tool-prefer Guide

![demo](../../assets/demo/memory-tool-prefer.gif)

Capability-aware routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why

Anthropic memory tool / OpenAI Agents API persistent memory gaps.

## Signals

Metadata keys: `requires_memory_tool`, `memory_tool`, `agent_memory`; allowlist `memory_tool_models`.

## Capability

Built-in map includes `memory_tool` on frontier models where applicable. Override with `metadata.model_capabilities` or `memory_tool_models`.

## Behavior

When the signal is truthy, rank healthy domain-eligible candidates by `memory_tool` support, then quality, then cost. Without a signal, stay quality-first.
