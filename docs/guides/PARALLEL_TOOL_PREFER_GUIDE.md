# parallel-tool-prefer Guide

![demo](../../assets/demo/parallel-tool-prefer.gif)

Capability-aware routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why

OpenAI parallel_tool_calls and Anthropic parallel tool use in multi-tool agent loops.

## Signals

Metadata keys: `requires_parallel_tool_calls`, `parallel_tool_calls`, `parallel_tools`; allowlist `parallel_tool_models`.

## Capability

Built-in map includes `parallel_tool_calls` on frontier models where applicable. Override with `metadata.model_capabilities` or `parallel_tool_models`.

## Behavior

When the signal is truthy, rank healthy domain-eligible candidates by `parallel_tool_calls` support, then quality, then cost. Without a signal, stay quality-first.
