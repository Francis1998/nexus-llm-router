# deep-research-prefer Guide

![demo](../../assets/demo/deep-research-prefer.gif)

Capability-aware routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why

OpenAI Deep Research / o-series research agents and Anthropic long-horizon research loops.

## Signals

Metadata keys: `requires_deep_research`, `deep_research`, `o_deep_research`; allowlist `deep_research_models`.

## Capability

Built-in map includes `deep_research` on frontier models where applicable. Override with `metadata.model_capabilities` or `deep_research_models`.

## Behavior

When the signal is truthy, rank healthy domain-eligible candidates by `deep_research` support, then quality, then cost. Without a signal, stay quality-first.
