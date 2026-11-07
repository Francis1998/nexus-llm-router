# mcp-prefer Guide

![demo](../../assets/demo/mcp-prefer.gif)

Capability-aware routing for GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro.

## Why

Model Context Protocol (MCP) tool-server routing so requests that need connected MCP servers land on Claude Opus 5.5 / GPT-6 Astra.

## Signals

Metadata keys: `requires_mcp`, `mcp_servers`, `model_context_protocol`; allowlist `mcp_models`.
