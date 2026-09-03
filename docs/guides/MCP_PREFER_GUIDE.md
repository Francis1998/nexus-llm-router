# mcp-prefer Guide

![demo](../../assets/demo/mcp-prefer.gif)

Capability-aware routing for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why

Model Context Protocol (MCP) tool-server routing so requests that need connected MCP servers land on Claude Sonnet 4.6 / GPT-5.5.

## Signals

Metadata keys: `requires_mcp`, `mcp_servers`, `model_context_protocol`; allowlist `mcp_models`.
