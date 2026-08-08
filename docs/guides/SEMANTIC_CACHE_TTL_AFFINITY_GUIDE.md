# Semantic-Cache TTL Affinity Routing Guide

Pin cacheable requests to providers with warm semantic-cache TTL remaining — a feature gap vs GPTCache / LiteLLM cache sticky routing for **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, and **Kimi K2**.

![Demo](../../assets/semantic-cache-ttl-affinity.gif)

## Usage

```http
X-Routing-Strategy: semantic-cache-ttl-affinity
```

Metadata: `cacheable=true` plus `cache_ttl_remaining:<provider>` (seconds). Window via `NEXUS_SEMANTIC_CACHE_TTL_SECONDS` (default `300`).
