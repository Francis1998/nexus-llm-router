# Semantic-Cache TTL Affinity Routing Guide

Pin cacheable requests to providers with warm semantic-cache TTL remaining — a feature gap vs GPTCache / LiteLLM cache sticky routing for **GPT-6 Astra**, **Claude Opus 5.5**, **Gemini 4 Argon**, and **Kimi K3.1 Pro**.

![Demo](../../assets/semantic-cache-ttl-affinity.gif)

## Usage

```http
X-Routing-Strategy: semantic-cache-ttl-affinity
```

Metadata: `cacheable=true` plus `cache_ttl_remaining:<provider>` (seconds). Window via `NEXUS_SEMANTIC_CACHE_TTL_SECONDS` (default `300`).
