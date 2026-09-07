# Semantic Fuzzy Cache Guide

Near-duplicate response caching for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 traffic using character-trigram Jaccard similarity — no embeddings
dependency. Closes the Portkey / LiteLLM semantic response-cache gap.

## Why

Exact-match caches miss when clients rephrase a prompt by a few characters.
Portkey and LiteLLM expose semantic caches (often embedding-backed). Nexus
adds `cache.SemanticFuzzyCache` as a lightweight trigram Jaccard layer that
works without vector DBs or embedding API calls.

## Distinct from

| Component | Role |
| --- | --- |
| `ResponseCache` | Exact SHA-256 match on model + messages + temperature |
| `semantic-cache` routing strategy | On `metadata.cache_hit`, prefer cheapest model |
| `SemanticFuzzyCache` | Near-duplicate *response* reuse via trigram similarity |

## How it works

1. Normalize chat messages to lowercase `role:content` text.
2. Build character trigrams and score Jaccard similarity against in-memory
   entries in the same `tenant|model` namespace.
3. On `get`, return the best hit at or above `threshold` (default `0.92`).
4. On `set`, store the value with TTL (`ttl_seconds`, default `300`) and evict
   oldest entries when `max_entries` (default `256`) is exceeded.

## Configuration

```bash
NEXUS_SEMANTIC_FUZZY_CACHE_ENABLED=true
NEXUS_SEMANTIC_FUZZY_CACHE_TTL_SECONDS=300.0
NEXUS_SEMANTIC_FUZZY_CACHE_THRESHOLD=0.92
NEXUS_SEMANTIC_FUZZY_CACHE_MAX_ENTRIES=256
```

## Example

```python
from cache.semantic_fuzzy_cache import SemanticFuzzyCache

cache = SemanticFuzzyCache(ttl_seconds=300.0, threshold=0.92)
messages = [{"role": "user", "content": "Summarize refund policy"}]
cache.set(messages, {"content": "30-day refunds"}, model="gpt-5.5", tenant="acme")
hit = cache.get(
    [{"role": "user", "content": "Summarize refund policy."}],
    model="gpt-5.5",
    tenant="acme",
)
assert hit is not None and hit.similarity >= 0.92
```

## Gap vs Portkey / LiteLLM

| Capability | Portkey / LiteLLM | Nexus |
| --- | --- | --- |
| Exact response cache | Yes | `ResponseCache` |
| Semantic / fuzzy response cache | Yes (often embeddings) | `SemanticFuzzyCache` (trigram Jaccard) |
| Tenant + model namespace | Yes | Yes |
| Embeddings dependency | Common | None |
| Semantic-cache *routing* | Separate | Existing `semantic-cache` strategy |
