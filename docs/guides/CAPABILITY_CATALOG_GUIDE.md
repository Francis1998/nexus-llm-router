# Model Capability Catalog Guide

Refreshable, thread-safe catalog of model capability frozensets for
GPT-6 / Claude Opus 5.5 / Gemini 4 / Kimi K3.1 routing. Strategies can
read a live snapshot instead of a process-frozen
`_KNOWN_MODEL_CAPABILITIES` map.

![Capability catalog demo](../../assets/demo/capability-catalog.gif)

## Why

Capability-aware strategies (`model-capability-gate`, `*-prefer`, …) fall back
to a static module-level map. Shipping a new model capability historically
required a restart. `ModelCapabilityCatalog` seeds the known frontier models,
supports live `upsert`, and can `refresh_from_static()` without bouncing the
process.

## How it works

1. Construction auto-seeds `gpt-6-astra`, `claude-opus-5-5`, `gemini-3.8-flash`,
   and `kimi-k3.1-pro` (or pass a custom `seed=` / `auto_seed=False`).
2. `get(model)` returns a frozenset (or `None`).
3. `upsert(model, capabilities)` inserts or replaces an entry.
4. `refresh_from_static()` replaces the map with the built-in seed.
5. `snapshot()` returns a copy safe for strategies to hold while the shared
   catalog continues to mutate.

Pass `catalog.snapshot()` (or the catalog itself via a thin adapter) as the
`capability_map=` argument many strategies already accept.

## Example

```python
from router.capability_catalog import ModelCapabilityCatalog

catalog = ModelCapabilityCatalog()
print(catalog.get("gpt-6-astra"))  # frozenset with tools, vision, ...
# GPT-6.1 Terra supports reasoning effort `none`, so logprobs are available;
# GPT-6 Astra and GPT-6.1 Sol never run with `none` and must not list them.
catalog.upsert("gpt-6.1-terra", {"json", "logprobs", "streaming", "tools", "vision"})
capability_map = catalog.snapshot()
# hand capability_map to ModelCapabilityGateStrategy / prefer strategies
catalog.refresh_from_static()
```

## Gap vs LiteLLM / OpenRouter

| Capability | LiteLLM / OpenRouter | Nexus |
| --- | --- | --- |
| Static known-model map | Yes | `_KNOWN_MODEL_CAPABILITIES` |
| Live refresh without restart | Yes (provider sync) | `ModelCapabilityCatalog` |
| Thread-safe shared catalog | Yes | RLock + `snapshot()` |
| Seed frontier SKUs | Yes | GPT-6 Astra / Claude Opus 5.5 / Gemini 3.8 Flash / Kimi K3.1 Pro |
