# Model-Capability-Gate Routing Guide

Use `model-capability-gate` to restrict candidate models to those that
declare every capability a request needs — `vision`, `tools`,
`long_context`, or any custom capability name — before falling back to
quality-first selection for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2.

## When to use it

- Multi-modal or tool-calling workloads where routing to a model that lacks
  the required capability would fail the request outright.
- Gateways that want LiteLLM / OpenRouter-style capability filtering without
  hand-maintaining per-request model allowlists.
- Fleets that need a per-request override for capabilities that are not yet
  reflected in the built-in known-model capability map.

## How it works

1. Parse `metadata.required_capabilities` — a comma-separated string (for
   example `"vision,tools"`) or a list of capability names.
2. Filter domain-eligible candidates through provider circuit health.
3. Resolve each candidate's capability set from a per-request
   `metadata.model_capabilities` override (mapping model name to a
   comma-separated string or list) or, absent an override, the built-in
   known-model capability map.
4. Keep candidates whose capability set is a superset of every required
   capability.
5. Among remaining candidates, pick the highest quality (cost as tie-break).
6. If no candidate satisfies every required capability, emergency-retain the
   highest-quality healthy candidate so the request still routes.
7. Requests that declare no required capabilities skip the gate and route
   quality-first.

## Quick start

```bash
export NEXUS_DEFAULT_STRATEGY=model-capability-gate
```

Or select it per request:

```http
X-Router-Strategy: model-capability-gate
```

```json
{
  "metadata": {
    "required_capabilities": "vision,tools"
  }
}
```
