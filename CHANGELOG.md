# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `complexity-tier` routing strategy: treats the classifier `complexity_score` directly as a required quality target and selects the cheapest domain-eligible model whose `quality_score` meets it — a catalog-adaptive quality-for-cost escalation ladder with no thresholds to tune. Unlike `rule-based`/`classifier` (which branch on hard-coded model names and misroute when the catalog changes), it adapts to whatever catalog is configured, and falls back to the highest-quality eligible model when the target is unreachable.
- `latency-budget` routing strategy: the latency-domain dual of `budget-aware`, it maximizes quality subject to a hard latency ceiling. It selects the highest-quality domain-eligible model whose provider rolling p95 latency stays within `NEXUS_LATENCY_SLA_MS`, treats providers with no observations yet as within the SLA (so a cold start routes to the best model), and falls back to the lowest-p95 eligible model when nothing meets the SLA. Unlike `latency-aware` (which always minimizes latency), it trades quality for speed only when the SLA requires it.
- `canary` routing strategy: progressive delivery that routes a configurable traffic fraction (`NEXUS_CANARY_WEIGHT`) onto a canary model (`NEXUS_CANARY_MODEL`) while the rest stays on a stable model (`NEXUS_CANARY_STABLE_MODEL`), bucketed by a stable hash of `request_id`. Unlike the symmetric `ab` strategy it is health-gated: when the canary provider's circuit breaker is open, all traffic falls back to the stable model, and the fallback chain is anchored on the stable model.

### Fixed
- OpenAI-compatible adapters (`openai`, `moonshot`) now surface a model's `refusal` message instead of returning an empty completion. When a model declines, OpenAI sets `message.content` to `null` and carries the explanation in a sibling `refusal` string; the shared `message_text` helper read only `content` and silently discarded the refusal. It now returns the refusal text when content is absent, preferring real content when both exist.
- Google Gemini adapter now skips thought-summary parts (`{"thought": true}`) when reconstructing the completion text. Gemini 2.5/3-series thinking models interleave thought parts carrying internal reasoning with the answer parts, so joining every `text` part leaked the model's reasoning into the user-facing content (the same class of bug fixed earlier for Anthropic `thinking` blocks). Only non-thought answer text is now returned.
- `nested_int` now coerces integer-valued usage counts serialized as JSON strings (for example `"12"`) instead of dropping them to the default. Some OpenAI-compatible gateways serialize usage token counts as strings, which previously zeroed token accounting, the cost estimate, and the audit record. Non-numeric strings still yield the default.

### Added (earlier this cycle)
- `sticky-session` routing strategy: consistent-hashes a request's `session_id` onto one domain-eligible model, so every turn in a session routes to the same model (stable context handling and provider prompt-cache affinity) while distinct sessions spread roughly uniformly across the eligible pool for session-level load balancing. Requires no additional configuration.
- `budget-aware` routing strategy: the dual of `cost-optimal`, it maximizes quality subject to a hard per-request cost ceiling. It selects the highest-quality domain-eligible model whose estimated cost stays within `NEXUS_REQUEST_COST_CEILING_USD`, and falls back to the cheapest eligible model when nothing fits the ceiling.
- `weighted-blend` routing strategy: selects the model maximizing a tunable composite of normalized quality, cost, and latency (min-max inverted so cheaper/faster scores higher). Weights are configurable via `NEXUS_BLEND_QUALITY_WEIGHT`, `NEXUS_BLEND_COST_WEIGHT`, and `NEXUS_BLEND_LATENCY_WEIGHT`, are normalized to sum to one, and fall back to pure quality when all are zero.
- `reliability-aware` routing strategy: routes to the highest-quality model whose provider circuit breaker is closed and orders the fallback chain healthy-providers-first, backed by a non-mutating `CircuitBreakerRegistry.is_available()` read.
- OpenAI-compatible FastAPI router API.
- Observe -> Decide -> Act routing engine.
- Rule-based, classifier, cost-optimal, latency-aware, and A/B routing strategies.
- OpenAI, Anthropic, Google Gemini, Moonshot, and mock provider adapters.
- Circuit breaker, budget guardrail, PII scrubber, and token-bucket rate limiter.
- Prometheus metrics, OpenTelemetry tracing, structlog JSON logging, and JSONL audit logs.
- Docker Compose stack with router, Prometheus, and Grafana.
- Demo GIF gallery and offline benchmark/demo script.

### Fixed
- Anthropic adapter now maps `tool`-role turns to the `user` role in the Messages API payload. The Anthropic API accepts only `user`/`assistant` turn roles (tool results are carried on a `user` turn), so forwarding a `{"role": "tool"}` entry verbatim caused the whole request to fail with a 400.
- Provider token-usage parsing (`nested_int`) now coerces integral JSON floats (for example `12.0`) to `int` instead of returning `0`. Some OpenAI-compatible gateways serialize usage counts with a decimal point, which previously zeroed token accounting, the cost estimate, and the audit record.
- Prompt code-feature detection now word-anchors its language keywords (`def`, `class`, `import`, `function`, `const`, `async`), so plain-English words that merely contain a keyword (e.g. `masterclass`, `subclass`) are no longer counted as code, which previously inflated complexity scores and biased the domain classifier toward code.
- OpenAI and Moonshot adapters now join structured `message.content` part lists (returned by OpenAI-compatible gateways such as LiteLLM, vLLM, and OpenRouter) via a shared `message_text()` helper, instead of silently returning an empty completion for the non-string shape.
- Anthropic adapter now joins all `content` text blocks (skipping non-text blocks such as `thinking`/`tool_use`) instead of reading only `content[0].text`, which returned an empty completion when a non-text block came first and truncated multi-block answers.
- Gemini adapter now concatenates all `content.parts` text segments (skipping non-text parts) instead of reading only the first, preventing silent truncation of multi-part completions.
- Restored README positioning after generated documentation overwrote project-specific content.
- Restored CI to supported Python versions and verified quality gates.
- Restored the real offline routing demo script after a placeholder benchmark overwrite.
- Removed automation artifact `DAILY_IMPROVEMENTS.md` and added repository integrity tests.
- Aligned contributor/security docs, Dockerfile production install, and full-source Bandit scan.

## [v0.1.0] — 2026-06-20

### Added
- Initial project scaffold with routing core, safety controls, observability, tests, docs, and deployment assets.
