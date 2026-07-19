# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `token-budget` routing strategy: selects the highest-quality domain-eligible model whose `context_window` can hold `prompt_tokens_estimate + max_tokens` within the request `token_budget` (`min(context_window, token_budget)`). Falls back to the largest-context eligible model when nothing fits. Catalog candidates now carry a `context_window` prior (GPT-5.5 / Claude Sonnet 4.6 at 200k, Gemini 2.5-class at 1M, Kimi K2 at 128k). See `docs/guides/TOKEN_BUDGET_GUIDE.md`.
- `epsilon-greedy` routing strategy: with probability `NEXUS_EPSILON` (default `0.1`) explores by picking uniformly among domain-eligible models via a second stable hash of `request_id`; otherwise exploits the highest-quality eligible model. Bucketing matches canary/A/B hashing so decisions stay deterministic and replayable while under-prioritized catalog entries still receive live traffic. See `docs/guides/EPSILON_GREEDY_GUIDE.md`.
- `cascade` routing strategy: routes the primary attempt to the cheapest domain-eligible model and orders the fallback chain by ascending cost, so a first-attempt failure escalates one price/capability rung at a time instead of jumping to the globally highest-quality model (as the base quality-ordered fallback does). This minimizes expected spend on the common first-attempt-succeeds path while still climbing toward stronger models on failure, with no thresholds to tune. Eligibility mirrors the sibling optimizers (domain + real-time, relaxing to domain-only then the full catalog). Requires no additional configuration.
- `round-robin` routing strategy: a provider load-balancer that distributes traffic evenly across every distinct provider offering a domain-eligible model, then routes each request to that provider's highest-quality eligible model. Where the cost/latency/quality/value strategies converge on one provider (concentrating rate-limit pressure and correlated failure), this spreads load across the pool. Balancing uses a stable hash of `request_id` rather than a mutable counter, so routing stays deterministic and replayable while distinct requests spread uniformly. Requires no additional configuration.
- `complexity-tier` routing strategy: treats the classifier `complexity_score` directly as a required quality target and selects the cheapest domain-eligible model whose `quality_score` meets it — a catalog-adaptive quality-for-cost escalation ladder with no thresholds to tune. Unlike `rule-based`/`classifier` (which branch on hard-coded model names and misroute when the catalog changes), it adapts to whatever catalog is configured, and falls back to the highest-quality eligible model when the target is unreachable.
- `latency-budget` routing strategy: the latency-domain dual of `budget-aware`, it maximizes quality subject to a hard latency ceiling. It selects the highest-quality domain-eligible model whose provider rolling p95 latency stays within `NEXUS_LATENCY_SLA_MS`, treats providers with no observations yet as within the SLA (so a cold start routes to the best model), and falls back to the lowest-p95 eligible model when nothing meets the SLA. Unlike `latency-aware` (which always minimizes latency), it trades quality for speed only when the SLA requires it.
- `canary` routing strategy: progressive delivery that routes a configurable traffic fraction (`NEXUS_CANARY_WEIGHT`) onto a canary model (`NEXUS_CANARY_MODEL`) while the rest stays on a stable model (`NEXUS_CANARY_STABLE_MODEL`), bucketed by a stable hash of `request_id`. Unlike the symmetric `ab` strategy it is health-gated: when the canary provider's circuit breaker is open, all traffic falls back to the stable model, and the fallback chain is anchored on the stable model.

### Fixed
- OpenAI adapter now sends `max_completion_tokens` for GPT-5.x models instead of the legacy `max_tokens` field. GPT-5.5 chat completions reject `max_tokens` with a 400; non-GPT-5 SKUs (for example `gpt-4.1-mini`) continue to use `max_tokens`.
- Medical and legal feature extraction now match common plurals (`patients`/`symptoms`/`treatments`, `contracts`/`clauses`/`statutes`, plus `diagnoses`). Singular-only patterns previously scored those prompts as zero domain hits and fell through to the general domain.
- Instruction-verb feature extraction now matches inflections precisely instead of using a permissive trailing `\w*`. The old pattern both over-matched unrelated words that merely start with a verb (`prove` → `proverb`, `design` → `designated`/`designation`) and, unable to model the silent-`e` drop, under-matched genuine `-ing` forms (`analyzing`/`optimizing`/`proving`/`comparing`). Both skewed `instruction_hits`, and therefore the complexity score and routing. The pattern now lists explicit inflectional endings (including British `-ise`/`-yse` spellings and consonant-doubling `debugged`/`debugging`) anchored by a trailing `\b`.
- Complexity/domain feature extraction now detects code keywords immediately followed by a newline, colon, or parenthesis. The `CODE_PATTERN` required a trailing literal space, so idiomatic snippets such as a SQL query beginning `SELECT\n` or a Python `class Foo:` scored zero `code_hits` and fell through to the `general` domain. Keywords are now bounded by `\b` on both sides, which detects the token regardless of the following character while still excluding substrings such as `subclass` and preserving the deliberately case-sensitive `SELECT` match.
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
