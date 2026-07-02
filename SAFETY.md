# Safety Controls

Nexus applies safety controls before and during provider dispatch.

## Circuit Breakers

Each provider has an independent circuit breaker:

- Opens after 3 consecutive failures.
- Stays open for 60 seconds.
- Recovers on the next allowed attempt after the recovery window.

## Request Timeouts

Provider calls are wrapped in `asyncio.wait_for` and adapter-level `httpx` timeouts. Default timeout is 30 seconds.

## Budget Guardrail

Spend is tracked by user key. Nexus rejects requests when estimated next spend would exceed `NEXUS_BUDGET_CAP_USD`.

## PII Scrubbing

PII scrubbing is disabled by default. When enabled, Nexus runs regex redaction for:

- Email addresses
- US phone numbers

Presidio can be installed with:

```bash
python -m pip install ".[pii]"
```

Then inject Presidio analyzer and anonymizer engines into `PiiScrubber` during application composition.

## Rate Limiting

Nexus uses a token bucket per API key identifier:

- Capacity: `NEXUS_RATE_LIMIT_CAPACITY`
- Refill rate: `NEXUS_RATE_LIMIT_REFILL_PER_SECOND`

The API key identifier is derived from the `Authorization` header.

## Failure Semantics

Provider failures do not immediately fail the request. Nexus records the failure, updates provider metrics, opens circuits when needed, and attempts the configured fallback chain. The request fails only when every eligible attempt fails.

### Guardrail rejections are not provider failures

Client-side guardrails are evaluated per candidate *before* dispatch:

- **Budget cap** (`BudgetExceededError`)
- **Open circuit** (`CircuitOpenError`)

When a guardrail rejects a candidate, Nexus **skips** that candidate and moves to
the next entry in the fallback chain **without** recording a provider failure or
incrementing `provider_error_rate`. This keeps a provider's circuit breaker a
signal of provider health only: a user hitting their spend cap, or a circuit that
is already open, can never accumulate spurious failures that would open (or keep
open) a circuit for otherwise-healthy traffic. Only genuine dispatch errors
(timeouts, adapter/HTTP errors) count toward the circuit breaker.
