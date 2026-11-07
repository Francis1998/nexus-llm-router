# ProviderCircuitBreakerCooldownAdvisor Guide

![ProviderCircuitBreakerCooldownAdvisor HITL flow](../../assets/demo/provider-circuit-breaker-cooldown-advisor.gif)

Offline HITL advisor. Never network I/O. Closes gaps vs LiteLLM/Portkey/Helicone provider circuit-breaker cooldown monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ProviderHealthHysteresisAdvisor` and `ProviderQuotaRemainingAdvisor`.

## Usage

```python
from safety.provider_circuit_breaker_cooldown import ProviderCircuitBreakerCooldownAdvisor

advice = ProviderCircuitBreakerCooldownAdvisor().advise(request_id="r1", cooldown_ratio=0.15)
print(advice.band)
```
