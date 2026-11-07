# RegionFailoverHysteresisAdvisor Guide

![RegionFailoverHysteresisAdvisor HITL flow](../../assets/demo/region-failover-hysteresis-advisor.gif)

Offline HITL advisor. Never network I/O. Closes gaps vs LiteLLM/OpenRouter region-failover hysteresis monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ProviderRegionAffinityAdvisor` and `ProviderHealthHysteresisAdvisor`.

## Usage

```python
from safety.region_failover_hysteresis import RegionFailoverHysteresisAdvisor

advice = RegionFailoverHysteresisAdvisor().advise(request_id="r1", failover_score=0.15)
print(advice.band)
```
