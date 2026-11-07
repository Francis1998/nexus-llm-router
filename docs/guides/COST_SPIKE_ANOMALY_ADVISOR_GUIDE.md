# CostSpikeAnomalyAdvisor Guide

![CostSpikeAnomalyAdvisor](../../assets/demo/cost-spike-anomaly.gif)

Offline advisory cost-spike anomaly bands vs rolling baseline. Never network
I/O. Gap vs Helicone / Portkey / LiteLLM cost-anomaly routing.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `CostForecastAdvisor` and `TenantSpendQuotaGuard`.

## Usage

```python
from safety.cost_spike_anomaly import CostSpikeAnomalyAdvisor

advice = CostSpikeAnomalyAdvisor().advise(
    tenant_id="t1",
    current_spend_usd=40.0,
    baseline_spend_usd=10.0,
)
print(advice.band, advice.spike_ratio)
```
