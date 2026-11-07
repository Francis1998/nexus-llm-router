# QueueWaitSloAdvisor Guide

![QueueWaitSloAdvisor](../../assets/demo/queue-wait-slo.gif)

Offline advisory queue-wait SLO bands. Closes OpenRouter / LiteLLM / Portkey
queue-wait gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `RequestPriorityAgingAdvisor`.

## Usage

```python
from safety.queue_wait_slo import QueueWaitSloAdvisor

advice = QueueWaitSloAdvisor().advise(request_id="r1", wait_s=3.5, slo_s=5.0)
print(advice.band, advice.ratio)
```
