        # AdaptiveRetryJitterAdvisor Guide

        ![AdaptiveRetryJitterAdvisor](../../assets/demo/adaptive-retry-jitter.gif)

        Offline advisory control for LLM routing. Never performs network I/O.

        Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

        ## Usage

        ```python
from safety.adaptive_retry_jitter import AdaptiveRetryJitterAdvisor

advice = AdaptiveRetryJitterAdvisor().advise(attempt=2, base_delay_ms=100.0)
print(advice.band, advice.jitter_ms)
```

