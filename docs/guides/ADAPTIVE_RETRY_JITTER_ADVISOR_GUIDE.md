        # AdaptiveRetryJitterAdvisor Guide

        ![AdaptiveRetryJitterAdvisor](../../assets/demo/adaptive-retry-jitter.gif)

        Offline advisory control for LLM routing. Never performs network I/O.

        Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

        ## Usage

        ```python
from safety.adaptive_retry_jitter import AdaptiveRetryJitterAdvisor

advice = AdaptiveRetryJitterAdvisor().advise(attempt=2, base_delay_ms=100.0)
print(advice.band, advice.jitter_ms)
```

