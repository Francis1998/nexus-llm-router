        # ResponseCacheTtlAdvisor Guide

        ![ResponseCacheTtlAdvisor](../../assets/demo/response-cache-ttl.gif)

        Offline advisory control for LLM routing. Never performs network I/O.

        Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

        ## Usage

        ```python
from safety.response_cache_ttl import ResponseCacheTtlAdvisor

advice = ResponseCacheTtlAdvisor().advise(request_class="stable")
print(advice.band, advice.ttl_seconds)
```

