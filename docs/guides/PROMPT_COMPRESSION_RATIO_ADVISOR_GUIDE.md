        # PromptCompressionRatioAdvisor Guide

        ![PromptCompressionRatioAdvisor](../../assets/demo/prompt-compression-ratio.gif)

        Offline advisory control for LLM routing. Never performs network I/O.

        Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

        ## Usage

        ```python
from safety.prompt_compression_ratio import PromptCompressionRatioAdvisor

advice = PromptCompressionRatioAdvisor().advise(
    prompt_tokens=5_000,
    context_window=8_000,
)
print(advice.band, advice.ratio)
```

