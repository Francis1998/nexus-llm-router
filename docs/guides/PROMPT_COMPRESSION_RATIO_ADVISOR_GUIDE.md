        # PromptCompressionRatioAdvisor Guide

        ![PromptCompressionRatioAdvisor](../../assets/demo/prompt-compression-ratio.gif)

        Offline advisory control for LLM routing. Never performs network I/O.

        Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

        ## Usage

        ```python
from safety.prompt_compression_ratio import PromptCompressionRatioAdvisor

advice = PromptCompressionRatioAdvisor().advise(
    prompt_tokens=5_000,
    context_window=8_000,
)
print(advice.band, advice.ratio)
```

