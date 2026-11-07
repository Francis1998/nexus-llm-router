# TokenizerMismatchAdvisor Guide

![TokenizerMismatchAdvisor](../../assets/demo/tokenizer-mismatch-advisor.gif)

Closes vLLM / OpenRouter / LiteLLM router-vs-provider tokenizer mismatch bands gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ToolSchemaDriftGate` and `MultimodalTokenTaxAdvisor`.

## Usage

```python
from safety.tokenizer_mismatch import TokenizerMismatchAdvisor

advice = TokenizerMismatchAdvisor().advise(request_id="r1", mismatch_ratio=0.08)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
