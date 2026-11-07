# EmbeddingIndexSkewAdvisor Guide

![EmbeddingIndexSkewAdvisor](../../assets/demo/embedding-index-skew-advisor.gif)

Offline advisor. Never network I/O. Gap vs vLLM/SGLang/embedding-serving index-skew monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `EmbeddingBatchSkewAdvisor` / `EmbeddingDimMismatchAdvisor`.

## Usage

```python
from safety.embedding_index_skew import EmbeddingIndexSkewAdvisor

advice = EmbeddingIndexSkewAdvisor().advise(request_id="r1", index_skew_ratio=0.0)
print(advice.band)
```

## Safety

Advisory bands only. Humans decide.
