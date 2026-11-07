# TokenizerVocabularyDriftAdvisor Guide

![TokenizerVocabularyDriftAdvisor flow](../../assets/demo/tokenizer-vocabulary-drift-advisor.gif)

Offline HITL advisor. Never network I/O.

Gap vs vLLM/HuggingFace/OpenAI tokenizer vocabulary-drift advisors. Distinct from ``TokenizerMismatchAdvisor`` and ``EmbeddingDimMismatchAdvisor``.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.tokenizer_vocabulary_drift import TokenizerVocabularyDriftAdvisor

advice = TokenizerVocabularyDriftAdvisor().advise(request_id="r1", drift_ratio=0.02)
print(advice.band)
```

## Safety

Advisory only. Humans decide routing actions.
