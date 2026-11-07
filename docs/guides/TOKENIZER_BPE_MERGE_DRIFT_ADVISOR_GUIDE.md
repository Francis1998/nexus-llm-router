# TokenizerBpeMergeDriftAdvisor Guide

![TokenizerBpeMergeDriftAdvisor](../../assets/demo/tokenizer-bpe-merge-drift-advisor.gif)

Offline advisor. Never network I/O. Gap vs vLLM/TGI/HuggingFace tokenizer BPE merge-drift monitors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `TokenizerVocabularyDriftAdvisor` / `TokenizerMismatchAdvisor`.

## Usage

```python
from safety.tokenizer_bpe_merge_drift import TokenizerBpeMergeDriftAdvisor

advice = TokenizerBpeMergeDriftAdvisor().advise(request_id="r1", bpe_merge_drift_score=0.0)
print(advice.band)
```

## Safety

Advisory bands only. Humans decide.
