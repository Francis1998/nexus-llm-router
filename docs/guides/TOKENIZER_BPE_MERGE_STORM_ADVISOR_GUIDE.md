# TokenizerBpeMergeStormAdvisor Guide

![TokenizerBpeMergeStormAdvisor](../../assets/demo/tokenizer-bpe-merge-storm-advisor.gif)

Offline HITL advisor for `bpe_merge_storm_rate` bands.
Gap vs vLLM/SGLang/TGI tokenizer BPE merge-storm monitors. Distinct from `TokenizerBpeMergeDriftAdvisor` / `PrefixCacheHitFloorAdvisor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.tokenizer_bpe_merge_storm import TokenizerBpeMergeStormAdvisor

advice = TokenizerBpeMergeStormAdvisor().advise(request_id="r1", bpe_merge_storm_rate=0.25)
print(advice.band)
```

## Safety

Advisory only. No network I/O. Humans decide.
