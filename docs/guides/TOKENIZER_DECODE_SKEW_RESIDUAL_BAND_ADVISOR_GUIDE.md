# TokenizerDecodeSkewResidualBandAdvisor Guide

![TokenizerDecodeSkewResidualBandAdvisor](../../assets/demo/tokenizer-decode-skew-residual-band-advisor.gif)

Offline `decode_skew_ratio` band advisor.
Gap vs vLLM/SGLang/TGI tokenizer decode-skew residual monitors. Distinct from `AttentionSinkResidualBandAdvisor` / `SpeculativeAcceptanceEntropyBandAdvisor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.tokenizer_decode_skew_residual_band import TokenizerDecodeSkewResidualBandAdvisor

advice = TokenizerDecodeSkewResidualBandAdvisor().advise(request_id="r1", decode_skew_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
