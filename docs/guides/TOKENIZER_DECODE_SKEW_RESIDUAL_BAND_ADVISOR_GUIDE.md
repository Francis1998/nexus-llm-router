# TokenizerDecodeSkewResidualBandAdvisor Guide

![TokenizerDecodeSkewResidualBandAdvisor](../../assets/demo/tokenizer-decode-skew-residual-band-advisor.gif)

Offline `decode_skew_ratio` band advisor.
Gap vs vLLM/SGLang/TGI tokenizer decode-skew residual monitors. Distinct from `AttentionSinkResidualBandAdvisor` / `SpeculativeAcceptanceEntropyBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.tokenizer_decode_skew_residual_band import TokenizerDecodeSkewResidualBandAdvisor

advice = TokenizerDecodeSkewResidualBandAdvisor().advise(request_id="r1", decode_skew_ratio=0.3)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
