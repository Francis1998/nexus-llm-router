# DisaggKvTransferStallBandAdvisor Guide

![DisaggKvTransferStallBandAdvisor](../../assets/demo/disagg-kv-transfer-stall-band-advisor.gif)

Safety band advisor for `stall_ratio`.
Gap vs vLLM/SGLang/TGI disaggregated KV-transfer stall monitors. Distinct from `DisaggPrefillDecodeSkewBandAdvisor` / `KvCacheOffloadThrashBandAdvisor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.disagg_kv_transfer_stall_band import DisaggKvTransferStallBandAdvisor

advice = DisaggKvTransferStallBandAdvisor().advise(request_id="req-1", stall_ratio=0.15)
print(advice.band)
```

## Safety

Advisory only. No network I/O.
