# MultimodalPayloadSizeGate Guide

![MultimodalPayloadSizeGate](../../assets/demo/multimodal-payload-size-gate.gif)

Offline advisory control for LLM routing. Never performs network I/O.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.multimodal_payload_size import MultimodalPayloadSizeGate

decision = MultimodalPayloadSizeGate().check(
    payload_bytes=2_000_000,
    max_bytes=5_000_000,
)
print(decision.band, decision.allowed)
```
