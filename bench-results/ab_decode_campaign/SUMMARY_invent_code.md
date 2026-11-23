# Invent-code campaign (mergesort)

**Prompt:** write a complete Python mergesort + docstring + `__main__` demo.

**Protocol:** cold+3 warm · `max_tokens=1536` · temp 0 · thinking off · wall tok/s  
**Quality:** extract Python · exec · must correctly sort fixed + random int lists (3 warm runs).

| Stack | A tok/s | B tok/s | Quality A | Quality B |
|-------|--------:|--------:|-----------|-----------|
| HyperQwen DFlash2 @96k | **234** | **177** | 3/3 PASS | 3/3 PASS |
| llama.cpp MTP @180k | **91** | — | 3/3 PASS | — |
| llama.cpp DSpine | **182** (n7@96k) | **196** (n15@65k) | 3/3 PASS | 3/3 PASS |
| FreeToken dense @48k | **40** | — | 3/3 PASS | — |
| FreeToken MoE @256k | **45** | — | 3/3 PASS | — |

Samples: `*_sample.py` next to each JSON.
