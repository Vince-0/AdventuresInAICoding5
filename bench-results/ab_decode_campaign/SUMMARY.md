# A/B invent-prose decode campaign

**When:** 2026-11-22 · **Box:** WSL2 RTX 4090 24 GB  
**Sensor:** invent prose (binary-search paragraph), cold+3 warm, `max_tokens=151`, temp 0, thinking off, wall tok/s → **warm median**

| Stack | Context | A (invent draft) | B (sticky-config draft) |
|-------|---------|-----------------:|------------------------:|
| HyperQwen DFlash2 | 96k | **149.8** (n=7) | **150.3** (n=15) |
| llama.cpp MTP | 180k | **75.5** (n=2) | — (no Mode B) |
| llama.cpp MTP (window-matched) | 96k | **77.5** (n=2) | — |
| llama.cpp DSpine | 96k / 65k | **79.8** (n=7) | **66.3** (n=15 @65k) |
| FreeToken dense NVFP4 | 48k | **38.1** | — |
| FreeToken MoE A3B NVFP4 | 256k | **29.5** | — |

**Read:** On invent-prose, HQ A≈B (~150). DSpine B (n=15) is *slower* than A on this sensor and needs a smaller window. MTP/FT have no Mode B draft knob.

JSON: `SUMMARY_ab_decode.json`, per-arm `*_A_*.json` / `*_B_*.json`.
