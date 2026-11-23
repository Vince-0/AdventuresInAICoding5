# Phase timings (TTFT / prefill / decode)

Streaming SSE. **Sticky warm prefill is cache-inflated** — compare **cold sticky prefill** and **warm invent decode**.

## Invent-code (Use case A shape; short prompt ~45 tok)

| Stack | Warm TTFT | Warm prefill tok/s | Warm decode tok/s |
|-------|----------:|-------------------:|------------------:|
| HQ A n=7 | 0.11 | 414.97 | **252.75** |
| HQ B n=15 | 0.08 | 547.32 | **256.19** |
| DSpine A n=7 | 0.11 | 411.03 | **196.43** |
| DSpine B n=15 | 0.1 | 426.93 | **219.16** |
| MTP | 0.1 | 439.8 | **100.2** |
| FT dense | 0.08 | 587.63 | **44.14** |
| FT MoE | 0.97 | 46.2 | **37.95** |

## Sticky ~25k quote (Use case B shape)

| Stack | Cold TTFT | Cold prefill tok/s | Warm decode tok/s* | Sticky pass |
|-------|----------:|-------------------:|-------------------:|------------:|
| HQ B n=15 | 13.63 | **1846.84** | 71.53 | 3/3 |
| HQ A n=7 (16k prompt**) | 7.185 | **2266.28** | 318.81 | 3/3 |
| DSpine A | 8.356 | **3012.66** | 167.64 | 3/3 |
| DSpine B | 8.636 | **2914.75** | 160.17 | 3/3 |
| MTP | 8.7 | **2893.54** | 71.3 | 3/3 |
| FT dense | 821.377 | **30.65** | 7.42 | 3/3 |
| FT MoE | 7.194 | **3498.97** | 33.73 | 3/3 |

\* sticky completion is only ~9 tokens — decode tok/s is noisy.

\*\* early HQ A sticky arm used uncalibrated target (~16k prompt). HQ B / others use ~25k.
