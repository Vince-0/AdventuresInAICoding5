# A/B decode campaign (Adventure #5 fit table)

**When:** 2026-11-22 · **Box:** WSL2 RTX 4090 24GB

## One sensor, two server modes

Same invent-prose short decode on every arm:

- prompt: binary-search paragraph (`layer_b_invent.py` / `rebench_power100`)
- cold + 3 warm · `max_tokens=151` · temp 0 · thinking off · wall tok/s
- report **warm median**

| Use case | Meaning | Server knob |
|----------|---------|-------------|
| **A invent** | Daily invent-loop draft | HQ `DFLASH_TOKENS=7`; DSpine `NMAX=7`; MTP `n-max=2`; FT = default (no draft modes) |
| **B sticky-config** | Reproduction / longer verify | HQ `DFLASH_TOKENS=15` @ same 96k; DSpine `NMAX=15` @ 65k (24GB ceiling); MTP/FT = **n/a** (no Mode B draft) |

Not prose-vs-code. Prose/code invent bakeoff stays a separate section.
