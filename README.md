# Adventures In AI Coding 5
## HyperQwen — two products on one dense 27B server

**I ran a patched vLLM stack that serves Qwen3.8-27B (W4A16) on one RTX 4090 as two deliberate products:** everyday chat (**Mode A**) and sticky reproduce/edit (**Mode B**). The interesting part is not a single tok/s number — it is the **Mode flip**, the **context ladder** to ≥128k, and keeping chat and reproduce honest as separate configs.

**Upstream:** [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) · [docs/wsl2-4090.md](https://github.com/syv-ai/HyperQwen/blob/main/docs/wsl2-4090.md) · lab plan: [`05_IMPLEMENTATION.md`](../05_IMPLEMENTATION.md)

**Primer (how LLMs / attention / KV work in plain language):** [AI Theory](https://github.com/Vince-0/AI_Theory)

**Hardware this run (`ran_here`):** WSL2 Ubuntu, RTX **4090 24 GB**, ~**27 GiB** host RAM, driver **610.88**, power limit **450 W** (100%), bare-metal venv (**no Docker** in this distro). API `:18020`.

---

## Key concepts

Read this first if the jargon below is new. Broader LLM / attention / KV vocabulary: **[AI Theory](https://github.com/Vince-0/AI_Theory#key-concepts)**.  
**New this adventure** (vs #3 MTP / #4 MoE-offload): block speculation, Mode A/B, lookup drafting, prefix cache, KV compression / context ladder.

### Glossary (ELI5)

| Term | ELI5 | On this adventure |
|------|------|-------------------|
| **VRAM** | Fast memory **on the GPU** — the desk where the model must sit to run quickly | **24 GB** desk. Holds weights + drafter + conversation memory (**KV**) |
| **Dense 27B** | One big brain that runs **all** its layers for every token (not a MoE specialist library) | Qwen3.8-**27B** — opposite of FreeToken’s “experts in RAM” story |
| **W4A16** | Pack the **recipe cards** (weights) into **4-bit** boxes; do the live math in **~16-bit** | Why 27B fits one 4090 without host-RAM MoE tricks |
| **KV cache** | The model’s **scratch notes** for “what did I already see in this chat?” | Longer chat / bigger sticky doc → bigger notes → less free VRAM |
| **Prefill vs decode** | **Prefill** = read the whole homework sheet once; **decode** = write the answer one word at a time | Sticky docs tax **prefill**; chat “feel” is mostly **decode** tok/s |
| **Frozen vLLM + HyperQwen** | Someone froze a known-good engine version and taped on tested patches + scripts | Install → patch → prepare → `start_qwen.sh` — not DIY patch soup |
| **Speculative decoding** | A **cheap guesser** drafts the next words; the **main model** checks them. Good guesses → many words per GPU step | Family includes #3 **MTP**; here the star is **DFlash2** |
| **DFlash2** | Guesser proposes a **block** of tokens in one go; main model **verifies** the block | `SPEC=dflash2` — shared by Mode A and B |
| **Mode A** | Short verify block (**~7** tokens) — tuned for normal chat | Daily driver on `:18020` |
| **Mode B** | Longer verify block (**~15**) — tuned when the answer **copies** the prompt | Quote / edit / sticky docs; costs headroom vs Mode A |
| **Lookup drafting** | If the answer is mostly “copy from above,” fill guesses from **history** instead of inventing | Why Mode B can fly on reproduce workloads |
| **Prefix cache** | Don’t re-read the same long prompt from scratch every turn — keep the **scratch notes** | Cold turn pays; turn-2 wall collapses when the prefix hits |
| **`-fast` variant** | Extra shrink/repack of a few sensitive pieces (output head / draft bits) | Auto-picked when present; better decode path on the same card |
| **Context ladder** | Different “how we store scratch notes” settings for short / long / huge chats | **fast** (~64k) → **long** (~128k+) → **huge** (~245k). This lab wants **≥128k** |
| **KVarN** | Squeeze the **scratch notes** (KV) into fewer bits so more history fits | `CTX=huge`: ~245k window in ~**4.9 GiB** KV pin |
| **Marlin repack** | At load time, rewrite 4-bit weights into a GPU-friendly layout | Can OOM on WSL if the memory allocator is wrong — see Issues |

### ELI5 walkthrough — ideas this stack teaches

1. **Two memory wallets**  
   **Weights** (the model file) and **KV** (chat/scratch notes) both spend **VRAM**. Making the chat longer does not change the file size — it grows the notes until the desk is full.

2. **Guess then check (DFlash2)**  
   Writing every token with the full 27B is like asking a professor to dictate one letter at a time. A small assistant **guesses a phrase**; the professor **marks it**. When the assistant is often right, you finish pages faster. That is **block speculative decoding**.

3. **Same brain, two jobs (Mode A vs B)**  
   Chat invents new words → short verify (**A**). Paste-a-doc and quote it back → longer verify + **lookup** (**B**). One server, two products — not one magic tok/s.

4. **Copy can be cheaper than invent (lookup + prefix cache)**  
   If the answer is already in the prompt, the stack can **match history** and **reuse KV** for the shared prefix. Cold sticky read is still expensive; the **second** turn on the same doc is where agents win.

5. **Long context is a KV problem**  
   “Support 256k” means little if the **notes** do not fit. The **context ladder** changes how notes are stored (**int8** / **KVarN**), so **≥128k** is a real product setting here, not a marketing ceiling.

6. **Quantization is not one switch**  
   **W4A16** shrinks weights; **`-fast`** tunes hot pieces; **KVarN** shrinks KV. Three different “make it smaller” stories.

Background primer: [AI Theory](https://github.com/Vince-0/AI_Theory) (tokenize → attention/KV → dense vs MoE → serve).

---

## Why

Local AI usually collapses to one number: “how many tok/s?” That hides product design. A server that is great at **chat** is often mediocre at **reproduce a long sticky document**, and the reverse.

HyperQwen’s bet on a **24 GB** card:

1. Ship a **frozen, patched** vLLM so dense **Qwen3.8-27B** is a supported product, not a weekend science fair.
2. Expose **two modes** on the same weights: chat (**A**) vs reproduce/edit (**B**).
3. Keep **long context** (here: **≥128k**) as a first-class knob via the **CTX** ladder — not an afterthought.

This write-up is a **lab notebook**: what booted under WSL, what Mode A/B actually felt like, what ≥128k required, and what I would leave running.

### Why not just stock vLLM / llama.cpp / buy another stack?

| Approach | Idea | Gap for *this* question |
|----------|------|-------------------------|
| Stock vLLM | Serve whatever fits | You rebuild the long-ctx / DFlash / requant recipe yourself; Mode A vs B is undocumented folklore |
| llama.cpp GGUF | Quantize dense 27B into VRAM | Fine for fitted GGUF; different stack — no HyperQwen DFlash2 Mode flip or shipped KVarN ladder |
| FreeToken MoE-offload | MoE **≫ VRAM** via host-RAM experts | Different wall (RAM + PCIe). Stronger C0 on gpt-oss here; not “dense 27B Mode A/B” |
| Buy more VRAM | Bigger card holds more | Expensive; HyperQwen asks what **one 24 GB** card can do as a dual product |
| **HyperQwen** | Patched vLLM + W4A16 + DFlash2 + CTX profiles | **Two products** + **≥128k** dense serve on consumer VRAM |

---

## How

**Path chosen:** bare-metal `venv` + HyperQwen `docs/install.md` (Docker missing in this WSL guest).

1. Clone HyperQwen; `.env` with `SPEC=dflash2`, `PREFIX_CACHE=1`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `HOST=127.0.0.1`, `GPU_UTIL=0.90`.
2. `pip install vllm==0.32.0` (+ pins); `bash patches/apply.sh`.
3. Prepare weights: AutoRound **W4A16** → **`-fast`** → **DFlash2** drafter (`prepare/` scripts).
4. Install **KVarN** for `CTX=huge` (`bash kvarn/install.sh`).
5. **Boot fix (WSL / driver 610.x):**  
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128`  
   Plain `:False` OOMs mid-Marlin; `:True` dies at KV init here.
6. `bash single-user/start_qwen.sh` → `http://127.0.0.1:18020/v1` (served name `qwen3.8-27b`).
7. **Mode B:** set `DFLASH_TOKENS=15`, restart. **Mode A:** remove it, restart.
8. **Context:** set `CTX=long` or `CTX=huge` (this notebook ignores “fast-only” as the product target).

```bash
cd ~/LLM/HyperQwen
set -a && source .env && set +a
# Prefer ≥128k product:
#   CTX=huge   # ~245k KVarN (pair-programming default here)
#   CTX=long   # ~128–150k int8 KV; optional SPEC=mtp control
export PATH="$PWD/venv/bin:$PATH"
bash single-user/start_qwen.sh
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:18020/health
```

**Sizing rule:** VRAM holds **weights + drafter + KV**. Leave shipped `KV_MEM` alone on stock `-fast` unless you measured a need. Host RAM is for OS/prepare — not an MoE expert pool.

**Thinking off for sensors:** pass `chat_template_kwargs.enable_thinking=false` or short quizzes fill `max_tokens` with reasoning and `content=null`.

---

## Measuring

Two different scores — do not mix them:

| Name | Plain meaning | How to read it |
|------|---------------|----------------|
| **Speed (Layer B)** | How fast the model *writes* after warmup | **tok/s** on short decode (wall incl. TTFT). Higher = snappier typing feel. |
| **Coding quiz (C0)** | Can it write small correct Python? | Fixed prompts → extract code → run checks → **passed/total**. |
| **Mode B sticky** | Exact marker quote + prefix-cache wall | Pass/fail quote at **≥8k–24k** docs under **CTX=huge**; log `cached_tokens`. |

Artifacts under `~/LLM/HyperQwen/bench-results/` (`c1_mode_a_decode.jsonl`, `c1_mode_a_ctx_huge.json`, `c0_mode_a.json`, `c3_c4_mode_b.json`, `exp_mode_b_ctx_huge.json`, …).

**Honesty labels:** `ran_here` = this 4090/WSL box · `cited_other_hw` = upstream 3090 / docs (incl. ~381 reproduce — **not** chat).

---

## Results

### Mode A vs Mode B (this adventure)

| Role | Config | Speed / sticky | Sensor | Takeaway |
|------|--------|----------------|--------|----------|
| **Mode A daily (re-bench)** | `dflash2` · n=7 · PC=1 · **CTX=fast** · stock graphs | Warm median **175.1 tok/s** (151 tok; `rebench_mode_a_ctx_fast_pl450_stock.json`) | C0 **14/24** (`c0_mode_a_pl450_stock.json`) | After PL **450 W** + driver/fw update. Short decode draws ~**285 W** (not PL-bound). |
| **Mode A (pre-update baseline)** | same · **CTX=fast** | Warm median **115.8 tok/s** (`c1_mode_a_decode.jsonl`) | C0 **14/24** | Same Layer B protocol; **+51%** decode after update |
| **Mode A @ huge (pre-update)** | · **CTX=huge** | Warm median **~113.6 tok/s** (`c1_mode_a_ctx_huge.json`) | — | Stock KVarN pin **~4.9 GiB** / window **245,760** worked then |
| **Mode A @ huge (re-bench)** | · **CTX=huge** | **Parked** — stock pin no longer fits (~4.51 GiB free after weights vs ~4.56–4.89 needed); trimmed boots OOM in draft/KVarN warmup | — | Retune `KV_MEM`/`MAX_LEN` before claiming ≥128k again on this driver/fw |
| **Mode B reproduce** | + `DFLASH_TOKENS=15` · **CTX=huge** | Marker quote **PASS @ 8k → 24k** (pre-update; `exp_mode_b_ctx_huge.json`) | Sticky quote PASS | Not re-run this pass — needs working huge boot first |
| Upstream cite | 3090 / docs | ~127 chat / ~381 reproduce | — | `cited_other_hw` — **381 ≠ chat** |

### Compared to other hosts on this lab

| Stack | Speed | C0 | Role vs HyperQwen |
|-------|-------|----|-------------------|
| **HyperQwen Mode A** (4090, re-bench) | **~175 tok/s** @ fast | **14/24** | **Daily dense chat** on `:18020` (huge parked pending KV retune) |
| gpt-oss FreeToken (4090 re-bench) | **~54.7 tok/s** | **20/24** | Stronger quiz; MoE-offload product — pin moe/KV (auto OOM’d on 24 GB) |
| gpt-oss FreeToken (3080, #4) | ~45.5 tok/s | 19/24 | Prior card; same recipe |
| Qwen3.5-4B-MTP (#3 class) | ~130–148 tok/s | ~5–8/28 | Snappy tiny; thinner quiz |

**4090 gpt-oss note:** `--moe-cache-auto` alone OOM’d; apples-to-apples speed config used **moe=316 / kv≈4125**. See `FreeToken/bench-results/compare_3080_vs_4090_gpt-oss.md`.

### Claims vs measured (4090 here)

| Claim / expectation | Measured (`ran_here`) |
|---------------------|------------------------|
| Boots on WSL2 4090 | Yes — after `max_split_size_mb:128` |
| Mode A chat ~120-class | **Was** 115.8 @ fast / ~113.6 @ huge → **now 175.1** @ fast (re-bench). Huge pending retune |
| Dual Mode A/B on one weights tree | Yes — env flip + restart |
| Sticky reproduce with long prefix | **Mode B + CTX=huge**: quote OK through **~24k** (pre-update). Re-bench blocked on huge boot |
| Useful context **≥128k** | Still the product bar; **CTX=huge** stock pin needs retune after this driver/fw |
| Prefix cache helps agents | Yes — large `cached_tokens` + much lower turn-2 wall |
| C0 vs gpt-oss on same 4090 | HyperQwen **14/24** (unchanged) · gpt-oss **20/24** — FT wins quiz; HQ wins decode speed (gap wider after re-bench) |
| Upstream ~381 reproduce = chat | **No** — label reproduction-only |
| PL 90%→100% alone explains +51% | **No** — short decode ~285 W ≪ 405/450 W; gain tracks driver/fw (+ quiet warm boot), not PL headroom |

### What I would run day to day

| Need | Pick |
|------|------|
| Dense capable chat / coding host on 24 GB | **HyperQwen Mode A · CTX=fast** for now (`:18020`); restore **CTX=huge** after KV retune |
| Quote / edit / sticky long prefix | **HyperQwen Mode B** + **CTX=huge** (flip `DFLASH_TOKENS=15`, restart) — blocked until huge boots again |
| Higher C0 quiz / MoE-offload | FreeToken **gpt-oss** (`:1919`) — different product |
| Snappy tiny agent feel | #3-class MTP 4B — different product |

One server, two Mode configs — not one fake “best tok/s.”

### Context ladder (why ≥128k mattered)

| Profile | Rough window | KV idea | When I use it |
|---------|--------------|---------|----------------|
| `CTX=fast` | ~64k | bf16-class KV | Control / short smoke only |
| `CTX=long` | ~128–150k | int8 / fp8 KV paths | ≥128k without KVarN |
| `CTX=huge` | ~245k | **KVarN** ~4.9 GiB pin (pre-update) | Pair-programming bar — **re-bench parked**: stock pin OOMs after driver/fw; leave **CTX=fast** until retuned |

Mode B sticky work that failed under short CTX **passed through ~24k** once `CTX=huge` was live. For this experiment I am **not** interested in “it works at 8k” as the success story.

Optional control: `SPEC=mtp` + `CTX=long` → window **150k**, warm short decode **~101.5 tok/s**, 8k TTFT ~3.4 s → ~0.65 s cached (`exp_mtp_ctx_long.json`).

---

## Agents (Hermes + OpenCode / OmO harness)

Same idea as prior adventures: drive a local OpenAI-compatible server from a harness — now including a full **coding-agent** loop, not only one-shot sensors.

```text
# Direct API / Hermes / short sensors
base_url = http://127.0.0.1:18020/v1
model    = qwen3.8-27b
# Prefer chat_template_kwargs.enable_thinking=false

# OpenCode + oh-my-openagent (OmO; successor to “ompi”)
# → thinking-off proxy (injects enable_thinking=false + floors max_tokens)
base_url = http://127.0.0.1:18021/v1   # proxy → :18020
model    = hyperqwen/qwen3.8-27b       # ~/.config/opencode/opencode.json
```

- Hermes one-shot → **`PHASE7_OK`** (temporary `model.base_url` override; restored afterward).
- Mode A coding smoke (`fizzbuzz`) — pass in **~0.64 s**.
- Mode B sticky quote — pass under **CTX=huge** through **~24k** (see Results).
- **OpenCode 1.19.5 + oh-my-openagent** → HyperQwen Mode A: built playable **`~/LLM/web-tetris`** (vanilla `index.html` / `game.js` / `style.css` / `README.md`) with real tool calls (`write` / `edit` / `bash` / …).

### Practical use case — local agent harness on HyperQwen

**Goal:** use HyperQwen as the model behind an OpenCode-style agent (plan → tools → files), the way Cursor/OmO users expect — not just chat tok/s.

**Stack that worked (`ran_here`):**

| Layer | Setting |
|-------|---------|
| Serve | Mode A `SPEC=dflash2` n=7 · `CTX=fast` · `:18020` |
| Harness | OpenCode + `oh-my-openagent@latest`; agents in `~/.omo/omo.jsonc` → `hyperqwen/qwen3.8-27b` |
| Proxy | `HyperQwen/bench-results/thinking_off_proxy.py` on **`:18021`** — forces `chat_template_kwargs.enable_thinking=false` and floors `max_tokens` |
| OpenCode model limits | `context: 65536`, **`output: 32768`** (was **8192** — too small) |
| GPU concurrency | `background_task.providerConcurrency.hyperqwen: 1` |

**Findings (tool calling / output budget / context):**

1. **Thinking vs tools** — With Qwen thinking **on** and `limit.output: 8192`, a Tetris agent step ended `finish_reason=length` with **`reasoning≈8192`, `output=0`**, **zero tool calls**. Same prompt with thinking **off** produced tool use immediately. Agent loops need thinking disabled (or a huge output budget that absorbs reasoning).
2. **Output budget** — `8192` was our OpenCode `limit.output`, not a HyperQwen ceiling. Cursor docs emphasize **large context** (often **~200k+**), not an 8k completion cap; provider models behind Cursor commonly allow **tens of thousands** of output tokens per step. For Mode A **65k** total window, a practical agent floor is **`output: 16k–32k`** (we use **32768**), leaving headroom for prompt + tools + history (often already **20k–40k** input on later steps). Don’t set output equal to `max_model_len`.
3. **Tool calling works** — HyperQwen’s `qwen3_coder` path through OpenAI-compatible OpenCode completed multi-step `write`/`edit`/`bash`. Smoke with a one-file write before a big job.
4. **Context / prefix cache** — Multi-step agent turns showed large **`cache.read`** on follow-ups; prefix cache is load-bearing for harness UX. **CTX=fast (65k)** was enough for a toy web game; bigger repos still want **CTX=huge** (parked pending KV retune).
5. **Recipe** — `:18020` for chat/sensors; **`:18021` proxy** for OpenCode/OmO; `--auto` for unattended runs; keep HyperQwen concurrency at **1** on this 4090.

```bash
# Boot HyperQwen, then:
export HYPERQWEN_API_KEY="$(grep '^VLLM_API_KEY=' ~/LLM/HyperQwen/.env | cut -d= -f2)"
python3 ~/LLM/HyperQwen/bench-results/thinking_off_proxy.py &   # :18021
export PATH="$HOME/.bun/bin:$HOME/.opencode/bin:$PATH"
opencode run --dir ~/LLM/web-tetris -m hyperqwen/qwen3.8-27b --auto "…"
```

---

## Issues

| Symptom | What I did / learned |
|---------|----------------------|
| No `docker` in WSL | Bare-metal venv per `docs/install.md` |
| Marlin OOM mid-load | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128` |
| `BIND=` ignored on bare-metal | Use **`HOST=127.0.0.1`** |
| Thinking fills `max_tokens`, `content=null` | `enable_thinking: false` for sensors **and** OpenCode/OmO (proxy `:18021`) |
| OpenCode agent: `length` with `reasoning=8192`, `output=0`, no tools | Thinking burned `limit.output`; disable thinking + raise **`output` to 32k** (not 8k) |
| Mode B ≥~12k OOM on short CTX | Raise to **`CTX=huge`** (or long); sticky **~24k** then OK |
| C0 wall-tok/s ~220–250 | Ignore for speed claims — use Layer B (**175.1** re-bench @ fast) |
| FreeToken `--moe-cache-auto` on 4090 | OOM — pin moe/KV for compare |
| Abliterated W4A16 “drop-in” (leminkozey) | Heavier than `-fast`; DFlash/MTP unpack OOMs / WSL hard-crash — not Mode A substitute without `SPEC=off` + smaller KV |
| WSL OOM sometimes hard-kills the distro | Same VRAM cliff as a CUDA OOM — treat as fatal; keep headroom |
| Cold AOT + stock graphs @ PL450 hard-reset WSL twice | Quiet boot (no `nvidia-smi` spam); let torch.compile cache warm; then stock FULL/CG=64 boots clean |
| `CTX=huge` stock `KV_MEM≈4.90 GiB` after update | Weights leave ~**4.51 GiB** free; pin needs ≥~**4.56 GiB** for 245k — retune `KV_MEM`/`MAX_LEN` (trimmed attempts still OOM’d in draft/KVarN path) |

---

## Lessons

1. **Two products, one server** — Mode A (chat, n=7) vs Mode B (reproduce, n=15). Never quote upstream ~381 as chat speed.
2. **Frozen + patched beats DIY** — HyperQwen’s value is the **recipe** (vLLM pin, patches, `-fast`, DFlash2, CTX profiles), not a random checkpoint alone.
3. **Lookup + prefix cache are the agent features** — Mode B helps when answers **reuse** context; prefix cache makes turn-2 cheap after the cold sticky tax.
4. **≥128k is a product requirement here** — `CTX=long` / **`CTX=huge`** (KVarN). Short-CTX Mode B cliffs are not the final word.
5. **Pass × speed × context** — Mode A is faster than gpt-oss on this 4090 but loses C0 (**14** vs **20**/24). Pick the product for the job.
6. **WSL allocator is part of the recipe** — `max_split_size_mb:128` was load-bearing on driver 610.x.
7. **Drop-in weights are not drop-in VRAM** — abliterated AutoRound without `-fast` ate the DFlash headroom; stock HyperQwen Mode A stays the daily dense host.
8. **Agent harness ≠ chat sensor** — OpenCode/OmO needs thinking **off**, a **large** `limit.output` (**~32k** here), and often a thin proxy; 8k + thinking-on silently produces zero tools.

### How HyperQwen differs from “just vLLM”

| | **Stock vLLM (typical)** | **HyperQwen** |
|---|--------------------------|---------------|
| Version story | You chase latest / DIY patches | **Frozen 0.32.0** + tested patch set |
| Weights | Bring your own quant | Shipped **W4A16** + **`-fast`** + **DFlash2** prepare path |
| Speculation | Optional MTP / various drafters | First-class **DFlash2** + **Mode A/B** verify length |
| Long context | Manual KV dtype / length fights | Documented **CTX=fast / long / huge** (KVarN on huge) |
| WSL | Often tribal knowledge | Explicit pin-memory + allocator gotchas |

### Why not just use llama.cpp?

Often you **should** — for GGUF ecosystem, MTP on small models, or abliterated GGUF experiments (see IDEAS watchlist **M-01**, ≥128k gate).

Reach for HyperQwen when the question is: **dense Qwen3.8-27B on one 24 GB card**, OpenAI API, **Mode A chat vs Mode B reproduce**, and a supported path to **≥128k**.

### Practical takeaway

- Leave **Mode A · CTX=fast** on `:18020` for daily dense chat/coding until **CTX=huge** KV is retuned.
- For **OpenCode / OmO agents**, use **`:18021` thinking-off proxy** and **`limit.output: 32768`**.
- Flip to **Mode B** only for sticky quote/edit workloads; needs **CTX=huge** again first.
- Keep **gpt-oss FreeToken** when C0 / MoE-offload quality matters more than dense decode speed.
- Label every upstream speed claim **`cited_other_hw`** until it is `ran_here`.

---

## Reflections

HyperQwen’s consumer dense story **holds** on a 4090 + ~27 GiB WSL box: Mode A re-bench is a real daily host (**~175 tok/s** @ fast after PL450/driver/fw), Mode B remains a real second product once **CTX=huge** boots again, and the interesting work was documenting the **Mode flip** and the **≥128k ladder** — not chasing a single reproduce tok/s. The PL bump did **not** explain the decode jump (short gen sits ~285 W); treat driver/firmware + boot path as the likely levers, and treat huge’s tighter free-VRAM as a real regression to fix next.

OpenCode/OmO is now a measured path (web Tetris harness test = IDEAS **I-04** **H-C** smoke). Next harness-axis work lives in **I-04** / ROADMAP **DD-01** / **P3** (Hermes · OpenCode · OmO · [omp.sh](https://omp.sh/) · OpenClaw on fixed HyperQwen). Still open: **CTX=huge** KV retune, optional upstream `bench/run_benchmarks.sh`, whether D-LSA ever beats this stack at length (RESEARCH **R-08**).

---

## Lab status

Checklist and logs: [`05_IMPLEMENTATION.md`](../05_IMPLEMENTATION.md) · artifacts under `~/LLM/HyperQwen/bench-results/`.

**Publish:** local draft only — do not commit `.env`, weights, or API keys.
