#!/usr/bin/env python3
"""Phase timings via SSE streaming: TTFT, prefill tok/s, decode tok/s.

Definitions (single-user, one request):
  TTFT          = time to first streamed content token
  prefill_tok/s = prompt_tokens / TTFT
  decode_tok/s  = (completion_tokens - 1) / (wall - TTFT)   # tokens after the first
  e2e_tok/s     = completion_tokens / wall                  # old wall metric

Jobs:
  invent_code — mergesort prompt (Use case A shape)
  sticky_quote — ~24k-target filler doc + quote marker (Use case B shape)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
MARKER = "UNIQUE_MARKER_QZ9_MODE_B"
CHARS_PER_TOK = 4

INVENT_CODE_PROMPT = (
    "Write a complete Python function that implements mergesort on a list of ints. "
    "Include docstring and a small __main__ demo. Keep going until the length limit."
)


def gpu_snapshot() -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,power.limit,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    keys = [
        "driver_version",
        "power_limit_w",
        "power_draw_w",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "gpu_util_pct",
        "mem_used_mib",
    ]
    return dict(zip(keys, parts))


def build_sticky_doc(target_tokens: int) -> str:
    unit = (
        "The laboratory notebook records iterative measurements of speculative decoding "
        "acceptance rates under Mode B. Paragraph index {i}. "
        "Context stickiness matters for reproduction tasks that quote earlier sentences.\n"
    )
    parts, i = [], 0
    budget = target_tokens * CHARS_PER_TOK
    while sum(len(p) for p in parts) < budget - 80:
        parts.append(unit.format(i=i))
        i += 1
    body = "".join(parts)
    return (
        f"DOCUMENT START\n{body}\n"
        f"NEAR THE END: the secret phrase is {MARKER}.\nDOCUMENT END\n\n"
        f"Quote the secret phrase from the document exactly, nothing else."
    )


def stream_chat(
    base: str,
    model: str,
    key: str | None,
    prompt: str,
    max_tokens: int,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=data, headers=headers
    )

    t0 = time.perf_counter()
    t_first = None
    pieces: list[str] = []
    usage = {}
    finish = None
    # Some servers omit stream_options; still parse SSE.
    with urllib.request.urlopen(req, timeout=900) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith(":"):  # comment / keepalive
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch0 = choices[0]
            finish = ch0.get("finish_reason") or finish
            delta = ch0.get("delta") or {}
            # content token(s)
            piece = delta.get("content")
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                pieces.append(piece)
            # some stacks put text in message on non-delta streams
            msg = ch0.get("message") or {}
            if msg.get("content") and t_first is None:
                t_first = time.perf_counter()
                pieces.append(msg["content"])
    t_end = time.perf_counter()

    content = "".join(pieces)
    wall = t_end - t0
    ttft = (t_first - t0) if t_first is not None else None
    decode_s = (t_end - t_first) if t_first is not None else None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        # rough fallback: whitespace split (not used when usage present)
        completion_tokens = max(1, len(content.split())) if content else 0

    prefill_tps = (
        round(prompt_tokens / ttft, 2)
        if prompt_tokens and ttft and ttft > 0
        else None
    )
    # tokens after the first streamed piece
    decode_tok = max(0, int(completion_tokens) - 1) if completion_tokens else 0
    decode_tps = (
        round(decode_tok / decode_s, 2)
        if decode_s and decode_s > 0 and decode_tok > 0
        else None
    )
    e2e_tps = (
        round(int(completion_tokens) / wall, 2)
        if completion_tokens and wall > 0
        else None
    )

    return {
        "wall_s": round(wall, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "decode_s": round(decode_s, 3) if decode_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tok_s": prefill_tps,
        "decode_tok_s": decode_tps,
        "e2e_tok_s": e2e_tps,
        "finish": finish,
        "content_prefix": content[:200],
        "sticky_ok": (MARKER in content) if MARKER in (prompt or "") else None,
        "gpu_after": gpu_snapshot(),
    }


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--usecase", choices=["A", "B"], required=True)
    ap.add_argument("--job", choices=["invent_code", "sticky_quote"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--note", default="")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--sticky-target", type=int, default=24576)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    key = args.key or None
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/health", headers=headers), timeout=30
        )
    except Exception:
        urllib.request.urlopen(
            urllib.request.Request(f"{base}/v1/models", headers=headers), timeout=30
        )

    model = args.model
    if not model:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base}/v1/models", headers=headers), timeout=30
        ) as r:
            models = json.loads(r.read().decode()).get("data") or []
            model = models[0]["id"] if models else "local"

    if args.job == "invent_code":
        prompt = INVENT_CODE_PROMPT
        max_tokens = args.max_tokens or 1536
    else:
        prompt = build_sticky_doc(args.sticky_target)
        max_tokens = args.max_tokens or 64

    runs = []
    cold = stream_chat(base, model, key, prompt, max_tokens)
    cold["label"] = "cold"
    runs.append(cold)
    print(
        f"cold: TTFT={cold['ttft_s']}s prefill={cold['prefill_tok_s']} "
        f"decode={cold['decode_tok_s']} e2e={cold['e2e_tok_s']} "
        f"prompt={cold['prompt_tokens']} compl={cold['completion_tokens']}",
        flush=True,
    )
    for i in range(args.warm):
        r = stream_chat(base, model, key, prompt, max_tokens)
        r["label"] = "warm"
        r["run"] = i + 1
        runs.append(r)
        print(
            f"warm{i+1}: TTFT={r['ttft_s']}s prefill={r['prefill_tok_s']} "
            f"decode={r['decode_tok_s']} e2e={r['e2e_tok_s']} "
            f"prompt={r['prompt_tokens']} compl={r['completion_tokens']} "
            f"sticky_ok={r['sticky_ok']}",
            flush=True,
        )

    warm = [r for r in runs if r["label"] == "warm"]
    summary = {
        "tag": args.tag,
        "usecase": args.usecase,
        "job": args.job,
        "note": args.note,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware_id": "wsl2-rtx4090-24g-27gib",
        "evidence": "ran_here",
        "base_url": base,
        "model": model,
        "max_tokens": max_tokens,
        "sticky_target": args.sticky_target if args.job == "sticky_quote" else None,
        "definitions": {
            "TTFT": "time to first streamed content token",
            "prefill_tok_s": "prompt_tokens / TTFT",
            "decode_tok_s": "(completion_tokens - 1) / (wall - TTFT)",
            "e2e_tok_s": "completion_tokens / wall",
        },
        "gpu_idle": gpu_snapshot(),
        "warm_median_ttft_s": median([r["ttft_s"] for r in warm]),
        "warm_median_prefill_tok_s": median([r["prefill_tok_s"] for r in warm]),
        "warm_median_decode_tok_s": median([r["decode_tok_s"] for r in warm]),
        "warm_median_e2e_tok_s": median([r["e2e_tok_s"] for r in warm]),
        "warm_median_prompt_tokens": median([r["prompt_tokens"] for r in warm]),
        "sticky_warm_pass": sum(1 for r in warm if r.get("sticky_ok"))
        if args.job == "sticky_quote"
        else None,
        "sticky_warm_total": len(warm) if args.job == "sticky_quote" else None,
        "runs": runs,
    }
    out = OUT / f"{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "tag",
                    "job",
                    "usecase",
                    "warm_median_ttft_s",
                    "warm_median_prefill_tok_s",
                    "warm_median_decode_tok_s",
                    "warm_median_e2e_tok_s",
                    "warm_median_prompt_tokens",
                    "sticky_warm_pass",
                )
            },
            indent=2,
        )
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
