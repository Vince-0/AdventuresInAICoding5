#!/usr/bin/env python3
"""Invent-code sensor: mergesort prompt → wall tok/s + extract/exec quality.

Prompt matches layer_b_invent_code.py. Default max_tokens=512 so a complete
function + docstring + small __main__ can finish (151 often truncates mid-body).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
PROMPT = (
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


def extract_python(text: str) -> str | None:
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if blocks:
        return max(blocks, key=len).strip()
    # Truncated replies often open a fence but never close it.
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    if "def " in text:
        # strip a leading bare fence line if present
        t = re.sub(r"^```(?:python|py)?\s*\n", "", text.strip(), flags=re.I)
        t = re.sub(r"\n```\s*$", "", t)
        return t.strip()
    return None


def quality_check(code: str | None) -> dict:
    if not code:
        return {"ok": False, "reason": "no_code_extracted"}
    # Drop __main__ block for quiet exec; keep defs
    body = re.sub(
        r"\nif\s+__name__\s*==\s*[\"']__main__[\"']\s*:.*",
        "\n",
        code,
        flags=re.S,
    )
    runner = (
        "import random, sys\n"
        + body
        + "\n"
        + """
# find a callable sorter
cands = [n for n,v in list(globals().items()) if callable(v) and n not in ('random',)]
# prefer names containing merge/sort
pref = [n for n in cands if 'merge' in n.lower() or n.lower() in ('sort','mergesort')]
name = (pref or cands or [None])[0]
if not name:
    print('FAIL no_callable'); sys.exit(2)
fn = globals()[name]
cases = [
    [],
    [1],
    [2,1],
    [3,1,2],
    [5,4,3,2,1],
    [1,2,3,4,5],
    [4,2,2,4,1],
]
for c in cases:
    got = fn(list(c))
    if got is None:
        print('FAIL returned_none', name); sys.exit(3)
    if list(got) != sorted(c):
        print('FAIL', name, c, '->', got); sys.exit(4)
# random
for _ in range(20):
    c = [random.randint(-50,50) for _ in range(random.randint(0,40))]
    got = fn(list(c))
    if list(got) != sorted(c):
        print('FAIL random', c, '->', got); sys.exit(5)
print('PASS', name)
"""
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(runner, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys_executable(), str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout"}
        ok = p.returncode == 0 and "PASS" in (p.stdout or "")
        return {
            "ok": ok,
            "reason": "pass" if ok else (p.stdout or p.stderr or f"rc={p.returncode}")[:300],
            "stdout": (p.stdout or "")[:200],
            "stderr": (p.stderr or "")[:200],
        }


def sys_executable() -> str:
    import sys

    return sys.executable


def chat(
    base: str,
    model: str,
    key: str | None,
    max_tokens: int,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
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
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    wall = time.perf_counter() - t0
    usage = d.get("usage") or {}
    tok = int(usage.get("completion_tokens") or 0)
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    # some servers put reasoning separately; content should be the code
    code = extract_python(content)
    q = quality_check(code)
    return {
        "wall_s": round(wall, 3),
        "tok": tok,
        "tok_s": round(tok / wall, 2) if wall > 0 else 0.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish": (d.get("choices") or [{}])[0].get("finish_reason"),
        "content": content,
        "code_extracted": bool(code),
        "code_chars": len(code or ""),
        "quality_ok": q["ok"],
        "quality_reason": q.get("reason"),
        "gpu_after": gpu_snapshot(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--usecase", choices=["A", "B"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--note", default="")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    key = args.key or None

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # health / models
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

    runs = []
    cold = chat(base, model, key, args.max_tokens)
    cold["label"] = "cold"
    runs.append(cold)
    print(
        f"cold: {cold['tok_s']} tok/s  quality={'PASS' if cold['quality_ok'] else 'FAIL'}  finish={cold['finish']}",
        flush=True,
    )
    for i in range(args.warm):
        r = chat(base, model, key, args.max_tokens)
        r["label"] = "warm"
        r["run"] = i + 1
        runs.append(r)
        print(
            f"warm{i+1}: {r['tok_s']} tok/s  quality={'PASS' if r['quality_ok'] else 'FAIL'}  finish={r['finish']}",
            flush=True,
        )

    warm = [r for r in runs if r["label"] == "warm"]
    rates = [r["tok_s"] for r in warm]
    q_ok = sum(1 for r in warm if r["quality_ok"])
    # persist one sample of code for inspection (best warm)
    best = max(warm, key=lambda r: (r["quality_ok"], r["tok_s"]))
    sample_path = OUT / f"{args.tag}_sample.py"
    code = extract_python(best["content"])
    if code:
        sample_path.write_text(code + "\n", encoding="utf-8")

    slim_runs = []
    for r in runs:
        row = {k: v for k, v in r.items() if k != "content"}
        row["content_chars"] = len(r.get("content") or "")
        slim_runs.append(row)

    summary = {
        "tag": args.tag,
        "usecase": args.usecase,
        "note": args.note,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware_id": "wsl2-rtx4090-24g-27gib",
        "evidence": "ran_here",
        "base_url": base,
        "model": model,
        "prompt": "code_mergesort",
        "prompt_text": PROMPT,
        "max_tokens": args.max_tokens,
        "protocol": f"cold+N warm; max_tokens={args.max_tokens}; temp=0; thinking=false; wall tok/s + extract/exec mergesort",
        "gpu_idle": gpu_snapshot(),
        "cold_tok_s": runs[0]["tok_s"],
        "warm_tok_s": rates,
        "warm_median_tok_s": round(statistics.median(rates), 2),
        "warm_mean_tok_s": round(statistics.mean(rates), 2),
        "quality_warm_pass": q_ok,
        "quality_warm_total": len(warm),
        "quality_warm_pass_rate": round(q_ok / len(warm), 3) if warm else 0.0,
        "sample_path": str(sample_path) if code else None,
        "runs": slim_runs,
    }

    out = OUT / f"{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "tag",
                    "usecase",
                    "warm_median_tok_s",
                    "quality_warm_pass",
                    "quality_warm_total",
                    "quality_warm_pass_rate",
                    "max_tokens",
                )
            },
            indent=2,
        )
    )
    print("wrote", out)
    if summary["sample_path"]:
        print("sample", summary["sample_path"])


if __name__ == "__main__":
    main()
