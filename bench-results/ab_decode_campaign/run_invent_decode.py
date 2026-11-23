#!/usr/bin/env python3
"""Use case A/B invent-prose short decode — cold+3 warm, max_tokens=151."""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, time, urllib.request
from pathlib import Path

PROMPT = (
    "Write a detailed but single-paragraph explanation of how binary search works. "
    "Do not use bullet points. Keep going until you hit the length limit."
)
OUT = Path(__file__).resolve().parent

def gpu_snapshot() -> dict:
    out = subprocess.check_output(
        ["nvidia-smi","--query-gpu=driver_version,power.limit,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()
    parts=[p.strip() for p in out.split(",")]
    keys=["driver_version","power_limit_w","power_draw_w","sm_clock_mhz","mem_clock_mhz","gpu_util_pct","mem_used_mib"]
    return dict(zip(keys, parts))

def one(base: str, model: str, key: str | None, max_tokens: int = 151) -> dict:
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
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=data, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    wall = time.perf_counter() - t0
    usage = d.get("usage") or {}
    tok = int(usage.get("completion_tokens") or 0)
    return {
        "wall_s": round(wall, 3),
        "tok": tok,
        "tok_s": round(tok / wall, 2) if wall > 0 else 0.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish": (d.get("choices") or [{}])[0].get("finish_reason"),
        "gpu_after": gpu_snapshot(),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--note", default="")
    ap.add_argument("--usecase", choices=["A","B"], required=True)
    ap.add_argument("--warm", type=int, default=3)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    key = args.key or None
    # health
    hreq = urllib.request.Request(f"{base}/health", headers={"Authorization": f"Bearer {key}"} if key else {})
    urllib.request.urlopen(hreq, timeout=30)
    model = args.model
    if not model:
        mreq = urllib.request.Request(f"{base}/v1/models", headers={"Authorization": f"Bearer {key}"} if key else {})
        with urllib.request.urlopen(mreq, timeout=30) as r:
            models = json.loads(r.read().decode()).get("data") or []
            model = models[0]["id"] if models else "local"
    runs = []
    cold = one(base, model, key)
    cold["label"] = "cold"
    runs.append(cold)
    print(f"cold: {cold['tok_s']} tok/s", flush=True)
    for i in range(args.warm):
        r = one(base, model, key)
        r["label"] = "warm"
        r["run"] = i + 1
        runs.append(r)
        print(f"warm{i+1}: {r['tok_s']} tok/s ({r['tok']} tok / {r['wall_s']}s)", flush=True)
    warm = [r["tok_s"] for r in runs if r["label"] == "warm"]
    summary = {
        "tag": args.tag,
        "usecase": args.usecase,
        "note": args.note,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware_id": "wsl2-rtx4090-24g-27gib",
        "evidence": "ran_here",
        "base_url": base,
        "model": model,
        "protocol": "cold+N warm; max_tokens=151; temp=0; thinking=false; wall tok/s; invent prose",
        "gpu_idle": gpu_snapshot(),
        "cold_tok_s": runs[0]["tok_s"],
        "warm_tok_s": warm,
        "warm_median_tok_s": round(statistics.median(warm), 2),
        "warm_mean_tok_s": round(statistics.mean(warm), 2),
        "runs": runs,
    }
    out = OUT / f"{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("tag","usecase","warm_median_tok_s","warm_mean_tok_s","cold_tok_s")}, indent=2))
    print("wrote", out)

if __name__ == "__main__":
    main()
