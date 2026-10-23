"""Latency physics of the local model server (mlx_lm.server or any OpenAI-compatible
endpoint that streams), measured from the client side with a raw SSE reader.

    python scripts/bench_llm_latency.py --model qwen3.5-4b-mlx [--out runs/_bench/bench.json] [--quick]

For every request it records: send time, prompt-processing keepalives (mlx emits
``: keepalive processed/total`` while prefilling), first content token, end of
stream, and the server's usage (prompt / completion / cached tokens). From these:

  queue+prefill = last keepalive - send   (prefill throughput: uncached tokens / time)
  TTFT          = first data chunk - send
  decode rate   = (completion_tokens - 1) / (end - first data chunk)

Experiments (run with nothing else using the server):
  prefill   -- prompt length scan, uncached (a random nonce opens the system message)
  decode    -- long generations at short / long context
  cache     -- identical prompt twice; same system prefix + new user turn
  parallel  -- 1/2/3 concurrent requests (HyMem's parallel dispatch / fold extraction)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import string
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = "/Users/hoi/project/models/Qwen3.5-4B/tokenizer.json"

FILLER = ("Turn {t}: opp Garganacl used Salt Cure on your Passimian; your Passimian 100%->71%; "
          "Salt Cure on your Passimian. your Passimian used Close Combat on opp Garganacl; super effective; "
          "opp Garganacl 100%->22%; your Passimian def -1; spd -1. ")


def _nonce() -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(12))


class Prompts:
    """Build chat prompts with an (approximately) exact number of tokens."""

    def __init__(self):
        try:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(TOKENIZER)
        except Exception:  # noqa: BLE001
            self.tok = None

    def n(self, text: str) -> int:
        return len(self.tok.encode(text).ids) if self.tok else int(len(text) / 3.5)

    def filler(self, n_tokens: int) -> str:
        out, t = [], 1
        while True:
            out.append(FILLER.format(t=t))
            t += 1
            text = "".join(out)
            if self.n(text) >= n_tokens:
                break
        if self.tok:                       # trim to the exact size
            ids = self.tok.encode(text).ids[:n_tokens]
            text = self.tok.decode(ids)
        return text

    def messages(self, system_tokens: int, user_tokens: int, task: str, nonce: bool = True) -> list[dict]:
        head = f"[{_nonce()}] " if nonce else ""
        system = head + "You are a Pokemon battle assistant.\n" + (self.filler(system_tokens) if system_tokens else "")
        user = (self.filler(user_tokens) + "\n\n" if user_tokens else "") + task
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


SHORT_TASK = "Reply with one word: ok"
LONG_TASK = "Count from 1 to 400, separated by commas and spaces. Output only the numbers."


async def one_request(client: httpx.AsyncClient, url: str, model: str, messages: list[dict],
                      max_tokens: int, label: str) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "top_k": 20, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    rec: dict = {"label": label, "max_tokens": max_tokens, "t_send": time.time(), "keepalives": 0}
    t0 = time.perf_counter()
    first_ka = last_ka = first_data = None
    usage, finish, text = {}, None, []
    async with client.stream("POST", url, json=body, headers={"Authorization": "Bearer EMPTY"}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            now = time.perf_counter() - t0
            if not line:
                continue
            if line.startswith(":"):
                if "keepalive" in line:
                    rec["keepalives"] += 1
                    first_ka = first_ka if first_ka is not None else now
                    last_ka = now
                    rec["progress"] = line.split("keepalive", 1)[1].strip()
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                if piece or delta.get("tool_calls"):
                    if first_data is None:
                        first_data = now
                    text.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    end = time.perf_counter() - t0
    ptok = int(usage.get("prompt_tokens") or 0)
    ctok = int(usage.get("completion_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    rec.update(prompt_tokens=ptok, completion_tokens=ctok, cached_tokens=cached, finish_reason=finish,
               t_first_keepalive=first_ka, t_prefill_end=last_ka, ttft=first_data, total=end,
               output_head="".join(text)[:80])
    uncached = max(0, ptok - cached)
    rec["uncached_tokens"] = uncached
    if last_ka is not None and uncached > 0:
        rec["prefill_tps"] = round(uncached / max(last_ka, 1e-6), 1)
    if first_data is not None and ctok > 1 and end > first_data:
        rec["decode_tps"] = round((ctok - 1) / (end - first_data), 2)
    return rec


def _fmt(rec: dict) -> str:
    g = lambda k: rec.get(k) if rec.get(k) is not None else float("nan")  # noqa: E731
    return (f"{rec['label']:<28} prompt={rec['prompt_tokens']:>5} cached={rec['cached_tokens']:>5} "
            f"out={rec['completion_tokens']:>4} | prefill_end={g('t_prefill_end'):6.2f}s ttft={g('ttft'):6.2f}s "
            f"total={rec['total']:6.2f}s | prefill {rec.get('prefill_tps', '-')} tok/s, "
            f"decode {rec.get('decode_tps', '-')} tok/s")


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-mlx")
    ap.add_argument("--out", default=str(ROOT / "runs" / "_bench" / "bench_llm_latency.json"))
    ap.add_argument("--quick", action="store_true", help="fewer sizes / reps")
    args = ap.parse_args()
    for v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

    from arena.llm.model_registry import ModelRegistry

    spec = ModelRegistry.load().get(args.model)
    if not spec.api_base:
        raise SystemExit(f"{args.model} has no api_base (needs a local OpenAI-compatible server)")
    url = spec.api_base.rstrip("/") + "/chat/completions"
    model = spec.litellm_id.split("/", 1)[1] if "/" in spec.litellm_id else spec.litellm_id
    P = Prompts()
    results: dict = {"model": args.model, "url": url, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "experiments": {}}
    sizes = [500, 2000, 4000] if args.quick else [250, 1000, 2000, 4000, 8000]
    reps = 1 if args.quick else 2

    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        # warm-up (model load, kernels)
        await one_request(client, url, model, P.messages(0, 50, SHORT_TASK), 4, "warmup")

        print("== prefill scan (uncached)")
        pre = []
        for n in sizes:
            for r in range(reps):
                rec = await one_request(client, url, model, P.messages(0, n, SHORT_TASK), 8, f"prefill_{n}_r{r}")
                print(_fmt(rec), flush=True)
                pre.append(rec)
        results["experiments"]["prefill"] = pre

        print("== decode (long generations)")
        dec = []
        for ctx in ([1000] if args.quick else [1000, 4000]):
            for max_tok in ([128] if args.quick else [64, 256]):
                rec = await one_request(client, url, model, P.messages(0, ctx, LONG_TASK), max_tok,
                                        f"decode_ctx{ctx}_out{max_tok}")
                print(_fmt(rec), flush=True)
                dec.append(rec)
        results["experiments"]["decode"] = dec

        print("== prefix cache")
        cache = []
        msgs = P.messages(1500, 500, SHORT_TASK)
        for label in ("cache_cold", "cache_identical_again"):
            rec = await one_request(client, url, model, msgs, 8, label)
            print(_fmt(rec), flush=True)
            cache.append(rec)
        new_user = [msgs[0], {"role": "user", "content": P.filler(500)[::-1] + "\n\n" + SHORT_TASK}]
        rec = await one_request(client, url, model, new_user, 8, "cache_same_system_new_user")
        print(_fmt(rec), flush=True)
        cache.append(rec)
        results["experiments"]["cache"] = cache

        print("== concurrency (each request: ~1500-token unique prompt, 128 output tokens)")
        par = []
        for k in (1, 2, 3):
            batch = [P.messages(0, 1500, LONG_TASK) for _ in range(k)]
            t0 = time.perf_counter()
            recs = await asyncio.gather(*(one_request(client, url, model, m, 128, f"parallel_k{k}_{i}")
                                          for i, m in enumerate(batch)))
            wall = time.perf_counter() - t0
            for rec in recs:
                rec["k"] = k
                rec["batch_wall"] = round(wall, 2)
                print(_fmt(rec), flush=True)
            par.extend(recs)
        results["experiments"]["parallel"] = par

    # --- summary / fitted model -----------------------------------------------------------
    pts = [(r["uncached_tokens"], r["t_prefill_end"]) for r in pre if r.get("t_prefill_end") is not None]
    a, b = _linfit([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else (0.0, 0.0)
    decode_rates = [r["decode_tps"] for r in dec if r.get("decode_tps")]
    summary = {
        "prefill_fixed_s": round(a, 3),
        "prefill_s_per_1k_tokens": round(b * 1000, 3),
        "prefill_tps_fit": round(1 / b, 1) if b > 0 else None,
        "decode_tps_mean": round(statistics.mean(decode_rates), 2) if decode_rates else None,
        "decode_tps_by_ctx": {r["label"]: r.get("decode_tps") for r in dec},
        "cache": {r["label"]: {"cached_tokens": r["cached_tokens"], "prefill_end": r.get("t_prefill_end"),
                               "ttft": r.get("ttft")} for r in cache},
        "parallel": {},
    }
    for k in (1, 2, 3):
        rs = [r for r in par if r["k"] == k]
        if rs:
            summary["parallel"][f"k{k}"] = {
                "batch_wall_s": rs[0]["batch_wall"],
                "mean_request_s": round(statistics.mean(r["total"] for r in rs), 2),
                "mean_ttft_s": round(statistics.mean(r["ttft"] for r in rs if r.get("ttft") is not None), 2),
                "mean_decode_tps_per_request": round(statistics.mean(r["decode_tps"] for r in rs if r.get("decode_tps")), 2),
                "aggregate_decode_tps": round(sum(r["completion_tokens"] for r in rs) / rs[0]["batch_wall"], 2),
            }
    results["summary"] = summary
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n== summary\n" + json.dumps(summary, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
