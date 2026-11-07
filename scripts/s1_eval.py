"""Evaluate a System-One backend against held-out teacher labels.

    python scripts/s1_eval.py --data data/s1/v1/valid.jsonl --model <base> [--adapter <dir>] [--backend mlx|hf]
    python scripts/s1_eval.py --data data/s1/v1/valid.jsonl --url http://127.0.0.1:8090     # a running server
    python scripts/s1_eval.py --data ... --teacher jev-1.14.0     # another TypeSafe model (needs the API key)
"""

from __future__ import annotations

import argparse
import asyncio
import json

from arena.s1.dataset import read_jsonl
from arena.s1.evaluate import evaluate


async def run(args) -> dict:
    if args.url:
        from arena.llm.typesafe import TypeSafeClient
        client = TypeSafeClient(base_url=args.url, api_key="local")
    elif args.teacher:
        from arena.llm.typesafe import TypeSafeClient
        client = TypeSafeClient(model=args.teacher)
    else:
        from arena.s1.scorer import ScorerClient, load_scorer
        kw = {"batch_size": args.batch_size, "merge": args.merge, "dtype": args.dtype} if args.backend == "hf" else {}
        client = ScorerClient(load_scorer(args.backend, args.model, args.adapter, **kw))
    res = await evaluate(client, read_jsonl(args.data), limit_requests=args.limit or None,
                         model=args.teacher or None, keep_predictions=bool(args.dump))
    if args.dump:
        with open(args.dump, "w") as f:
            for row in res.pop("predictions"):
                f.write(json.dumps(row) + "\n")
    if hasattr(client, "aclose"):
        await client.aclose()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max requests")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="write every request's predicted distributions (JSONL)")
    ap.add_argument("--batch-size", type=int, default=16, help="hf: questions per forward pass")
    ap.add_argument("--merge", action="store_true", help="hf: merge the adapter into the base weights")
    ap.add_argument("--dtype", default=None, help="hf: bfloat16 (CUDA default) | float32 | float16")
    args = ap.parse_args()
    res = asyncio.run(run(args))
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
