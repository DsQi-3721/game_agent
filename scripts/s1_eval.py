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
        client = ScorerClient(load_scorer(args.backend, args.model, args.adapter))
    res = await evaluate(client, read_jsonl(args.data), limit_requests=args.limit or None,
                         model=args.teacher or None)
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
    args = ap.parse_args()
    res = asyncio.run(run(args))
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
