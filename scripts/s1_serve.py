"""Serve a local System-One student over the TypeSafe protocol.

    python scripts/s1_serve.py --model /Users/hoi/project/models/Qwen3.5-0.8B [--adapter runs/s1_models/<id>] \
        [--backend mlx|hf] [--port 8090] [--temperature 1.0]

Then point an agent at it: ``student_url: http://127.0.0.1:8090`` (cascade first tier) or
``TYPESAFE_BASE_URL=http://127.0.0.1:8090`` (replace the teacher entirely).
"""

from __future__ import annotations

import argparse
import os

for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)

from aiohttp import web  # noqa: E402

from arena.s1.scorer import load_scorer  # noqa: E402
from arena.s1.server import make_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()
    scorer = load_scorer(args.backend, args.model, args.adapter, temperature=args.temperature, model_id=args.model_id)
    print(f"[s1_serve] {scorer.model_id} on http://{args.host}:{args.port}", flush=True)
    web.run_app(make_app(scorer), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
