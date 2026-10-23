"""Distil a System-One student from a dataset (LoRA).

    python scripts/s1_train.py --data data/s1/v1 --base /Users/hoi/project/models/Qwen3.5-0.8B \
        --out runs/s1_models/s1-v1 [--backend mlx|hf] [--mode auto|head|full] [--epochs 1] [--max-train 2000]

mlx (Mac): ``--mode head`` (default for Qwen3.5 hybrids) adapts the last attention block only and
runs ~1 s/example on an M4; ``--mode full`` back-propagates through the linear-attention layers
with reference ops (~17 s/example). hf (GPU servers): full LoRA on every linear layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--mode", default="auto", choices=["auto", "head", "full"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    data = Path(args.data)
    kw = {k: v for k, v in (("lr", args.lr), ("rank", args.rank)) if v is not None}
    if args.backend == "mlx":
        from arena.s1.train_mlx import train
        meta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, args.out, epochs=args.epochs,
                     max_train=args.max_train, max_tokens=args.max_tokens, mode=args.mode, **kw)
    else:
        from arena.s1.train_hf import train
        meta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, args.out, epochs=args.epochs,
                     max_train=args.max_train, max_tokens=args.max_tokens, **kw)
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
