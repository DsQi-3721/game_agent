"""Distil a System-One student from a dataset (LoRA).

    python scripts/s1_train.py --data data/s1/v1 --base /Users/hoi/project/models/Qwen3.5-0.8B \
        --out runs/s1_models/s1-v1 [--backend mlx|hf] [--mode auto|head|full] [--epochs 1] [--max-train 2000]

mlx (Mac): ``--mode head`` (default for Qwen3.5 hybrids) adapts the last attention block only and
runs ~1 s/example on an M4; ``--mode full`` back-propagates through the linear-attention layers
with reference ops (~17 s/example). hf (GPU servers): LoRA on every linear layer, length-bucketed batches (``--batch-size``), warm-up +
cosine schedule; ``s1_train_meta.json`` in the output holds the loss curve and per-epoch validation.
"""

from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--batch-size", type=int, default=None, help="hf: examples per forward pass (default 8)")
    ap.add_argument("--grad-accum", type=int, default=None, help="hf: forward passes per optimizer step (default 1)")
    ap.add_argument("--warmup-frac", type=float, default=None, help="hf: linear warm-up share of the steps")
    ap.add_argument("--grad-ckpt", action="store_true", help="hf: gradient checkpointing (less memory, ~30%% slower)")
    ap.add_argument("--eval-n", type=int, default=None, help="validation examples scored after every epoch")
    ap.add_argument("--eval-batch-size", type=int, default=None, help="hf: validation examples per forward pass (default 16)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None, help="hf: optimizer steps between loss lines")
    ap.add_argument("--save-every-epoch", action="store_true", help="hf: also keep <out>/epoch<k> adapters")
    ap.add_argument("--init-adapter", default=None, help="hf: continue training this adapter (e.g. the champion)")
    ap.add_argument("--eval-every", type=int, default=0, help="hf: validate and save a resumable checkpoint every N steps")
    ap.add_argument("--until-converged", action="store_true", help="hf: train until held-out action loss reaches a plateau")
    ap.add_argument("--min-epochs", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=4, help="hf: validation checks at minimum LR without meaningful improvement")
    ap.add_argument("--min-delta", type=float, default=0.002, help="hf: relative validation loss improvement")
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--bucket-window", type=int, default=8, help="hf: batches per length-sorting window")
    ap.add_argument("--resume-from", default=None, help="hf: checkpoint with adapter, optimizer, data position and RNG state")
    args = ap.parse_args()
    data = Path(args.data)
    kw = {k: v for k, v in (("lr", args.lr), ("rank", args.rank)) if v is not None}
    if args.backend == "hf":
        kw.update({k: v for k, v in (("batch_size", args.batch_size), ("grad_accum", args.grad_accum),
                                     ("warmup_frac", args.warmup_frac), ("eval_n", args.eval_n), ("eval_batch_size", args.eval_batch_size),
                                     ("seed", args.seed), ("log_every", args.log_every)) if v is not None})
        if args.grad_ckpt:
            kw["grad_ckpt"] = True
        if args.save_every_epoch:
            kw["save_every_epoch"] = True
        if args.init_adapter:
            kw["init_adapter"] = args.init_adapter
        kw.update(eval_every=args.eval_every, until_converged=args.until_converged, min_epochs=args.min_epochs,
                  patience=args.patience, min_delta=args.min_delta, min_lr_frac=args.min_lr_frac,
                  bucket_window=args.bucket_window, resume_from=args.resume_from)
    if args.backend == "mlx":
        from arena.s1.train_mlx import train
        meta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, args.out, epochs=args.epochs,
                     max_train=args.max_train, max_tokens=args.max_tokens, mode=args.mode, **kw)
    else:
        from arena.s1.train_hf import train
        meta = train(str(data / "train.jsonl"), str(data / "valid.jsonl"), args.base, args.out, epochs=args.epochs,
                     max_train=args.max_train, max_tokens=args.max_tokens, log=lambda m: print(m, flush=True), **kw)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({k: v for k, v in meta.items() if k not in ("history", "loss_curve")}, indent=1))


if __name__ == "__main__":
    main()
