"""LoRA distillation of a System-One student on mlx (Apple silicon).

Loss per example: weight * cross-entropy between the target distribution (teacher soft labels,
optionally mixed with System 2's choice) and the student's softmax over the option letters at
the answer position -- i.e. the student is trained on exactly the next-token distribution the
scorer reads. Only that position is supervised; the prompt is never a training target.

Adapters are written in mlx_lm's format (adapters.safetensors + adapter_config.json), so
``mlx_lm.load(base, adapter_path=out)`` and ``MlxScorer(base, out)`` load them directly.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

from arena.s1.prompts import render
from arena.s1.scorer import _chat_ids, _label_ids


def trailing_dense_layers(model) -> int:
    """Number of last blocks after the last linear-attention block (Qwen3.5: 1)."""
    n = 0
    for layer in reversed(model.layers):
        if getattr(layer, "is_linear", False):
            break
        n += 1
    return n


def train(train_file: str, valid_file: Optional[str], base_model: str, out_dir: str, *, rank: int = 8,
          num_layers: Optional[int] = None, scale: float = 20.0, lr: float = 2e-4, epochs: int = 1,
          max_tokens: int = 4096, max_train: Optional[int] = None, eval_n: int = 200, eval_every: int = 0,
          mode: str = "auto", seed: int = 0, log: Callable[[str], None] = print) -> dict:
    """``mode``: "full" back-propagates through every block (for Qwen3.5 that means mlx_lm's
    reference gated-delta ops: ~17 s per 800-token example on an M4 -- use a GPU server);
    "head" adapts only the trailing attention block(s) after the last linear-attention block and
    keeps the fused kernels for the frozen part (fast); "auto" picks "head" for hybrid models."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx_lm
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from arena.s1.dataset import read_jsonl

    rng = random.Random(seed)
    mx.random.seed(seed)
    model, tok = mlx_lm.load(base_model)
    model.freeze()
    hybrid = any(getattr(layer, "is_linear", False) for layer in model.layers)
    if mode == "auto":
        mode = "head" if hybrid else "full"
    if mode == "head":
        num_layers = max(1, trailing_dense_layers(model)) if hybrid else (num_layers or 4)
    elif num_layers is None:
        num_layers = 16
    lora_cfg = {"rank": rank, "scale": scale, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers, lora_cfg)
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    label_ids = _label_ids(tok)

    def encode(examples):
        out, skipped = [], 0
        for ex in examples:
            msgs, keys = render(ex["state"], ex["question"])
            ids = _chat_ids(tok, msgs)
            if len(ids) > max_tokens or len(keys) != len(ex["target"]):
                skipped += 1
                continue
            out.append((ids, len(keys), ex["target"], float(ex.get("weight", 1.0))))
        return out, skipped

    train_ex = read_jsonl(train_file)
    rng.shuffle(train_ex)
    if max_train:
        train_ex = train_ex[:max_train]
    train_set, skipped = encode(train_ex)
    valid_set, _ = encode(read_jsonl(valid_file)[:eval_n]) if valid_file and Path(valid_file).exists() else ([], 0)
    log(f"[train] {len(train_set)} train ({skipped} skipped > {max_tokens} tok), {len(valid_set)} valid, "
        f"{n_trainable / 1e6:.2f}M trainable params")

    def loss_fn(model, ids, n, target, weight):
        logits = model(ids[None])[0, -1]
        lab = logits[mx.array(label_ids[:n])].astype(mx.float32)
        return -weight * (target * (lab - mx.logsumexp(lab))).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=lr)

    def evaluate() -> dict:
        if not valid_set:
            return {}
        model.eval()                          # fused Metal kernels for the linear-attention layers
        ce, agree = 0.0, 0
        for ids, n, target, _w in valid_set:
            logits = model(mx.array(ids)[None])[0, -1]
            lab = logits[mx.array(label_ids[:n])].astype(mx.float32)
            logp = (lab - mx.logsumexp(lab)).tolist()
            ce -= sum(t * lp for t, lp in zip(target, logp))
            agree += int(max(range(n), key=lambda j: logp[j]) == max(range(n), key=lambda j: target[j]))
        if mode == "full":
            model.train()
        return {"valid_ce": round(ce / len(valid_set), 4), "valid_agree": round(agree / len(valid_set), 4)}

    # "full": training mode switches Qwen3.5's gated-delta layers from the fused Metal kernel (no
    # backward) to the differentiable reference ops. "head": everything below the adapted block is
    # frozen and constant w.r.t. the adapters, so the fused kernels stay (eval mode)
    if mode == "full":
        model.train()
    else:
        model.eval()
    log(f"[train] mode={mode}, LoRA on the last {num_layers} block(s)")
    history = [{"step": 0, **evaluate()}]
    log(f"[train] before: {history[-1]}")
    step, t0, running = 0, time.perf_counter(), []
    for epoch in range(epochs):
        rng.shuffle(train_set)
        for ids, n, target, w in train_set:
            loss, grads = loss_and_grad(model, mx.array(ids), n, mx.array(target, dtype=mx.float32), w)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            step += 1
            running.append(float(loss))
            if step % 50 == 0:
                log(f"[train] epoch {epoch} step {step}/{len(train_set) * epochs} loss "
                    f"{sum(running[-50:]) / len(running[-50:]):.4f} ({(time.perf_counter() - t0) / step:.2f} s/step)")
            if eval_every and step % eval_every == 0:
                history.append({"step": step, **evaluate()})
                log(f"[train] eval: {history[-1]}")
    final = {"step": step, **evaluate()}
    history.append(final)
    log(f"[train] after: {final}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    meta = {"num_layers": num_layers, "lora_parameters": lora_cfg, "fine_tune_type": "lora",
            "base_model": base_model, "train_file": str(train_file), "n_train": len(train_set), "epochs": epochs,
            "mode": mode,
            "lr": lr, "trainer": "arena.s1.train_mlx", "train_seconds": round(time.perf_counter() - t0, 1),
            "train_loss_last": (sum(running[-50:]) / max(1, len(running[-50:]))) if running else math.nan,
            "history": history}
    (out / "adapter_config.json").write_text(json.dumps(meta, indent=2))
    return meta
