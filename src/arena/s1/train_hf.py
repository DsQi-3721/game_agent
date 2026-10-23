"""LoRA distillation of a System-One student with torch / transformers / peft (GPU servers).

Same data, rendering and loss as ``train_mlx`` (soft-label cross-entropy over the option
letters at the answer position). Writes a peft adapter directory that ``HFScorer`` loads.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Optional

from arena.s1.prompts import render
from arena.s1.scorer import _chat_ids, _label_ids


def train(train_file: str, valid_file: Optional[str], base_model: str, out_dir: str, *, rank: int = 16,
          lr: float = 1e-4, epochs: int = 1, max_tokens: int = 4096, max_train: Optional[int] = None,
          grad_accum: int = 8, eval_n: int = 300, seed: int = 0, device: Optional[str] = None,
          log: Callable[[str], None] = print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arena.s1.dataset import read_jsonl

    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    label_ids = torch.tensor(_label_ids(tok), device=device)

    def encode(examples):
        out = []
        for ex in examples:
            msgs, keys = render(ex["state"], ex["question"])
            ids = _chat_ids(tok, msgs)
            if len(ids) <= max_tokens and len(keys) == len(ex["target"]):
                out.append((ids, len(keys), ex["target"], float(ex.get("weight", 1.0))))
        return out

    train_ex = read_jsonl(train_file)
    rng.shuffle(train_ex)
    train_set = encode(train_ex[:max_train] if max_train else train_ex)
    valid_set = encode(read_jsonl(valid_file)[:eval_n]) if valid_file and Path(valid_file).exists() else []
    log(f"[train_hf] device={device} {len(train_set)} train, {len(valid_set)} valid")

    def label_logp(ids, n):
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float()
        return torch.log_softmax(logits[label_ids[:n]], dim=-1)

    def evaluate() -> dict:
        if not valid_set:
            return {}
        model.eval()
        ce, agree = 0.0, 0
        with torch.no_grad():
            for ids, n, target, _w in valid_set:
                lp = label_logp(ids, n)
                t = torch.tensor(target, device=device)
                ce -= float((t * lp).sum())
                agree += int(int(lp.argmax()) == int(t.argmax()))
        model.train()
        return {"valid_ce": round(ce / len(valid_set), 4), "valid_agree": round(agree / len(valid_set), 4)}

    history = [{"step": 0, **evaluate()}]
    log(f"[train_hf] before: {history[-1]}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    step, t0 = 0, time.perf_counter()
    for _epoch in range(epochs):
        rng.shuffle(train_set)
        for ids, n, target, w in train_set:
            lp = label_logp(ids, n)
            loss = -w * (torch.tensor(target, device=device) * lp).sum() / grad_accum
            loss.backward()
            step += 1
            if step % grad_accum == 0:
                opt.step()
                opt.zero_grad()
            if step % 200 == 0:
                log(f"[train_hf] step {step}/{len(train_set) * epochs} loss {float(loss) * grad_accum:.4f}")
    opt.step()
    opt.zero_grad()
    history.append({"step": step, **evaluate()})
    log(f"[train_hf] after: {history[-1]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    meta = {"base_model": base_model, "rank": rank, "trainer": "arena.s1.train_hf", "n_train": len(train_set),
            "epochs": epochs, "lr": lr, "train_seconds": round(time.perf_counter() - t0, 1), "history": history}
    (out / "s1_train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
