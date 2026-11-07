"""LoRA distillation of a System-One student with torch / transformers / peft (GPU servers).

Weighted soft-label cross-entropy and optional pairwise action ranking over the option letters
at the answer position. Outcome and resource prediction share the same typed-question interface.
Examples are trained in right-padded, length-bucketed batches through ``scorer.label_logits`` --
the computation ``HFScorer`` serves with, so training and serving cannot drift apart, and only the
label rows of the output head are ever computed. AdamW uses linear warm-up followed by cosine
decay or validation-driven learning-rate reductions. Training supports distributed batches,
periodic validation, resumable checkpoints, and exporting the best validation adapter.

Measured on an A800 (Qwen3.5-0.8B, ~1k-token prompts): one example per forward pass costs a flat
~0.28 s forward+backward whatever its length -- kernel-launch bound -- so batching is nearly free.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from datetime import datetime
from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

from arena.s1.prompts import render
from arena.s1.scorer import _chat_ids_batch, _label_ids, label_logits


def length_batches(lengths: list[int], batch_size: int, rng: random.Random, window: int = 8) -> list[list[int]]:
    """Every index exactly once: shuffle, sort by length inside windows of ``window`` batches
    (little padding per batch), then shuffle the batch order."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    span = batch_size * window
    out: list[list[int]] = []
    for s in range(0, len(idx), span):
        w = sorted(idx[s:s + span], key=lambda i: lengths[i])
        out += [w[j:j + batch_size] for j in range(0, len(w), batch_size)]
    rng.shuffle(out)
    return out


def lr_factor(step: int, total: int, warmup: int, min_frac: float) -> float:
    """Linear warm-up over ``warmup`` optimizer steps, then cosine decay to ``min_frac``."""
    if step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def soft_ce(logits, batch):
    """(B, L) label logits + [(ids, n_options, target, weight)] -> (per-example CE, log-probs over the n options)."""
    import torch

    n = torch.tensor([b[1] for b in batch], device=logits.device)
    valid = torch.arange(logits.shape[1], device=logits.device)[None, :] < n[:, None]
    logp = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1).masked_fill(~valid, 0.0)
    target = torch.tensor([b[2] + [0.0] * (logits.shape[1] - b[1]) for b in batch],
                          dtype=logp.dtype, device=logp.device)
    return -(target * logp).sum(-1), logp


def objective_loss(logits, batch):
    """Soft-label CE, or logistic preference loss when the row carries a preferred/rejected pair."""
    import torch
    from torch.nn.functional import softplus

    ce, logp = soft_ce(logits, batch)
    pairs = [row[4] if len(row) > 4 else None for row in batch]
    if not any(pairs):
        return ce, logp
    indices = torch.tensor([[pair["preferred"], pair["rejected"]] if pair else [0, 0]
                            for pair in pairs], device=logp.device)
    scores = logp.gather(1, indices)
    mask = torch.tensor([pair is not None for pair in pairs], device=logp.device)
    return torch.where(mask, softplus(scores[:, 1] - scores[:, 0]), ce), logp


def train(train_file: str, valid_file: Optional[str], base_model: str, out_dir: str, *, rank: int = 16,
          alpha: Optional[float] = None, lr: float = 1e-4, epochs: int = 1, max_tokens: int = 4096,
          max_train: Optional[int] = None, batch_size: int = 8, grad_accum: int = 1, warmup_frac: float = 0.05,
          min_lr_frac: float = 0.1, weight_decay: float = 0.0, max_grad_norm: float = 1.0, grad_ckpt: bool = False,
          eval_n: int = 300, eval_batch_size: int = 16, log_every: int = 10, seed: int = 0,
          save_every_epoch: bool = False, init_adapter: Optional[str] = None, device: Optional[str] = None,
          eval_every: int = 0, until_converged: bool = False, min_epochs: float = 2.0,
          patience: int = 4, min_delta: float = 0.002, resume_from: Optional[str] = None,
          bucket_window: int = 8, action_only: bool = False, permute_each_visit: bool = False,
          max_epochs: Optional[int] = None, event_callback=None, family_mass: Optional[dict] = None,
          log: Callable[[str], None] = print) -> dict:
    """``save_every_epoch`` also keeps ``<out>/epoch<k>`` adapters (the final one is ``<out>`` itself).
    ``init_adapter`` continues training an existing adapter (e.g. the champion's) instead of a fresh
    LoRA; its rank / targets then come from that adapter."""
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from arena.s1.dataset import read_jsonl

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank_id, world_size, local_rank = 0, 1, 0
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank_id, world_size = dist.get_rank(), dist.get_world_size()
        device = "cuda"
    if rank_id:
        log = lambda _: None
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    if init_adapter or resume_from:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_from or init_adapter, is_trainable=True)
        rank = model.peft_config["default"].r
        alpha = model.peft_config["default"].lora_alpha
    else:
        model = get_peft_model(model, lora)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    label_ids = _label_ids(tok)

    class LabelForward(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, sequences):
            return label_logits(self.policy, sequences, label_ids, pad_id)

    forward = LabelForward(model)
    if distributed:
        forward = torch.nn.parallel.DistributedDataParallel(forward, device_ids=[local_rank],
                                                           broadcast_buffers=False)

    def encode(examples, quiet=False):
        out, skipped = [], 0
        for start in range(0, len(examples), 256):
            chunk = examples[start:start + 256]
            rendered = [render(ex["state"], ex["question"]) for ex in chunk]
            encoded = _chat_ids_batch(tok, [messages for messages, _ in rendered])
            for ex, (_, keys), ids in zip(chunk, rendered, encoded):
                if len(ids) > max_tokens:
                    raise ValueError(f"{ex.get('id', 'example')} requires {len(ids)} tokens; "
                                     f"max_tokens={max_tokens} would discard its policy input")
                if len(keys) != len(ex["target"]):
                    skipped += 1
                    continue
                if ex.get("keys", keys) != keys:
                    raise ValueError(f"option order differs for {ex.get('id')}")
                pair = ex.get("preference")
                if pair and not (0 <= pair["preferred"] < len(keys) and 0 <= pair["rejected"] < len(keys)
                                 and pair["preferred"] != pair["rejected"]):
                    raise ValueError(f"invalid action preference for {ex.get('id')}")
                out.append((ids, len(keys), ex["target"], float(ex.get("weight", 1.0)), pair, ex.get("purpose", "unknown")))
            completed = start + len(chunk)
            if not quiet and (completed % 16384 == 0 or completed == len(examples)):
                log(f"[train_hf] encoded {completed}/{len(examples)} examples")
        return out, skipped

    train_ex = read_jsonl(train_file)
    rng.shuffle(train_ex)
    train_ex = train_ex[:max_train] if max_train else train_ex
    train_set, skipped = encode(train_ex)
    valid_examples = read_jsonl(valid_file) if valid_file and Path(valid_file).exists() else []
    rng.shuffle(valid_examples)
    valid_examples = valid_examples[:eval_n] if eval_n else valid_examples
    valid_set = encode(valid_examples)[0]
    if action_only:
        for ex in train_ex + valid_examples:
            if ex['purpose'] != 'action' or ex.get('preference') or ex['question']['type'] != 'choice':
                raise ValueError('Action-only training requires full legal-action choice targets')
        if skipped or len(valid_set) != len(valid_examples):
            raise ValueError('Action supervision cannot silently skip malformed targets')
    ordering = sorted(range(len(valid_set)), key=lambda i: len(valid_set[i][0]))
    valid_set = [valid_set[i] for i in ordering]
    valid_examples = [valid_examples[i] for i in ordering]
    if not train_set:
        raise ValueError(f"no usable training examples in {train_file}")
    if until_converged and (not valid_set or eval_every <= 0):
        raise ValueError("Convergence training needs a validation set and eval_every > 0")
    lengths = [len(b[0]) for b in train_set]
    global_batch_size = batch_size * world_size
    n_batches = math.ceil(len(train_set) / global_batch_size)
    steps_per_epoch = math.ceil(n_batches / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = int(round(warmup_frac * (steps_per_epoch if until_converged else total_steps)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(f"[train_hf] device={device} {len(train_set)} train ({skipped} skipped), {len(valid_set)} valid, "
        f"{n_trainable / 1e6:.2f}M trainable, batch {batch_size}x{world_size}x{grad_accum}, {total_steps} steps "
        f"(warm-up {warmup}), lr {lr}")

    def evaluate() -> dict:
        if not valid_set:
            return {}
        model.eval()
        from collections import defaultdict
        metrics = defaultdict(lambda: {"n": 0, "loss": 0.0, "agree": 0})
        ce_sum = ce_agree = ce_n = 0
        predictions = []
        with torch.no_grad():
            for b in range(rank_id * eval_batch_size, len(valid_set), eval_batch_size * world_size):
                chunk = valid_set[b:b + eval_batch_size]
                losses, logp = objective_loss(label_logits(model, [x[0] for x in chunk], label_ids, pad_id), chunk)
                losses_cpu = losses.float().cpu().tolist()
                pair_scores = logp.detach().cpu()
                if action_only:
                    predictions.extend((b+i, row.exp().tolist()) for i,row in enumerate(pair_scores))
                for loss, row, x in zip(losses_cpu, pair_scores, chunk):
                    m = metrics[x[5]]
                    m["n"] += 1
                    m["loss"] += loss
                    if x[4]:
                        m["agree"] += int(row[x[4]["preferred"]] > row[x[4]["rejected"]])
                    else:
                        agrees = int(int(row[:x[1]].argmax()) == max(range(x[1]), key=lambda j: x[2][j]))
                        m["agree"] += agrees
                        ce_sum += loss
                        ce_agree += agrees
                        ce_n += 1
        if distributed:
            purposes = sorted({row[5] for row in valid_set})
            packed = torch.tensor([[metrics[name][key] for key in ("n", "loss", "agree")]
                                   for name in purposes] + [[ce_n, ce_sum, ce_agree]], dtype=torch.float64, device=device)
            dist.all_reduce(packed)
            combined = packed.cpu().tolist()
            metrics = {name: dict(zip(("n", "loss", "agree"), values)) for name, values in zip(purposes, combined)}
            ce_n, ce_sum, ce_agree = combined[-1]
        model.train()
        action_metrics = {}
        if action_only:
            from arena.s1.action_training import policy_metrics
            if distributed:
                gathered = [None] * world_size
                dist.all_gather_object(gathered, predictions)
                predictions = [p for part in gathered for p in part]
            predictions.sort()
            detail = policy_metrics(valid_examples, [p for _,p in predictions])
            balance = family_mass or {'kakuna_executed':.4,'native':.4,'historical':.2}
            families, masses = tuple(balance), tuple(balance.values())
            balanced = sum(detail[k]['ce']*w for k,w in zip(families,masses) if k in detail)
            balanced /= sum(w for k,w in zip(families,masses) if k in detail)
            action_metrics = {'valid_action_metrics': detail, 'valid_policy_ce': balanced,
                              'valid_top1': detail['all']['top1']}
        by_purpose = {name: {"n": m["n"], "loss": round(m["loss"] / m["n"], 4),
                             "agree": round(m["agree"] / m["n"], 4)} for name, m in metrics.items()}
        return {"valid_loss": round(sum(m["loss"] for m in metrics.values()) / len(valid_set), 4),
                "valid_ce": round(ce_sum / ce_n, 4) if ce_n else None,
                "valid_agree": round(ce_agree / ce_n, 4) if ce_n else None,
                "valid_by_purpose": by_purpose, **action_metrics}

    from arena.s1.convergence import ConvergenceControl
    control = ConvergenceControl(lr, lr * min_lr_frac, min_epochs, min_delta, patience) if until_converged else None
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=device == "cuda")
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, warmup, min_lr_frac))
             if not until_converged else None)
    history, curve, epoch, step, n_seen, n_tok = [], [], 0, 0, 0, 0
    eval_seconds = loop_eval_seconds = 0.0
    saved_order, next_batch, last_evaluated = None, 0, -1
    best_directory = None
    converged = False
    if resume_from:
        saved = torch.load(Path(resume_from) / "training_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(saved["optimizer"])
        if sched is not None:
            sched.load_state_dict(saved["scheduler"])
        if control is not None:
            control = ConvergenceControl(**saved["control"])
            best_directory = Path(saved.get("best_adapter") or
                                  Path(resume_from).parent / f"step_{control.best_step:07d}")
        history, curve = saved["history"], saved["loss_curve"]
        epoch, step = saved["epoch_index"], saved["step"]
        n_seen, n_tok = saved["n_seen"], saved["n_tok"]
        saved_order, next_batch = saved["order"], saved["next_batch"]
        last_evaluated = saved["last_evaluated"]
        rng.setstate(saved["random_state"])
        random_states = saved["rank_random_states"][rank_id]
        torch.set_rng_state(random_states["torch"])
        if device == "cuda":
            torch.cuda.set_rng_state(random_states["cuda"])
        log(f"[train_hf] resumed step {step}, next batch {next_batch}")

    def save_checkpoint(row, order, batch_index):
        random_state = {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state() if device == "cuda" else None}
        rank_states = [None] * world_size
        if distributed:
            dist.all_gather_object(rank_states, random_state)
        else:
            rank_states = [random_state]
        if rank_id:
            return
        directory = out / "checkpoints" / f"step_{step:07d}"
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(directory))
        state = {"optimizer": opt.state_dict(), "scheduler": sched.state_dict() if sched else None,
                 "control": asdict(control) if control else None, "epoch_index": epoch, "step": step,
                 "n_seen": n_seen, "n_tok": n_tok, "order": order, "next_batch": batch_index,
                 "last_evaluated": step, "history": history, "loss_curve": curve,
                 "random_state": rng.getstate(), "rank_random_states": rank_states,
                 "best_adapter": str(best_directory.resolve()) if best_directory else None,
                 "world_size": world_size, "global_batch_size": global_batch_size}
        torch.save(state, directory / "training_state.pt")
        entry = {**row, "adapter": str(directory.resolve()), "examples_seen": n_seen}
        (directory / "validation.json").write_text(json.dumps(entry, indent=2))
        with (out / "checkpoints.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")

    def validate(epoch_fraction, order, batch_index, label):
        nonlocal eval_seconds, loop_eval_seconds, last_evaluated, converged, best_directory
        started = time.perf_counter()
        row = {"epoch": epoch_fraction, "step": step,
               "timestamp": datetime.now().astimezone().isoformat(), **evaluate()}
        if control is not None:
            action = row["valid_by_purpose"].get("action")
            if not action:
                raise ValueError("Policy convergence requires action validation examples")
            row["convergence"] = control.observe(row.get('valid_policy_ce', action["loss"]), epoch_fraction, step)
            if row["convergence"]["new_best"]:
                best_directory = out / "checkpoints" / f"step_{step:07d}"
            converged = row["convergence"]["converged"]
            for group in opt.param_groups:
                group["lr"] = control.learning_rate
        history.append(row)
        if event_callback and rank_id == 0:
            event_callback(row)
        if eval_every:
            save_checkpoint(row, order, batch_index)
            if rank_id == 0:
                with (out / "validation.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
            if distributed:
                dist.barrier()
        last_evaluated = step
        duration = time.perf_counter() - started
        eval_seconds += duration
        loop_eval_seconds += duration
        log(f"[train_hf] {label}: {row}")

    if not resume_from:
        validate(0, None, 0, "before")
    loop_eval_seconds = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    micro, t0, start_seen, start_tok = 0, time.perf_counter(), n_seen, n_tok
    window_loss, window_ce, window_n = 0.0, 0.0, 0
    while (epoch < epochs or until_converged) and (max_epochs is None or epoch < max_epochs):
        order = saved_order if saved_order is not None else length_batches(lengths, global_batch_size, rng, bucket_window)
        start_batch = next_batch if saved_order is not None else 0
        saved_order, next_batch = None, 0
        for bi in range(start_batch, len(order)):
            global_idx = order[bi]
            idx = global_idx[rank_id::world_size]
            local_n = len(idx)
            batch = [train_set[i] for i in (idx or global_idx[:1])]
            if permute_each_visit:
                from arena.s1.action_training import permute_action
                # The common batch-order RNG must stay identical on every DDP rank.
                batch = encode([permute_action(train_ex[i], random.Random(seed+(epoch+1)*len(train_ex)+i))
                                for i in (idx or global_idx[:1])], quiet=True)[0]
            if control is not None and step < warmup:
                for group in opt.param_groups:
                    group["lr"] = lr * (step + 1) / warmup
            accumulation_count = min(grad_accum, len(order) - (bi // grad_accum) * grad_accum)
            synchronize = (bi+1) % grad_accum == 0 or bi == len(order)-1
            with forward.no_sync() if distributed and not synchronize else nullcontext():
                ce, _ = objective_loss(forward([b[0] for b in batch]), batch)
                w = torch.tensor([b[3] for b in batch], device=ce.device)
                if not local_n:
                    w.zero_()
                loss = (w * ce).sum() * world_size / (len(global_idx) * accumulation_count)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite training objective at optimizer step {step}")
                loss.backward()
            micro += 1
            n_seen += len(global_idx)
            n_tok += sum(len(train_set[index][0]) for index in global_idx)
            window_loss += float((w * ce).detach().sum())
            window_ce += float(ce.detach().sum()) if local_n else 0
            window_n += local_n
            if synchronize:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
                opt.step()
                if sched is not None:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or (not until_converged and step == total_steps):
                    if distributed:
                        window = torch.tensor([window_loss, window_ce, window_n], dtype=torch.float64, device=device)
                        dist.all_reduce(window)
                        window_loss, window_ce, window_n = window.tolist()
                    el = time.perf_counter() - t0 - loop_eval_seconds
                    curve.append({"step": step, "epoch": round(epoch + (bi + 1) / len(order), 3),
                                  "loss": round(window_loss / window_n, 4), "objective": round(window_ce / window_n, 4),
                                  "lr": round(opt.param_groups[0]["lr"], 7)})
                    limit = "convergence" if until_converged else total_steps
                    log(f"[train_hf] step {step}/{limit} epoch {curve[-1]['epoch']:.2f} loss "
                        f"{curve[-1]['loss']:.4f} objective {curve[-1]['objective']:.4f} lr {curve[-1]['lr']:.2e} "
                        f"({(n_seen-start_seen) / el:.1f} ex/s, {(n_tok-start_tok) / el:.0f} tok/s)")
                    progress = {"state": "training", **curve[-1], "examples_seen": n_seen,
                                "examples_per_s": (n_seen-start_seen)/el, "tokens_per_s": (n_tok-start_tok)/el,
                                "best_step": control.best_step if control else None,
                                "validation_checks": len(history),
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "peak_gpu_gb": torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else None}
                    if rank_id == 0:
                        (out / "progress.json").write_text(json.dumps(progress, indent=2))
                        with (out / "training.jsonl").open("a") as file:
                            file.write(json.dumps(progress) + "\n")
                        if event_callback:
                            event_callback(progress)
                    window_loss = window_ce = 0.0
                    window_n = 0
                if eval_every and step % eval_every == 0:
                    validate(epoch + (bi+1)/len(order), order, bi+1, f"validation step {step}")
                    if converged:
                        break
        if converged:
            break
        if step != last_evaluated and (not until_converged or (max_epochs is not None and epoch+1 >= max_epochs)):
            validate(epoch + 1, order, len(order), f"after epoch {epoch + 1}")
        if converged:
            break
        if save_every_epoch and epoch + 1 < epochs and rank_id == 0:
            model.save_pretrained(str(Path(out_dir) / f"epoch{epoch + 1}"))
        epoch += 1
    train_seconds = time.perf_counter() - t0 - loop_eval_seconds
    peak_memory = torch.cuda.max_memory_allocated()/1e9 if device == 'cuda' else None
    rank_peak_memory = [peak_memory]
    if distributed:
        rank_peak_memory = [None]*world_size
        dist.all_gather_object(rank_peak_memory,peak_memory)
    if control is not None and rank_id == 0:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(best_directory / name, out / name)
    elif rank_id == 0:
        model.save_pretrained(str(out))
    meta = {"base_model": base_model, "trainer": "arena.s1.train_hf", "train_file": str(train_file),
            "init_adapter": init_adapter,
            "n_train": len(train_set), "skipped": skipped, "n_valid_eval": len(valid_set), "epochs": history[-1]["epoch"],
            "rank": rank, "alpha": alpha or 2 * rank, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
            "warmup_steps": warmup, "min_lr_frac": min_lr_frac, "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm, "grad_ckpt": grad_ckpt, "seed": seed, "steps": step,
            "trainable_params": n_trainable, "train_tokens": n_tok,
            "train_seconds": round(train_seconds, 1), "eval_seconds": round(eval_seconds, 1),
            "examples_per_s": round((n_seen-start_seen) / max(train_seconds, 1e-9), 2),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            "history": history, "loss_curve": curve, "until_converged": until_converged,
            "converged": converged, "best_step": control.best_step if control else step,
            "eval_every": eval_every, "bucket_window": bucket_window, "resume_from": resume_from,
            "convergence": asdict(control) if control else None}
    meta['optimizer'] = {'name':'AdamW','betas':opt.defaults['betas'],'eps':opt.defaults['eps']}
    meta.update(action_only=action_only, permute_each_visit=permute_each_visit, max_epochs=max_epochs,
                stopping_reason='validation_plateau' if converged else 'epoch_limit',
                peak_mem_gb_by_rank=rank_peak_memory)
    meta.update(world_size=world_size, global_batch_size=global_batch_size, effective_batch_size=global_batch_size*grad_accum)
    if rank_id == 0:
        (out / "s1_train_meta.json").write_text(json.dumps(meta, indent=2))
        (out / "progress.json").write_text(json.dumps({"state": "converged" if converged else
            "epoch_limit" if until_converged and max_epochs is not None else "completed",
            "step": step, "epoch": history[-1]["epoch"], "best_step": meta["best_step"]}, indent=2))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return meta
