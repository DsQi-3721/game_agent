"""Fast EVE-RLCD decision model for MicroRTS: LoRA + 26-letter readout + SharedPrefixCache."""

import copy
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from rlcd.decide import SharedPrefixCache, check_split
from rlcd.export import _write_export, load_decision_only
from rlcd.rewards import REWARDS, policy_gradient_loss
from rlcd.schema import MAX_CHOICES, Question, render_prefix, render_prompt, render_suffix
from torch import nn

from .skills import skill_text
from .state import state_to_text

MODEL_ID = "anthonym21/qwen3-0.6b-rlcd-decision"
MODEL_REVISION = "b327ec5efb5fdbf8bfafa3b369720ac5f6434b05"
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / ".runtime/models/qwen3-0.6b-rlcd-decision"


def resolve_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype, device):
    if dtype == "auto":
        dev_type = torch.device(device).type
        return torch.bfloat16 if dev_type in ("cuda", "mps") else torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]


def shuffle_candidates(candidates, chosen_idx=None, rng=None):
    """Randomly permute candidates to remove positional and A/B letter bias during training."""
    n = len(candidates)
    perm = list(range(n))
    (rng or random).shuffle(perm)
    shuffled = [candidates[i] for i in perm]
    if chosen_idx is None:
        return shuffled, perm, None
    return shuffled, perm, perm.index(chosen_idx)


def unit_choice_question(state, unit, candidates, state_text=None):
    """Per-unit ChoiceQ over a compact, deduplicated tactical skill shortlist."""
    ctx = state_text if state_text is not None else state_to_text(state)
    options = [skill_text(c, state) for c in candidates]
    if len(options) == 1:
        return Question(
            "noul",
            ctx,
            f"For {unit['type']} U{unit['id']} at ({unit['x']},{unit['y']}), wait is the only option. Wait?",
            ["true", "false"],
        ).validate()
    return Question(
        "choice",
        ctx,
        f"Best skill for {unit['type']} U{unit['id']} at ({unit['x']},{unit['y']}) to win this match?",
        options,
    ).validate()


def choice_question(state, candidates, state_text=None):
    """Legacy global ChoiceQ across all idle units (kept for compatibility)."""
    ctx = state_text if state_text is not None else state_to_text(state)
    if len(candidates) == 1:
        return Question(
            "noul",
            ctx,
            "Waiting is the only legal choice. Must we wait?",
            ["true", "false"],
        ).validate()
    return Question(
        "choice",
        ctx,
        "Which legal skill should our player execute to win this MicroRTS match?",
        [skill_text(c, state) for c in candidates],
    ).validate()


def win_question(state, candidate=None, greedy=False, state_text=None):
    """Binary NoulQ asking whether our player will win this match (draw counts as false)."""
    ctx = state_text if state_text is not None else state_to_text(state)
    if candidate is None:
        q = (
            f"Following the current {'greedy' if greedy else 'sampled'} policy against this "
            "opponent, will our player win this match? A draw counts as false."
        )
    else:
        q = (
            f"If we execute {skill_text(candidate, state)} and continue using the current "
            f"{'greedy' if greedy else 'sampled'} policy, will our player win? A draw counts as false."
        )
    return Question("noul", ctx, q, ["true", "false"]).validate()


def rlcd_bandit_loss(logits, actions, outcomes, arm="rlcd"):
    """Exact eve-rlcd bandit loss: r = c - stopgrad(p_a) with leave-one-out group baseline."""
    logp = logits.log_softmax(-1)
    p_taken = logp.detach().exp().gather(1, actions)
    return policy_gradient_loss(logp, actions, REWARDS[arm](outcomes, p_taken))


def forecast_loss(logits, sampled_answers, won, arm="rlcd"):
    """Exact eve-rlcd RLCD/RLVR loss over a binary win forecast (single row or batch)."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        actions = torch.as_tensor(sampled_answers, dtype=torch.long, device=logits.device).unsqueeze(0)
        targets = torch.tensor([0 if won else 1], dtype=torch.long, device=logits.device)
    else:
        actions = torch.as_tensor(sampled_answers, dtype=torch.long, device=logits.device)
        targets = torch.as_tensor(
            [0 if w else 1 for w in won], dtype=torch.long, device=logits.device
        )
    correct = (actions == targets[:, None]).float()
    return rlcd_bandit_loss(logits, actions, correct, arm=arm)


class EveRLCDPolicy(nn.Module):
    def __init__(
        self,
        init=DEFAULT_MODEL,
        device="cpu",
        rank=8,
        max_len=2048,
        dtype="auto",
        target_modules=("q_proj", "v_proj"),
    ):
        super().__init__()
        self.init = str(Path(init).resolve())
        self.device_name = resolve_device(device)
        self.rank = int(rank)
        self.max_len = int(max_len)
        self.dtype_name = dtype
        self.backbone_dtype = resolve_dtype(dtype, self.device_name)
        self.target_modules = list(target_modules)
        self.greedy = False

        self.rlcd = load_decision_only(self.init, device="cpu")
        body = self.rlcd.model.to(device=self.device_name, dtype=self.backbone_dtype)
        body.requires_grad_(False)
        if self.rank > 0:
            body = get_peft_model(
                body,
                LoraConfig(
                    r=self.rank,
                    lora_alpha=2 * self.rank,
                    lora_dropout=0.0,
                    target_modules=self.target_modules,
                ),
            )
            # Keep trainable LoRA weights in fp32 for stable AdamW updates.
            for p in body.parameters():
                if p.requires_grad:
                    p.data = p.data.to(device=self.device_name, dtype=torch.float32)

        self.backbone = body.to(self.device_name)
        self.letter_weight = nn.Parameter(
            self.rlcd.weight.detach().clone().to(device=self.device_name, dtype=torch.float32)
        )
        self.register_buffer(
            "letter_bias",
            None
            if self.rlcd.bias is None
            else self.rlcd.bias.detach().clone().to(device=self.device_name, dtype=torch.float32),
        )
        self.value_head = nn.Linear(self.letter_weight.shape[1], 1).to(
            device=self.device_name, dtype=torch.float32
        )
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self.rlcd.model = self.backbone
        self.rlcd.weight = self.letter_weight
        self.rlcd.bias = self.letter_bias
        self.eval()

        # Verify tokenizer supports exact prefix/suffix split for SharedPrefixCache.
        try:
            check_split(self.rlcd.tok)
            self._can_prefix_cache = True
        except ValueError:
            self._can_prefix_cache = False

    def param_groups(self, lr, head_lr_mult=5.0, weight_decay=0.0):
        """Separate LR groups so the value & letter heads adapt faster than LoRA adapters."""
        lora_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("value_head.") or name == "letter_weight":
                head_params.append(p)
            else:
                lora_params.append(p)
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": lr, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": lr * head_lr_mult, "weight_decay": 0.0})
        return groups

    def _encode_questions(self, questions, strict_len=True):
        """Tokenize a batch of Questions ONCE (right-padded, optional strict length check)."""
        tok = self.rlcd.tok
        prefix = [tok.bos_token_id] if self.rlcd.prepend_bos else []
        room = self.max_len - len(prefix)
        seqs = []
        for q in questions:
            ids = tok.encode(render_prompt(q), add_special_tokens=False)
            if strict_len and len(ids) > room:
                raise ValueError(
                    f"Prompt has {len(ids) + len(prefix)} tokens, max_len={self.max_len}; "
                    "increase --max-len or reduce --candidate-limit. State/options are never silently cut."
                )
            seqs.append(prefix + ids[-room:])
        width = max(len(s) for s in seqs)
        ids_t = torch.full((len(seqs), width), self.rlcd.pad_id, dtype=torch.long)
        mask_t = torch.zeros((len(seqs), width), dtype=torch.long)
        last_t = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask_t[i, : len(s)] = 1
            last_t[i] = len(s) - 1
        dev = self.device_name
        return ids_t.to(dev), mask_t.to(dev), last_t.to(dev)

    def _readout_batch(self, questions, strict_len=True):
        ids, mask, last = self._encode_questions(questions, strict_len=strict_len)
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]

    def forward_questions(self, questions, return_values=False, strict_len=True):
        """Differentiable batched forward: returns logits (B, 26) and optionally values (B,)."""
        rows = self._readout_batch(questions, strict_len=strict_len)
        logits = self.rlcd.logits_from_hidden(rows, [q.k for q in questions])
        if return_values:
            return logits, self.value_head(rows.float()).squeeze(-1)
        return logits

    def _forward_shared_prefix(self, state_text, questions):
        """Fast no-grad inference: prefill the shared state ONCE, then batch all question suffixes."""
        tok = self.rlcd.tok
        dev = self.device_name
        prefix_str = render_prefix(state_text)
        prefix_ids = ([tok.bos_token_id] if self.rlcd.prepend_bos else []) + tok.encode(
            prefix_str, add_special_tokens=False
        )
        suffix_seqs = [tok.encode(render_suffix(q), add_special_tokens=False) for q in questions]
        max_total = len(prefix_ids) + max(len(s) for s in suffix_seqs)
        if max_total > self.max_len:
            raise ValueError(
                f"Prompt has {max_total} tokens, max_len={self.max_len}; "
                "increase --max-len or reduce --candidate-limit. State/options are never silently cut."
            )

        t0 = time.perf_counter()
        p_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
        prefill_out = self.backbone(input_ids=p_tensor, use_cache=True)
        cache = SharedPrefixCache(prefill_out.past_key_values)
        t1 = time.perf_counter()

        m = len(suffix_seqs)
        width = max(len(s) for s in suffix_seqs)
        s_ids = torch.full((m, width), self.rlcd.pad_id, dtype=torch.long)
        s_mask = torch.zeros((m, width), dtype=torch.long)
        s_last = torch.zeros(m, dtype=torch.long)
        for i, s in enumerate(suffix_seqs):
            s_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            s_mask[i, : len(s)] = 1
            s_last[i] = len(s) - 1
        s_ids, s_mask, s_last = s_ids.to(dev), s_mask.to(dev), s_last.to(dev)
        p_len = p_tensor.shape[1]
        attn_mask = torch.cat([torch.ones((m, p_len), dtype=torch.long, device=dev), s_mask], dim=1)
        pos_ids = torch.arange(p_len, p_len + width, device=dev).unsqueeze(0).expand(m, -1)

        out = self.backbone(
            input_ids=s_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=cache,
            use_cache=False,
        )
        rows = out[0][torch.arange(m, device=dev), s_last]
        logits = self.rlcd.logits_from_hidden(rows, [q.k for q in questions])
        values = self.value_head(rows.float()).squeeze(-1)
        t2 = time.perf_counter()
        return logits, values, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    @torch.no_grad()
    def decide_state(
        self,
        state,
        unit_candidates_list,
        *,
        include_win_forecast=True,
        greedy=False,
        forecast_group=4,
        forecast_rng=None,
        shuffle_rng=None,
    ):
        """Decide skills for all idle units AND forecast win probability in one cached pass."""
        state_text = state_to_text(state)
        questions = []
        prepared = []
        for unit, cands in unit_candidates_list:
            if shuffle_rng is not None and len(cands) > 1:
                cands_use, _, _ = shuffle_candidates(cands, rng=shuffle_rng)
            else:
                cands_use = cands
            q = unit_choice_question(state, unit, cands_use, state_text=state_text)
            questions.append(q)
            prepared.append((unit, cands_use, q))

        win_q = None
        if include_win_forecast:
            win_q = win_question(state, candidate=None, greedy=greedy, state_text=state_text)
            questions.append(win_q)

        if not questions:
            return {"state_text": state_text, "unit_decisions": [], "total_ms": 0.0}

        if self._can_prefix_cache and len(questions) >= 2:
            logits, values, prefill_ms, suffix_ms = self._forward_shared_prefix(
                state_text, questions
            )
        else:
            t0 = time.perf_counter()
            logits, values = self.forward_questions(questions, return_values=True)
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            suffix_ms = 0.0

        unit_decisions = []
        for i, (unit, cands_use, q) in enumerate(prepared):
            k = len(cands_use)
            unit_logits = logits[i, :k]
            dist = torch.distributions.Categorical(logits=unit_logits)
            idx = int(unit_logits.argmax().item()) if greedy else int(dist.sample().item())
            probs = dist.probs.cpu().tolist()
            log_prob = float(dist.log_prob(torch.tensor(idx, device=unit_logits.device)).item())
            unit_decisions.append(
                {
                    "unit_id": unit["id"],
                    "unit_type": unit["type"],
                    "question": q,
                    "candidates": cands_use,
                    "chosen_index": idx,
                    "chosen_skill": cands_use[idx],
                    "action_probs": probs,
                    "log_prob": log_prob,
                    "value": float(values[i].item()),
                }
            )

        result = {
            "state_text": state_text,
            "unit_decisions": unit_decisions,
            "state_value": float(values[-1].item()) if include_win_forecast else float(values.mean().item()),
            "prefill_ms": prefill_ms,
            "suffix_ms": suffix_ms,
            "total_ms": prefill_ms + suffix_ms,
        }
        if include_win_forecast:
            win_probs = logits[-1, :2].softmax(-1).cpu()
            answers = torch.multinomial(
                win_probs, forecast_group, replacement=True, generator=forecast_rng
            )
            result.update(
                win_question=win_q,
                win_probability=float(win_probs[0].item()),
                forecast_probs=win_probs.tolist(),
                forecast_answers=answers.tolist(),
            )
        return result

    # --- Backward-compatible single-question methods ---

    def rows(self, question):
        return self._readout_batch([question], strict_len=True)

    def forward(self, state, candidates):
        if not 1 <= len(candidates) <= MAX_CHOICES:
            raise ValueError(f"eve-rlcd supports at most {MAX_CHOICES} candidates")
        q = choice_question(state, candidates)
        rows = self._readout_batch([q], strict_len=True)
        logits = self.rlcd.logits_from_hidden(rows, [len(candidates)])[0, : len(candidates)]
        return logits, self.value_head(rows.float()).squeeze()

    def win_logits(self, state, candidate=None):
        q = win_question(state, candidate, self.greedy)
        rows = self._readout_batch([q], strict_len=True)
        return self.rlcd.logits_from_hidden(rows, [2])[0, :2]

    def forecast(self, state, candidate, group, generator):
        with torch.no_grad():
            probs = self.win_logits(state, candidate).softmax(-1).cpu()
            answers = torch.multinomial(probs, group, replacement=True, generator=generator)
        return {
            "win_probability": probs[0].item(),
            "forecast_probs": probs.tolist(),
            "forecast_answers": answers.tolist(),
        }

    # --- Serialization, fast in-place adapter swap, and decision-only export ---

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state_dict):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if params.keys() != state_dict.keys():
            raise ValueError("Checkpoint trainable parameter names do not match the policy")
        with torch.no_grad():
            for name, value in state_dict.items():
                params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))

    def load_adapter(self, path):
        """Swap adapter/head weights in-place without reloading the 2.4 GB base backbone."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "microrts-eve-rlcd-v1":
            raise ValueError("Expected a microRTS eve-rlcd adapter checkpoint")
        if self.rlcd.record["sha256"] != checkpoint["base_sha256"]:
            raise ValueError("Adapter checkpoint belongs to a different base export")
        self.load_trainable_state(checkpoint["trainable"])
        return checkpoint

    def save(self, path, training_config, training_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format": "microrts-eve-rlcd-v1",
                "model": {
                    "init": self.init,
                    "rank": self.rank,
                    "max_len": self.max_len,
                    "target_modules": self.target_modules,
                },
                "base_sha256": self.rlcd.record["sha256"],
                "trainable": self.trainable_state(),
                "training_config": training_config,
                "training_state": training_state,
            },
            temporary,
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu", init=None, dtype="auto"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "microrts-eve-rlcd-v1":
            raise ValueError("Expected a microRTS eve-rlcd adapter checkpoint")
        model_cfg = dict(checkpoint["model"])
        if init:
            model_cfg["init"] = init
        policy = cls(**model_cfg, device=device, dtype=dtype)
        if policy.rlcd.record["sha256"] != checkpoint["base_sha256"]:
            raise ValueError("Adapter checkpoint belongs to a different base export")
        policy.load_trainable_state(checkpoint["trainable"])
        return policy

    def export_decision_only(self, out_dir, meta=None):
        """Merge LoRA into the Qwen body and write a standard eve-rlcd decision-only export.

        The exported directory loads directly with `rlcd.decide.Decider.load(out_dir)` with
        zero LoRA overhead.
        """
        out_path = Path(out_dir)
        # Merge on CPU in fp32 so the exported safetensors match eve-rlcd's fp32 export standard.
        body_cpu = copy.deepcopy(self.backbone).to(device="cpu", dtype=torch.float32)
        if self.rank > 0:
            body_cpu = body_cpu.merge_and_unload()
        body_cpu.eval()
        record = copy.deepcopy(self.rlcd.record)
        record["source"] = str(out_path)
        if meta:
            record["training"] = {**(record.get("training") or {}), **meta}
        return _write_export(
            out_path,
            body_cpu,
            self.rlcd.tok,
            self.letter_weight.detach().cpu().float(),
            None if self.letter_bias is None else self.letter_bias.detach().cpu().float(),
            record,
        )
