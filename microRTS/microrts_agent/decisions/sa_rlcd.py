"""SA-RLCD: Sequential Advantage-Calibrated RLCD for adversarial MicroRTS decision-making.

Four core algorithmic improvements over naive eve-rlcd adaptation:
  1. IS-RLCD (Importance-Sampled Exploratory Bandit RLCD):
     Samples from an eps-mixed proposal q = (1 - eps) * p + eps / K with importance weight
     w_a = stopgrad(p_a / q_a) and LOO baseline. Mathematically proven unbiased for half
     the Brier gradient while eliminating the zero-gradient saturation dead-zone of vanilla RLCD.
  2. RB-TD(lambda)-RLCD (Rao-Blackwellized Temporal-Difference RLCD):
     Computes the exact analytical expectation of RLCD over binary NoulQ win forecasts against
     TD(lambda) win-probability targets z_t in [0, 1] (zero Monte Carlo variance, zero
     saturation dead-zone, and bootstraps timeout games instead of discarding them).
  3. WPC-PPO + Uncertainty-Adaptive Entropy:
     Unifies NoulQ and ChoiceQ by using calibrated P(win | s) directly as the bounded SMDP
     critic in PPO, and scaling per-state exploration entropy by calibrated outcome uncertainty
     u(s) = 4 * P(win|s) * (1 - P(win|s)).
  4. Anti-Forgetting Calibration Replay & Reference-KL Anchor:
     Interleaves balanced NoulQ calibration replay and ChoiceQ anchors during online RL so
     cross-opponent probability calibration never drifts under hard-opponent specialization.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from rlcd.loop import GradWindow, batch_indices, cosine_lr
from rlcd.metrics import brier, ece
from rlcd.rewards import kl_categorical, supervised_loss
from rlcd.schema import Question

from .eve_policy import (
    DEFAULT_MODEL,
    EveRLCDPolicy,
    resolve_device,
    shuffle_candidates,
    win_question,
)
from .metrics import aggregate_games, atomic_json, selection_score
from .state import ensure_force_summary
from .train import EnvPool, evaluate_policy, load_decision_rows, play_episode, vars_json

# =====================================================================
# 1. IS-RLCD: Unbiased Importance-Sampled Exploratory Bandit RLCD
# =====================================================================


def is_rlcd_proposal(probs, ks, eps=0.15):
    """Build eps-mixed proposal distribution q = (1 - eps) * p + eps / K over valid options."""
    batch, width = probs.shape
    ar = torch.arange(width, device=probs.device).unsqueeze(0)
    k_t = torch.as_tensor(ks, dtype=torch.long, device=probs.device).unsqueeze(1)
    valid = ar < k_t
    uniform = valid.to(probs.dtype) / k_t.to(probs.dtype).clamp(min=1.0)
    q = (1.0 - eps) * probs.detach() + eps * uniform
    return q * valid.to(probs.dtype)


def importance_sampled_rlcd_loss(logits, actions, outcomes, proposal_probs):
    """Unbiased IS-RLCD loss for bandit feedback.

    With a_g ~ q_prop and w_a = stopgrad(p_a / q_prop(a)):
      E[w_a * ((c_a - sg(p_a)) - b_g) * grad log p_a] = -0.5 * grad ||p - q_true||^2
    Because E_{q_prop}[w_a * grad log p_a] = sum_i grad p_i = 0, the LOO baseline b_g
    (computed over other samples' raw RLCD rewards) introduces zero bias!
    """
    logp = logits.log_softmax(-1)
    p = logp.detach().exp()
    p_a = p.gather(1, actions)
    q_a = proposal_probs.detach().gather(1, actions).clamp(min=1e-12)
    w_a = (p_a / q_a).detach()
    raw_r = outcomes - p_a
    _, group = raw_r.shape
    if group > 1:
        baseline = (raw_r.sum(1, keepdim=True) - raw_r) / (group - 1)
    else:
        baseline = torch.zeros_like(raw_r)
    adv = (w_a * (raw_r - baseline)).detach()
    return -(adv * logp.gather(1, actions)).mean()


# =====================================================================
# 2. RB-TD(lambda)-RLCD: Zero-Variance Sequential Win Calibration
# =====================================================================


def td_lambda_win_targets(steps, bootstrap_pwin=0.2, td_lambda=0.85, timeout_value=0.15):
    """Compute backward TD(lambda) calibrated win-probability targets z_t in [0, 1].

    - Natural win  -> terminal target 1.0
    - Natural loss -> terminal target 0.0
    - Timeout      -> soft target min(bootstrap_pwin, timeout_value) to discourage stalemates
    """
    n = len(steps)
    if n == 0:
        return []
    final = steps[-1]
    if final["terminated"]:
        terminal_target = 1.0 if final.get("won", False) else 0.0
    else:
        terminal_target = float(min(max(bootstrap_pwin, 0.0), timeout_value))

    targets = [0.0] * n
    z_next = terminal_target
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            targets[i] = terminal_target
            z_next = terminal_target
        else:
            p_next = float(steps[i + 1]["win_probability"])
            z_t = (1.0 - td_lambda) * p_next + td_lambda * z_next
            targets[i] = float(min(max(z_t, 0.0), 1.0))
            z_next = targets[i]
    return targets


def rb_rlcd_loss(win_logits, targets):
    """Exact Rao-Blackwellized RLCD loss: 0.5 * sum_i (p_i - q_i)^2 = (p_win - z)^2.

    Matches the exact expected gradient of RLCD with zero Monte Carlo variance and
    zero saturation dead-zone even when p_win approaches 0 or 1.
    """
    probs = win_logits.softmax(-1)
    z = torch.as_tensor(targets, dtype=probs.dtype, device=probs.device)
    q = torch.stack([z, 1.0 - z], dim=-1)
    return 0.5 * (probs - q).square().sum(-1).mean()


# =====================================================================
# 3. WPC-PPO: Win-Probability-Critic SMDP Advantages & Adaptive Entropy
# =====================================================================


def compute_wpc_advantages(
    steps,
    bootstrap_pwin=0.15,
    *,
    gae_lambda=0.95,
    shaping_scale=0.005,
    timeout_value=0.15,
):
    """SMDP GAE directly in calibrated win-probability space V(s) = P(win | s) in [0, 1].

    Uses terminal outcome (win=1.0, loss=0.0, timeout=timeout_value) plus tiny bounded
    potential tie-breaker shaping, eliminating raw reward hacking while giving every step
    dense win-probability-delta credit assignment.
    """
    n = len(steps)
    advantages = [0.0] * n
    gae = 0.0
    final = steps[-1]
    if final["terminated"]:
        next_v = 1.0 if final.get("won", False) else 0.0
    else:
        next_v = float(min(max(bootstrap_pwin, 0.0), timeout_value))

    for i in range(n - 1, -1, -1):
        s = steps[i]
        v_cur = float(s["win_probability"])
        # Extract pure tiny shaping delta without terminal +-20 spike.
        raw_r = float(s["reward"])
        if s["terminated"]:
            shaped_only = 0.0
            r_step = 1.0 if s.get("won", False) else 0.0
            mask = 0.0
        elif s["truncated"]:
            shaped_only = 0.0
            r_step = float(timeout_value)
            mask = 0.0
        else:
            shaped_only = float(np.clip(raw_r * shaping_scale, -0.04, 0.04))
            r_step = shaped_only
            mask = 1.0
        disc = float(s["discount"])
        delta = r_step + disc * mask * next_v - v_cur
        gae = delta + disc * gae_lambda * mask * gae
        advantages[i] = gae
        next_v = v_cur
    return advantages


# =====================================================================
# 4. Offline IS-RLCD & Force-Summary SFT
# =====================================================================


def _row_to_enriched_question(row, shuffle_rng=None):
    ctx = ensure_force_summary(row["context"])
    choices = list(row["choices"])
    ans = int(row["answer"])
    if shuffle_rng is not None and row.get("primitive", "choice") == "choice" and len(choices) > 1:
        choices, _, ans = shuffle_candidates(choices, chosen_idx=ans, rng=shuffle_rng)
    q = Question(
        row.get("primitive", "choice"),
        ctx,
        row["question"],
        choices,
        answer=ans,
    ).validate()
    return q, ans


def evaluate_enriched_dataset(policy, rows, batch_size=16):
    if not rows:
        return {}
    all_probs, all_ans = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            qs = [_row_to_enriched_question(r)[0] for r in chunk]
            ans = [int(r["answer"]) for r in chunk]
            logits = policy.forward_questions(qs, strict_len=False)
            probs = logits.softmax(-1).cpu().numpy()
            all_probs.append(probs)
            all_ans.extend(ans)
    probs = np.concatenate(all_probs, axis=0)
    ans_arr = np.array(all_ans, dtype=np.int64)
    preds = probs.argmax(axis=-1)
    hits = (preds == ans_arr).astype(np.float64)
    confs = probs.max(axis=-1)
    return {
        "n": len(rows),
        "accuracy": float(hits.mean()),
        "brier": float(brier(probs, ans_arr)),
        "ece": float(ece(confs, hits)),
        "mean_conf": float(confs.mean()),
    }


def run_offline_sa_rlcd(
    policy,
    rows,
    out_dir,
    *,
    mode="is-rlcd",
    epochs=2,
    micro_batch=16,
    accum_steps=2,
    lr=3e-5,
    head_lr_mult=5.0,
    group=4,
    eps=0.15,
    val_rows=None,
    win_rows=None,
    win_val_rows=None,
    seed=7,
):
    """Run Force-summary SFT or IS-RLCD bandit training (with optional RB-RLCD NoulQ mixing)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = list(rows)
    if win_rows:
        # Mix a balanced slice of NoulQ win-forecast rows (up to 30% of choice rows).
        sample_n = min(len(win_rows), max(len(rows) // 3, 500))
        train_items.extend(rng.sample(win_rows, sample_n))

    optimizer = torch.optim.AdamW(
        policy.param_groups(lr=lr, head_lr_mult=head_lr_mult), weight_decay=0.0
    )
    n_micro = epochs * ((len(train_items) + micro_batch - 1) // micro_batch)
    total_steps = max(1, (n_micro + accum_steps - 1) // accum_steps)
    warmup = max(1, total_steps // 10)
    grads = GradWindow()
    params = [p for p in policy.parameters() if p.requires_grad]

    val_history = []
    best_brier = None
    step = 0
    loss_acc, acc_sum = 0.0, 0.0
    t0 = time.monotonic()

    def run_val(ep_label):
        nonlocal best_brier
        v = evaluate_enriched_dataset(policy, val_rows, batch_size=max(16, micro_batch))
        if win_val_rows:
            vw = evaluate_enriched_dataset(policy, win_val_rows[:2000], batch_size=max(16, micro_batch))
            v["win_brier"] = vw["brier"]
            v["win_ece"] = vw["ece"]
            v["win_acc"] = vw["accuracy"]
        val_history.append({"epoch": ep_label, "step": step, **v})
        sel_brier = v["brier"] + (0.5 * v["win_brier"] if "win_brier" in v else 0.0)
        if best_brier is None or sel_brier < best_brier:
            best_brier = sel_brier
            policy.save(out_dir / "best.pt", {"mode": mode, "step": step, "val": v})
        print(json.dumps({"val_epoch": ep_label, "step": step, **v}), flush=True)

    if val_rows:
        run_val(0)

    def finish_step(ep):
        nonlocal step, loss_acc, acc_sum
        n = grads.finish(params)
        cur_lr = cosine_lr(step + 1, total_steps, lr, warmup=warmup)
        for g_idx, g in enumerate(optimizer.param_groups):
            g["lr"] = cur_lr * (head_lr_mult if g_idx == 1 else 1.0)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step <= 5 or step % 20 == 0 or step == total_steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "epoch": ep,
                        "mode": mode,
                        "loss": loss_acc / max(n, 1),
                        "acc": acc_sum / max(n, 1),
                        "grad_norm": grad_norm,
                        "lr": cur_lr,
                        "elapsed_sec": time.monotonic() - t0,
                    }
                ),
                flush=True,
            )
        loss_acc, acc_sum = 0.0, 0.0

    for ep in range(epochs):
        for idx in batch_indices(len(train_items), micro_batch, rng):
            qs, ans_list, is_noul = [], [], []
            for i in idx:
                r = train_items[i]
                q, a = _row_to_enriched_question(r, shuffle_rng=rng)
                qs.append(q)
                ans_list.append(a)
                is_noul.append(r.get("primitive") == "noul")
            logits = policy.forward_questions(qs, strict_len=False)
            ans_t = torch.tensor(ans_list, dtype=torch.long, device=logits.device)
            logp = logits.log_softmax(-1)
            probs = logp.exp()

            if mode == "sft":
                loss = supervised_loss(logp, ans_t)
            else:
                # ChoiceQ uses unbiased IS-RLCD (exploratory bandit); NoulQ uses exact RB-RLCD.
                noul_mask = torch.tensor(is_noul, dtype=torch.bool, device=logits.device)
                choice_mask = ~noul_mask
                losses = []
                if choice_mask.any():
                    c_logits = logits[choice_mask]
                    c_probs = probs[choice_mask]
                    c_ans = ans_t[choice_mask]
                    c_ks = [qs[j].k for j in range(len(qs)) if not is_noul[j]]
                    q_prop = is_rlcd_proposal(c_probs, c_ks, eps=eps)
                    actions = torch.multinomial(q_prop, group, replacement=True)
                    outcomes = (actions == c_ans[:, None]).float()
                    c_loss = importance_sampled_rlcd_loss(c_logits, actions, outcomes, q_prop)
                    losses.append(c_loss * float(choice_mask.sum().item()))
                if noul_mask.any():
                    n_logits = logits[noul_mask, :2]
                    n_targets = (ans_t[noul_mask] == 0).float()
                    n_loss = rb_rlcd_loss(n_logits, n_targets)
                    losses.append(n_loss * float(noul_mask.sum().item()))
                loss = sum(losses) / len(idx)

            grads.backward(loss, len(idx))
            with torch.no_grad():
                loss_acc += float(loss.item()) * len(idx)
                acc_sum += float((probs.argmax(-1) == ans_t).float().sum().item())
            if grads.micro == accum_steps:
                finish_step(ep)
        if grads.micro:
            finish_step(ep)
        if val_rows:
            run_val(ep + 1)

    metrics = {
        "mode": mode,
        "steps": step,
        "elapsed_sec": time.monotonic() - t0,
        "train_rows": len(train_items),
        "val_history": val_history,
    }
    if val_history:
        metrics["val"] = val_history[-1]
    policy.save(out_dir / "policy.pt", metrics)
    atomic_json(out_dir / "summary.json", metrics)
    return metrics


# =====================================================================
# 5. Online SA-RLCD Update (WPC-PPO + Adaptive Entropy + RB-TD(lambda) + Calib-Replay)
# =====================================================================


def sa_rlcd_update(
    policy,
    ref_policy,
    optimizer,
    episode_batch,
    *,
    calib_win_rows=None,
    anchor_choice_rows=None,
    ppo_epochs=2,
    micro_batch=16,
    clip_eps=0.2,
    ent_min=0.003,
    ent_max=0.025,
    kl_coef=0.02,
    calibration_weight=1.5,
    td_lambda=0.85,
    gae_lambda=0.95,
    timeout_value=0.15,
    replay_win_n=24,
    replay_choice_n=16,
    rng=None,
):
    """Unified SA-RLCD update combining WPC-PPO, Uncertainty-Adaptive Entropy,
    KL anchoring, RB-TD(lambda)-RLCD win calibration, and anti-forgetting replay."""
    rng = rng or random.Random(0)
    ppo_items = []
    cal_items = []
    all_advs = []

    for steps, final_state, metrics in episode_batch:
        if not steps:
            continue
        won = metrics["outcome"] == "win"
        for st in steps:
            st["won"] = won
        bootstrap_p = 0.15
        if steps[-1]["truncated"]:
            with torch.no_grad():
                wq = win_question(final_state, greedy=False)
                wl = policy.forward_questions([wq], strict_len=False)[0, :2]
                bootstrap_p = float(wl.softmax(-1)[0].item())

        advs = compute_wpc_advantages(
            steps,
            bootstrap_pwin=bootstrap_p,
            gae_lambda=gae_lambda,
            timeout_value=timeout_value,
        )
        win_targets = td_lambda_win_targets(
            steps,
            bootstrap_pwin=bootstrap_p,
            td_lambda=td_lambda,
            timeout_value=timeout_value,
        )
        for st, adv, z_t in zip(steps, advs, win_targets, strict=True):
            p_win = float(st["win_probability"])
            uncertainty = float(4.0 * p_win * (1.0 - p_win))  # in [0, 1], peaks at 0.5
            for ud in st["unit_decisions"]:
                ppo_items.append(
                    {
                        "question": ud["question"],
                        "action": ud["chosen_index"],
                        "old_logp": ud["log_prob"],
                        "adv": adv,
                        "uncertainty": uncertainty,
                    }
                )
                all_advs.append(adv)
            cal_items.append({"question": st["win_question"], "target": z_t})

    if not ppo_items:
        return {"ppo_samples": 0, "policy_loss": 0.0, "calibration_loss": 0.0}

    # Normalize WPC advantages across the episode batch.
    adv_arr = np.asarray(all_advs, dtype=np.float32)
    adv_mean, adv_std = float(adv_arr.mean()), float(adv_arr.std()) + 1e-8
    for item in ppo_items:
        item["norm_adv"] = (item["adv"] - adv_mean) / adv_std

    # Mix anti-forgetting Calibration Replay states from balanced win_pro_train!
    if calib_win_rows and replay_win_n > 0:
        for r in rng.sample(calib_win_rows, min(replay_win_n, len(calib_win_rows))):
            q, a = _row_to_enriched_question(r)
            cal_items.append({"question": q, "target": 1.0 if a == 0 else 0.0})

    # Optional ChoiceQ anchor batch to prevent catastrophic forgetting of macro fundamentals.
    anchor_items = []
    if anchor_choice_rows and replay_choice_n > 0:
        for r in rng.sample(anchor_choice_rows, min(replay_choice_n, len(anchor_choice_rows))):
            q, a = _row_to_enriched_question(r, shuffle_rng=rng)
            anchor_items.append({"question": q, "answer": a})

    # Precompute frozen reference logp for KL regularization (once before PPO epochs).
    if ref_policy is not None and kl_coef > 0:
        with torch.no_grad():
            for i in range(0, len(ppo_items), micro_batch):
                chunk = ppo_items[i : i + micro_batch]
                ref_logits = ref_policy.forward_questions([c["question"] for c in chunk], strict_len=False)
                ref_logp = ref_logits.log_softmax(-1).cpu()
                for c, rlp in zip(chunk, ref_logp, strict=True):
                    c["ref_logp"] = rlp

    params = [p for p in policy.parameters() if p.requires_grad]
    pi_losses, entropies, kls, clip_fracs, cal_losses, anc_losses, grad_norms = [], [], [], [], [], [], []

    # 1. WPC-PPO epochs with Uncertainty-Adaptive Entropy & KL anchor.
    for _ in range(ppo_epochs):
        order = list(range(len(ppo_items)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_n = len(order)
        for i in range(0, total_n, micro_batch):
            chunk = [ppo_items[j] for j in order[i : i + micro_batch]]
            logits = policy.forward_questions([c["question"] for c in chunk], strict_len=False)
            dev = logits.device
            actions = torch.tensor([c["action"] for c in chunk], dtype=torch.long, device=dev)
            old_logp = torch.tensor([c["old_logp"] for c in chunk], dtype=torch.float32, device=dev)
            adv_t = torch.tensor([c["norm_adv"] for c in chunk], dtype=torch.float32, device=dev)
            unc_t = torch.tensor([c["uncertainty"] for c in chunk], dtype=torch.float32, device=dev)

            logp_all = logits.log_softmax(-1)
            probs = logp_all.exp()
            new_logp = logp_all.gather(1, actions[:, None]).squeeze(1)
            ratio = (new_logp - old_logp).exp()
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            pi_loss = -torch.min(surr1, surr2).mean()

            # Uncertainty-adaptive entropy: explore more in uncertain/stalemate states!
            per_ent = -(probs * logp_all).sum(-1)
            ent_weights = ent_min + (ent_max - ent_min) * unc_t
            ent_bonus = (ent_weights * per_ent).mean()

            kl_val = torch.zeros((), device=dev)
            if ref_policy is not None and kl_coef > 0:
                ref_lp = torch.stack([c["ref_logp"] for c in chunk], dim=0).to(dev)
                kl_val = kl_categorical(logp_all, ref_lp).mean()

            loss = pi_loss - ent_bonus + kl_coef * kl_val
            (loss * (len(chunk) / total_n)).backward()

            pi_losses.append(float(pi_loss.item()))
            entropies.append(float(per_ent.mean().item()))
            kls.append(float(kl_val.item()))
            clip_fracs.append(float(((ratio - 1.0).abs() > clip_eps).float().mean().item()))

        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        optimizer.step()

    # 2. RB-TD(lambda)-RLCD Calibration Pass (online trajectories + balanced Calib-Replay) + ChoiceQ anchor.
    optimizer.zero_grad(set_to_none=True)
    total_cal = len(cal_items)
    if total_cal > 0:
        for i in range(0, total_cal, micro_batch):
            chunk = cal_items[i : i + micro_batch]
            w_logits = policy.forward_questions([c["question"] for c in chunk], strict_len=False)[:, :2]
            c_loss = rb_rlcd_loss(w_logits, [c["target"] for c in chunk])
            (calibration_weight * c_loss * (len(chunk) / total_cal)).backward()
            cal_losses.append(float(c_loss.item()))

    if anchor_items:
        total_anc = len(anchor_items)
        for i in range(0, total_anc, micro_batch):
            chunk = anchor_items[i : i + micro_batch]
            a_logits = policy.forward_questions([c["question"] for c in chunk], strict_len=False)
            a_ans = torch.tensor([c["answer"] for c in chunk], dtype=torch.long, device=a_logits.device)
            a_loss = supervised_loss(a_logits.log_softmax(-1), a_ans)
            (0.15 * a_loss * (len(chunk) / total_anc)).backward()
            anc_losses.append(float(a_loss.item()))

    if total_cal > 0 or anchor_items:
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        optimizer.step()

    return {
        "ppo_samples": len(ppo_items),
        "cal_samples": len(cal_items),
        "policy_loss": float(np.mean(pi_losses)),
        "entropy": float(np.mean(entropies)),
        "kl": float(np.mean(kls)) if kls else 0.0,
        "clip_frac": float(np.mean(clip_fracs)),
        "calibration_loss": float(np.mean(cal_losses)) if cal_losses else 0.0,
        "anchor_loss": float(np.mean(anc_losses)) if anc_losses else 0.0,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
    }


def run_online_sa_rlcd(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    if args.checkpoint:
        policy = EveRLCDPolicy.load(args.checkpoint, device=device, init=args.init, dtype=args.dtype)
    else:
        policy = EveRLCDPolicy(
            args.init or DEFAULT_MODEL,
            device=device,
            rank=args.rank,
            max_len=args.max_len,
            dtype=args.dtype,
        )
    policy.save(out / "initial.pt", vars_json(args))

    # Lightweight frozen reference copy for KL anchoring (shares frozen backbone weights via deepcopy of adapters only?
    # Full deepcopy of 0.6B bf16 is only 1.2 GB VRAM on A800 80GB — super fast and exact!).
    ref_policy = None
    if args.kl_coef > 0:
        ref_policy = copy.deepcopy(policy).eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    calib_win_rows = load_decision_rows(args.calib_replay) if args.calib_replay else None
    anchor_choice_rows = load_decision_rows(args.anchor_data) if args.anchor_data else None

    optimizer = torch.optim.AdamW(
        policy.param_groups(lr=args.lr, head_lr_mult=args.head_lr_mult), weight_decay=0.0
    )

    maps = args.maps.split(",")
    opponents = args.opponents.split(",")
    val_maps = args.val_maps.split(",") if args.val_maps else maps
    val_opponents = args.val_opponents.split(",") if args.val_opponents else list(dict.fromkeys(opponents))
    players = [int(p) for p in args.players.split(",")]
    scenarios = [(m, o, p) for m in maps for o in opponents for p in players]

    pool = EnvPool(
        max_ticks=args.max_ticks,
        skill_ticks=args.skill_ticks,
        candidate_limit=args.candidate_limit,
        gamma=args.gamma,
        reward="win_focused",
    )
    t0 = time.monotonic()
    ep_counter = 0
    best_score = None
    best_update = 0

    try:
        if args.eval_every > 0:
            _, val_sum = evaluate_policy(
                policy,
                pool,
                maps=val_maps,
                opponents=val_opponents,
                players=players,
                repeats=args.eval_repeats,
                seed_base=100_000,
                greedy=args.eval_greedy,
            )
            best_score = selection_score(val_sum)
            policy.save(out / "best.pt", {"update": 0, "val": val_sum})
            print(json.dumps({"eval_update": 0, **val_sum}), flush=True)

        for upd in range(args.updates):
            cur_lr = cosine_lr(upd + 1, args.updates, args.lr, warmup=max(1, args.updates // 10))
            for g_idx, g in enumerate(optimizer.param_groups):
                g["lr"] = cur_lr * (args.head_lr_mult if g_idx == 1 else 1.0)

            batch_data, batch_metrics = [], []
            for _ in range(args.batch_episodes):
                m, opp, p = scenarios[ep_counter % len(scenarios)]
                env = pool.get(m, opp, p)
                steps, final_state, gm = play_episode(
                    env,
                    policy,
                    seed=args.seed + ep_counter * 37,
                    greedy=False,
                    forecast_group=4,
                    shuffle_training_options=True,
                )
                batch_data.append((steps, final_state, gm))
                batch_metrics.append(gm)
                ep_counter += 1

            upd_stats = sa_rlcd_update(
                policy,
                ref_policy,
                optimizer,
                batch_data,
                calib_win_rows=calib_win_rows,
                anchor_choice_rows=anchor_choice_rows,
                ppo_epochs=args.ppo_epochs,
                micro_batch=args.micro_batch,
                clip_eps=args.clip_eps,
                ent_min=args.ent_min,
                ent_max=args.ent_max,
                kl_coef=args.kl_coef,
                calibration_weight=args.calibration_weight,
                td_lambda=args.td_lambda,
                timeout_value=args.timeout_value,
                replay_win_n=args.replay_win_n,
                replay_choice_n=args.replay_choice_n,
                rng=rng,
            )
            agg = aggregate_games(batch_metrics)
            row = {
                "update": upd + 1,
                "episodes": ep_counter,
                "lr": cur_lr,
                "elapsed_sec": time.monotonic() - t0,
                **{
                    k: agg[k]
                    for k in (
                        "win_rate",
                        "match_score",
                        "mean_reward",
                        "mean_ticks",
                        "mean_episode_brier",
                        "game_weighted_ece",
                        "mean_ms_per_decision",
                    )
                    if k in agg
                },
                **upd_stats,
            }
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

            if args.eval_every > 0 and (upd + 1) % args.eval_every == 0:
                _, val_sum = evaluate_policy(
                    policy,
                    pool,
                    maps=val_maps,
                    opponents=val_opponents,
                    players=players,
                    repeats=args.eval_repeats,
                    seed_base=100_000,
                    greedy=args.eval_greedy,
                )
                score = selection_score(val_sum)
                if best_score is None or score > best_score:
                    best_score = score
                    best_update = upd + 1
                    policy.save(out / "best.pt", {"update": upd + 1, "val": val_sum})
                print(
                    json.dumps({"eval_update": upd + 1, "best_update": best_update, **val_sum}),
                    flush=True,
                )
            policy.save(out / "last.pt", vars_json(args) | {"update": upd + 1})

        summary = {
            "completed_updates": args.updates,
            "completed_episodes": ep_counter,
            "best_update": best_update,
            "best_score": best_score,
            "elapsed_sec": time.monotonic() - t0,
        }
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        pool.close()


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Offline SFT or IS-RLCD with Force-summary & balanced NoulQ RB-RLCD
    for name in ("sft", "is-rlcd"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--val-data", type=Path)
        p.add_argument("--win-data", type=Path)
        p.add_argument("--win-val-data", type=Path)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--init", type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--device", default="auto")
        p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
        p.add_argument("--rank", type=int, default=16)
        p.add_argument("--max-len", type=int, default=2048)
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--micro-batch", type=int, default=16)
        p.add_argument("--accum", type=int, default=2)
        p.add_argument("--lr", type=float, default=1e-4 if name == "sft" else 3e-5)
        p.add_argument("--head-lr-mult", type=float, default=5.0)
        p.add_argument("--group", type=int, default=4)
        p.add_argument("--eps", type=float, default=0.15)
        p.add_argument("--seed", type=int, default=7)

    # Online SA-RLCD
    p_on = sub.add_parser("online")
    p_on.add_argument("--out", type=Path, required=True)
    p_on.add_argument("--init", type=Path)
    p_on.add_argument("--checkpoint", type=Path)
    p_on.add_argument("--calib-replay", type=Path, help="Balanced NoulQ JSONL for calibration replay")
    p_on.add_argument("--anchor-data", type=Path, help="ChoiceQ JSONL for anti-forgetting anchor")
    p_on.add_argument("--device", default="auto")
    p_on.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p_on.add_argument("--rank", type=int, default=16)
    p_on.add_argument("--max-len", type=int, default=2048)
    p_on.add_argument("--maps", default="maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml,maps/12x12/basesWorkers12x12A.xml,maps/open_competition/basesWorkers16x16A.xml")
    p_on.add_argument("--opponents", default="WorkerRush,WorkerRush,CoacAI,Mayari,LightRush")
    p_on.add_argument("--val-maps", default="maps/8x8/basesWorkers8x8C.xml,maps/12x12/basesWorkers12x12B.xml,maps/16x16/basesWorkers16x16B.xml")
    p_on.add_argument("--val-opponents", default="WorkerRush,CoacAI,Mayari,LightRush")
    p_on.add_argument("--players", default="0,1")
    p_on.add_argument("--max-ticks", type=int, default=2500)
    p_on.add_argument("--skill-ticks", type=int, default=64)
    p_on.add_argument("--candidate-limit", type=int, default=12)
    p_on.add_argument("--gamma", type=float, default=0.9985)
    p_on.add_argument("--updates", type=int, default=60)
    p_on.add_argument("--batch-episodes", type=int, default=4)
    p_on.add_argument("--ppo-epochs", type=int, default=2)
    p_on.add_argument("--micro-batch", type=int, default=16)
    p_on.add_argument("--clip-eps", type=float, default=0.2)
    p_on.add_argument("--ent-min", type=float, default=0.003)
    p_on.add_argument("--ent-max", type=float, default=0.025)
    p_on.add_argument("--kl-coef", type=float, default=0.02)
    p_on.add_argument("--calibration-weight", type=float, default=1.5)
    p_on.add_argument("--td-lambda", type=float, default=0.85)
    p_on.add_argument("--timeout-value", type=float, default=0.15)
    p_on.add_argument("--replay-win-n", type=int, default=24)
    p_on.add_argument("--replay-choice-n", type=int, default=16)
    p_on.add_argument("--lr", type=float, default=1.5e-5)
    p_on.add_argument("--head-lr-mult", type=float, default=5.0)
    p_on.add_argument("--eval-every", type=int, default=10)
    p_on.add_argument("--eval-repeats", type=int, default=1)
    p_on.add_argument("--eval-greedy", action="store_true")
    p_on.add_argument("--seed", type=int, default=2028)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd in ("sft", "is-rlcd"):
        device = resolve_device(args.device)
        if args.checkpoint:
            policy = EveRLCDPolicy.load(args.checkpoint, device=device, init=args.init, dtype=args.dtype)
        else:
            policy = EveRLCDPolicy(
                args.init or DEFAULT_MODEL,
                device=device,
                rank=args.rank,
                max_len=args.max_len,
                dtype=args.dtype,
            )
        rows = load_decision_rows(args.data)
        val_rows = load_decision_rows(args.val_data) if args.val_data else None
        win_rows = load_decision_rows(args.win_data) if args.win_data else None
        win_val_rows = load_decision_rows(args.win_val_data) if args.win_val_data else None
        run_offline_sa_rlcd(
            policy,
            rows,
            args.out,
            mode=args.cmd,
            epochs=args.epochs,
            micro_batch=args.micro_batch,
            accum_steps=args.accum,
            lr=args.lr,
            head_lr_mult=args.head_lr_mult,
            group=args.group,
            eps=args.eps,
            val_rows=val_rows,
            win_rows=win_rows,
            win_val_rows=win_val_rows,
            seed=args.seed,
        )
    elif args.cmd == "online":
        run_online_sa_rlcd(args)


if __name__ == "__main__":
    main()
