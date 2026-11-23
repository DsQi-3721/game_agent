"""Unified EVE-RLCD MicroRTS pipeline: collect -> sft -> bandit-rlcd / online-rl -> eval / export / bench."""

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from rlcd.loop import GradWindow, batch_indices, cosine_lr
from rlcd.metrics import brier, ece
from rlcd.rewards import supervised_loss
from rlcd.schema import Question

from microrts_agent.registries.ai import AI_MAPPING

from .env import SkillEnv
from .eve_policy import (
    DEFAULT_MODEL,
    MODEL_ID,
    MODEL_REVISION,
    EveRLCDPolicy,
    forecast_loss,
    resolve_device,
    rlcd_bandit_loss,
    shuffle_candidates,
    unit_choice_question,
    win_question,
)
from .metrics import (
    aggregate_games,
    atomic_json,
    compute_smdp_gae,
    episode_outcome,
    forecast_summary,
    selection_score,
)
from .state import state_potential, state_to_text


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cpu_tree(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, dict):
        return {k: _cpu_tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_cpu_tree(x) for x in v]
    return v


class EnvPool:
    """Reuse SkillEnv instances keyed by (map, opponent, player) to avoid JVM churn."""

    def __init__(self, **common_kwargs):
        self.common = common_kwargs
        self._envs = {}

    def get(self, map_path, opponent, player):
        key = (map_path, opponent, int(player))
        if key not in self._envs:
            self._envs[key] = SkillEnv(
                map_path=map_path, opponent=opponent, player=int(player), **self.common
            )
        return self._envs[key]

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs.clear()


# =====================================================================
# Stage 0: Collect expert skill trajectories from built-in MicroRTS bots
# =====================================================================


def collect_expert_dataset(
    out_path,
    *,
    teachers=("WorkerRush", "CoacAI"),
    opponents=("RandomBiasedAI", "LightRush", "WorkerRush"),
    maps=("maps/8x8/basesWorkers8x8A.xml", "maps/8x8/basesWorkers8x8B.xml"),
    players=(0, 1),
    games_per_combo=2,
    max_ticks=2000,
    skill_ticks=64,
    candidate_limit=12,
    seed=7,
    keep_wins_only=False,
    dedup_waits=True,
    all_out_path=None,
):
    """Run Java expert bots inside multi-unit SkillEnv and export clean Question JSONL rows."""
    from microrts_agent.envs.base_vec_env import suppress_java_output

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_out_path is not None:
        all_out_path = Path(all_out_path)
        all_out_path.parent.mkdir(parents=True, exist_ok=True)

    pool = EnvPool(
        max_ticks=max_ticks, skill_ticks=skill_ticks, candidate_limit=candidate_limit
    )
    rows = []
    all_rows = []
    game_stats = []
    ep_idx = 0
    skipped_waits = 0
    silenced = False
    try:
        for map_path in maps:
            map_games = 0
            map_wins = 0
            for teacher_name in teachers:
                for opp_name in opponents:
                    for player in players:
                        env = pool.get(map_path, opp_name, player)
                        if not silenced:
                            suppress_java_output()
                            silenced = True
                        for _rep in range(games_per_combo):
                            ep_seed = seed + ep_idx * 17
                            state = env.reset(seed=ep_seed)
                            map_size = f"{state['width']}x{state['height']}"
                            teacher = AI_MAPPING[teacher_name](env.base.real_utt)
                            has_abs = hasattr(teacher, "getAbstractAction")
                            ep_rows = []
                            last_skill_by_unit = {}
                            while not env.done:
                                idle = env.idle_unit_candidates(state)
                                assignments = {}
                                if idle:
                                    ctx = state_to_text(state)
                                    cloned = env.gs.clone()
                                    pa = teacher.getAction(env.player, cloned)
                                    by_id = {
                                        int(pair.m_a.getID()): pair.m_b
                                        for pair in pa.getActions()
                                    }
                                    for u, cands in idle:
                                        ju = env.gs.getUnit(u["id"])
                                        cju = cloned.getUnit(u["id"])
                                        bact = by_id.get(u["id"])
                                        aact = (
                                            teacher.getAbstractAction(cju)
                                            if (has_abs and cju is not None)
                                            else None
                                        )
                                        idx, match_kind = env.skills.match_bot_skill(
                                            env.gs,
                                            ju,
                                            bact,
                                            cands,
                                            abstract_action=aact,
                                            return_reason=True,
                                        )
                                        chosen = cands[idx]
                                        assignments[u["id"]] = chosen
                                        skill_name = chosen["skill"]
                                        prev = last_skill_by_unit.get(u["id"])
                                        last_skill_by_unit[u["id"]] = skill_name
                                        # Keep first wait in an idle streak, drop spam repeats.
                                        if dedup_waits and skill_name == "wait" and prev == "wait":
                                            skipped_waits += 1
                                            continue
                                        q = unit_choice_question(
                                            state, u, cands, state_text=ctx
                                        )
                                        ep_rows.append(
                                            {
                                                "primitive": "choice",
                                                "context": q.context,
                                                "question": q.question,
                                                "choices": q.choices,
                                                "answer": idx,
                                                "skill": skill_name,
                                                "unit_type": u["type"],
                                                "match_kind": match_kind,
                                                "teacher": teacher_name,
                                                "opponent": opp_name,
                                                "map": map_path,
                                                "map_size": map_size,
                                                "player": player,
                                                "episode": ep_idx,
                                                "seed": ep_seed,
                                                "tick": state["time"],
                                            }
                                        )
                                res = env.step_units(assignments)
                                state = res["next_state"]
                            outcome = episode_outcome(
                                res["terminated"], res["truncated"], res["winner"], player
                            )
                            won = outcome == "win"
                            map_games += 1
                            map_wins += int(won)
                            game_stats.append(
                                {
                                    "episode": ep_idx,
                                    "seed": ep_seed,
                                    "teacher": teacher_name,
                                    "opponent": opp_name,
                                    "map": map_path,
                                    "map_size": map_size,
                                    "player": player,
                                    "outcome": outcome,
                                    "terminated": res["terminated"],
                                    "ticks": state["time"],
                                    "decisions": len(ep_rows),
                                }
                            )
                            for r in ep_rows:
                                r["won"] = won
                                r["outcome"] = outcome
                                r["terminated"] = res["terminated"]
                            if all_out_path is not None:
                                all_rows.extend(ep_rows)
                            if (not keep_wins_only) or won:
                                rows.extend(ep_rows)
                            ep_idx += 1
            print(
                json.dumps(
                    {
                        "map": Path(map_path).name,
                        "games": map_games,
                        "wins": map_wins,
                        "total_rows": len(rows),
                    }
                ),
                flush=True,
            )
    finally:
        pool.close()

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    if all_out_path is not None:
        with all_out_path.open("w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    wins = sum(g["outcome"] == "win" for g in game_stats)
    losses = sum(g["outcome"] == "loss" for g in game_stats)
    draws = sum(g["outcome"] == "draw" for g in game_stats)
    timeouts = sum(g["outcome"] == "timeout" for g in game_stats)
    summary = {
        "path": str(out_path),
        "all_path": str(all_out_path) if all_out_path else None,
        "games": len(game_stats),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": wins / max(len(game_stats), 1),
        "rows": len(rows),
        "all_rows": len(all_rows) if all_out_path else len(rows),
        "skipped_consecutive_waits": skipped_waits,
    }
    atomic_json(out_path.with_suffix(".summary.json"), summary | {"games_detail": game_stats})
    return summary


def load_decision_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_question(row, shuffle_rng=None):
    choices = list(row["choices"])
    ans = int(row["answer"])
    if shuffle_rng is not None and row.get("primitive", "choice") == "choice" and len(choices) > 1:
        choices, _, ans = shuffle_candidates(choices, chosen_idx=ans, rng=shuffle_rng)
    q = Question(
        row.get("primitive", "choice"),
        row["context"],
        row["question"],
        choices,
        answer=ans,
    ).validate()
    return q, ans


def evaluate_dataset(policy, rows, batch_size=8):
    """Evaluate classification accuracy, Brier, and ECE on offline MicroRTS decision rows."""
    if not rows:
        return {}
    all_probs, all_ans = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            qs = [_row_to_question(r)[0] for r in chunk]
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


# =====================================================================
# Stage 1: SFT Warmup & Stage 2a: Faithful Offline Bandit RLCD
# =====================================================================


def run_offline_stage(
    policy,
    rows,
    out_dir,
    *,
    mode="sft",
    arm="rlcd",
    epochs=2,
    micro_batch=4,
    accum_steps=4,
    lr=5e-5,
    head_lr_mult=5.0,
    group=4,
    include_win_noul=True,
    val_rows=None,
    seed=7,
):
    """Unified offline trainer for both Stage 1 SFT Warmup and Stage 2a Bandit-RLCD/RLVR/Oracle."""
    _seed_all(seed)
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally include binary win-forecast NoulQ rows from terminated games.
    train_items = list(rows)
    if include_win_noul:
        seen_states = set()
        for r in rows:
            if not r.get("terminated"):
                continue
            key = (r.get("episode"), r.get("tick"))
            if key in seen_states:
                continue
            seen_states.add(key)
            train_items.append(
                {
                    "primitive": "noul",
                    "context": r["context"],
                    "question": (
                        "Following the current sampled policy against this opponent, "
                        "will our player win this match? A draw counts as false."
                    ),
                    "choices": ["true", "false"],
                    "answer": 0 if r.get("won") else 1,
                }
            )

    optimizer = torch.optim.AdamW(
        policy.param_groups(lr=lr, head_lr_mult=head_lr_mult), weight_decay=0.0
    )
    n_micro = epochs * ((len(train_items) + micro_batch - 1) // micro_batch)
    total_steps = max(1, (n_micro + accum_steps - 1) // accum_steps)
    warmup = max(1, total_steps // 10)
    grads = GradWindow()
    params = [p for p in policy.parameters() if p.requires_grad]

    history = []
    val_history = []
    best_val_brier = None
    step = 0
    loss_acc, acc_sum, n_seen = 0.0, 0.0, 0
    t0 = time.monotonic()

    if val_rows:
        v0 = evaluate_dataset(policy, val_rows, batch_size=max(8, micro_batch))
        val_history.append({"epoch": 0, "step": 0, **v0})
        best_val_brier = v0["brier"]
        policy.save(out_dir / "best.pt", {"mode": mode, "arm": arm, "step": 0, "val": v0})
        print(json.dumps({"val_epoch": 0, "step": 0, **v0}), flush=True)

    def finish_step(ep):
        nonlocal step, loss_acc, acc_sum, n_seen
        n = grads.finish(params)
        cur_lr = cosine_lr(step + 1, total_steps, lr, warmup=warmup)
        for g_idx, g in enumerate(optimizer.param_groups):
            g["lr"] = cur_lr * (head_lr_mult if g_idx == 1 else 1.0)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {
            "step": step,
            "epoch": ep,
            "mode": mode,
            "arm": arm if mode == "bandit" else "sft",
            "loss": loss_acc / max(n, 1),
            "acc": acc_sum / max(n, 1),
            "grad_norm": grad_norm,
            "lr": cur_lr,
            "elapsed_sec": time.monotonic() - t0,
        }
        history.append(rec)
        loss_acc, acc_sum, n_seen = 0.0, 0.0, 0
        if step <= 5 or step % 10 == 0 or step == total_steps:
            print(json.dumps(rec), flush=True)

    for ep in range(epochs):
        for idx in batch_indices(len(train_items), micro_batch, rng):
            qs, ans_list = [], []
            for i in idx:
                q, a = _row_to_question(train_items[i], shuffle_rng=rng)
                qs.append(q)
                ans_list.append(a)
            logits = policy.forward_questions(qs, strict_len=False)
            ans_t = torch.tensor(ans_list, dtype=torch.long, device=logits.device)
            logp = logits.log_softmax(-1)
            probs = logp.exp()

            if mode == "sft" or arm == "oracle":
                loss = supervised_loss(logp, ans_t)
            else:
                actions = torch.multinomial(probs.detach(), group, replacement=True)
                outcomes = (actions == ans_t[:, None]).float()
                loss = rlcd_bandit_loss(logits, actions, outcomes, arm=arm)

            grads.backward(loss, len(idx))
            with torch.no_grad():
                loss_acc += float(loss.item()) * len(idx)
                acc_sum += float((probs.argmax(-1) == ans_t).float().sum().item())
                n_seen += len(idx)
            if grads.micro == accum_steps:
                finish_step(ep)
        if grads.micro:
            finish_step(ep)
        if val_rows:
            ve = evaluate_dataset(policy, val_rows, batch_size=max(8, micro_batch))
            val_history.append({"epoch": ep + 1, "step": step, **ve})
            if best_val_brier is None or ve["brier"] < best_val_brier:
                best_val_brier = ve["brier"]
                policy.save(
                    out_dir / "best.pt", {"mode": mode, "arm": arm, "step": step, "val": ve}
                )
            print(json.dumps({"val_epoch": ep + 1, "step": step, **ve}), flush=True)

    metrics = {"steps": step, "elapsed_sec": time.monotonic() - t0, "train_rows": len(train_items)}
    if val_history:
        metrics["val"] = val_history[-1]
        metrics["val_history"] = val_history
    policy.save(out_dir / "policy.pt", {"mode": mode, "arm": arm, **metrics})
    atomic_json(out_dir / "summary.json", metrics)
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    return metrics


# =====================================================================
# Multi-unit Rollout & Evaluation (SharedPrefixCache fast inference)
# =====================================================================


def play_episode(
    env,
    policy,
    *,
    seed=0,
    greedy=False,
    forecast_group=4,
    shuffle_training_options=False,
):
    """Play one full MicroRTS episode with multi-unit concurrent SkillEnv + SharedPrefixCache."""
    state = env.reset(seed=seed)
    forecast_rng = torch.Generator(device="cpu").manual_seed(seed + 10_000_000)
    shuffle_rng = random.Random(seed + 20_000_000) if shuffle_training_options else None

    steps = []
    win_probs = []
    latencies_ms = []
    prefill_ms_list = []
    suffix_ms_list = []
    total_unit_decisions = 0

    while not env.done:
        idle = env.idle_unit_candidates(state)
        if not idle:
            res = env.step_units({})
            state = res["next_state"]
            if steps:
                # Fold trailing tick rewards after the last decision into the last step.
                steps[-1]["reward"] += res["reward"]
                steps[-1]["discounted_reward"] += steps[-1]["discount"] * res["discounted_reward"]
                steps[-1]["duration"] += res["duration"]
                steps[-1]["discount"] *= res["discount"]
                steps[-1]["terminated"] = res["terminated"]
                steps[-1]["truncated"] = res["truncated"]
                steps[-1]["winner"] = res["winner"]
            continue

        dec = policy.decide_state(
            state,
            idle,
            include_win_forecast=True,
            greedy=greedy,
            forecast_group=forecast_group,
            forecast_rng=forecast_rng,
            shuffle_rng=shuffle_rng,
        )
        assignments = {d["unit_id"]: d["chosen_skill"] for d in dec["unit_decisions"]}
        res = env.step_units(assignments)

        latencies_ms.append(dec["total_ms"])
        prefill_ms_list.append(dec["prefill_ms"])
        suffix_ms_list.append(dec["suffix_ms"])
        win_probs.append(dec["win_probability"])
        total_unit_decisions += len(dec["unit_decisions"])

        steps.append(
            {
                "unit_decisions": dec["unit_decisions"],
                "win_question": dec["win_question"],
                "win_probability": dec["win_probability"],
                "forecast_probs": dec["forecast_probs"],
                "forecast_answers": dec["forecast_answers"],
                "value": dec["state_value"],
                "state_potential": state_potential(state, scale=1.0),
                "reward": res["reward"],
                "discounted_reward": res["discounted_reward"],
                "duration": res["duration"],
                "discount": res["discount"],
                "terminated": res["terminated"],
                "truncated": res["truncated"],
                "winner": res["winner"],
            }
        )
        state = res["next_state"]

    final = steps[-1] if steps else res
    outcome = episode_outcome(
        final["terminated"], final["truncated"], final["winner"], env.player
    )
    won = outcome == "win"
    metrics = {
        "map": env.map_path,
        "opponent": env.opponent,
        "player": env.player,
        "seed": seed,
        "outcome": outcome,
        "winner": final["winner"],
        "terminated": final["terminated"],
        "truncated": final["truncated"],
        "ticks": state["time"],
        "decisions": len(steps),
        "unit_decisions": total_unit_decisions,
        "reward": sum(s["reward"] for s in steps),
        "ms_per_decision": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "prefill_ms": float(np.mean(prefill_ms_list)) if prefill_ms_list else 0.0,
        "suffix_ms": float(np.mean(suffix_ms_list)) if suffix_ms_list else 0.0,
        **forecast_summary(win_probs, final["terminated"], won),
    }
    return steps, state, metrics


def evaluate_policy(
    policy,
    pool,
    *,
    maps,
    opponents,
    players,
    repeats=4,
    seed_base=100_000,
    greedy=False,
    forecast_group=4,
):
    """Evaluate policy across all (map, opponent, player, repeat) matchups."""
    games = []
    idx = 0
    for m in maps:
        for opp in opponents:
            for p in players:
                env = pool.get(m, opp, p)
                for rep in range(repeats):
                    _, _, gm = play_episode(
                        env,
                        policy,
                        seed=seed_base + idx * 100 + rep,
                        greedy=greedy,
                        forecast_group=forecast_group,
                    )
                    games.append(gm)
                idx += 1
    return games, aggregate_games(games)


# =====================================================================
# Stage 2b: Online MicroRTS PPO + RLCD Win-Calibration
# =====================================================================


def ppo_rlcd_update(
    policy,
    optimizer,
    episode_batch,
    *,
    ppo_epochs=2,
    micro_batch=4,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    arm="rlcd",
    calibration_weight=1.0,
    gae_lambda=0.95,
):
    """Batched clipped PPO on unit skill decisions + exact RLCD on win-probability forecasts."""
    # Flatten unit-level PPO samples and state-level RLCD forecast samples across the episode batch.
    ppo_items = []
    forecast_items = []
    all_advs = []

    for steps, final_state, metrics in episode_batch:
        if not steps:
            continue
        bootstrap = 0.0
        if steps[-1]["truncated"]:
            with torch.no_grad():
                wq = win_question(final_state, greedy=False)
                _, v = policy.forward_questions([wq], return_values=True, strict_len=False)
                bootstrap = float(v[0].item())
        advs, rets = compute_smdp_gae(steps, bootstrap=bootstrap, gae_lambda=gae_lambda)
        won = metrics["outcome"] == "win"
        for step, adv, ret in zip(steps, advs, rets, strict=True):
            for ud in step["unit_decisions"]:
                ppo_items.append(
                    {
                        "question": ud["question"],
                        "k": len(ud["candidates"]),
                        "action": ud["chosen_index"],
                        "old_logp": ud["log_prob"],
                        "adv": adv,
                        "ret": ret,
                    }
                )
                all_advs.append(adv)
            if arm != "none" and metrics["terminated"] and calibration_weight > 0:
                forecast_items.append(
                    {
                        "question": step["win_question"],
                        "answers": step["forecast_answers"],
                        "won": won,
                    }
                )

    if not ppo_items:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "calibration_loss": None}

    adv_arr = np.asarray(all_advs, dtype=np.float32)
    adv_mean = float(adv_arr.mean())
    adv_std = float(adv_arr.std()) + 1e-8
    for item in ppo_items:
        item["norm_adv"] = (item["adv"] - adv_mean) / adv_std

    params = [p for p in policy.parameters() if p.requires_grad]
    rng = random.Random(len(ppo_items))
    pi_losses, v_losses, entropies, clip_fracs, cal_losses = [], [], [], [], []
    grad_norms = []

    # 1. PPO epochs over unit decisions (batched forward/backward, gradient accumulation).
    for _ in range(ppo_epochs):
        order = list(range(len(ppo_items)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_items = len(order)
        for i in range(0, total_items, micro_batch):
            chunk = [ppo_items[j] for j in order[i : i + micro_batch]]
            qs = [c["question"] for c in chunk]
            logits, values = policy.forward_questions(qs, return_values=True, strict_len=False)
            dev = logits.device
            actions = torch.tensor([c["action"] for c in chunk], dtype=torch.long, device=dev)
            old_logp = torch.tensor([c["old_logp"] for c in chunk], dtype=torch.float32, device=dev)
            adv_t = torch.tensor([c["norm_adv"] for c in chunk], dtype=torch.float32, device=dev)
            ret_t = torch.tensor([c["ret"] for c in chunk], dtype=torch.float32, device=dev)

            logp_all = logits.log_softmax(-1)
            probs = logp_all.exp()
            new_logp = logp_all.gather(1, actions[:, None]).squeeze(1)
            ratio = (new_logp - old_logp).exp()
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = torch.nn.functional.smooth_l1_loss(values, ret_t)
            entropy = -(probs * logp_all).sum(-1).mean()

            loss = pi_loss + value_coef * v_loss - entropy_coef * entropy
            (loss * (len(chunk) / total_items)).backward()

            pi_losses.append(float(pi_loss.item()))
            v_losses.append(float(v_loss.item()))
            entropies.append(float(entropy.item()))
            clip_fracs.append(float(((ratio - 1.0).abs() > clip_eps).float().mean().item()))

        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        optimizer.step()

    # 2. Exact RLCD win-forecast calibration pass (1 on-policy pass over terminated states).
    if forecast_items:
        optimizer.zero_grad(set_to_none=True)
        total_f = len(forecast_items)
        for i in range(0, total_f, micro_batch):
            chunk = forecast_items[i : i + micro_batch]
            qs = [c["question"] for c in chunk]
            logits = policy.forward_questions(qs, strict_len=False)[:, :2]
            c_loss = forecast_loss(
                logits,
                [c["answers"] for c in chunk],
                [c["won"] for c in chunk],
                arm=arm,
            )
            (calibration_weight * c_loss * (len(chunk) / total_f)).backward()
            cal_losses.append(float(c_loss.item()))
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        optimizer.step()

    return {
        "ppo_samples": len(ppo_items),
        "forecast_samples": len(forecast_items),
        "policy_loss": float(np.mean(pi_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": float(np.mean(entropies)),
        "clip_frac": float(np.mean(clip_fracs)),
        "calibration_loss": float(np.mean(cal_losses)) if cal_losses else None,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
    }


def run_online_rl(args):
    _seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    resume_path = out / "resume.pt"
    if args.resume and resume_path.exists():
        policy = EveRLCDPolicy.load(resume_path, device=device, init=args.init, dtype=args.dtype)
        saved_state = torch.load(resume_path, map_location="cpu", weights_only=True)["training_state"]
        start_update = saved_state["update"]
        best_score = saved_state["best_score"]
        best_update = saved_state["best_update"]
    else:
        if args.checkpoint:
            policy = EveRLCDPolicy.load(
                args.checkpoint, device=device, init=args.init, dtype=args.dtype
            )
        else:
            policy = EveRLCDPolicy(
                args.init or DEFAULT_MODEL,
                device=device,
                rank=args.rank,
                max_len=args.max_len,
                dtype=args.dtype,
            )
        saved_state = None
        start_update = 0
        best_score = None
        best_update = 0
        policy.save(out / "initial.pt", vars_json(args))

    optimizer = torch.optim.AdamW(
        policy.param_groups(lr=args.lr, head_lr_mult=args.head_lr_mult), weight_decay=0.0
    )
    if saved_state and "optimizer" in saved_state:
        optimizer.load_state_dict(saved_state["optimizer"])

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
        reward=args.reward,
    )
    t0 = time.monotonic()
    ep_counter = start_update * args.batch_episodes

    try:
        # Initial validation baseline at step 0.
        if start_update == 0 and args.eval_every > 0:
            _, val_sum = evaluate_policy(
                policy,
                pool,
                maps=val_maps,
                opponents=val_opponents,
                players=players,
                repeats=args.eval_repeats,
                seed_base=100_000,
                greedy=args.eval_greedy,
                forecast_group=args.group,
            )
            best_score = selection_score(val_sum)
            policy.save(out / "best.pt", {"update": 0, "val": val_sum})
            print(json.dumps({"eval_update": 0, **val_sum}), flush=True)

        for upd in range(start_update, args.updates):
            if args.max_minutes > 0 and (time.monotonic() - t0) >= args.max_minutes * 60:
                break

            # Cosine LR decay across updates.
            cur_lr = cosine_lr(
                upd + 1, args.updates, args.lr, warmup=max(1, args.updates // 10)
            )
            for g_idx, g in enumerate(optimizer.param_groups):
                g["lr"] = cur_lr * (args.head_lr_mult if g_idx == 1 else 1.0)

            batch_data = []
            batch_metrics = []
            for _ in range(args.batch_episodes):
                m, opp, p = scenarios[ep_counter % len(scenarios)]
                env = pool.get(m, opp, p)
                steps, final_state, gm = play_episode(
                    env,
                    policy,
                    seed=args.seed + ep_counter * 31,
                    greedy=False,
                    forecast_group=args.group,
                    shuffle_training_options=True,
                )
                batch_data.append((steps, final_state, gm))
                batch_metrics.append(gm)
                ep_counter += 1

            upd_stats = ppo_rlcd_update(
                policy,
                optimizer,
                batch_data,
                ppo_epochs=args.ppo_epochs,
                micro_batch=args.micro_batch,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy,
                arm=args.arm,
                calibration_weight=args.calibration_weight,
            )
            agg = aggregate_games(batch_metrics)
            row = {
                "update": upd + 1,
                "episodes": ep_counter,
                "lr": cur_lr,
                "elapsed_sec": time.monotonic() - t0,
                **{k: agg[k] for k in ("win_rate", "match_score", "mean_reward", "mean_ticks", "mean_ms_per_decision") if k in agg},
                **{k: agg[k] for k in ("mean_episode_brier", "game_weighted_ece") if k in agg},
                **upd_stats,
            }
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

            # Periodic validation + best checkpoint selection.
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
                    forecast_group=args.group,
                )
                score = selection_score(val_sum)
                if best_score is None or score > best_score:
                    best_score = score
                    best_update = upd + 1
                    policy.save(out / "best.pt", {"update": upd + 1, "val": val_sum})
                print(json.dumps({"eval_update": upd + 1, "best_update": best_update, **val_sum}), flush=True)

            # Save resumable checkpoint atomically after every update.
            policy.save(
                resume_path,
                vars_json(args) | {"update": upd + 1},
                {
                    "update": upd + 1,
                    "best_score": best_score,
                    "best_update": best_update,
                    "optimizer": _cpu_tree(optimizer.state_dict()),
                },
            )
            policy.save(out / "last.pt", vars_json(args) | {"update": upd + 1})

        summary = {
            "completed_updates": ep_counter // args.batch_episodes,
            "completed_episodes": ep_counter,
            "best_update": best_update,
            "best_score": best_score,
            "elapsed_sec": time.monotonic() - t0,
        }
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        pool.close()


# =====================================================================
# Fast Inference Benchmark
# =====================================================================


def run_benchmark(args):
    device = resolve_device(args.device)
    if args.checkpoint:
        policy = EveRLCDPolicy.load(args.checkpoint, device=device, init=args.init, dtype=args.dtype)
    else:
        policy = EveRLCDPolicy(
            args.init or DEFAULT_MODEL, device=device, rank=args.rank, dtype=args.dtype
        )
    env = SkillEnv(map_path=args.map, opponent=args.opponent, player=0, candidate_limit=12)
    try:
        state = env.reset(seed=args.seed)
        idle = env.idle_unit_candidates(state)
        state_text = state_to_text(state)
        qs = [unit_choice_question(state, u, c, state_text=state_text) for u, c in idle]
        qs.append(win_question(state, state_text=state_text))

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

        # Warmup
        with torch.no_grad():
            policy.decide_state(state, idle, include_win_forecast=True, greedy=True)
            sync()

        # 1. Cached SharedPrefixCache path
        t0 = time.perf_counter()
        prefills, suffixes = [], []
        for _ in range(args.iters):
            res = policy.decide_state(state, idle, include_win_forecast=True, greedy=True)
            sync()
            prefills.append(res["prefill_ms"])
            suffixes.append(res["suffix_ms"])
        cached_ms = (time.perf_counter() - t0) / args.iters * 1000.0

        # 2. Uncached full-prompt per-question path (old behavior)
        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.iters):
                for q in qs:
                    policy.forward_questions([q], strict_len=False)
                sync()
        uncached_ms = (time.perf_counter() - t1) / args.iters * 1000.0

        report = {
            "device": device,
            "dtype": str(policy.backbone_dtype),
            "questions": len(qs),
            "prefix_tokens": len(policy.rlcd.tok.encode(state_text, add_special_tokens=False)),
            "cached_total_ms": round(cached_ms, 2),
            "cached_ms_per_question": round(cached_ms / len(qs), 2),
            "prefill_ms": round(float(np.mean(prefills)), 2),
            "suffix_ms": round(float(np.mean(suffixes)), 2),
            "uncached_total_ms": round(uncached_ms, 2),
            "speedup": round(uncached_ms / max(cached_ms, 1e-6), 2),
        }
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


def vars_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download the pinned Qwen3-0.6B RLCD decision export")
    p_dl.add_argument("--init", type=Path, default=DEFAULT_MODEL)

    # collect
    p_col = sub.add_parser("collect", help="Collect expert MicroRTS skill decisions from Java bots")
    p_col.add_argument("--out", type=Path, required=True)
    p_col.add_argument("--all-out", type=Path, help="Optional path to also write all games (wins+losses)")
    p_col.add_argument("--teachers", default="WorkerRush,CoacAI")
    p_col.add_argument("--opponents", default="RandomBiasedAI,LightRush,WorkerRush")
    p_col.add_argument("--maps", default="maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml")
    p_col.add_argument("--players", default="0,1")
    p_col.add_argument("--games-per-combo", type=int, default=2)
    p_col.add_argument("--max-ticks", type=int, default=2000)
    p_col.add_argument("--skill-ticks", type=int, default=64)
    p_col.add_argument("--candidate-limit", type=int, default=12)
    p_col.add_argument("--keep-wins-only", action="store_true")
    p_col.add_argument("--no-dedup-waits", action="store_true")
    p_col.add_argument("--seed", type=int, default=7)

    # sft
    p_sft = sub.add_parser("sft", help="Stage 1: SFT warmup on collected MicroRTS decisions")
    p_sft.add_argument("--data", type=Path, required=True)
    p_sft.add_argument("--val-data", type=Path)
    p_sft.add_argument("--out", type=Path, required=True)
    p_sft.add_argument("--init", type=Path)
    p_sft.add_argument("--checkpoint", type=Path)
    p_sft.add_argument("--device", default="auto")
    p_sft.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p_sft.add_argument("--rank", type=int, default=8)
    p_sft.add_argument("--max-len", type=int, default=2048)
    p_sft.add_argument("--epochs", type=int, default=2)
    p_sft.add_argument("--micro-batch", type=int, default=4)
    p_sft.add_argument("--accum", type=int, default=4)
    p_sft.add_argument("--lr", type=float, default=5e-5)
    p_sft.add_argument("--head-lr-mult", type=float, default=5.0)
    p_sft.add_argument("--no-win-noul", action="store_true")
    p_sft.add_argument("--seed", type=int, default=7)

    # bandit-rlcd
    p_br = sub.add_parser("bandit-rlcd", help="Stage 2a: Exact eve-rlcd bandit training (rlcd/rlvr/oracle)")
    p_br.add_argument("--data", type=Path, required=True)
    p_br.add_argument("--val-data", type=Path)
    p_br.add_argument("--out", type=Path, required=True)
    p_br.add_argument("--init", type=Path)
    p_br.add_argument("--checkpoint", type=Path)
    p_br.add_argument("--arm", choices=["rlcd", "rlvr", "oracle"], default="rlcd")
    p_br.add_argument("--group", type=int, default=4)
    p_br.add_argument("--device", default="auto")
    p_br.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p_br.add_argument("--rank", type=int, default=8)
    p_br.add_argument("--max-len", type=int, default=2048)
    p_br.add_argument("--epochs", type=int, default=2)
    p_br.add_argument("--micro-batch", type=int, default=4)
    p_br.add_argument("--accum", type=int, default=4)
    p_br.add_argument("--lr", type=float, default=2e-5)
    p_br.add_argument("--head-lr-mult", type=float, default=5.0)
    p_br.add_argument("--no-win-noul", action="store_true")
    p_br.add_argument("--seed", type=int, default=7)

    # online-rl
    p_rl = sub.add_parser("online-rl", help="Stage 2b: Online MicroRTS PPO + RLCD win calibration")
    p_rl.add_argument("--out", type=Path, required=True)
    p_rl.add_argument("--init", type=Path)
    p_rl.add_argument("--checkpoint", type=Path)
    p_rl.add_argument("--resume", action="store_true")
    p_rl.add_argument("--device", default="auto")
    p_rl.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p_rl.add_argument("--rank", type=int, default=8)
    p_rl.add_argument("--max-len", type=int, default=2048)
    p_rl.add_argument("--maps", default="maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml")
    p_rl.add_argument("--opponents", default="RandomBiasedAI,WorkerRush")
    p_rl.add_argument("--val-maps", default="", help="Optional held-out maps for periodic validation")
    p_rl.add_argument("--val-opponents", default="", help="Optional opponents for periodic validation")
    p_rl.add_argument("--players", default="0,1")
    p_rl.add_argument("--max-ticks", type=int, default=2000)
    p_rl.add_argument("--skill-ticks", type=int, default=64)
    p_rl.add_argument("--candidate-limit", type=int, default=12)
    p_rl.add_argument("--gamma", type=float, default=0.99)
    p_rl.add_argument("--reward", choices=["shaped", "win_focused", "sparse"], default="shaped")
    p_rl.add_argument("--updates", type=int, default=50)
    p_rl.add_argument("--max-minutes", type=float, default=0.0)
    p_rl.add_argument("--batch-episodes", type=int, default=4)
    p_rl.add_argument("--ppo-epochs", type=int, default=2)
    p_rl.add_argument("--micro-batch", type=int, default=4)
    p_rl.add_argument("--clip-eps", type=float, default=0.2)
    p_rl.add_argument("--value-coef", type=float, default=0.5)
    p_rl.add_argument("--entropy", type=float, default=0.01)
    p_rl.add_argument("--lr", type=float, default=2e-5)
    p_rl.add_argument("--head-lr-mult", type=float, default=5.0)
    p_rl.add_argument("--arm", choices=["rlcd", "rlvr", "none"], default="rlcd")
    p_rl.add_argument("--group", type=int, default=4)
    p_rl.add_argument("--calibration-weight", type=float, default=1.0)
    p_rl.add_argument("--eval-every", type=int, default=5)
    p_rl.add_argument("--eval-repeats", type=int, default=2)
    p_rl.add_argument("--eval-greedy", action="store_true")
    p_rl.add_argument("--seed", type=int, default=7)

    # eval
    p_ev = sub.add_parser("eval", help="Evaluate a checkpoint on MicroRTS games")
    p_ev.add_argument("--checkpoint", type=Path)
    p_ev.add_argument("--init", type=Path)
    p_ev.add_argument("--out", type=Path)
    p_ev.add_argument("--device", default="auto")
    p_ev.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p_ev.add_argument("--rank", type=int, default=8)
    p_ev.add_argument("--maps", default="maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml")
    p_ev.add_argument("--opponents", default="RandomBiasedAI,WorkerRush,LightRush")
    p_ev.add_argument("--players", default="0,1")
    p_ev.add_argument("--repeats", type=int, default=2)
    p_ev.add_argument("--max-ticks", type=int, default=2000)
    p_ev.add_argument("--skill-ticks", type=int, default=64)
    p_ev.add_argument("--candidate-limit", type=int, default=12)
    p_ev.add_argument("--greedy", action="store_true")
    p_ev.add_argument("--seed", type=int, default=200_000)

    # export
    p_ex = sub.add_parser("export", help="Merge LoRA and export a Decider-ready decision-only model")
    p_ex.add_argument("--checkpoint", type=Path, required=True)
    p_ex.add_argument("--init", type=Path)
    p_ex.add_argument("--out", type=Path, required=True)

    # bench
    p_bn = sub.add_parser("bench", help="Benchmark SharedPrefixCache vs uncached decision latency")
    p_bn.add_argument("--checkpoint", type=Path)
    p_bn.add_argument("--init", type=Path)
    p_bn.add_argument("--device", default="auto")
    p_bn.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p_bn.add_argument("--rank", type=int, default=8)
    p_bn.add_argument("--map", default="maps/8x8/basesWorkers8x8A.xml")
    p_bn.add_argument("--opponent", default="WorkerRush")
    p_bn.add_argument("--iters", type=int, default=5)
    p_bn.add_argument("--seed", type=int, default=7)

    return ap


def _load_or_init_policy(args):
    device = resolve_device(args.device)
    if getattr(args, "checkpoint", None):
        return EveRLCDPolicy.load(args.checkpoint, device=device, init=args.init, dtype=args.dtype)
    return EveRLCDPolicy(
        args.init or DEFAULT_MODEL,
        device=device,
        rank=args.rank,
        max_len=getattr(args, "max_len", 2048),
        dtype=args.dtype,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "download":
        from huggingface_hub import snapshot_download

        print(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=args.init,
                allow_patterns=["*.json", "*.safetensors", "merges.txt", "*.jinja"],
                max_workers=2,
            )
        )
        return

    if args.cmd == "collect":
        summary = collect_expert_dataset(
            args.out,
            teachers=args.teachers.split(","),
            opponents=args.opponents.split(","),
            maps=args.maps.split(","),
            players=[int(p) for p in args.players.split(",")],
            games_per_combo=args.games_per_combo,
            max_ticks=args.max_ticks,
            skill_ticks=args.skill_ticks,
            candidate_limit=args.candidate_limit,
            seed=args.seed,
            keep_wins_only=args.keep_wins_only,
            dedup_waits=not args.no_dedup_waits,
            all_out_path=args.all_out,
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    if args.cmd in ("sft", "bandit-rlcd"):
        policy = _load_or_init_policy(args)
        rows = load_decision_rows(args.data)
        val_rows = load_decision_rows(args.val_data) if args.val_data else None
        metrics = run_offline_stage(
            policy,
            rows,
            args.out,
            mode="sft" if args.cmd == "sft" else "bandit",
            arm=getattr(args, "arm", "rlcd"),
            epochs=args.epochs,
            micro_batch=args.micro_batch,
            accum_steps=args.accum,
            lr=args.lr,
            head_lr_mult=args.head_lr_mult,
            group=getattr(args, "group", 4),
            include_win_noul=not args.no_win_noul,
            val_rows=val_rows,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2), flush=True)
        return

    if args.cmd == "online-rl":
        run_online_rl(args)
        return

    if args.cmd == "eval":
        policy = _load_or_init_policy(args)
        pool = EnvPool(
            max_ticks=args.max_ticks,
            skill_ticks=args.skill_ticks,
            candidate_limit=args.candidate_limit,
        )
        try:
            games, summary = evaluate_policy(
                policy,
                pool,
                maps=args.maps.split(","),
                opponents=args.opponents.split(","),
                players=[int(p) for p in args.players.split(",")],
                repeats=args.repeats,
                seed_base=args.seed,
                greedy=args.greedy,
            )
            if args.out:
                atomic_json(args.out / "eval.json", {"summary": summary, "games": games})
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            pool.close()
        return

    if args.cmd == "export":
        policy = EveRLCDPolicy.load(args.checkpoint, device="cpu", init=args.init, dtype="fp32")
        record = policy.export_decision_only(args.out, meta={"exported_from": str(args.checkpoint)})
        print(json.dumps({"exported": str(args.out), "sha256": record["sha256"]}, indent=2))
        return

    if args.cmd == "bench":
        run_benchmark(args)


if __name__ == "__main__":
    main()
