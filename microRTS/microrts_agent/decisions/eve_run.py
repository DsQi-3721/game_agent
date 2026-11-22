"""Real eve-rlcd model: skill policy gradient plus a separate RLCD win forecast."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from rlcd.metrics import brier, ece
from rlcd.schema import render_prompt

from .env import SkillEnv
from .eve_policy import (
    DEFAULT_MODEL,
    MODEL_ID,
    MODEL_REVISION,
    EveRLCDPolicy,
    choice_question,
    forecast_loss,
)
from .run import collect_episode, train_episode


def forecast_metrics(transitions):
    if not transitions[-1]["terminated"]:
        return {"forecast_labeled_decisions": 0}  # timeouts are censored, not losses
    won = transitions[-1]["winner"] == transitions[0]["state"]["player"]
    probs = np.array([t["forecast_probs"] for t in transitions])
    answers = np.full(len(probs), 0 if won else 1)
    return {
        "forecast_labeled_decisions": len(probs),
        "win_brier": brier(probs, answers),
        "win_ece": ece(probs[:, 0], np.full(len(probs), won)),
        "mean_win_probability": float(probs[:, 0].mean()),
    }


def parameter_changes(before, after):
    groups = {
        "all": list(before),
        "lora": [n for n in before if "lora_" in n],
        "letter_head": ["letter_weight"],
        "value_head": [n for n in before if n.startswith("value_head.")],
    }
    return {
        name + "_l2_delta": sum(float((after[n] - before[n]).square().sum()) for n in names) ** 0.5
        for name, names in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "train", "eval"])
    parser.add_argument("--init", type=Path, help="local upstream decision-only export")
    parser.add_argument("--checkpoint", type=Path, help="our LoRA/head checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank; 0 freezes the backbone")
    parser.add_argument(
        "--max-len", type=int, help="whole prompt budget; default 2048; never silently truncated"
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--map", default="maps/8x8/basesWorkers8x8A.xml")
    parser.add_argument("--opponent", default="WorkerRush")
    parser.add_argument("--player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-ticks", type=int, default=3000)
    parser.add_argument("--skill-ticks", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, choices=range(2, 27), default=26)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--arm", choices=["rlcd", "rlvr", "none"], default="rlcd")
    parser.add_argument("--group", type=int, default=4, help="independent binary forecast samples")
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--greedy", action="store_true", help="evaluation only")
    parser.add_argument("--out", type=Path, default=Path("runs/eve-rlcd"))
    args = parser.parse_args()
    if args.mode == "download":
        from huggingface_hub import snapshot_download

        print(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=args.init or DEFAULT_MODEL,
                allow_patterns=["*.json", "*.safetensors", "merges.txt", "*.jinja"],
                max_workers=2,
            )
        )
        return
    if (
        args.episodes < 1
        or args.group < 1
        or args.rank < 0
        or (args.max_len is not None and args.max_len < 1)
        or args.lr <= 0
        or args.calibration_weight < 0
        or (args.greedy and args.mode == "train")
    ):
        parser.error(
            "Need positive episodes/group/max-len/lr, nonnegative rank/calibration-weight, "
            "and sampled actions for training"
        )
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 10000)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (args.out / "transitions.jsonl").open("x", encoding="utf-8") as writer:
        if args.checkpoint:
            policy = EveRLCDPolicy.load(args.checkpoint, device, init=args.init)
            if args.max_len:
                policy.max_len = args.max_len
        else:
            policy = EveRLCDPolicy(
                args.init or DEFAULT_MODEL, device, args.rank, args.max_len or 2048
            )
        policy.greedy = args.greedy
        trainable = [p for p in policy.parameters() if p.requires_grad]
        initial = policy.trainable_state()
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(
            device=device,
            init=policy.init,
            rank=policy.rank,
            max_len=policy.max_len,
            base_export={
                k: policy.rlcd.record.get(k) for k in ("base_model", "source", "sha256", "training")
            },
            trainable_parameters=sum(p.numel() for p in trainable),
            total_parameters=sum(p.numel() for p in policy.parameters()),
        )
        if Path(policy.init) == DEFAULT_MODEL:
            config.update(model_id=MODEL_ID, model_revision=MODEL_REVISION)
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "loaded": policy.init,
                    "device": device,
                    "trainable_parameters": config["trainable_parameters"],
                }
            ),
            flush=True,
        )
        if args.mode == "train":
            policy.save(args.out / "initial.pt", config)
        optimizer = (
            torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
            if args.mode == "train"
            else None
        )
        env = SkillEnv(
            map_path=args.map,
            opponent=args.opponent,
            player=args.player,
            max_ticks=args.max_ticks,
            skill_ticks=args.skill_ticks,
            candidate_limit=args.candidate_limit,
            gamma=args.gamma,
            reward=args.reward,
            seed=args.seed,
        )
        episodes, labeled = [], []
        try:
            for episode in range(args.episodes):
                transitions, metrics = collect_episode(
                    env,
                    policy,
                    episode,
                    writer,
                    args.greedy,
                    forecast=lambda state, c: policy.forecast(state, c, args.group, rng),
                )
                metrics.update(forecast_metrics(transitions))
                won = transitions[-1]["winner"] == env.player
                if transitions[-1]["terminated"]:
                    labeled.extend((t["forecast_probs"], 0 if won else 1) for t in transitions)
                if episode == 0:
                    (args.out / "example_prompt.txt").write_text(
                        render_prompt(
                            choice_question(transitions[0]["state"], transitions[0]["candidates"])
                        )
                        + "\n"
                    )
                if optimizer is not None:
                    calibration_losses = []

                    def auxiliary(t, won=won, calibration_losses=calibration_losses):
                        logits = policy.win_logits(t["state"], t["chosen_skill"])
                        loss = forecast_loss(logits, t["forecast_answers"], won, args.arm)
                        calibration_losses.append(loss.item())
                        return args.calibration_weight * loss

                    use_calibration = args.arm != "none" and transitions[-1]["terminated"]
                    metrics.update(
                        train_episode(
                            policy,
                            optimizer,
                            transitions,
                            env,
                            args.entropy,
                            auxiliary_loss=auxiliary if use_calibration else None,
                        )
                    )
                    metrics["calibration_loss"] = (
                        float(np.mean(calibration_losses)) if calibration_losses else None
                    )
                    probe = next(
                        (t for t in transitions if len(t["candidates"]) > 1), transitions[0]
                    )
                    with torch.no_grad():
                        after = policy(probe["state"], probe["candidates"])[0].softmax(-1).cpu()
                        win_after = (
                            policy.win_logits(probe["state"], probe["chosen_skill"])
                            .softmax(-1)[0]
                            .item()
                        )
                    metrics["policy_probability_l1_change"] = float(
                        (after - torch.tensor(probe["action_probs"])).abs().sum()
                    )
                    metrics["probe_win_probability_change"] = win_after - probe["win_probability"]
                    # Save after each completed update so a long run retains useful work.
                    policy.save(
                        args.out / "policy.pt", config | {"completed_episodes": episode + 1}
                    )
                episodes.append(metrics)
                with (args.out / "episodes.jsonl").open("a") as log:
                    log.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
        finally:
            env.close()
        summary = {
            "episodes": episodes,
            "arm": args.arm,
            "win_rate": sum(e["outcome"] == "win" for e in episodes) / len(episodes),
            "timeouts": sum(e["truncated"] for e in episodes),
            "elapsed_seconds": time.monotonic() - started,
            **parameter_changes(initial, policy.trainable_state()),
        }
        if labeled:
            probs = np.array([p for p, _ in labeled])
            answers = np.array([a for _, a in labeled])
            summary.update(
                win_brier=brier(probs, answers),
                win_ece=ece(probs[:, 0], answers == 0),
                forecast_labeled_decisions=len(labeled),
            )
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
