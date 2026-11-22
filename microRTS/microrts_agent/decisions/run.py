"""python -m microrts_agent.decisions.run {train,eval} --help"""

import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .env import REWARD_NAMES, SkillEnv
from .policy import discounted_returns, select
from .state import decision_prompt

DEFAULT_POLICY = "microrts_agent.decisions.policy:TinyPolicy"


def collect_episode(env, policy, episode, writer, greedy=False, forecast=None):
    state = env.reset()
    transitions = []
    while not env.done:
        candidates = env.candidates(state)
        with torch.no_grad():
            logits, _ = policy(state, candidates)
            index, distribution = select(logits, greedy)
        probs = distribution.probs.cpu().tolist()
        extra = forecast(state, candidates[index]) if forecast is not None else {}
        transition = {
            "episode": episode,
            "decision": len(transitions),
            "state": state,
            "candidates": candidates,
            "chosen_index": index,
            "chosen_skill": candidates[index],
            "action_probs": probs,
            "action_probability": probs[index],
            # Greedy execution is deterministic even if the model's softmax is diffuse.
            "behavior_probability": 1.0 if greedy else probs[index],
            "selection_mode": "greedy" if greedy else "sample",
            **extra,
            **env.execute(candidates[index]),
        }
        writer.write(json.dumps(transition, separators=(",", ":")) + "\n")
        transitions.append(transition)
        state = transition["next_state"]
    writer.flush()
    final = transitions[-1]
    outcome = (
        "timeout"
        if final["truncated"]
        else "draw"
        if final["winner"] == -1
        else "win"
        if final["winner"] == env.player
        else "loss"
    )
    return transitions, {
        "episode": episode,
        "outcome": outcome,
        "winner": final["winner"],
        "terminated": final["terminated"],
        "truncated": final["truncated"],
        "ticks": state["time"],
        "decisions": len(transitions),
        "reward": sum(t["reward"] for t in transitions),
    }


def train_episode(policy, optimizer, transitions, env, entropy_weight, auxiliary_loss=None):
    bootstrap = 0.0
    if transitions[-1]["truncated"]:
        state = transitions[-1]["next_state"]
        with torch.no_grad():
            _, value = policy(state, env.candidates(state))
            bootstrap = value.item()
    returns = discounted_returns(transitions, bootstrap)
    loss_sum = 0.0
    optimizer.zero_grad()
    for transition, target in zip(transitions, returns, strict=True):
        logits, value = policy(transition["state"], transition["candidates"])
        distribution = torch.distributions.Categorical(logits=logits)
        selected = torch.tensor(transition["chosen_index"], device=logits.device)
        advantage = target - value
        loss = (
            -distribution.log_prob(selected) * advantage.detach()
            + 0.5 * advantage.square()
            - entropy_weight * distribution.entropy()
        )
        (loss / len(transitions)).backward()
        loss_sum += loss.item()
        if auxiliary_loss is not None:
            auxiliary = auxiliary_loss(transition)
            (auxiliary / len(transitions)).backward()
            loss_sum += auxiliary.item()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": loss_sum / len(transitions),
        "grad_norm": float(grad_norm),
        "bootstrap_value": bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--map", default="maps/8x8/basesWorkers8x8A.xml")
    parser.add_argument("--opponent", default="WorkerRush")
    parser.add_argument("--player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-ticks", type=int, default=3000)
    parser.add_argument("--skill-ticks", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=26)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount per game tick")
    parser.add_argument("--reward", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", help="module:factory returning a torch module; see TinyPolicy")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/decisions"))
    parser.add_argument(
        "--greedy", action="store_true", help="evaluation only; default samples the policy"
    )
    args = parser.parse_args()
    if args.episodes < 1 or (args.mode == "train" and args.greedy):
        parser.error("Need positive episodes; on-policy training requires sampled actions")
    args.out.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting a prior research run or its collected dataset.
    trajectory_path = args.out / "transitions.jsonl"
    with trajectory_path.open("x", encoding="utf-8") as writer:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        checkpoint = (
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if args.checkpoint
            else None
        )
        spec = args.policy or (checkpoint["policy"] if checkpoint else DEFAULT_POLICY)
        module, factory = spec.split(":")
        policy = getattr(importlib.import_module(module), factory)()
        if checkpoint:
            policy.load_state_dict(checkpoint["state_dict"])
        # Disable dropout for both collection and gradient recomputation; gradients still work.
        policy.eval()
        initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(policy=spec, reward_components=REWARD_NAMES)
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if args.mode == "train":
            torch.save({"policy": spec, "state_dict": initial}, args.out / "initial.pt")
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
        episodes = []
        optimizer = (
            torch.optim.Adam(policy.parameters(), lr=args.lr) if args.mode == "train" else None
        )
        try:
            for episode in range(args.episodes):
                transitions, metrics = collect_episode(env, policy, episode, writer, args.greedy)
                if optimizer:
                    probe = next(
                        (t for t in transitions if len(t["candidates"]) > 1), transitions[0]
                    )
                    with torch.no_grad():
                        before = policy(probe["state"], probe["candidates"])[0].softmax(-1)
                    metrics.update(train_episode(policy, optimizer, transitions, env, args.entropy))
                    with torch.no_grad():
                        after = policy(probe["state"], probe["candidates"])[0].softmax(-1)
                    metrics["policy_probability_l1_change"] = float((after - before).abs().sum())
                if episode == 0:
                    (args.out / "example_prompt.txt").write_text(
                        decision_prompt(transitions[0]["state"], transitions[0]["candidates"])
                        + "\n"
                    )
                episodes.append(metrics)
                print(json.dumps(metrics), flush=True)
        finally:
            env.close()
        delta = (
            sum(
                float((v.detach() - initial[k]).float().square().sum())
                for k, v in policy.state_dict().items()
            )
            ** 0.5
        )
        summary = {
            "episodes": episodes,
            "parameter_l2_delta": delta,
            "wins": sum(e["outcome"] == "win" for e in episodes),
            "losses": sum(e["outcome"] == "loss" for e in episodes),
            "draws": sum(e["outcome"] == "draw" for e in episodes),
            "timeouts": sum(e["outcome"] == "timeout" for e in episodes),
        }
        # Empirical match result, never the confidence of a chosen action.
        summary["win_rate"] = summary["wins"] / len(episodes)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if optimizer:
            torch.save(
                {"policy": spec, "state_dict": policy.state_dict(), "config": config},
                args.out / "policy.pt",
            )
        print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
