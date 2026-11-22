"""Resumable training, fixed periodic validation, then untouched final tests."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from jpype import JClass

from .env import SkillEnv
from .eve_policy import DEFAULT_MODEL, MODEL_REVISION, EveRLCDPolicy, forecast_loss
from .eve_run import forecast_metrics
from .experiment_metrics import (
    aggregate,
    atomic_json,
    forecast_bins,
    selection_score,
    write_progress,
    write_report,
)
from .run import collect_episode, train_episode

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


class SeededSkillEnv(SkillEnv):
    """Independent game seeds also reset Java bot RNGs and produced-unit IDs."""

    def __init__(self, *, seed, **kwargs):
        self.episode_seed = seed
        super().__init__(seed=seed, **kwargs)

    def reset(self):
        super().reset()
        JClass("rts.UnitAction").r.setSeed(self.episode_seed)
        JClass("rts.units.Unit").next_ID = 1 + max(
            int(u.getID()) for u in self.client.pgs.getUnits()
        )
        cls = self.client.ai2.getClass()
        # WorkerRush and its abstraction layer have non-public java.util.Random fields.
        while cls is not None:
            for field in cls.getDeclaredFields():
                if str(field.getType().getName()) == "java.util.Random":
                    field.setAccessible(True)
                    rng = field.get(self.client.ai2)
                    if rng is not None:
                        rng.setSeed(self.episode_seed)
            cls = cls.getSuperclass()
        return self.state()


def scenarios(config, maps):
    return [{"map": m, "player": p} for m in maps for p in config["players"]]


def evaluation_cases(config, split):
    maps = config["train_maps"] if split == "validation" else config["test_maps"]
    return [
        dict(
            scenario,
            seed=config[f"{split}_seed_base"] + i * 1000 + repeat,
            case=f"{i:02d}-{repeat:03d}",
        )
        for i, scenario in enumerate(scenarios(config, maps))
        for repeat in range(config[f"{split}_repeats"])
    ]


class ProgressWriter:
    def __init__(self, stream, progress):
        self.stream, self.progress = stream, progress

    def write(self, line):
        self.stream.write(line)
        row = json.loads(line)
        self.progress(tick=row["next_state"]["time"], decisions=row["decision"] + 1)

    def flush(self):
        self.stream.flush()


class Experiment:
    def __init__(self, config, out, resume):
        self.config, self.out = config, out
        self.status = {"pid": os.getpid(), "started_at": utc_now()}
        self.state = {
            "completed": 0,
            "training_seconds": 0.0,
            "best_step": 0,
            "best_score": None,
            "validated_step": -1,
            "validations": [],
            "train_metrics": [],
            "stop_reason": None,
        }
        out.mkdir(parents=True, exist_ok=True)
        self.lock = (out / "process.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.write(str(os.getpid()))
        self.lock.flush()
        protocol = out / "protocol.json"
        if protocol.exists():
            if not resume:
                raise ValueError("Run already exists; use --resume to recover it")
            previous = json.loads(protocol.read_text())
            if previous["config"] != config:
                raise ValueError("Config differs from the frozen run protocol")
        else:
            atomic_json(
                protocol,
                {
                    "config": config,
                    "created_at": utc_now(),
                    "model_revision": MODEL_REVISION,
                    "selection": "match_score, then mean game reward; earliest exact tie",
                },
            )
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        device = config["device"]
        self.progress(phase="loading", device=device)
        try:
            self.load_model()
        except BaseException as exc:
            self.progress(phase="failed", error=f"{type(exc).__name__}: {exc}")
            (out / "error.txt").write_text(traceback.format_exc())
            raise

    def load_model(self):
        config, out = self.config, self.out
        device = config["device"]
        saved = out / "resume.pt"
        if saved.exists():
            self.policy = EveRLCDPolicy.load(saved, device=device)
            self.state = torch.load(saved, map_location="cpu", weights_only=True)["training_state"]
        else:
            self.policy = EveRLCDPolicy(
                config.get("init", DEFAULT_MODEL),
                device=device,
                rank=config["rank"],
                max_len=config["max_len"],
            )
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=config["lr"],
            weight_decay=0,
        )
        if "optimizer" in self.state:
            self.optimizer.load_state_dict(self.state.pop("optimizer"))
        if not (out / "initial.pt").exists():
            self.policy.save(out / "initial.pt", config | {"completed_episodes": 0})
        self.save_state()

    def progress(self, **details):
        self.status.update(details)
        self.status.update(
            updated_at=utc_now(),
            completed_episodes=self.state["completed"],
            training_seconds=self.state["training_seconds"],
            best_step=self.state["best_step"],
        )
        atomic_json(self.out / "status.json", self.status)

    def save_state(self):
        self.policy.save(
            self.out / "resume.pt",
            self.config | {"completed_episodes": self.state["completed"]},
            self.state | {"optimizer": cpu_tree(self.optimizer.state_dict())},
        )

    def game(self, case, directory, *, training=False, checkpoint_sha256=None):
        cfg = self.config
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"game-{case['case']}.json"
        if not training and destination.exists():
            cached = json.loads(destination.read_text())
            if cached["checkpoint_sha256"] == checkpoint_sha256:
                return cached
        seed = case["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        forecast_rng = torch.Generator().manual_seed(seed + 10000000)
        self.progress(case=case, tick=0, decisions=0)
        started = time.monotonic()
        env = SeededSkillEnv(
            seed=seed,
            map_path=case["map"],
            player=case["player"],
            **{
                k: cfg[k]
                for k in (
                    "opponent",
                    "max_ticks",
                    "skill_ticks",
                    "candidate_limit",
                    "gamma",
                    "reward",
                )
            },
        )
        trajectory = directory / f"game-{case['case']}.jsonl"
        temporary = trajectory.with_suffix(".jsonl.tmp")
        try:
            with temporary.open("w") as stream:
                transitions, metrics = collect_episode(
                    env,
                    self.policy,
                    self.state["completed"],
                    ProgressWriter(stream, self.progress),
                    forecast=lambda state, c: self.policy.forecast(
                        state, c, cfg["group"], forecast_rng
                    ),
                )
            metrics.update(forecast_metrics(transitions), forecast_bins=forecast_bins(transitions))
            if training:
                won = transitions[-1]["winner"] == env.player
                calibration = []

                def auxiliary(t):
                    loss = forecast_loss(
                        self.policy.win_logits(t["state"], t["chosen_skill"]),
                        t["forecast_answers"],
                        won,
                        cfg["arm"],
                    )
                    calibration.append(loss.item())
                    return cfg["calibration_weight"] * loss

                self.progress(phase="updating")
                metrics.update(
                    train_episode(
                        self.policy,
                        self.optimizer,
                        transitions,
                        env,
                        cfg["entropy"],
                        auxiliary_loss=auxiliary
                        if cfg["arm"] != "none" and transitions[-1]["terminated"]
                        else None,
                    )
                )
                metrics["calibration_loss"] = float(np.mean(calibration)) if calibration else None
        finally:
            env.close()
        temporary.replace(trajectory)
        metrics.update(
            case, seconds=time.monotonic() - started, checkpoint_sha256=checkpoint_sha256
        )
        atomic_json(destination, metrics)
        return metrics

    def evaluate(self, checkpoint, split, directory):
        fingerprint = digest(checkpoint)
        games = []
        cases = evaluation_cases(self.config, split)
        self.progress(
            phase=split, evaluation_checkpoint=str(checkpoint), evaluation_games=len(cases)
        )
        for i, case in enumerate(cases):
            self.progress(evaluation_completed=i)
            games.append(self.game(case, directory, checkpoint_sha256=fingerprint))
        self.progress(evaluation_completed=len(cases))
        summary = aggregate(games)
        atomic_json(directory / "summary.json", summary)
        return games, summary

    def validate(self):
        step = self.state["completed"]
        folder = self.out / "checkpoints"
        folder.mkdir(exist_ok=True)
        checkpoint = folder / f"step-{step:06d}.pt"
        if step == 0:
            shutil.copyfile(self.out / "initial.pt", checkpoint)
        else:
            self.policy.save(checkpoint, self.config | {"completed_episodes": step})
        _, summary = self.evaluate(
            checkpoint, "validation", self.out / "validation" / f"step-{step:06d}"
        )
        score = selection_score(summary)
        if self.state["best_score"] is None or score > self.state["best_score"]:
            temporary = self.out / "best.pt.tmp"
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(self.out / "best.pt")
            self.state.update(best_score=score, best_step=step)
        self.state["validated_step"] = step
        self.state["validations"].append({"step": step, "summary": summary})
        self.save_state()
        atomic_json(self.out / "validation_history.json", self.state["validations"])
        write_progress(self.out, self.state)
        print(
            json.dumps({"validation_step": step, "best_step": self.state["best_step"], **summary}),
            flush=True,
        )

    def run(self):
        cfg = self.config
        if self.state["validated_step"] < 0:
            self.validate()
        train_scenarios = scenarios(cfg, cfg["train_maps"])
        while (
            self.state["completed"] < cfg["episodes"]
            and self.state["training_seconds"] < cfg["train_hours"] * 3600
        ):
            step = self.state["completed"]
            if step and step % cfg["validate_every"] == 0 and self.state["validated_step"] != step:
                self.validate()
            case = dict(
                train_scenarios[step % len(train_scenarios)],
                seed=cfg["train_seed_base"] + step,
                case=f"{step:06d}",
            )
            self.progress(
                phase="training",
                next_validation=((step // cfg["validate_every"]) + 1) * cfg["validate_every"],
            )
            metrics = self.game(case, self.out / "training", training=True)
            self.state["completed"] += 1
            self.state["training_seconds"] += metrics["seconds"]
            self.state["train_metrics"].append(metrics)
            self.save_state()
            self.progress(phase="training")
            print(
                json.dumps(
                    {
                        "trained_episode": self.state["completed"],
                        **{
                            k: metrics[k]
                            for k in ("outcome", "reward", "ticks", "decisions", "loss", "seconds")
                        },
                    }
                ),
                flush=True,
            )
        self.state["stop_reason"] = (
            "episode_budget"
            if self.state["completed"] >= cfg["episodes"]
            else "training_time_budget"
        )
        if self.state["validated_step"] != self.state["completed"]:
            self.validate()
        self.save_state()
        shutil.copyfile(
            self.out / "checkpoints" / f"step-{self.state['completed']:06d}.pt",
            self.out / "last.pt",
        )
        # Final tests reuse the already-loaded backbone via fast in-place adapter swaps.
        results = {}
        del self.optimizer
        for name in ("initial", "best", "last"):
            checkpoint = self.out / f"{name}.pt"
            self.policy.load_adapter(checkpoint)
            games, _ = self.evaluate(
                checkpoint, "test", self.out / "test" / digest(checkpoint)[:16]
            )
            results[name] = games
        self.progress(phase="reporting")
        write_report(self.out, self.state, results)
        self.progress(phase="completed", completed_at=utc_now(), report=str(self.out / "REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "setup/eve-rlcd-overnight.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["episodes"] < 1 or config["train_hours"] <= 0 or config["validate_every"] < 1:
        parser.error("Need positive episode, time and validation budgets")
    experiment = Experiment(config, args.out.resolve(), args.resume)
    try:
        experiment.run()
    except BaseException as exc:
        experiment.progress(phase="failed", error=f"{type(exc).__name__}: {exc}")
        (experiment.out / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
