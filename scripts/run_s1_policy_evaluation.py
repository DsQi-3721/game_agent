"""Evaluate frozen policy adapters after training, with concurrent old/new team blocks.

    python scripts/run_s1_policy_evaluation.py prepare --out <evaluation directory>
    python scripts/run_s1_policy_evaluation.py run --out <evaluation directory>
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROUND = Path("/data/wangxinqi/game_agent_runs/s1_policy_retraining/round1")
POOL = Path("/data/wangxinqi/teams/gen9ou_eval12")
ARMS = ("old", "new")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def configuration(opponent, arm, team, games, pool, port, seed):
    cfg = yaml.safe_load((ROOT / f"configs/experiments/strong_{opponent}_ou.yaml").read_text())
    student = next(a for a in cfg["agents"] if a["name"] == "s1_student")
    student["name"] = arm
    student["params"].update(
        teacher=False, student_url=f"http://127.0.0.1:{port}", student_model=f"s1-policy-{arm}",
        mode="split" if arm == "old" else "flat", history_events=0 if arm == "old" else 8,
        ask_hard=False, debias=True, tier0=False, escalate="none", shadow_rate=0.0,
        opponent_model="prior", s1_timeout_s=20.0, record=True)
    opponent_agent = cfg["agents"][-1]
    if opponent == "kakuna":
        opponent_agent["external"]["env"].update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    cfg.update(name=f"s1_policy_{opponent}_{arm}_t{team:02d}", agents=[student, opponent_agent], seed=seed+team)
    cfg["teams"].update(mode="mirror", pool=str(pool), per_team=games, offset=team*games)
    cfg["evaluation"]["n_battles"] = games
    return cfg


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError(f"Evaluation already prepared: {out}")
    team_dir = out / "teams"
    team_dir.mkdir(exist_ok=True)
    teams = sorted(POOL.glob("*.txt"))
    if len(teams) != 12:
        raise ValueError(f"Expected 12 evaluation teams, found {len(teams)}")
    for path in teams:
        shutil.copyfile(path, team_dir / path.name)
    training = json.loads((ROUND / "training_process.json").read_text())
    adapters = {"old": str(ROOT / "runs/s1_models/s1-v7-2b"), "new": training["output"]}
    services = {}
    for arm, port in zip(ARMS, (8114, 8115)):
        services[arm] = {
            "url": f"http://127.0.0.1:{port}", "adapter": adapters[arm],
            "command": [sys.executable, "-u", str(ROOT / "scripts/s1_serve.py"), "--backend", "hf",
                        "--model", "/data/models/Qwen3.5-2B", "--adapter", adapters[arm],
                        "--port", str(port), "--merge", "--model-id", f"s1-policy-{arm}",
                        "--batch-size", "8", "--temperature", "1.0", "--action-temperature", "1.55"],
            "env": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2"}}
    schedule = []
    for team in range(12):
        for opponent in ("kakuna", "foulplay"):
            for arm in (ARMS if team % 2 == 0 else tuple(reversed(ARMS))):
                directory = Path(opponent) / f"team{team:02d}" / arm
                (out / directory).mkdir(parents=True, exist_ok=True)
                cfg = configuration(opponent, arm, team, 10, team_dir,
                                    8114 if arm == "old" else 8115, 61206)
                (out / directory / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                schedule.append({"opponent": opponent, "arm": arm, "team": team, "games": 10,
                                 "directory": str(directory)})
    write_json(out / "schedule.json", schedule)
    protocol = {
        "created_at": datetime.now().astimezone().isoformat(), "round": 1,
        "expected_games": 480, "games_per_arm_opponent": 120, "parallel_pairs": 2,
        "training": training, "planned_steps": 8341, "services": services,
        "teams": [p.name for p in teams], "team_pool": str(team_dir),
        "collection_team_selection": str(ROUND.parent / "teams/selection.json"),
        "selection": "global joint probability argmax; both option orders averaged",
        "old_input": "trained split questions, current state; joint probability decoding",
        "new_input": "full action question, current state and eight completed decisions",
        "online_assistance": "none: no teacher, search, HyMem or learned opponent model",
        "service": "same GPU, HF backend, merged LoRA, batch size 8, action temperature 1.55",
        "win_difference_ci": "Newcombe difference of independent Wilson score intervals, 95%",
        "latency": "action request received to final action sent; team preview excluded",
        "passing": "Kakuna win difference > 0 and CI lower bound > 0; mean and p90 latency ratios <= 1.10",
        "heldout_preferences": str(ROUND / "training/valid.jsonl"),
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"state": "prepared", "expected_games": 480})
    print(f"Prepared 480 games in {out}", flush=True)


def training_progress(log):
    with open(log, "rb") as file:
        file.seek(max(0, os.fstat(file.fileno()).st_size - 8192))
        matches = re.findall(rb"\[train_hf\] step (\d+)/", file.read())
    return int(matches[-1]) if matches else 0


async def wait_for_model(protocol, status):
    training = protocol["training"]
    output = Path(training["output"])
    metadata = output / "s1_train_meta.json"
    while not metadata.exists():
        step = training_progress(training["log"])
        status("waiting_training", training_step=step, planned_steps=protocol.get("planned_steps"),
               stopping="validation_convergence" if protocol.get("until_converged") else "fixed_epochs")
        try:
            os.kill(training["pid"], 0)
            state = Path(f"/proc/{training['pid']}/stat").read_text().split(") ", 1)[1][0]
            if state == "Z":
                raise ProcessLookupError("training exited")
        except (ProcessLookupError, FileNotFoundError):
            if metadata.exists():
                break
            raise RuntimeError("Training ended without a completed model; see training.log")
        await asyncio.sleep(30)
    meta = json.loads(metadata.read_text())
    ready = meta.get("converged") if protocol.get("until_converged") else meta["steps"] == protocol["planned_steps"]
    if not ready or not (output / "adapter_model.safetensors").exists():
        raise RuntimeError("Final training metadata or adapter is incomplete")
    return meta


async def start_service(out, arm, service, children):
    log_path = out / f"service_{arm}.log"
    with log_path.open("a") as log:
        process = await asyncio.create_subprocess_exec(
            *service["command"], cwd=ROOT, env={**os.environ, **service["env"]},
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    children.append(process)
    write_json(out / f"service_{arm}.json", {**service, "pid": process.pid})
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"{arm} inference service exited; see {log_path}")
            try:
                health = (await client.get(service["url"] + "/health")).json()
                if health.get("model") == f"s1-policy-{arm}":
                    return
                raise RuntimeError(f"Port for {arm} is serving a different model")
            except (httpx.HTTPError, ValueError):
                await asyncio.sleep(2)
    raise TimeoutError(f"{arm} inference service did not become ready; see {log_path}")


async def heldout_preferences(out, protocol):
    from arena.llm.typesafe import TypeSafeClient
    clients = {arm: TypeSafeClient(base_url=service["url"], api_key="local", model=f"s1-policy-{arm}")
               for arm, service in protocol["services"].items()}
    cases = [json.loads(line) for line in Path(protocol["heldout_preferences"]).read_text().splitlines()
             if json.loads(line)["purpose"] == "action_rank"]
    results = []
    try:
        for case in cases:
            question = case["question"]
            reverse = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
            answers = await asyncio.gather(*(clients[arm].system_one(
                case["state"], {"best": question, "best_rev": reverse}) for arm in ARMS))
            scores = {}
            for arm, answer in zip(ARMS, answers):
                probs = {key: (answer.answers["best"].probabilities[key]
                               + answer.answers["best_rev"].probabilities[key]) / 2 for key in case["keys"]}
                preferred = case["keys"][case["preference"]["preferred"]]
                rejected = case["keys"][case["preference"]["rejected"]]
                scores[arm] = {"correct": probs[preferred] > probs[rejected],
                               "margin": probs[preferred] - probs[rejected]}
            results.append({"id": case["id"], "group": case["group"],
                            "categories": case.get("critical_reasons", []), "scores": scores})
    finally:
        await asyncio.gather(*(client.aclose() for client in clients.values()))
    result = {"n": len(results), "input": "identical full-action questions and causal history for both adapters",
              "reference": "held-out replicated Foul Play action preferences",
              "arms": {arm: {"correct": sum(row["scores"][arm]["correct"] for row in results),
                              "accuracy": sum(row["scores"][arm]["correct"] for row in results) / len(results)}
                       for arm in ARMS}, "cases": results}
    write_json(out / "heldout_preferences.json", result)


async def run(out):
    protocol = json.loads((out / "protocol.json").read_text())
    schedule = json.loads((out / "schedule.json").read_text())
    children = []
    completed = []

    def status(state, **extra):
        write_json(out / "status.json", {"state": state, "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(), "expected_games": 480,
            "completed_games": sum(item["games"] for item in completed), **extra})

    async def match(item):
        directory = out / item["directory"]
        report_path = directory / "report.json"
        if not report_path.exists():
            env = {**os.environ, "ARENA_SHOWDOWN_URL": "http://127.0.0.1:8100",
                   "ARENA_OPPONENT_MODEL": "prior", "ARENA_STREAM_DIR": str(directory / "streams"),
                   "LITELLM_LOCAL_MODEL_COST_MAP": "True", "OMP_NUM_THREADS": "1"}
            print(f"start {item['directory']}", flush=True)
            with (directory / "run.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "arena.cli", "run", str(directory / "config.yaml"),
                    "--run-dir", str(directory), "--emit-events", cwd=ROOT, env=env,
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                code = await process.wait()
            if code:
                raise RuntimeError(f"Match exited with {code}: {directory}")
        report = json.loads(report_path.read_text())
        agent = report["agents"][item["arm"]]
        if agent["games"] != item["games"] or report.get("aborted"):
            raise RuntimeError(f"Incomplete match: {directory}")
        completed.append(item)
        status("battles", last_completed=item["directory"])
        print(f"finish {item['directory']} {agent['wins']}/{agent['games']}", flush=True)

    try:
        meta = await wait_for_model(protocol, status)
        write_json(out / "training_metrics.json", {"steps": meta["steps"], "history": meta["history"]})
        status("starting_services")
        for arm in ARMS:
            await start_service(out, arm, protocol["services"][arm], children)
        status("heldout_preferences")
        await heldout_preferences(out, protocol)
        pairs = [schedule[i:i+2] for i in range(0, len(schedule), 2)]
        limit = asyncio.Semaphore(protocol["parallel_pairs"])

        async def pair(items):
            async with limit:
                results = await asyncio.gather(*(match(item) for item in items), return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

        results = await asyncio.gather(*(pair(items) for items in pairs), return_exceptions=True)
        failures = [str(result) for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("Incomplete evaluation: " + "; ".join(failures))
        from s1_policy_evaluation_results import summarize, write_report
        result = summarize(out)
        write_json(out / "summary.json", result)
        write_report(out / "results.md", result)
        status("completed", passed=result["passed"], summary=str(out / "summary.json"))
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for process in children:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 30)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        prepare(out)
    else:
        asyncio.run(run(out))


if __name__ == "__main__":
    main()
