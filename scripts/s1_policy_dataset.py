"""Build full-action, causal-history policy samples from Kakuna-labelled trajectories."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from arena.s1.convert_metamon import (action_key, convert_state, history_extras, is_selfplay,
                                      read_trajectory)
from arena.s1.identity import Splits, battle_identity
from arena.s1.policy_data import (DecisionHistory, full_action_question, full_action_target,
                                 omit_unrecorded_estimates)
from arena.s1.resources import ResourceLedger


def build_battle(job):
    row, split, per_battle = job
    trajectory = read_trajectory(row["file"])
    name = Path(row["file"]).name
    identity = battle_identity(name)
    states, actions = trajectory["states"], trajectory["actions"]
    history = DecisionHistory(8)
    resources = ResourceLedger()
    output, errors = [], Counter()
    n = min(len(row["probs"]), len(states), len(actions))
    picked = set(range(n)) if n <= per_battle else {
        round(i * (n - 1) / (per_battle - 1)) for i in range(per_battle)}
    for step in range(n):
        state = states[step]
        try:
            view, questions, keys, battle = convert_state(
                state, tag=name, turn=step + 1, own_exact=is_selfplay(name),
                extras=history_extras(states, step, actions), resource_ledger=resources)
            recent = history.observe(step, view)
            played = action_key(state, actions[step])
            contradicted = row["chosen"][step] in (row.get("extra_illegal") or {}).get(str(step), [])
            if row["missing"][step] or contradicted:
                played = None
            history.acted(played)
            if step not in picked or len(keys) < 2:
                continue
            if contradicted:
                errors["recorded_action_contradicts_derived_legality"] += 1
                continue
            question = full_action_question(questions)
            keys = list(question["criteria"])
            target, dropped = full_action_target(state, row["probs"][step], keys)
            view, question = omit_unrecorded_estimates(view, question)
            view["recent_decisions"] = recent
            output.append({
                "id": f"{identity}:{step}:best", "request": f"{name}:{step}", "group": identity,
                "purpose": "action", "qid": "best", "state": view, "question": question,
                "keys": keys, "target": target, "weight": 1.0, "source": "kakuna_trajectory",
                "teacher": "kakuna-34", "split": split, "played_key": played,
                "outcome": {"won": name.removesuffix(".lz4").endswith("_WIN.json")},
                "provenance": {"record": row["file"], "step": step, "battle": identity,
                               "tier": "reconstructed", "history_decisions": len(recent),
                               "unrecorded": (battle._last_request or {}).get("unrecorded", []),
                               "teacher_removed_mass": dropped},
            })
        except (ValueError, KeyError, TypeError, AttributeError, AssertionError) as exc:
            errors[f"{type(exc).__name__}: {str(exc)[:100]}"] += 1
            history = DecisionHistory(8)
    return split, output, dict(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="/data/wangxinqi/s1v6/labels/*.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-battle", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    splits = Splits.load()
    rows = [json.loads(line) for path in sorted(glob.glob(args.labels)) for line in open(path) if line.strip()]
    jobs = []
    for row in rows:
        identity = battle_identity(Path(row["file"]).name)
        split = splits.of(identity)
        if split in ("train", "valid"):
            jobs.append((row, split, args.per_battle))
    if args.limit:
        jobs = jobs[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, errors = Counter(), Counter()
    stats = {"train": Counter(), "valid": Counter()}
    files = {s: open(out / f"{s}.jsonl", "w") for s in stats}
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, (split, examples, failures) in enumerate(pool.map(build_battle, jobs, chunksize=2)):
                errors.update(failures)
                stats[split]["battles"] += 1
                for example in examples:
                    files[split].write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    stats[split]["with_history"] += bool(example["state"]["recent_decisions"])
                    stats[split]["tera_available"] += any(k.endswith("_tera") for k in example["keys"])
                    stats[split]["teacher_removed_mass_sum"] += example["provenance"]["teacher_removed_mass"]
                if (index + 1) % 50 == 0:
                    print(json.dumps({"battles": index + 1, "positions": dict(counts)}, ensure_ascii=False), flush=True)
    finally:
        for file in files.values():
            file.close()
    meta = {"positions": dict(counts), "distribution": {k: dict(v) for k, v in stats.items()},
            "excluded": dict(errors), "per_battle": args.per_battle,
            "split_rule": "data/splits/s1-v6.json", "history_events": 8,
            "actions": "all legal ordinary moves, Tera moves and switches in a single distribution"}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
