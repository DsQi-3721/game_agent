"""Select critical student decisions and build separate search and trajectory supervision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from arena.llm.typesafe import choice, noul
from fp_judge import export_team, find_log, messages


SETUP = {"swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "bulkup",
         "irondefense", "shellsmash", "agility", "shiftgear", "tailglow"}


def rows(path):
    with open(path) as file:
        yield from (json.loads(line) for line in file)


def policies(root):
    for split in ("train", "valid"):
        yield from (r for r in rows(Path(root) / f"{split}.jsonl") if r["source"].startswith("student_"))


def critical_reasons(row):
    keys, target = row["keys"], row["target"]
    teacher = keys[max(range(len(keys)), key=target.__getitem__)]
    played = row["played_key"]
    reasons = []
    if teacher != played:
        reasons.append("teacher_student_disagreement")
    if any(k.endswith("_tera") for k in (played, teacher)):
        reasons.append("tera_resource")
    if any(k.removeprefix("move_").removesuffix("_tera") in SETUP for k in (played, teacher)):
        reasons.append("setup")
    if sum("played switch_" in event for event in row["state"]["recent_decisions"]) >= 2:
        reasons.append("repeated_switching")
    if min(row["state"][side]["pokemon_left"] for side in ("you", "opponent")) <= 3:
        reasons.append("endgame")
    return reasons


def select(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games = defaultdict(list)
    for row in policies(args.policy):
        reasons = critical_reasons(row)
        if reasons:
            games[row["group"]].append({**row, "critical_reasons": reasons})
    cases = []
    for candidates in games.values():
        selected = []
        for category in ("tera_resource", "teacher_student_disagreement", "endgame", "setup", "repeated_switching"):
            eligible = [r for r in candidates if category in r["critical_reasons"]
                        and all(abs(r["provenance"]["step"] - s["provenance"]["step"]) >= 4 for s in selected)]
            if not eligible:
                continue
            def importance(row):
                played = row["keys"].index(row["played_key"])
                return max(row["target"]) - row["target"][played]
            selected.append(max(eligible, key=importance))
            if len(selected) >= args.per_battle:
                break
        cases.extend({**row, "policy_id": row["id"], "id": row["request"]} for row in selected)
    with open(out / "cases.jsonl", "w") as file:
        for row in cases:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    observations = defaultdict(list)
    selected_games = {row["provenance"]["record"] for row in cases}
    for row in rows(Path(args.native) / "observations.jsonl"):
        if row["trajectory"] in selected_games:
            observations[row["trajectory"]].append(row)
    with open(out / "streams.jsonl", "w") as file:
        for observed in observations.values():
            first = observed[0]
            tag = first["battle_tag"]
            stream = list(rows(first["player_stream"]))
            meta = stream[0]
            side = first["request"]["side"]["id"]
            log_path = find_log(tag)
            if log_path is None:
                raise ValueError(f"missing own team record: {tag}")
            log = json.loads(Path(log_path).read_text())
            visible = [line for event in stream if event["k"] == "in"
                       for line in event["data"].splitlines()
                       if line.startswith("|") and not line.startswith(("|init|", "|title|"))]
            record = {"battle_tag": tag, "our_name": meta["username"],
                      "team": export_team(log[f"{side}team"]),
                      "messages": messages(tag, f"{log['p1']} vs. {log['p2']}", visible),
                      "decisions": [{"rqid": r["request"]["rqid"], "final": r["played_key"]} for r in observed]}
            file.write(json.dumps(record) + "\n")
    meta = {"battles": len(observations), "positions": len(cases),
            "reasons": dict(Counter(reason for row in cases for reason in row["critical_reasons"])),
            "splits": dict(Counter(r["split"] for r in cases)),
            "input": "captured player messages and the player's own team; opponent sets are sampled by Foul Play"}
    (out / "selection.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = {r["id"]: r for r in rows(Path(args.search) / "cases.jsonl")}
    comparisons = {r["id"]: r for r in rows(Path(args.search) / "comparisons.jsonl")}
    native = defaultdict(list)
    for row in rows(Path(args.native) / "observations.jsonl"):
        native[row["trajectory"]].append(row)
    counts = defaultdict(Counter)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "valid")}

    def write(row):
        files[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[row["split"]][row["purpose"]] += 1

    try:
        for case_id, comparison in comparisons.items():
            case = cases[case_id]
            for pair in comparison["pairs"]:
                preferred, rejected = pair["preferred"], pair["rejected"]
                target = [float(key == preferred) for key in case["keys"]]
                write({**case, "id": f"{case['policy_id']}:rank:{preferred}:{rejected}",
                       "purpose": "action_rank", "loss_kind": "pairwise", "source": "fp_action_comparison",
                       "teacher": "Foul Play equal-budget search on fixed sampled sets",
                       "teacher_target": case["target"], "target": target,
                       "preference": {"preferred": case["keys"].index(preferred),
                                      "rejected": case["keys"].index(rejected)},
                       "comparison": {**pair, "evidence": str(Path(args.search) / "engine.jsonl"),
                                      "replication": str(Path(args.search) / "comparisons.jsonl"), "case": case_id}})
        for row in policies(args.policy):
            observed = native[row["provenance"]["record"]]
            step = row["provenance"]["step"]
            state = {**deepcopy(row["state"]), "proposed_action": row["played_key"]}
            question = noul("Will you win the complete battle after the proposed action and subsequent play?")
            won = row["outcome"]["won"]
            write({**row, "id": row["id"] + ":outcome", "state": state, "purpose": "outcome",
                   "qid": "battle_outcome", "question": question, "keys": ["true", "false"],
                   "target": [float(won), float(not won)], "teacher": "observed_terminal_result",
                   "source": "student_trajectory_result"})
            if step + args.horizon >= len(observed):
                continue
            future = observed[step + args.horizon]
            current_balance = row["state"]["you"]["pokemon_left"] - row["state"]["opponent"]["pokemon_left"]
            next_balance = future["state"]["you"]["pokemon_left"] - future["state"]["opponent"]["pokemon_left"]
            delta = next_balance - current_balance
            keys = ["improves", "unchanged", "worsens"]
            answer = "improves" if delta > 0 else "worsens" if delta < 0 else "unchanged"
            question = choice(f"After the proposed action and {args.horizon} subsequent decision points, "
                              "how does your surviving-Pokemon advantage (yours minus theirs) change?", {
                                  "improves": "The difference increases.", "unchanged": "The difference stays the same.",
                                  "worsens": "The difference decreases."})
            write({**row, "id": row["id"] + ":resource", "state": state, "purpose": "resource",
                   "qid": "future_material", "question": question, "keys": keys,
                   "target": [float(key == answer) for key in keys], "teacher": "observed_continuation",
                   "source": "student_trajectory_result",
                   "continuation": {"horizon_decisions": args.horizon, "future_rqid": future["request"]["rqid"],
                                    "current_balance": current_balance, "future_balance": next_balance}})
    finally:
        for file in files.values():
            file.close()
    meta = {"counts": {s: dict(c) for s, c in counts.items()}, "search_positions": len(comparisons),
            "ranked_positions": sum(bool(c["pairs"]) for c in comparisons.values()),
            "horizon_decisions": args.horizon, "future_information": "targets and provenance only"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--policy", required=True)
    select_parser.add_argument("--native", required=True)
    select_parser.add_argument("--out", required=True)
    select_parser.add_argument("--per-battle", type=int, default=3)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--policy", required=True)
    build_parser.add_argument("--native", required=True)
    build_parser.add_argument("--search", required=True)
    build_parser.add_argument("--out", required=True)
    build_parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    (select if args.command == "select" else build)(args)


if __name__ == "__main__":
    main()
