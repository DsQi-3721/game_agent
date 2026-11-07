"""Aggregate historical and newly relabelled student decisions into policy training data."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import groupby
from pathlib import Path

from arena.s1.convert_metamon import read_trajectory
from arena.s1.identity import Splits, battle_identity
from arena.s1.policy_data import full_action_target
from arena.s1.policy_replay import rebuild_historical, rebuild_native


@lru_cache(maxsize=4)
def trajectory(path):
    return read_trajectory(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    original = {row["file"]: row for p in glob.glob("/data/wangxinqi/s1v6/labels/*.jsonl")
                for row in (json.loads(line) for line in open(p))}
    first_ambiguous = {}
    for path, label in original.items():
        ambiguous = [i for i, missing in enumerate(label["missing"]) if missing]
        ambiguous += [int(step) for step, illegal in label.get("extra_illegal", {}).items()
                      if int(step) < len(label["chosen"]) and label["chosen"][int(step)] in illegal]
        first_ambiguous[path] = min(ambiguous) if ambiguous else len(label["chosen"])
    native_labels = {row["file"]: row for row in (json.loads(line) for line in open(args.labels))}
    splits = Splits.load()
    counts, groups = defaultdict(Counter), defaultdict(set)
    sources, single_options = Counter(), 0
    identities = set()
    excluded_prefixes = Counter()
    historical_mass = Counter()
    files = {split: open(root / f"{split}.jsonl", "w") for split in ("train", "valid")}

    def write(row):
        split = row["split"]
        if split not in files:
            raise ValueError(f"held-out battle entered aggregation: {row['group']}")
        target = row["target"]
        if (list(row["question"]["criteria"]) != row["keys"] or len(target) != len(row["keys"])
                or any(not math.isfinite(p) or p < 0 for p in target) or abs(sum(target) - 1.0) > 1e-5):
            raise ValueError(f"unaligned action target: {row['id']}")
        if row["id"] in identities:
            raise ValueError(f"duplicate decision: {row['id']}")
        identities.add(row["id"])
        files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split]["positions"] += 1
        counts[split]["with_history"] += bool(row["state"].get("recent_decisions"))
        counts[split]["tera_available"] += any(key.endswith("_tera") for key in row["keys"])
        counts[split]["teacher_removed_mass_sum"] += row["provenance"]["teacher_removed_mass"]
        groups[split].add(row["group"])
        sources[f"{split}:{row['source']}"] += 1

    try:
        for split in files:
            historical = Path(args.historical) / f"{split}.jsonl"
            with open(historical.with_suffix(".normalized.jsonl"), "w") as normalized:
                records = (json.loads(line) for line in open(historical))
                refreshed = (row for _, battle_rows in groupby(records, key=lambda r: r['provenance']['record'])
                             for row in rebuild_historical(battle_rows))
                for row in refreshed:
                    path, step = row["provenance"]["record"], row["provenance"]["step"]
                    target, removed = full_action_target(trajectory(path)["states"][step],
                                                          original[path]["probs"][step], row["keys"])
                    row["target"] = target
                    row["id"] = f"{row['group']}:{Path(path).name}:{step}:best"
                    row["outcome"] = {"won": Path(path).name.removesuffix(".lz4").endswith("_WIN.json")}
                    row["provenance"]["teacher_removed_mass"] = removed
                    row["provenance"]["teacher_history_complete"] = step <= first_ambiguous[path]
                    historical_mass[split] += removed
                    normalized.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if row["provenance"]["teacher_history_complete"]:
                        write(row)
                    else:
                        excluded_prefixes[split] += 1
            historical.with_suffix(".normalized.jsonl").replace(historical)
        historical_meta = json.loads((Path(args.historical) / "meta.json").read_text())
        for split in files:
            historical_meta["distribution"][split]["teacher_removed_mass_sum"] = historical_mass[split]
            historical_meta["distribution"][split]["unique_battles"] = len(groups[split])
            historical_meta["distribution"][split]["complete_prefix_positions"] = counts[split]["positions"]
        (Path(args.historical) / "meta.json").write_text(json.dumps(historical_meta, indent=2))
        records = (json.loads(line) for line in open(Path(args.native) / 'observations.jsonl'))
        refreshed = (row for _, battle_rows in groupby(records, key=lambda row: (row['battle_tag'], row['collector_file']))
                     for row in rebuild_native(battle_rows))
        for row in refreshed:
            identity = battle_identity(row["battle_tag"], platform="local-gpu8100")
            split = splits.of(identity)
            state = row['state']
            if len(row["keys"]) < 2:
                single_options += 1
                continue
            label = native_labels[row["trajectory"]]
            target, removed = full_action_target(row["universal"], label["probs"][row["step"]], row["keys"],
                                                 index_keys=label["action_keys"][row["step"]])
            source = next(parent.name for parent in Path(row["collector_file"]).parents
                          if parent.name in ("kakuna", "foulplay", "student"))
            write({"id": f"{identity}:{row['request']['rqid']}:best",
                   "request": f"{row['battle_tag']}:{row['request']['rqid']}", "group": identity,
                   "purpose": "action", "qid": "best", "state": state, "question": row["question"],
                   "keys": row["keys"], "target": target, "weight": 1.0,
                   "source": f"student_{source}", "teacher": "kakuna-34", "split": split,
                   "played_key": row["played_key"], "outcome": row["outcome"],
                   "provenance": {"battle": identity, "record": row["trajectory"], "step": row["step"],
                                  "rqid": row["request"]["rqid"], "tier": "native_player_view",
                                  "history_decisions": len(state["recent_decisions"]),
                                  "teacher_history_complete": True,
                                  "teacher_removed_mass": removed, "collector_file": row["collector_file"]}})
    finally:
        for file in files.values():
            file.close()
    overlap = groups["train"] & groups["valid"]
    if overlap:
        raise ValueError(f"train/valid battle overlap: {sorted(overlap)[:3]}")
    meta = {"positions": {split: dict(value) for split, value in counts.items()},
            "battles": {split: len(value) for split, value in groups.items()},
            "sources": dict(sources), "single_option_native_positions": single_options,
            "historical_excluded_ambiguous_teacher_prefix": dict(excluded_prefixes),
            "train_valid_battle_overlap": 0, "split_rule": "data/splits/s1-v6.json"}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
