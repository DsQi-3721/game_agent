"""Assemble policy, action preferences, trajectory outcomes and framework replay for LoRA training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random

from arena.s1.identity import Splits
from arena.s1.prompts import render
from arena.s1.scorer import _chat_ids


def read(path):
    with open(path) as file:
        yield from (json.loads(line) for line in file)


def framework_examples(path, limit, rng):
    buckets = defaultdict(list)
    for row in read(path):
        if row["purpose"] in ("memory", "tool_plan", "step_control", "route_task"):
            buckets[row["purpose"]].append(row)
    selected, spare = [], []
    for purpose, fraction in (("memory", 0.7), ("tool_plan", 0.2), ("step_control", 0.07), ("route_task", 0.03)):
        bucket = buckets[purpose]
        rng.shuffle(bucket)
        count = min(len(bucket), round(limit * fraction))
        selected.extend(bucket[:count])
        spare.extend(bucket[count:])
    rng.shuffle(spare)
    return selected + spare[:max(0, limit - len(selected))]


def category(row):
    return row["purpose"] if row["purpose"] in ("action", "action_rank", "outcome", "resource") else "framework"


def permute_options(row, order):
    """Vary action presentation without changing action probabilities or preference identities."""
    row = deepcopy(row)
    old_keys = row["keys"]
    row["keys"] = [old_keys[i] for i in order]
    row["question"]["criteria"] = {key: row["question"]["criteria"][key] for key in row["keys"]}
    row["target"] = [row["target"][i] for i in order]
    if "teacher_target" in row:
        row["teacher_target"] = [row["teacher_target"][i] for i in order]
    if row.get("preference"):
        row["preference"] = {name: order.index(index) for name, index in row["preference"].items()}
    return row


def quantiles(values):
    values = sorted(values)
    return {"n": len(values), "min": values[0], "p50": values[len(values) // 2],
            "p90": values[int(len(values) * .9)], "p99": values[int(len(values) * .99)], "max": values[-1]}


def build_split(args, split, tokenizer, splits, out, budget):
    rng = random.Random(args.seed + int(split == "valid"))
    candidates = list(read(Path(args.policy) / f"{split}.jsonl"))
    candidates.extend(read(Path(args.supervision) / f"{split}.jsonl"))
    candidates.extend(framework_examples(Path(args.framework) / f"{split}.jsonl",
                                         10000 if split == "train" else 1000, rng))
    retained, lengths, histories, skipped, seen = [], [], [], Counter(), set()
    for row in candidates:
        if row["purpose"] in ("action", "action_rank"):
            order = list(range(len(row["keys"])))
            rng.shuffle(order)
            row = permute_options(row, order)
        if splits.of(row["group"]) != split:
            raise ValueError(f"battle split differs: {row['id']}")
        if row["id"] in seen:
            raise ValueError(f"repeated training example: {row['id']}")
        seen.add(row["id"])
        messages, keys = render(row["state"], row["question"])
        target = row["target"]
        if (keys != row["keys"] or len(keys) != len(target) or abs(sum(target) - 1) > 1e-5
                or any(not math.isfinite(p) or p < 0 for p in target)):
            raise ValueError(f"target differs from the presented options: {row['id']}")
        length = len(_chat_ids(tokenizer, messages))
        if length > args.max_tokens:
            raise ValueError(f"{row['id']} requires {length} tokens; increase --max-tokens "
                             "to keep the complete policy input")
        if row["purpose"] == "action":
            if not row["provenance"]["teacher_history_complete"]:
                raise ValueError(f"teacher history is incomplete: {row['id']}")
            histories.append(len(tokenizer.encode(json.dumps(row["state"].get("recent_decisions", []),
                                                               separators=(",", ":")), add_special_tokens=False)))
        retained.append(row)
        lengths.append(length)
    masses = Counter()
    for row in retained:
        mass = 2.0 if row["purpose"] == "action" and row["source"].startswith("student_") else 1.0
        row["weight"] = mass
        masses[category(row)] += mass
    for key in budget:
        if not masses[key] and (split == "train" or key == "action"):
            raise ValueError(f"no usable {key} supervision in {split}")
    n = len(retained)
    budget_total = sum(budget[key] for key in masses)
    weighted, counts, sources = Counter(), Counter(), Counter()
    groups = set()
    rng.shuffle(retained)
    with open(out / f"{split}.jsonl", "w") as file:
        for row in retained:
            key = category(row)
            row["weight"] *= budget[key] * n / (masses[key] * budget_total)
            weighted[key] += row["weight"]
            counts[row["purpose"]] += 1
            sources[row["source"]] += 1
            groups.add(row["group"])
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {"examples": n, "purposes": dict(counts), "sources": dict(sources),
                "effective_budget": {key: value / n for key, value in weighted.items()},
                "policy_unique_positions": counts["action"], "groups": len(groups),
                "prompt_tokens": quantiles(lengths), "history_tokens": quantiles(histories),
                "overlength_excluded": dict(skipped)}
    print(json.dumps({"split": split, **metadata}), flush=True)
    return metadata, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--framework", default="data/s1/s1-v7")
    parser.add_argument("--base", default="/data/models/Qwen3.5-2B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=61106)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    splits = Splits.load()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata, groups = {}, {}
    budget = {"action": .75, "action_rank": .05, "outcome": .05, "resource": .05, "framework": .10}
    for split in ("train", "valid"):
        metadata[split], groups[split] = build_split(args, split, tokenizer, splits, out, budget)
    write_metadata(args, out, metadata, groups)


def write_metadata(args, out, metadata, groups):
    if groups["train"] & groups["valid"]:
        raise ValueError("a battle group occurs in both training and validation")
    metadata.update({"train_valid_group_overlap": 0, "max_tokens": args.max_tokens,
                     "policy_option_order": "randomized per example with aligned targets and preference indices",
                     "policy_native_relative_weight": 2.0, "epochs_planned": 1,
                     "deployment": {"mode": "flat", "history_events": 8, "teacher": False, "ask_hard": False},
                     "losses": {"action": "soft cross entropy", "action_rank": "logistic pairwise preference",
                                "outcome": "terminal win prediction", "resource": "four-decision material change",
                                "framework": "soft cross entropy"}})
    (out / "meta.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
