"""Two equal-budget searches of fixed player-view hypotheses for policy preferences."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import random
import time

from evidence import aggregate, read_rows, run_search
import analyze as replay


def paired_search(state, milliseconds):
    return run_search(state, milliseconds), run_search(state, milliseconds)


def compare(case, samples, first, second, margin=0.03):
    policies = [aggregate(samples, results) for results in (first, second)]
    eligible = [key for key in case["keys"]
                if all(key in result["ours"] and result["ours"][key]["visits"] > 0
                       for result in first + second)
                and all(policy[key]["visits"] >= 32 for policy in policies)]
    pairs = []
    if eligible:
        preferred = max(eligible, key=lambda key: sum(policy[key]["score"] for policy in policies))
        rejected = sorted((key for key in eligible if key != preferred),
                          key=lambda key: (key != case["played_key"],
                                           -case["target"][case["keys"].index(key)]))
        for key in rejected:
            gaps = [policy[preferred]["score"] - policy[key]["score"] for policy in policies]
            if min(gaps) < margin:
                continue
            pairs.append({"preferred": preferred, "rejected": key, "score_gaps": gaps,
                          "preferred_scores": [p[preferred]["score"] for p in policies],
                          "rejected_scores": [p[key]["score"] for p in policies],
                          "preferred_visits": [p[preferred]["visits"] for p in policies],
                          "rejected_visits": [p[key]["visits"] for p in policies]})
            if len(pairs) >= 3:
                break
    return {"id": case["id"], "pairs": pairs, "policies": policies,
            "legal_options": len(case["keys"]), "comparable_options": len(eligible),
            "sample_count": len(samples), "minimum_score_gap": margin,
            "coverage": "both actions visited under every fixed hypothesis in each search"}


async def battle_evidence(record, cases, settings):
    replay.FoulPlayConfig.pokemon_format = "gen9ou"
    replay.FoulPlayConfig.smogon_stats = None
    replay.FoulPlayConfig.parallelism = settings["samples"]
    replay.FoulPlayConfig.log_to_file = False
    replay.FoulPlayConfig.username = record["our_name"]
    needed = {int(case_id.rsplit(":", 1)[-1]) for case_id in cases}
    ws = replay.ReplayWS(record["messages"])
    mode = replay.standard_battle.StandardBattleMode()
    battle = await mode.start_battle(ws, "gen9ou", replay.export_to_dict(record["team"]))
    decisions = {d["rqid"]: d for d in record["decisions"]}
    output = []
    while (message := ws.next()) is not None:
        if not replay.update_battle(battle, message):
            continue
        decision = decisions.get(battle.rqid)
        if decision is None:
            continue
        if battle.rqid in needed:
            case_id = f"{record['battle_tag']}:{battle.rqid}"
            observed = deepcopy(battle)
            observed.user.update_from_request_json(observed.request_json)
            random.seed(f"{settings['seed']}:{case_id}")
            sampled = mode.prepare_battles(observed, settings["samples"])
            samples = [{"state": replay.battle_to_poke_engine_state(b).to_string(), "weight": weight}
                       for b, weight in sampled]
            results = [paired_search(sample["state"], settings["search_ms"]) for sample in samples]
            first, second = [r[0] for r in results], [r[1] for r in results]
            for sample, result in zip(samples, first):
                sample["initial"] = result
            comparison = compare(cases[case_id], samples, first, second)
            comparison.update(replication=second, search_ms=settings["search_ms"])
            evidence = {"id": case_id, "samples": samples, "search_ms": settings["search_ms"],
                        "sampling_seed": settings["seed"]}
            output.append((evidence, comparison))
        if decision.get("final"):
            battle.user.last_selected_move = replay.LastUsedMove(
                battle.user.active.name, replay.fp_name(decision["final"]), battle.turn)
        if battle.rqid >= max(needed):
            break
    return output


def battle_worker(record, cases, settings):
    return asyncio.run(battle_evidence(record, cases, settings))


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = {r["id"]: r for r in read_rows(args.cases)}
    compared_path = out / "comparisons.jsonl"
    previous = read_rows(compared_path) if compared_path.exists() else []
    done = {r["id"] for r in previous}
    counts = Counter(positions=len(done), pairs=sum(len(r["pairs"]) for r in previous),
                     ranked_positions=sum(bool(r["pairs"]) for r in previous))
    started = time.time()
    with open(out / "engine.jsonl", "a") as engine, open(compared_path, "a") as output, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        settings = {"samples": args.samples, "search_ms": args.search_ms, "seed": args.seed}
        for record in read_rows(args.streams):
            ids = {f"{record['battle_tag']}:{d['rqid']}" for d in record["decisions"]}
            needed = ids & (cases.keys() - done)
            if not needed:
                continue
            futures.append(pool.submit(battle_worker, record, {case_id: cases[case_id] for case_id in needed}, settings))
        for future in as_completed(futures):
            for evidence, comparison in future.result():
                engine.write(json.dumps(evidence) + "\n")
                engine.flush()
                output.write(json.dumps(comparison) + "\n")
                output.flush()
                done.add(comparison["id"])
                counts["positions"] += 1
                counts["pairs"] += len(comparison["pairs"])
                counts["ranked_positions"] += bool(comparison["pairs"])
                if counts["positions"] % 20 == 0:
                    print(json.dumps({**counts, "seconds": round(time.time() - started, 1)}), flush=True)
    missing = cases.keys() - done
    if missing:
        raise ValueError(f"{len(missing)} requested decisions were not reconstructed: {sorted(missing)[:5]}")
    meta = {**counts, "seconds": round(time.time() - started, 1), "search_ms_per_hypothesis_per_run": args.search_ms,
            "replications": 2, "samples_requested": args.samples, "workers": args.workers,
            "comparison": "same sampled states and weights, equal time budgets; directional agreement in both runs",
            "score_units": "engine search value, not a calibrated winning probability"}
    (out / "search_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=61106)
    main(parser.parse_args())
