"""Offline evidence tools on Foul Play's native samples and poke-engine.

Run with external/foulplay/venv/bin/python. Subcommands: extract, tools, reference.
Private engine states are separate from the player-visible cases and model prompts.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import random
import time

import analyze as replay
from poke_engine import State, generate_instructions, monte_carlo_tree_search


def read_rows(path):
    return [json.loads(line) for line in open(path)]


def to_key(name):
    if name.startswith("switch "):
        return "switch_" + "".join(c for c in name[7:].lower() if c.isalnum())
    return "move_" + name[:-5] + "_tera" if name.endswith("-tera") else "move_" + name


def native_key(key):
    if key == "move_No Move":
        return "none"
    if key.startswith("switch_"):
        return key[7:]
    return key[5:-5] + "-tera" if key.endswith("_tera") else key[5:]


def run_search(state, milliseconds):
    t = time.perf_counter()
    result = monte_carlo_tree_search(State.from_string(state), milliseconds, threads=1)
    def side(opts):
        return {to_key(o.move_choice): {"visits": o.visits, "total_score": o.total_score}
                for o in opts}
    return {"ours": side(result.side_one), "theirs": side(result.side_two),
            "visits": result.total_visits, "seconds": time.perf_counter() - t}


def merge_search(a, b):
    out = {"visits": a["visits"] + b["visits"], "seconds": a["seconds"] + b["seconds"]}
    for side in ("ours", "theirs"):
        out[side] = {}
        for k in a[side].keys() | b[side].keys():
            out[side][k] = {f: a[side].get(k, {}).get(f, 0) + b[side].get(k, {}).get(f, 0)
                            for f in ("visits", "total_score")}
    return out


def aggregate(samples, results):
    """Each hypothesis keeps its original weight, regardless of allocated search time."""
    z = sum(s["weight"] for s in samples)
    policy = {}
    for s, r in zip(samples, results):
        w = s["weight"] / z
        for key, v in r["ours"].items():
            p = policy.setdefault(key, {"share": 0., "score": 0., "score_weight": 0., "visits": 0})
            p["share"] += w * v["visits"] / max(1, r["visits"])
            p["visits"] += v["visits"]
            if v["visits"]:
                p["score"] += w * v["total_score"] / v["visits"]
                p["score_weight"] += w
    for p in policy.values():
        p["score"] = p["score"] / p["score_weight"] if p["score_weight"] else None
        del p["score_weight"]
    return policy


def sample_description(state):
    s = State.from_string(state)
    active = s.side_two.pokemon[int(s.side_two.active_index)]
    return {"active": active.id, "item": active.item, "ability": active.ability,
            "speed": active.speed, "moves": [m.id for m in active.moves if m.id != "none"],
            "tera": active.tera_type}


def public_evidence(record, results=None):
    results = results or [s["initial"] for s in record["samples"]]
    return {"policy": aggregate(record["samples"], results), "samples": [
        {"sample": i, "weight": s["weight"], "assumed_active_set": sample_description(s["state"]),
         "policy": {k: {"share": v["visits"] / max(1, r["visits"]),
                        "score": v["total_score"] / v["visits"] if v["visits"] else None,
                        "visits": v["visits"]} for k, v in r["ours"].items()}}
        for i, (s, r) in enumerate(zip(record["samples"], results))]}


async def extract(args):
    selected = {r["id"] for r in read_rows(args.cases)}
    done = {r["id"] for r in read_rows(args.out)} if args.out.exists() else set()
    replay.FoulPlayConfig.pokemon_format = "gen9ou"
    replay.FoulPlayConfig.smogon_stats = None
    replay.FoulPlayConfig.parallelism = args.samples
    replay.FoulPlayConfig.log_to_file = False
    with open(args.out, "a") as f, open(args.out.with_suffix(".public.jsonl"), "a") as public, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in read_rows(args.streams):
            needed = {d["rqid"] for d in rec["decisions"]
                      if f"{rec['battle_tag']}:{d['rqid']}" in selected - done}
            if not needed:
                continue
            replay.FoulPlayConfig.username = rec["our_name"]
            ws = replay.ReplayWS(rec["messages"])
            mode = replay.standard_battle.StandardBattleMode()
            battle = await mode.start_battle(ws, "gen9ou", replay.export_to_dict(rec["team"]))
            decs = {d["rqid"]: d for d in rec["decisions"]}
            while (msg := ws.next()) is not None:
                if not replay.update_battle(battle, msg):
                    continue
                d = decs.get(battle.rqid)
                if d is None:
                    continue
                if battle.rqid in needed:
                    case_id = f"{rec['battle_tag']}:{battle.rqid}"
                    b = deepcopy(battle)
                    b.user.update_from_request_json(b.request_json)
                    random.seed(f"{args.seed}:{case_id}")
                    sampled = mode.prepare_battles(b, args.samples)
                    samples = [{"state": replay.battle_to_poke_engine_state(bb).to_string(), "weight": w}
                               for bb, w in sampled]
                    futures = [pool.submit(run_search, s["state"], args.search_ms) for s in samples]
                    for s, future in zip(samples, futures):
                        s["initial"] = future.result()
                    row = {"id": case_id, "battle_tag": rec["battle_tag"], "samples": samples,
                           "search_ms": args.search_ms, "sampling_seed": args.seed}
                    f.write(json.dumps(row) + "\n"); f.flush()
                    public.write(json.dumps({"id": case_id, **public_evidence(row)}) + "\n"); public.flush()
                    done.add(case_id)
                    print(json.dumps({"extracted": len(done), "id": case_id}), flush=True)
                if d.get("final"):
                    battle.user.last_selected_move = replay.LastUsedMove(
                        battle.user.active.name, replay.fp_name(d["final"]), battle.turn)
                if battle.rqid >= max(needed):
                    break


def branches(state, actions, opponent_actions):
    """One-turn consequences, conditional on each sampled set and opponent action."""
    s = State.from_string(state)
    before = [[p.hp for p in side.pokemon] for side in (s.side_one, s.side_two)]
    out = []
    for action in actions:
        for opp in opponent_actions:
            outcomes = generate_instructions(s, native_key(action), native_key(opp))
            expected_loss = [0., 0.]
            ko = [0., 0.]
            effects = Counter()
            for inst in outcomes:
                child = State.from_string(state)
                child = child.apply_instructions(inst)
                weight = inst.percentage / 100
                for i, side in enumerate((child.side_one, child.side_two)):
                    expected_loss[i] += weight * sum((hp - p.hp) / p.maxhp for hp, p in zip(before[i], side.pokemon)
                                                    if p.maxhp)
                    ko[i] += weight * sum(hp > 0 and p.hp == 0 for hp, p in zip(before[i], side.pokemon))
                for instruction in inst.instruction_list:
                    name = repr(instruction)
                    if not name.startswith(("Damage", "Heal")):
                        effects[name] += weight
            out.append({"action": action, "opponent_action": opp, "our_hp_loss": expected_loss[0],
                        "their_hp_loss": expected_loss[1], "our_expected_kos": ko[0], "their_expected_kos": ko[1],
                        "effects": dict(effects.most_common(8))})
    return out


def tools(args):
    cases = {r["id"] for r in read_rows(args.cases)}
    plans = {r["id"]: r for r in read_rows(args.plans)}
    done = {r["id"] for r in read_rows(args.out)} if args.out.exists() else set()
    with open(args.out, "a") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in read_rows(args.engine):
            if rec["id"] not in plans or rec["id"] not in cases or rec["id"] in done:
                continue
            n = len(rec["samples"])
            plan = plans[rec["id"]]["plan"]
            allocation = Counter(plan["sample_ids"])
            initial = [s["initial"] for s in rec["samples"]]
            consequences = []
            branch_start = time.process_time()
            for i in sorted(allocation):
                s = rec["samples"][i]
                valid_actions = [k for k in plan["candidates"] if k in initial[i]["ours"]]
                opposing = sorted(initial[i]["theirs"], key=lambda k: -initial[i]["theirs"][k]["visits"])[:2]
                consequences.append({"sample": i, "conditional_outcomes": branches(s["state"], valid_actions, opposing)})
            branch_seconds = time.process_time() - branch_start
            branch_ms = math.ceil(1000 * branch_seconds)
            remaining = args.search_ms * n - branch_ms
            if remaining < len(allocation):
                raise ValueError("Conditional simulation exhausted the supplemental budget")
            units_total = sum(allocation.values())
            budgets = {i: remaining * units // units_total for i, units in allocation.items()}
            for i in sorted(budgets)[:remaining - sum(budgets.values())]:
                budgets[i] += 1
            # Conditional simulation is charged to the targeted arm's compute budget.
            control_f = [pool.submit(run_search, s["state"], args.search_ms) for s in rec["samples"]]
            targeted_f = {i: pool.submit(run_search, rec["samples"][i]["state"], ms) for i, ms in budgets.items()}
            control = [merge_search(a, ft.result()) for a, ft in zip(initial, control_f)]
            targeted = [merge_search(a, targeted_f[i].result()) if i in targeted_f else a
                        for i, a in enumerate(initial)]
            row = {"id": rec["id"], "control_policy": aggregate(rec["samples"], control),
                   "evidence": public_evidence(rec, targeted), "branches": consequences,
                   "search_budget_ms_per_arm": (rec["search_ms"] + args.search_ms) * n,
                   "branch_seconds": branch_seconds, "branch_budget_ms": branch_ms,
                   "targeted_allocations_ms": budgets,
                   "targeted_search_seconds": sum(r["seconds"] for r in targeted),
                   "control_search_seconds": sum(r["seconds"] for r in control)}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(json.dumps({"tools": rec["id"], "allocation": allocation}), flush=True)


def reference(args):
    selected = {r["id"] for r in read_rows(args.cases)}
    done = {r["id"] for r in read_rows(args.out)} if args.out.exists() else set()
    with open(args.out, "a") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in read_rows(args.engine):
            if rec["id"] not in selected or rec["id"] in done:
                continue
            futures = [pool.submit(run_search, s["state"], args.search_ms) for s in rec["samples"]]
            results = [ft.result() for ft in futures]
            row = {"id": rec["id"], "policy": aggregate(rec["samples"], results),
                   "search_ms": args.search_ms, "seconds": sum(r["seconds"] for r in results)}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(json.dumps({"reference": rec["id"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("extract", "tools", "reference"):
        p = sub.add_parser(name)
        p.add_argument("--cases", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--search-ms", type=int, default=5000 if name == "reference" else 700)
        if name == "extract":
            p.add_argument("--streams", type=Path, required=True)
            p.add_argument("--samples", type=int, default=8)
            p.add_argument("--seed", type=int, default=20261105)
        else:
            p.add_argument("--engine", type=Path, required=True)
        if name == "tools":
            p.add_argument("--plans", type=Path, required=True)
    args = ap.parse_args()
    # analyze.py changes cwd to the external repository; CLI paths refer to the arena root.
    root = Path(__file__).resolve().parents[2]
    for k, v in vars(args).items():
        if isinstance(v, Path) and not v.is_absolute():
            setattr(args, k, root / v)
    logging.disable(logging.CRITICAL)
    if args.command == "extract":
        asyncio.run(extract(args))
    elif args.command == "tools":
        tools(args)
    else:
        reference(args)


if __name__ == "__main__":
    main()
