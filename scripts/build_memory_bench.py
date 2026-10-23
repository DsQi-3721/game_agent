"""Build the memory-routing benchmark from a logged HyMem battle.

Replays every fold of the run (the planner's <summary> + the three auxiliary JSON extractions,
exactly as ``MemoryManager.fold`` merged them), so each fold yields the memory items HyMem's
contexts actually received at that point, together with the battle state the planner saw.
Each fold also gets control items whose correct routing is known by construction:

  true_fact      a key fact extracted by rule from the battle log      -> valid
  contradicted   "opp <active> has fainted" while it is on the field     -> invalid
  wrong_battle   a reveal by a species that is not in this battle        -> invalid
  stale_numbers  "your <active> is at 5% HP" while it is healthy         -> current_numbers / invalid
  generic        battle-independent advice                               -> generic
  tool_tip       advice about calling the analysis tools                 -> tool_tip

    python scripts/build_memory_bench.py --run runs/hymem_latency_randbats --out data/jev_eval/memory_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from arena.agents.hymem.memory import (
    HyMemStore,
    merge_tool_memories,
    norm_episode,
    norm_tool,
    norm_working,
    parse_json,
    render_memory,
    render_tool_memory,
    rule_episode,
)
from arena.agents.hymem.messages import extract_summary, parse_results
from arena.agents.system1.memory_router import collect_items
from arena.state.battle_log import KeyFact

GENERIC = ["Always consider type matchups before attacking.",
           "Keep your Pokemon healthy and avoid unnecessary damage.",
           "Think about what the opponent might do next turn."]
TOOL_TIPS = ["Call damage_calc with the exact move name from the entity sheet to get KO ranges.",
             "Use switch_eval before switching to see the hazard damage on entry."]
OUTSIDERS = [("garchomp", "Earthquake"), ("kingambit", "Sucker Punch"), ("dragonite", "Extreme Speed"),
             ("gholdengo", "Make It Rain")]


def state_from_text(text: str, turn: int) -> dict:
    """The planner's rendered state -> the field names the router questions refer to."""
    you, opp, field = [], [], []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("Turn"):
            head, _, rest = line.partition("|")
            if rest.strip():
                field.append(rest.strip())
        elif line.startswith("Your") or line.startswith("You must"):
            you.append(line)
        elif line.startswith("Opp"):
            opp.append(line)
    return {"battle": f"Pokemon Showdown gen9randombattle singles, turn {turn}", "you": " | ".join(you),
            "opponent": " | ".join(opp), "field": " | ".join(field) or "nothing on the field"}


def species_in(text: str, prefix: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith(prefix):
            body = line.split(":", 1)[1]
            out += re.findall(r"([a-z][a-z0-9-]+) (?:L\d+ )?\d+%", body)
    return out


def controls(state_text: str, key_facts: list[KeyFact], fold: int) -> list[dict]:
    mine = species_in(state_text, "Your active")
    opp = species_in(state_text, "Opp active")
    my_hp = re.search(r"Your active: \S+ (\d+)%", state_text)
    everyone = " ".join(species_in(state_text, "Your") + species_in(state_text, "Opp"))
    out = []
    for kf in key_facts[-3:]:
        out.append({"text": kf.text, "control": "true_fact", "expect_valid": True})
    if opp:
        out.append({"text": f"opp {opp[0]} has fainted.", "control": "contradicted", "expect_valid": False})
    sp, mv = next(((s, m) for s, m in OUTSIDERS if s not in everyone), OUTSIDERS[0])
    out.append({"text": f"opp {sp.title()} revealed {mv}.", "control": "wrong_battle", "expect_valid": False})
    if mine and my_hp and int(my_hp.group(1)) >= 40:
        out.append({"text": f"Your {mine[0]} is at 5% HP.", "control": "stale_numbers", "expect_valid": False,
                    "expect_kind": "current_numbers"})
    out.append({"text": GENERIC[fold % len(GENERIC)], "control": "generic", "expect_kind": "generic"})
    out.append({"text": TOOL_TIPS[fold % len(TOOL_TIPS)], "control": "tool_tip", "expect_kind": "tool_tip"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hymem_latency_randbats")
    ap.add_argument("--agent", default="hymem")
    ap.add_argument("--out", default="data/jev_eval/memory_bench.jsonl")
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(Path(args.run) / f"calls_{args.agent}.jsonl", encoding="utf-8")]
    end = next(r for r in rows if r["space"] == "event:battle_end")["extra"]
    all_facts = [KeyFact(int(k["turn"] or 0), k["kind"], k["text"]) for k in end["key_facts"]]
    fold_events = end["fold_events"]

    store = HyMemStore()
    folds, cur = [], None
    for r in rows:
        sp = r["space"]
        if sp == "executor":
            parsed = parse_results(r.get("output") or "")
            if parsed and parsed.get("findings"):
                store.exec_log.append({"task": "", "status": parsed.get("status", ""),
                                       "findings": parsed["findings"], "window": store.folds})
        elif sp == "fold_summary":
            cur = {"decision": r["decision"], "summary": extract_summary(r["output"] or "") or (r["output"] or ""),
                   "user": r["messages"][1]["content"], "mem": []}
            folds.append(cur)
        elif sp == "memory" and cur is not None:
            prompt = r["messages"][-1]["content"]
            kind = "episode" if "EPISODE memory" in prompt else "working" if "WORKING memory" in prompt else "tool"
            cur["mem"].append((kind, parse_json(r.get("output") or "")))
            if len(cur["mem"]) == 3:
                ev = fold_events[len([f for f in folds if len(f["mem"]) == 3]) - 1]
                turn = int(ev["turn"])
                store.key_facts = [k for k in all_facts if k.turn <= turn]
                store.summary = cur["summary"][:2000]
                mem = dict(cur["mem"])
                ep, wm, tm = norm_episode(mem.get("episode")), norm_working(mem.get("working")), norm_tool(mem.get("tool"))
                store.episode = rule_episode(store, ep or store.episode)
                if wm is not None:
                    store.working = wm
                store.tool = merge_tool_memories(store.tool, tm)
                store.folds += 1
                u = cur["user"]
                i, j = u.find("## Current state"), u.find("## Legal actions")
                state_text = u[i + len("## Current state"):j].strip() if i != -1 else ""
                cur.update(turn=turn, state_text=state_text, items=[
                    {"id": m.id, "text": m.text, "source": m.source, "turn": m.turn, "trusted": m.trusted,
                     "kind_hint": m.kind_hint} for m in collect_items(store)],
                    controls=controls(state_text, store.key_facts, store.folds),
                    rule_planner_chars=len(render_memory(store)), rule_executor_chars=len(render_tool_memory(store)))
                cur = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_items = n_ctrl = 0
    with open(out, "w", encoding="utf-8") as f:
        for k, fo in enumerate(x for x in folds if "items" in x):
            rec = {"fold": k + 1, "decision": fo["decision"], "turn": fo["turn"], "state_text": fo["state_text"],
                   "state": state_from_text(fo["state_text"], fo["turn"]), "items": fo["items"],
                   "controls": fo["controls"], "rule_planner_chars": fo["rule_planner_chars"],
                   "rule_executor_chars": fo["rule_executor_chars"]}
            n_items += len(fo["items"])
            n_ctrl += len(fo["controls"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len([x for x in folds if 'items' in x])} folds, {n_items} memory items, {n_ctrl} controls -> {out}")


if __name__ == "__main__":
    main()
