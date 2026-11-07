"""Export complete player-view student trajectories for causal Kakuna relabelling."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from arena.observation.capture import choice_attempts, load_stream
from arena.observation.trace import choice_action_id
from arena.s1.policy_replay import rebuild_native


def exported_games(runs):
    games, outcomes = defaultdict(dict), {}
    for path in sorted(Path(runs).rglob("s1exp_*.jsonl")):
        for line in open(path):
            row = json.loads(line)
            if row["type"] == "policy":
                games[(row["battle_tag"], path)][row["request"]["rqid"]] = {**row, "collector_file": str(path)}
            elif row["type"] == "outcome":
                outcomes[(row["battle_tag"], path)] = row
    for (tag, path), by_request in games.items():
        outcome = outcomes.get((tag, path))
        if outcome is None:
            continue
        first = next(iter(by_request.values()))
        side = first["request"]["side"]["id"]
        username = next(event[3] for event in first["public_prefix"]
                        if len(event) > 3 and event[1] == "player" and event[2] == side)
        streams = []
        for candidate in Path(runs).rglob(f"{tag}.*.jsonl"):
            with open(candidate) as stream:
                if json.loads(next(stream)).get("username") == username:
                    streams.append(candidate)
        if len(streams) != 1:
            raise ValueError(f"expected one collected player stream for {tag}, got {len(streams)}")
        attempts = choice_attempts(load_stream(streams[0]))
        accepted = {r['rqid']: r['data'].split('|', 1)[0] for r in attempts if r['error'] is None}
        rejected = {r['rqid'] for r in attempts if r['error'] is not None}
        rows, actions, masks = [], [], {}
        for rqid, row in sorted(by_request.items()):
            choice = accepted.get(rqid)
            if choice is None:
                if rqid in rejected:
                    continue
                raise ValueError(f"missing accepted action for {tag}:{rqid}")
            aid = choice_action_id(choice, row["request"]) or ""
            if aid.startswith("move:"):
                played = "move_" + aid[5:].replace("+tera", "_tera")
            elif aid.startswith("switch:"):
                from poke_env.data import to_id_str
                ident = aid[7:]
                mon = next(m for m in row["request"]["side"]["pokemon"]
                           if to_id_str(m["ident"].split(": ", 1)[-1]) == ident)
                played = "switch_" + to_id_str(mon["details"].split(",", 1)[0])
            else:
                raise ValueError(f"unsupported accepted action {tag}:{rqid}: {choice}")
            mapping = row["action_map"]
            legal = [i for i, key in enumerate(mapping) if key in row["keys"]
                     and (i < 9 or row["universal"]["can_tera"])]
            action = next((i for i in legal if mapping[i] == played), None)
            if action is None:
                raise ValueError(f"accepted action has no native index {tag}:{rqid}: {played}")
            masks[str(len(rows))] = [i for i in range(13) if i not in legal]
            rows.append({**row, "played_key": played, "accepted_order": choice, "player_stream": str(streams[0]),
                         "native_action": action})
            actions.append(action)
        if len(rows) < 2:
            raise ValueError(f"trajectory {tag} has fewer than two observed decisions")
        yield tag, rows, actions, masks, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limits", default="", help="per-opponent completed-game quotas as a JSON object")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    count = positions = 0
    limits = json.loads(args.limits) if args.limits else {}
    by_opponent = defaultdict(int)
    games = sorted(exported_games(args.runs), key=lambda game: int(game[0].rsplit("-", 1)[-1]))
    with open(out / "files.txt", "w") as names, open(out / "masks.jsonl", "w") as masks_out, \
            open(out / "observations.jsonl", "w") as records:
        for tag, rows, actions, masks, outcome in games:
            source = next((parent.name for parent in Path(rows[0]["collector_file"]).parents
                           if parent.name in ("kakuna", "foulplay", "student")), "probe")
            if source in limits and by_opponent[source] >= limits[source]:
                continue
            path = out / "trajectories" / f"{tag}.json"
            if not path.exists():
                path.write_text(json.dumps({"source_stream": rows[0]["player_stream"],
                                           "request_ids": [r["request"]["rqid"] for r in rows],
                                           "legal_keys": [r["keys"] for r in rows],
                                           "played_keys": [r["played_key"] for r in rows],
                                           "requests": [r["request"] for r in rows]}, ensure_ascii=False))
            names.write(str(path) + "\n")
            masks_out.write(json.dumps({"file": str(path), "steps": masks}) + "\n")
            for step, row in enumerate(rebuild_native(rows)):
                records.write(json.dumps({**row, "step": step, "trajectory": str(path),
                                          "outcome": {"won": outcome["won"], "turns": outcome["turns"]}},
                                         ensure_ascii=False) + "\n")
            count += 1
            by_opponent[source] += 1
            positions += len(rows)
    meta = {"completed_battles": count, "decision_positions": positions,
            "opponents": dict(by_opponent), "selection": "completed battles in creation order within each opponent",
            "teacher_context": "observed student states and accepted student actions; all policy logits kept",
            "terminal_input": "none; last submitted action is not part of its own teacher input"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
