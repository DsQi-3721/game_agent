"""Coverage and teacher alignment of complete student trajectories."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    native = Path(args.native)
    labels = {r["file"]: r for r in (json.loads(x) for x in (native / "labels.jsonl").open())}
    selected = (native / "files.txt").read_text().splitlines()
    counts, by_opponent, games = Counter(), defaultdict(Counter), {}
    for line in (native / "observations.jsonl").open():
        row = json.loads(line)
        label = labels[row["trajectory"]]
        step = row["step"]
        if set(row["keys"]) != {key for key in label["action_keys"][step] if key is not None}:
            raise ValueError(f"teacher action coverage differs: {row['battle_tag']}:{row['request']['rqid']}")
        group = next(parent.name for parent in Path(row["collector_file"]).parents
                     if parent.name in ("kakuna", "foulplay", "student"))
        game = games.setdefault(row["trajectory"], {"group": group, "positions": 0,
                    "won": row["outcome"]["won"], "stream": row["player_stream"], "behind": False, "recovered": False})
        game["positions"] += 1
        own, opp = row["state"]["you"]["pokemon_left"], row["state"]["opponent"]["pokemon_left"]
        game["recovered"] |= game["behind"] and own >= opp
        game["behind"] |= own < opp
        counts["positions"] += 1
        counts["behind_positions"] += own < opp
        counts["endgame_positions"] += min(own, opp) <= 2
        counts["tera_spent_positions"] += bool(row["state"]["you"]["tera"]["used_this_battle"])
        counts["actual_tera_actions"] += row["played_key"].endswith("_tera")
        counts["actual_switch_actions"] += row["played_key"].startswith("switch_")
    if set(selected) != games.keys():
        raise ValueError("selected trajectories and observed games differ")
    for path, game in games.items():
        if len(labels[path]["probs"]) != game["positions"]:
            raise ValueError(f"teacher position count differs: {path}")
        terminal = any(event.get("k") == "in" and any(line.startswith(("|win|", "|tie"))
                       for line in event["data"].splitlines())
                       for event in (json.loads(x) for x in open(game["stream"])))
        if not terminal:
            raise ValueError(f"trajectory has no observed terminal result: {path}")
        by_opponent[game["group"]]["battles"] += 1
        by_opponent[game["group"]]["wins"] += game["won"]
        counts["battles"] += 1
        counts["wins"] += game["won"]
        counts["games_with_material_recovery"] += game["recovered"]
        counts["wins_after_material_deficit"] += game["behind"] and game["won"]
        counts["games_over_50_decisions"] += game["positions"] > 50
    policy = json.loads((Path(args.policy) / "meta.json").read_text())
    meta = {"native": dict(counts), "opponents": {k: dict(v) for k, v in by_opponent.items()},
            "policy": policy, "teacher_coverage": "every legal action at every selected decision",
            "terminal_coverage": "every selected game has a player-observed win/tie event",
            "collection_results": "coverage statistics; not a controlled model-strength evaluation"}
    Path(args.out).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
