"""Kakuna labels at every observed student decision, including the final decision."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from kakuna_label import _cut, _label, _loader, _sdpa_window
from kakuna_local import kakuna


def native_trajectory(raw):
    """Use exactly the Battle parser, state encoder and action conversion used by live Kakuna."""
    from poke_env.environment import Battle, Move, Pokemon
    from poke_env.player import Player
    from metamon.interface import UniversalAction, UniversalState

    def clean(text):
        return "".join(c for c in text.lower() if c.isalnum())

    records = [json.loads(line) for line in open(raw["source_stream"])]
    meta = records[0]
    battle = Battle(meta["battle_tag"], meta["username"], logging.getLogger("policy_native"), 9)
    wanted = {rqid: index for index, rqid in enumerate(raw["request_ids"])}
    ignored = set(Player.MESSAGES_TO_IGNORE) | {
        "t:", "expire", "uhtmlchange", "tempnotify", "tempnotifyoff", "error", "bigerror", "showteam"}
    snapshots = {}
    for record in records:
        if record["k"] != "in":
            continue
        for line in record["data"].splitlines():
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            kind = parts[1]
            if kind == "request":
                if not parts[2]:
                    continue
                request = json.loads("|".join(parts[2:]))
                battle.parse_request(request)
                # Reflected/called moves can enter the parser's move dictionary even
                # when an unchanged side request skips its normal refresh. Own move
                # slots and PP are explicit private observations in every request.
                for entry in request["side"]["pokemon"]:
                    mon = battle.team[entry["ident"]]
                    reported = request.get("active", [{}])[0].get("moves", []) if entry["active"] else []
                    names = entry["moves"]
                    if reported and all(Move.should_be_stored(Move.retrieve_id(m["id"]), 9) for m in reported):
                        names = [m["id"] for m in reported]
                    own_moves = [mon._add_move(name) for name in names]
                    mon._moves = {move.id: move for move in own_moves if move is not None}
                    for move in reported:
                        key = Move.retrieve_id(move["id"])
                        if key in mon.moves and "pp" in move:
                            mon.moves[key]._current_pp = move["pp"]
                    if reported:
                        battle._available_moves = mon.available_moves_from_request(request["active"][0])
                index = wanted.get(request.get("rqid"))
                if index is None:
                    continue
                state = UniversalState.from_Battle(battle).to_dict()
                mapping = []
                for action in range(13):
                    order = UniversalAction.action_idx_to_BattleOrder(battle, action)
                    key = None
                    if order is not None and not (action >= 9 and not state["can_tera"]):
                        if isinstance(order.order, Pokemon):
                            ident = next(k for k, mon in battle.team.items() if mon is order.order)
                            entry = next(mon for mon in request["side"]["pokemon"] if mon["ident"] == ident)
                            key = "switch_" + clean(entry["details"].split(",")[0])
                        else:
                            key = "move_" + clean(order.order.id) + ("_tera" if order.terastallize else "")
                    mapping.append(key if key in raw["legal_keys"][index] else None)
                played = raw["played_keys"][index]
                missing = set(raw["legal_keys"][index]) - set(mapping)
                if missing:
                    raise ValueError(f"native Kakuna cannot represent {meta['battle_tag']}:{request['rqid']} {sorted(missing)}")
                snapshots[request["rqid"]] = state, mapping, mapping.index(played)
            elif kind == "win":
                battle.won_by(parts[2])
            elif kind == "tie":
                battle.tied()
            elif kind not in ignored:
                battle.parse_message(parts)
    ordered = [snapshots[rqid] for rqid in raw["request_ids"]]
    masks = {index: [i for i, key in enumerate(mapping) if key is None]
             for index, (_, mapping, _) in enumerate(ordered)}
    return {**raw, "states": [s for s, _, _ in ordered], "actions": [a for _, _, a in ordered],
            "action_keys": [mapping for _, mapping, _ in ordered]}, masks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from amago.nets.transformer import FlexAttention
    from metamon.rl.metamon_to_amago import MetamonAMAGODataset

    model = kakuna(34)
    policy = model.initialize_agent(checkpoint=34, log=False).policy.eval()
    for module in policy.modules():
        if isinstance(module, FlexAttention):
            module.train()
            module.forward = _sdpa_window(module)
    device = next(policy.parameters()).device
    loader = _loader(model)
    wrapper = MetamonAMAGODataset.__new__(MetamonAMAGODataset)
    wrapper.parsed_replay_dset = loader
    root = Path(args.data)
    started = time.time()
    existing = [json.loads(line) for line in open(args.out)] if Path(args.out).exists() else []
    done = {row["file"] for row in existing}
    positions = sum(len(row["probs"]) for row in existing)
    with open(args.out, "a") as output:
        for index, line in enumerate(open(root / "files.txt")):
            path = line.strip()
            if path in done:
                continue
            raw, mask = native_trajectory(json.loads(Path(path).read_text()))
            Path(path).write_text(json.dumps(raw))
            data = loader.load_filename(path)
            # The replay loader leaves the final observation without an action mask.
            # Native collection records a genuine decision there, so all N masks come
            # from the corresponding player requests, including that final decision.
            data[1]["legal"] = [set(range(13)) - set(mask[t]) for t in range(len(raw["states"]))]
            # N observed states have N-1 completed transitions. The N-th logits use the
            # actual N-1 past actions and rewards, and predict the final observed decision.
            probs, _ = _label(policy, wrapper, data, device)
            if len(probs) != len(raw["states"]):
                raise ValueError(f"teacher prefix alignment differs for {path}")
            record = {"file": path, "probs": probs.tolist(), "chosen": raw["actions"],
                      "teacher": "kakuna-34", "mask": mask, "action_keys": raw["action_keys"],
                      "observation_encoder": "Metamon native Battle and UniversalState; own move slots and PP from requests",
                      "legal_masks": "native request at every observed decision"}
            if index == 0 and len(raw["states"]) >= 8:
                prefix = _cut(data, 5)
                prefix[1]["legal"] = data[1]["legal"][:len(next(iter(prefix[0].values())))]
                short, _ = _label(policy, wrapper, prefix, device)
                delta = float((probs[:len(short)] - short).abs().max())
                if delta > 0.001:
                    raise ValueError(f"teacher prefix causality differs by {delta}")
                record["prefix_max_abs_difference"] = delta
            output.write(json.dumps(record) + "\n")
            output.flush()
            done.add(path)
            positions += len(probs)
            print(json.dumps({"battles": len(done), "positions": positions,
                              "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
