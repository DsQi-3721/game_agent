"""Lossless MicroRTS engine state extraction and compact text rendering."""

import contextlib
import json

import jpype

ACTION_NAMES = ("wait", "move", "harvest", "return", "produce", "attack")


def read_state(gs, player, unit_types):
    """Extract a JSON-serializable, fully observable snapshot from MicroRTS GameState."""
    writer = jpype.JClass("java.io.StringWriter")()
    gs.toJSON(writer)
    raw = json.loads(str(writer.toString()))
    pgs = raw["pgs"]
    actions = {a["ID"]: a for a in raw["actions"]}
    units = []
    for u in sorted(pgs["units"], key=lambda u: u["ID"]):
        action = None
        if u["ID"] in actions:
            a = actions[u["ID"]]
            assignment = gs.getActionAssignment(gs.getUnit(u["ID"]))
            action = {
                **a["action"],
                "name": ACTION_NAMES[a["action"]["type"]],
                "started_at": a["time"],
                "remaining_ticks": int(
                    a["time"] + assignment.action.ETA(assignment.unit) - raw["time"]
                ),
            }
        units.append(
            {
                "id": u["ID"],
                "owner": u["player"],
                "type": u["type"],
                "x": u["x"],
                "y": u["y"],
                "hp": u["hitpoints"],
                "resources": u["resources"],
                "action": action,
            }
        )
    usage = gs.getResourceUsage()
    return {
        "time": raw["time"],
        "player": player,
        "width": pgs["width"],
        "height": pgs["height"],
        "terrain": pgs["terrain"],
        "players": [
            {
                "id": p["ID"],
                "resources": p["resources"],
                "available_resources": p["resources"] - int(usage.getResourcesUsed(p["ID"])),
            }
            for p in pgs["players"]
        ],
        "units": units,
        "unit_types": unit_types,
    }


def _fmt_unit(u, unit_types):
    max_hp = unit_types.get(u["type"], {}).get("hp")
    hp = f"{u['hp']}/{max_hp}" if max_hp else str(u["hp"])
    parts = [f"U{u['id']} {u['type']}({u['x']},{u['y']}) hp={hp}"]
    if u["resources"]:
        parts.append(f"carry={u['resources']}")
    act = u["action"]
    if act:
        parts.append(f"act={act['name']}({act['remaining_ticks']}t)")
    else:
        parts.append("idle")
    return " ".join(parts)


def _force_counts(units):
    b = sum(1 for u in units if u["type"] == "Base")
    rax = sum(1 for u in units if u["type"] == "Barracks")
    w = sum(1 for u in units if u["type"] == "Worker")
    mil = sum(1 for u in units if u["type"] in ("Light", "Heavy", "Ranged"))
    hp = sum(int(u["hp"]) for u in units)
    return f"B={b} Rax={rax} W={w} Mil={mil} hp={hp}"


def ensure_force_summary(context_text):
    """Idempotently inject the 1-line Force summary into an existing state_to_text string."""
    if "\nForce: " in context_text:
        return context_text
    lines = context_text.splitlines()
    if not lines:
        return context_text
    section = None
    me_u, opp_u = [], []
    tick = 0
    for token in lines[0].split():
        if token.startswith("t="):
            with contextlib.suppress(ValueError):
                tick = int(token[2:])
    for ln in lines[1:]:
        if ln.startswith("My units"):
            section = "me"
            continue
        if ln.startswith("Enemy units"):
            section = "opp"
            continue
        if ln.startswith("Resources:") or ln.startswith("Walls:"):
            section = None
            continue
        if ln.startswith("- U") and section in ("me", "opp"):
            parts = ln[2:].split()
            if len(parts) >= 3:
                utype = parts[1].split("(")[0]
                hp_tok = next((p for p in parts if p.startswith("hp=")), "hp=1")
                try:
                    hp = int(hp_tok[3:].split("/")[0])
                except ValueError:
                    hp = 1
                (me_u if section == "me" else opp_u).append({"type": utype, "hp": hp})
    phase = "early" if tick < 150 else "mid" if tick < 500 else "late"
    summary = f"Force: me[{_force_counts(me_u)}] vs opp[{_force_counts(opp_u)}] phase={phase}"
    insert_at = 2 if len(lines) > 1 and lines[1].startswith("Res:") else 1
    lines.insert(insert_at, summary)
    return "\n".join(lines)


def state_to_text(state):
    """Render a compact, high-signal MicroRTS state description (~150-260 tokens on 8x8)."""
    player = state["player"]
    opp = 1 - player
    unit_types = state.get("unit_types") or {}
    header = f"MicroRTS t={state['time']} map={state['width']}x{state['height']} me=P{player}"
    if "opponent" in state:
        header += f" opp={state['opponent']}"
    lines = [header]

    players = {p["id"]: p for p in state.get("players", [])}
    if player in players and opp in players:
        me_p, opp_p = players[player], players[opp]
        lines.append(
            f"Res: me={me_p['resources']}(avail={me_p['available_resources']}) "
            f"opp={opp_p['resources']}(avail={opp_p['available_resources']})"
        )

    own = [u for u in state["units"] if u["owner"] == player]
    enemies = [u for u in state["units"] if u["owner"] == opp]
    resources = [u for u in state["units"] if u["type"] == "Resource" and u["resources"] > 0]

    phase = "early" if state["time"] < 150 else "mid" if state["time"] < 500 else "late"
    lines.append(f"Force: me[{_force_counts(own)}] vs opp[{_force_counts(enemies)}] phase={phase}")

    terrain = state.get("terrain", "")
    walls = [(i % state["width"], i // state["width"]) for i, v in enumerate(terrain) if int(v)]
    if walls:
        wall_str = " ".join(f"({x},{y})" for x, y in walls[:24])
        if len(walls) > 24:
            wall_str += f" (+{len(walls) - 24})"
        lines.append(f"Walls: {wall_str}")

    lines.append(f"My units ({len(own)}):")
    for u in own:
        lines.append("- " + _fmt_unit(u, unit_types))

    lines.append(f"Enemy units ({len(enemies)}):")
    for u in enemies:
        lines.append("- " + _fmt_unit(u, unit_types))

    if resources:
        res_str = ", ".join(f"R{r['id']}({r['x']},{r['y']})[{r['resources']}]" for r in resources)
        lines.append(f"Resources: {res_str}")

    return "\n".join(lines)


def candidate_text(candidate):
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def decision_prompt(state, candidates):
    options = "\n".join(f"{i}: {candidate_text(c)}" for i, c in enumerate(candidates))
    return state_to_text(state) + (
        "\nChoose one skill by its index. Output a probability distribution over these candidates. "
        "These are action-selection probabilities, not win probabilities.\n" + options
    )
