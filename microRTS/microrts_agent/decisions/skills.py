"""Deduplicated tactical skills backed by MicroRTS abstract actions and BFS pathfinding."""

from collections import defaultdict, deque

import jpype

KINDS = ("wait", "move", "harvest", "return", "build", "produce", "attack")
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def skill_text(c, state=None):
    """Concise, human- and model-readable description of a candidate skill."""
    kind = c["skill"]
    if kind == "wait":
        return f"Wait {c['ticks']}t"
    units = {u["id"]: u for u in state["units"]} if state else {}
    actor = f"U{c['actor']}"
    if kind == "harvest":
        r = units.get(c["resource_id"])
        loc = f"({r['x']},{r['y']})" if r else ""
        return f"{actor}: harvest R{c['resource_id']}{loc} -> B{c['base_id']}"
    if kind == "return":
        return f"{actor}: return cargo -> B{c['base_id']}"
    if kind == "attack":
        if c.get("target_id") is None:
            return f"{actor}: attack nearest enemy"
        t = units.get(c["target_id"])
        desc = f"{t['type']} U{t['id']}({t['x']},{t['y']})" if t else f"U{c['target_id']}"
        return f"{actor}: attack {desc}"
    if kind == "move":
        tx, ty = c["target"]
        return f"{actor}: move to ({tx},{ty})"
    tx, ty = c["target"]
    return f"{actor}: {kind} {c['unit_type']} at ({tx},{ty})"


class Skills:
    def __init__(self, utt):
        self.utt = utt
        self.pathfinder = jpype.JClass("ai.abstraction.pathfinding.BFSPathFinding")()
        self.Move = jpype.JClass("ai.abstraction.Move")
        self.Attack = jpype.JClass("ai.abstraction.Attack")
        self.Harvest = jpype.JClass("ai.abstraction.Harvest")
        self.Build = jpype.JClass("ai.abstraction.Build")

    def next_action(self, gs, c, usage=None):
        """Return a complete, engine-legal UnitAction, or None if blocked."""
        u = gs.getUnit(c["actor"])
        if u is None or gs.getActionAssignment(u) is not None:
            return None
        kind = c["skill"]
        pf = self.pathfinder
        if kind == "move":
            plan = self.Move(u, *c["target"], pf)
        elif kind == "attack":
            target_id = c.get("target_id")
            if target_id is None:
                # Dynamic nearest-enemy targeting (auto-switches to closest live threat).
                owner = int(u.getPlayer())
                ux, uy = int(u.getX()), int(u.getY())
                best, best_d = None, 10**9
                for other in gs.getPhysicalGameState().getUnits():
                    if int(other.getPlayer()) == 1 - owner:
                        d = abs(int(other.getX()) - ux) + abs(int(other.getY()) - uy)
                        if d < best_d:
                            best, best_d = other, d
                target = best
            else:
                target = gs.getUnit(target_id)
            if target is None:
                return None
            plan = self.Attack(u, target, pf)
        elif kind in ("harvest", "return"):
            base = gs.getUnit(c["base_id"])
            resource = gs.getUnit(c["resource_id"]) if kind == "harvest" else None
            if base is None or (int(u.getResources()) == 0 and resource is None):
                return None
            plan = self.Harvest(u, resource, base, pf)
        else:
            plan = self.Build(u, self.utt.getUnitType(c["unit_type"]), *c["target"], pf)
        ru = usage if usage is not None else gs.getResourceUsage()
        action = plan.execute(gs, ru)
        if (
            action is not None
            and u.getUnitActions(gs).contains(action)
            and gs.isUnitActionAllowed(u, action)
        ):
            if usage is not None:
                action_usage = action.resourceUsage(u, gs.getPhysicalGameState())
                if not ru.consistentWith(action_usage, gs):
                    return None
            return action
        return None

    def unit_candidates(self, state, gs, u, limit=12, wait_ticks=None):
        """Compact, deduplicated tactical skill shortlist for a single idle unit."""
        player = state["player"]
        own = [x for x in state["units"] if x["owner"] == player]
        enemies = [x for x in state["units"] if x["owner"] == 1 - player]
        resources = [x for x in state["units"] if x["type"] == "Resource" and x["resources"] > 0]
        bases = [x for x in own if state["unit_types"][x["type"]]["isStockpile"]]
        own_barracks = [x for x in own if x["type"] == "Barracks"]
        money = next(p["available_resources"] for p in state["players"] if p["id"] == player)
        occupied = {(x["x"], x["y"]) for x in state["units"]}
        width, height = state["width"], state["height"]
        terrain = state["terrain"]
        center = (width // 2, height // 2)

        def free(x, y):
            return (
                0 <= x < width
                and 0 <= y < height
                and (x, y) not in occupied
                and not int(terrain[y * width + x])
            )

        out = []

        def try_add(c):
            if len(out) < limit - 1 and self.next_action(gs, c) is not None:
                out.append(c)
                return True
            return False

        t = state["unit_types"][u["type"]]
        actor = u["id"]
        ux, uy = u["x"], u["y"]
        if wait_ticks is None:
            # Buildings saving gold check every 16 ticks; mobile units keep crisp 4-tick micro.
            wait_ticks = 4 if t["canMove"] else 16

        # 1. Harvest / return (nearest base + closest safe resource to that base)
        if t["canHarvest"] and bases:
            nearest_base = min(bases, key=lambda b: _manhattan(ux, uy, b["x"], b["y"]))
            if u["resources"] > 0:
                try_add({"skill": "return", "actor": actor, "base_id": nearest_base["id"]})
            elif resources:
                nearest_res = min(
                    resources,
                    key=lambda r: _manhattan(nearest_base["x"], nearest_base["y"], r["x"], r["y"])
                    * 2
                    + _manhattan(ux, uy, r["x"], r["y"]),
                )
                try_add(
                    {
                        "skill": "harvest",
                        "actor": actor,
                        "resource_id": nearest_res["id"],
                        "base_id": nearest_base["id"],
                    }
                )

        # 2. Produce / build (one best valid cell per affordable unit type)
        anchor = center
        if enemies:
            nearest_enemy = min(enemies, key=lambda e: _manhattan(ux, uy, e["x"], e["y"]))
            anchor = (nearest_enemy["x"], nearest_enemy["y"])
        elif resources:
            nearest_res = min(resources, key=lambda r: _manhattan(ux, uy, r["x"], r["y"]))
            anchor = (nearest_res["x"], nearest_res["y"])

        for name in t["produces"]:
            product = state["unit_types"][name]
            if product["cost"] > money:
                continue
            kind = "produce" if product["canMove"] else "build"
            if kind == "build" and name == "Barracks" and len(own_barracks) >= 2:
                continue
            if kind == "build" and name == "Base" and len(bases) >= 2:
                continue
            # Prefer adjacent cells for produce; up to distance 2 for buildings.
            cells = []
            max_dist = 3 if kind == "build" else 2
            for d in range(1, max_dist):
                for dx, dy in DIRECTIONS:
                    x, y = ux + d * dx, uy + d * dy
                    if free(x, y):
                        cells.append((x, y))
            # Prefer immediately adjacent cells first (no wasted walk ticks), then closest to anchor.
            cells.sort(
                key=lambda xy: (
                    _manhattan(ux, uy, xy[0], xy[1]),
                    _manhattan(xy[0], xy[1], anchor[0], anchor[1]),
                )
            )
            for x, y in cells:
                if try_add({"skill": kind, "actor": actor, "unit_type": name, "target": [x, y]}):
                    break

        # 3. Attack (dynamic nearest-enemy engagement + specific enemy base/building rush)
        if t["canAttack"] and enemies:
            try_add({"skill": "attack", "actor": actor, "target_id": None})
            buildings = [e for e in enemies if e["type"] in ("Base", "Barracks")]
            if buildings:
                eb = min(buildings, key=lambda x: (_manhattan(ux, uy, x["x"], x["y"]), x["hp"]))
                try_add({"skill": "attack", "actor": actor, "target_id": eb["id"]})

        # 4. Tactical moves (at most 2: advance to center/mid, retreat toward own base)
        if t["canMove"]:
            move_goals = []
            if free(*center) and (ux, uy) != center:
                move_goals.append(center)
            if bases:
                bx, by = bases[0]["x"], bases[0]["y"]
                if _manhattan(ux, uy, bx, by) > 2:
                    for dx, dy in DIRECTIONS:
                        rx, ry = bx + dx, by + dy
                        if free(rx, ry):
                            move_goals.append((rx, ry))
                            break
            if not move_goals:
                for dx, dy in DIRECTIONS:
                    nx, ny = ux + dx, uy + dy
                    if free(nx, ny):
                        move_goals.append((nx, ny))
                        break
            for mx, my in dict.fromkeys(move_goals):
                if len(out) >= limit - 1:
                    break
                try_add({"skill": "move", "actor": actor, "target": [mx, my]})

        out.append({"skill": "wait", "actor": actor, "ticks": wait_ticks})
        return out

    def candidates(self, state, gs, limit=26):
        """Round-robin global shortlist across all idle own units (legacy single-decision mode)."""
        player = state["player"]
        own = [u for u in state["units"] if u["owner"] == player and u["action"] is None]
        groups = defaultdict(deque)
        for u in own:
            for c in self.unit_candidates(state, gs, u, limit=limit):
                if c["skill"] != "wait":
                    groups[u["id"], c["skill"]].append(c)
        result = []
        queues = deque(groups.values())
        while queues and len(result) < limit - 1:
            queue = queues.popleft()
            result.append(queue.popleft())
            if queue:
                queues.append(queue)
        return result + [{"skill": "wait", "ticks": 8}]

    def match_bot_skill(
        self, gs, unit, bot_action, candidates, abstract_action=None, return_reason=False
    ):
        """Map a Java expert bot's abstract/primitive action to the closest candidate index."""

        def _ret(idx, reason):
            return (idx, reason) if return_reason else idx

        # 0. If the bot exposes its AbstractAction (AbstractionLayerAI), match directly by intent.
        if abstract_action is not None:
            cls_name = str(abstract_action.getClass().getSimpleName())
            if cls_name == "Harvest":
                kind = "return" if int(unit.getResources()) > 0 else "harvest"
                for i, c in enumerate(candidates):
                    if c["skill"] == kind:
                        return _ret(i, "abstract")
            elif cls_name == "Attack":
                tgt = getattr(abstract_action, "target", None)
                tgt_id = int(tgt.getID()) if tgt is not None else None
                for i, c in enumerate(candidates):
                    if c["skill"] == "attack" and c.get("target_id") == tgt_id:
                        return _ret(i, "abstract")
                for i, c in enumerate(candidates):
                    if c["skill"] == "attack":
                        return _ret(i, "abstract")
            elif cls_name in ("Train", "Build"):
                ut = getattr(abstract_action, "type", None)
                ut_name = str(ut.name) if ut is not None else ""
                for i, c in enumerate(candidates):
                    if c["skill"] in ("produce", "build") and c.get("unit_type") == ut_name:
                        return _ret(i, "abstract")
            elif cls_name == "Move" and hasattr(abstract_action, "x"):
                mx, my = int(abstract_action.x), int(abstract_action.y)
                moves = [
                    (_manhattan(mx, my, c["target"][0], c["target"][1]), i)
                    for i, c in enumerate(candidates)
                    if c["skill"] == "move"
                ]
                if moves:
                    return _ret(min(moves)[1], "abstract")

        wait_idx = next(
            (i for i, c in enumerate(candidates) if c["skill"] == "wait"),
            len(candidates) - 1,
        )
        if bot_action is None or int(bot_action.getType()) == 0:
            return _ret(wait_idx, "wait_native")

        b_type = int(bot_action.getType())
        b_dir = int(bot_action.getDirection())
        b_loc = (int(bot_action.getLocationX()), int(bot_action.getLocationY()))
        b_utype = str(bot_action.getUnitType().name) if bot_action.getUnitType() is not None else ""

        # Precompute each candidate's immediate primitive action.
        cand_actions = []
        for i, c in enumerate(candidates):
            if c["skill"] == "wait":
                cand_actions.append((i, c, None))
            else:
                cand_actions.append((i, c, self.next_action(gs, c)))

        # 1. Exact primitive match (prefer tactical skills over generic move).
        priority = {"produce": 0, "build": 0, "return": 0, "attack": 1, "harvest": 1, "move": 2}
        exact = []
        for i, c, act in cand_actions:
            if act is None or int(act.getType()) != b_type:
                continue
            if b_type in (1, 2, 3) and int(act.getDirection()) == b_dir:
                exact.append((priority.get(c["skill"], 3), i))
            elif b_type == 4 and str(act.getUnitType().name) == b_utype:
                dir_bonus = 0 if int(act.getDirection()) == b_dir else 1
                exact.append((dir_bonus, i))
            elif b_type == 5 and (int(act.getLocationX()), int(act.getLocationY())) == b_loc:
                exact.append((0, i))
        if exact:
            exact.sort()
            return _ret(exact[0][1], "exact")

        # 2. Semantic fallback by action type.
        if b_type == 4:
            for i, c, _ in cand_actions:
                if c["skill"] in ("produce", "build") and c.get("unit_type") == b_utype:
                    return _ret(i, "semantic")
        elif b_type == 2:
            for i, c, _ in cand_actions:
                if c["skill"] == "harvest":
                    return _ret(i, "semantic")
        elif b_type == 3:
            for i, c, _ in cand_actions:
                if c["skill"] == "return":
                    return _ret(i, "semantic")
        elif b_type == 5:
            for i, c, _ in cand_actions:
                if c["skill"] == "attack":
                    return _ret(i, "semantic")
        elif b_type == 1:
            # Bot is moving in direction b_dir: pick candidate whose target lies best along b_dir.
            dx, dy = DIRECTIONS[b_dir] if 0 <= b_dir < 4 else (0, 0)
            ux, uy = int(unit.getX()), int(unit.getY())
            nx, ny = ux + dx, uy + dy
            best = None
            for i, c, _act in cand_actions:
                if c["skill"] == "wait":
                    continue
                if "target" in c:
                    tx, ty = c["target"]
                elif c["skill"] == "attack":
                    tid = c.get("target_id")
                    if tid is None:
                        owner = int(unit.getPlayer())
                        enemies_j = [
                            x for x in gs.getPhysicalGameState().getUnits()
                            if int(x.getPlayer()) == 1 - owner
                        ]
                        if not enemies_j:
                            continue
                        tu = min(
                            enemies_j,
                            key=lambda x: _manhattan(ux, uy, int(x.getX()), int(x.getY())),
                        )
                    else:
                        tu = gs.getUnit(tid)
                        if tu is None:
                            continue
                    tx, ty = int(tu.getX()), int(tu.getY())
                elif "resource_id" in c:
                    ru = gs.getUnit(c["resource_id"])
                    if ru is None:
                        continue
                    tx, ty = int(ru.getX()), int(ru.getY())
                elif "base_id" in c:
                    bu = gs.getUnit(c["base_id"])
                    if bu is None:
                        continue
                    tx, ty = int(bu.getX()), int(bu.getY())
                else:
                    continue
                gain = _manhattan(ux, uy, tx, ty) - _manhattan(nx, ny, tx, ty)
                score = (gain, -priority.get(c["skill"], 3))
                if best is None or score > best[0]:
                    best = (score, i)
            if best is not None and best[0][0] > 0:
                return _ret(best[1], "semantic")

        return _ret(wait_idx, "wait_fallback")


def action_row(unit, action, width, attack_diameter):
    """JNI layout: [cell, type, move, harvest, return, prod_dir, prod_type, attack]."""
    kind = int(action.getType())
    row = [int(unit.getY()) * width + int(unit.getX()), kind, 0, 0, 0, 0, 0, 0]
    if kind in (1, 2, 3):
        row[kind + 1] = int(action.getDirection())
    elif kind == 4:
        row[5:7] = [int(action.getDirection()), int(action.getUnitType().ID)]
    elif kind == 5:
        center = attack_diameter // 2
        row[7] = (
            (int(action.getLocationY()) - int(unit.getY()) + center) * attack_diameter
            + int(action.getLocationX())
            - int(unit.getX())
            + center
        )
    return row
