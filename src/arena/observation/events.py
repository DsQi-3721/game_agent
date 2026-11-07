"""Every battle-protocol message kind (sim/SIM-PROTOCOL.md, Showdown 1636bf6), what it does to a player's
knowledge, and who consumes it.

``ROLE`` -- what the message can do (several may apply):
  state       updates the current state (HP, status, boosts, field ...)
  reveal      shows information the player did not have (a move, item, ability, Pokemon, rule)
  invalidate  makes earlier information false (item lost, hazard removed, effect ended, status cured)
  evidence    a mechanism outcome that supports inference (miss, crit, immunity, hit count, failure)
  phase       starts or ends a choice phase / the battle
  history     part of the public record of what happened (who did what, in order)
  ignore      presentation, chat, timing or a format not in scope -- with the reason in ``WHY``

``DIGEST`` -- how ``state.battle_log.LogDigester`` (the turn narrative + key facts) uses it:
  line / attach (to the action line) / observation (consumed by ``arena.observation`` instead of the
  narrative: rules, preview, requests) / skip (``ignore`` only).
``scripts/audit_protocol_events.py`` checks that every kind seen in the strict battles is listed here.
"""

from __future__ import annotations

ROLE: dict[str, tuple[str, ...]] = {
    # battle start, format, rules, preview: observation only
    "player": ("reveal",), "teamsize": ("reveal",), "gametype": ("reveal",), "gen": ("reveal",),
    "tier": ("reveal",), "rated": ("reveal",), "rule": ("reveal",), "clearpoke": ("reveal", "phase"),
    "poke": ("reveal",), "teampreview": ("phase",), "start": ("phase",), "request": ("phase", "state", "reveal"),
    "turn": ("phase", "history"), "upkeep": ("phase", "history"), "win": ("phase",), "tie": ("phase",),
    # major actions
    "move": ("history", "reveal", "state"), "switch": ("history", "state", "reveal"),
    "drag": ("history", "state", "reveal"), "detailschange": ("state", "reveal"), "-formechange": ("state", "reveal"),
    "replace": ("reveal", "invalidate"),
    "swap": ("state",), "cant": ("history", "evidence"), "faint": ("state", "history", "invalidate"),
    # minor actions
    "-fail": ("evidence", "history"), "-block": ("evidence", "reveal"), "-notarget": ("evidence",),
    "-miss": ("evidence",), "-damage": ("state", "reveal"), "-heal": ("state", "reveal"), "-sethp": ("state",),
    "-status": ("state", "reveal"), "-curestatus": ("state", "invalidate"), "-cureteam": ("state", "invalidate"),
    "-boost": ("state",), "-unboost": ("state",), "-setboost": ("state",), "-swapboost": ("state",),
    "-invertboost": ("state",), "-clearboost": ("state", "invalidate"), "-clearallboost": ("state", "invalidate"),
    "-clearpositiveboost": ("state", "invalidate"), "-clearnegativeboost": ("state", "invalidate"),
    "-copyboost": ("state",), "-weather": ("state", "invalidate", "reveal"), "-fieldstart": ("state", "reveal"),
    "-fieldend": ("state", "invalidate"), "-sidestart": ("state",), "-sideend": ("state", "invalidate"),
    "-swapsideconditions": ("state",), "-start": ("state", "reveal"), "-end": ("state", "invalidate"),
    "-crit": ("evidence",), "-supereffective": ("evidence",), "-resisted": ("evidence",), "-immune": ("evidence", "reveal"),
    "-item": ("state", "reveal"), "-enditem": ("state", "invalidate", "reveal"), "-ability": ("reveal", "state"),
    "-endability": ("state", "invalidate"), "-transform": ("state", "reveal"), "-activate": ("evidence", "reveal", "state"),
    "-hint": ("evidence",), "-prepare": ("state", "reveal"), "-mustrecharge": ("state",), "-nothing": ("evidence",),
    "-hitcount": ("evidence",), "-singlemove": ("state",), "-singleturn": ("state",), "-terastallize": ("state", "reveal"),
    "-ohko": ("evidence",), "-center": ("state",), "-combine": ("evidence",), "-waiting": ("state",),
    # presentation / chat / timing / out of scope
    "": ("ignore",), "t:": ("ignore",), "-anim": ("ignore",), "-message": ("ignore",), "j": ("ignore",),
    "J": ("ignore",), "l": ("ignore",), "L": ("ignore",), "n": ("ignore",), "N": ("ignore",), "c": ("ignore",),
    "c:": ("ignore",), "chat": ("ignore",), "raw": ("ignore",), "html": ("ignore",), "uhtml": ("ignore",),
    "uhtmlchange": ("ignore",), "inactive": ("ignore",), "inactiveoff": ("ignore",), "init": ("ignore",),
    "title": ("ignore",), "badge": ("ignore",), "debug": ("ignore",), "bigerror": ("ignore",), "split": ("ignore",),
    "timestamp": ("ignore",), "-mega": ("ignore",), "-primal": ("ignore",), "-burst": ("ignore",),
    "-zpower": ("ignore",), "-zbroken": ("ignore",), "-candynamax": ("ignore",),
}

WHY: dict[str, str] = {
    "": "blank separator line", "t:": "wall-clock timestamp", "-anim": "client animation hint",
    "-message": "free-text message (the facts it reports also come as structured lines)",
    "j": "room join", "J": "room join", "l": "room leave", "L": "room leave", "n": "name change", "N": "name change",
    "c": "chat", "c:": "chat", "chat": "chat", "raw": "server HTML", "html": "server HTML", "uhtml": "server HTML",
    "uhtmlchange": "server HTML", "inactive": "timer warning", "inactiveoff": "timer off", "init": "room init",
    "title": "room title", "badge": "user badge", "debug": "simulator debug", "split": "channel split marker (the "
    "player's own private line follows it)", "timestamp": "timestamp", "bigerror": "server error text",
    "-mega": "Mega Evolution: not in Gen 9 OU / Random Battle", "-primal": "Primal Reversion: not in scope",
    "-burst": "Ultra Burst: not in scope", "-zpower": "Z-Moves: not in scope", "-zbroken": "Z-Moves: not in scope",
    "-candynamax": "Dynamax: not in scope",
}

# kinds the turn narrative leaves to the observation (they are not "noise": rules, preview and requests are
# read by arena.observation and reach every consumer from there)
OBSERVATION_ONLY = {"player", "teamsize", "gametype", "gen", "tier", "rated", "rule", "clearpoke", "poke",
                    "teampreview", "start", "request", "upkeep"}


def roles(kind: str) -> tuple[str, ...]:
    return ROLE.get(kind, ("unclassified",))


def is_ignorable(kind: str) -> bool:
    return roles(kind) == ("ignore",)
