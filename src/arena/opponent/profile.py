"""Team-preview profile of the opponent: what kind of team it is and how it wants to win.

From the six species alone (team preview) each Pokemon gets its belief (``arena.opponent.posterior``, teammates
included), and the team-level reading is computed from those probabilities:

- tags with probabilities: rain / sun / sand / snow (setter abilities and moves, Smogon's classifier rules),
  Trick Room, dual screens / Aurora Veil, Sticky Web, hazard stack, hazard removal, VoltTurn (>= 3 pivots);
- playstyle: hyper offense / offense / balance / stall from the expected "stalliness" of the six sets (Smogon's
  formula, pkmn/stats ``classifier.ts``) with thresholds re-fitted to Smogon's current sample teams (the original
  ones date from older generations and put most current teams into stall; see ``STYLE_CUTS``);
- the roles behind the tags (who sets hazards, who removes them, who pivots), likely leads (Smogon lead rates),
  win conditions (setup sweepers, Booster Energy / Choice attackers, priority);
- with our team known: the opponent's Pokemon that threaten most of our team, and our best answer to each;
- short counterplay notes per tag (``TIPS``, written from Smogon's SV OU resources: sample-team notes, Good Cores,
  Role Compendium; see ``reports/opponent_modeling_resources.md`` appendix C).

``profile_for(battle)`` caches one profile per battle (team preview does not change) and ``render`` gives the
text the LLM brief carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from poke_env.data import to_id_str

from arena.knowledge import store
from arena.opponent.kb import kb_for, resolve_species
from arena.opponent.posterior import DEFAULT_PARAMS, Belief, Params, belief

LOG3 = math.log2(3)

# --- Smogon classifier.ts tables (gen 9 subset) -----------------------------------------------------------------
GREATER_OFF_AB = {"purepower", "hugepower", "speedboost", "moody"}
LESSER_OFF_AB = {"chlorophyll", "download", "hustle", "moxie", "reckless", "sandrush", "solarpower", "swiftswim",
                 "technician", "tintedlens", "darkaura", "fairyaura", "infiltrator", "parentalbond", "protean",
                 "strongjaw", "sweetveil", "toughclaws", "aerilate", "normalize", "pixilate", "refrigerate"}
LESSER_DEF_AB = {"dryskin", "filter", "hydration", "icebody", "intimidate", "ironbarbs", "marvelscale", "naturalcure",
                 "magicguard", "multiscale", "raindish", "roughskin", "solidrock", "thickfat", "unaware", "aromaveil",
                 "bulletproof", "cheekpouch", "gooey", "regenerator"}
GREATER_DEF_AB = {"slowstart", "truant", "furcoat", "harvest"}
LESSER_BOOST_ITEMS = {"expertbelt", "wiseglasses", "muscleband", "blackglasses", "charcoal", "dragonfang", "hardstone",
                      "magnet", "metalcoat", "miracleseed", "mysticwater", "nevermeltice", "poisonbarb", "sharpbeak",
                      "silkscarf", "silverpowder", "softsand", "spelltag", "twistedspoon", "fairyfeather"}
GREATER_BOOST_ITEMS = {"focussash", "mentalherb", "powerherb", "whiteherb", "absorbbulb", "cellbattery", "airballoon",
                       "ejectbutton", "shedshell", "custapberry", "salacberry", "liechiberry", "petayaberry",
                       "sitrusberry", "lumberry", "choiceband", "choicescarf", "choicespecs", "lifeorb", "snowball",
                       "boosterenergy", "throatspray", "loadeddice", "ejectpack", "mirrorherb", "covertcloak"}
RECOVERY = {"recover", "slackoff", "roost", "moonlight", "morningsun", "synthesis", "softboiled", "milkdrink",
            "shoreup", "wish", "rest", "junglehealing", "lifedew", "lunarblessing", "healorder", "swallow",
            "leechseed", "aquaring", "strengthsap"}
PROTECT = {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker", "obstruct", "silktrap",
           "burningbulwark"}
PHAZING = {"whirlwind", "roar", "dragontail", "circlethrow"}
PARALYSIS = {"thunderwave", "stunspore", "glare", "nuzzle"}
CONFUSION = {"confuseray", "supersonic", "swagger", "flatter", "teeterdance"}
LESSER_OFF_MOVES = {"flareblitz", "bravebird", "wavecrash", "headsmash", "woodhammer", "volttackle", "doubleedge",
                    "wildcharge", "takedown", "highjumpkick", "jumpkick", "axekick", "supercellslam", "outrage",
                    "petaldance", "thrash", "ragingfury", "closecombat", "headlongrush", "dragonascent", "armorcannon",
                    "chloroblast", "mindblown", "steelbeam", "submission"}
GREATER_OFF_MOVES = {"selfdestruct", "explosion", "mistyexplosion", "destinybond", "perishsong", "memento",
                     "healingwish", "lunardance", "finalgambit"}
GREATER_SETUP = {"dragondance", "swordsdance", "nastyplot", "quiverdance", "shiftgear", "tailglow", "geomancy",
                 "growth", "curse", "tidyup", "victorydance", "filletaway", "clangoroussoul", "noretreat", "torchsong"}
LESSER_SETUP = {"bulkup", "calmmind", "coil", "howl", "workup", "honeclaws", "agility", "rockpolish", "autotomize",
                "flamecharge", "trailblaze", "aquastep", "chargebeam", "poweruppunch", "meditate", "sharpen",
                "irondefense", "cosmicpower", "takeheart", "acidarmor", "amnesia"}
SETUP = GREATER_SETUP | LESSER_SETUP | {"bellydrum", "shellsmash"}

WEATHER = {  # tag: (setter abilities, setter move, its rock)
    "rain": ({"drizzle"}, "raindance", "damprock"),
    "sun": ({"drought", "orichalcumpulse"}, "sunnyday", "heatrock"),
    "sand": ({"sandstream"}, "sandstorm", "smoothrock"),
    "snow": ({"snowwarning"}, "snowscape", "icyrock"),
}
HAZARD_MOVES = {"stealthrock": "Stealth Rock", "spikes": "Spikes", "toxicspikes": "Toxic Spikes",
                "stickyweb": "Sticky Web", "stoneaxe": "Stealth Rock (Stone Axe)", "ceaselessedge": "Spikes (Ceaseless Edge)"}
REMOVAL = {"defog": "Defog", "rapidspin": "Rapid Spin", "mortalspin": "Mortal Spin", "tidyup": "Tidy Up",
           "courtchange": "Court Change"}
PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "teleport", "shedtail"}
PRIORITY = {"extremespeed", "suckerpunch", "aquajet", "bulletpunch", "iceshard", "machpunch", "vacuumwave",
            "shadowsneak", "jetpunch", "accelerock", "grassyglide", "firstimpression", "quickattack", "thunderclap",
            "upperhand", "fakeout"}
SCREENS = {"reflect", "lightscreen", "auroraveil"}

# playstyle cut points on the mean expected stalliness at team preview, fitted on Smogon's 21 SV OU sample teams
# (6 offense, 6 bulky offense, 6 balance, 3 stall): 18/21 right, 14/21 leave-one-out; the same formula on the
# teams' real sets separates all 21 (``scripts/eval_opponent_model.py archetype``, reports/opponent_modeling.md)
STYLE_CUTS = (("offense", -0.17), ("bulky offense", 0.18), ("balance", 2.33), ("stall", 99.0))

TIPS = {
    "rain": "Rain: its setter (Pelipper) brings rain on entry; Swift Swim users double their Speed and Water moves hit "
            "1.5x. Pressure the setter (hazards, Knock Off), keep a weather changer or bulky Water/Grass/Dragon answers.",
    "sun": "Sun: Drought setter; Protosynthesis/Chlorophyll users get faster or stronger and Fire moves hit 1.5x. "
           "Change the weather or wear the setter down with Stealth Rock; bulky Water/Dragon/Fire answers.",
    "sand": "Sand: Tyranitar/Hippowdon; Sand Rush users double their Speed. Fighting/Water/Ground moves, change the "
            "weather.",
    "snow": "Snow: Ice types gain 1.5x Defense, Slush Rush users double their Speed, Aurora Veil becomes available.",
    "trickroom": "Trick Room: slow attackers move first for 5 turns. KO or Taunt the setter, or stall the turns out "
                 "(Protect, switching).",
    "screens": "Screens: setup behind Reflect / Light Screen (5 turns, 8 with Light Clay). Remove the setter early, "
               "use Defog / Brick Break / Psychic Fangs, or stall the turns out before trading.",
    "webs": "Sticky Web: -1 Speed on every grounded switch-in. Remove it (Defog, Rapid Spin, Mortal Spin, Tidy Up), "
            "use Magic Bounce, or rely on bulk and priority.",
    "hazardstack": "Hazard stack: Spikes + Stealth Rock wear down every switch. Keep a remover healthy, Heavy-Duty "
                   "Boots on switch-heavy Pokemon, and avoid needless switching.",
    "voltturn": "VoltTurn: many pivots keep momentum. Punish with hazards, Rocky Helmet / Rough Skin, and switch-ins "
                "that resist the pivot move.",
    "offense": "Offense: hazard or screen lead, then fast, strong attackers. Preserve priority and speed control, "
               "do not give a setup sweeper free turns, trade efficiently.",
    "bulky offense": "Bulky offense: sturdy attackers and pivots. Look for wallbreakers and long-term damage.",
    "balance": "Balance: defensive core plus a few win conditions. Hazards, status and wallbreakers wear the core down; "
               "identify its win condition early.",
    "stall": "Stall: walls with recovery, status and hazard removal. Wallbreakers, Taunt, Knock Off and setup moves; "
             "avoid feeding status, keep up pressure.",
}


def _any(q: dict, moves) -> float:
    p = 1.0
    for m in moves:
        p *= 1.0 - min(1.0, q.get(m, 0.0))
    return 1.0 - p


def _at_least(ps: list[float], k: int) -> float:
    """P(at least k successes) of independent Bernoulli(p_i) (Poisson binomial)."""
    dp = [1.0] + [0.0] * len(ps)
    for p in ps:
        for j in range(len(ps), 0, -1):
            dp[j] = dp[j] * (1 - p) + dp[j - 1] * p
        dp[0] *= 1 - p
    return sum(dp[k:])


def _stats(sid: str, spread) -> Optional[dict]:
    from poke_env.data import GenData
    from poke_env.stats import compute_raw_stats
    data = GenData.from_gen(9)
    if sid not in data.pokedex:
        return None
    nature, evs = (spread[0], spread[1]) if spread else ("serious", [84] * 6)
    if nature not in data.natures:
        nature = "serious"
    vals = compute_raw_stats(sid, list(evs), [31] * 6, 100, nature, data)
    return dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), vals))


def stalliness(b: Belief) -> float:
    """Expected Smogon stalliness of one Pokemon under its belief (classifier.ts ``classifyForme``)."""
    st = _stats(b.species, b.spreads[0] if b.spreads else None)
    if st is None:
        return 0.0
    base = -math.log2(((0.84 * max(st["atk"], st["spa"]) / max(st["def"], st["spd"])) * 120 + 2) * 0.925 / st["hp"])
    ab = sum(p * (-1.0 if a in GREATER_OFF_AB else -0.5 if a in LESSER_OFF_AB else 0.5 if a in LESSER_DEF_AB
                  else 1.0 if a in GREATER_DEF_AB else 0.0) for a, p in b.abilities.items())
    it = 0.0
    for i, p in b.items.items():
        if i in ("weaknesspolicy", "lightclay"):
            v = -1.0
        elif i in ("rockyhelmet", "eviolite"):
            v = 0.5
        elif i == "toxicorb":
            v = 0.5 * b.abilities.get("poisonheal", 0.0) - 1.0 * sum(b.abilities.get(a, 0.0) for a in
                                                                      ("toxicboost", "guts", "quickfeet"))
        elif i in LESSER_BOOST_ITEMS:
            v = -0.25
        elif i in GREATER_BOOST_ITEMS:
            v = -0.5
        else:
            v = 0.0
        it += p * v
    q = b.moves
    mv = sum(q.get(m, 0.0) * w for m, w in (("toxic", 1.0), ("spikes", 0.5), ("toxicspikes", 0.5),
                                              ("willowisp", 0.5), ("haze", 0.5), ("clearsmog", 0.5),
                                              ("substitute", -0.5), ("superfang", -0.5), ("trick", -0.5),
                                              ("endeavor", -1.0)))
    mv += 1.0 * _any(q, RECOVERY) + 1.0 * _any(q, PROTECT) + 0.5 * _any(q, PHAZING) + 0.5 * _any(q, PARALYSIS)
    mv += 0.5 * _any(q, CONFUSION) - 0.5 * _any(q, LESSER_OFF_MOVES) - 1.0 * _any(q, GREATER_OFF_MOVES)
    p_bd, p_ss = q.get("bellydrum", 0.0), q.get("shellsmash", 0.0)
    p_gs, p_ls = _any(q, GREATER_SETUP), _any(q, LESSER_SETUP)
    rest = max(0.0, 1 - p_bd - p_ss)
    mv += -2.0 * p_bd - 1.5 * p_ss - rest * (1.0 * p_gs + 0.5 * (1 - p_gs) * p_ls)
    extra = 0.5 * (b.abilities.get("sandstream", 0.0) + b.abilities.get("snowwarning", 0.0))
    return base + ab + it + mv + extra - LOG3


@dataclass
class TeamProfile:
    species: list                                   # canonical ids, preview order
    names: dict                                     # id -> display name
    tags: dict = field(default_factory=dict)        # tag -> probability
    style: str = ""
    stalliness: float = 0.0
    roles: dict = field(default_factory=dict)       # role -> [(id, p, what)]
    leads: list = field(default_factory=list)       # [(id, p)]
    wincons: list = field(default_factory=list)     # [(id, why)]
    threats: list = field(default_factory=list)     # [text]
    beliefs: dict = field(default_factory=dict)     # id -> Belief at preview

    def top_tags(self, threshold: float = 0.5) -> list[str]:
        return [t for t, p in sorted(self.tags.items(), key=lambda kv: -kv[1]) if p >= threshold]


def style_of(score: float) -> str:
    return next(name for name, cut in STYLE_CUTS if score < cut)


def build_profile(kb: dict, species: list[str], params: Params = DEFAULT_PARAMS) -> Optional[TeamProfile]:
    sids = [s for s in (resolve_species(kb, x) for x in species) if s]
    if len(sids) < 2:
        return None
    beliefs = {s: belief(kb, s, [m for m in sids if m != s], {}, params=params) for s in sids}
    beliefs = {s: b for s, b in beliefs.items() if b is not None}
    names = {s: b.name for s, b in beliefs.items()}
    prof = TeamProfile(species=list(beliefs), names=names, beliefs=beliefs)
    tags, roles = {}, {}

    def role(name, sid, p, what):
        if p >= 0.15:
            roles.setdefault(name, []).append((sid, round(p, 2), what))

    for tag, (abil, move, rock) in WEATHER.items():
        ps = []
        for s, b in beliefs.items():
            pa = sum(b.abilities.get(a, 0.0) for a in abil)
            pm = b.moves.get(move, 0.0)
            pr = b.items.get(rock, 0.0)
            p = min(1.0, pa + (1 - pa) * pm * (pr + 0.5 * (1 - pr)))
            ps.append(p)
            role(f"{tag} setter", s, p, "ability" if pa >= pm else move)
        tags[tag] = round(1 - math.prod(1 - p for p in ps), 3)
    tr = [b.moves.get("trickroom", 0.0) for b in beliefs.values()]
    slow = [1.0 if (_stats(s, b.spreads[0] if b.spreads else None) or {}).get("spe", 999) <= 180 else 0.0
            for s, b in beliefs.items()]
    tags["trickroom"] = round(max(_at_least(tr, 3), _at_least(tr, 2) * (1.0 if sum(slow) >= 2 else 0.3)), 3)
    for s, b in beliefs.items():
        role("Trick Room setter", s, b.moves.get("trickroom", 0.0), "Trick Room")
        dual = min(b.moves.get("reflect", 0.0), b.moves.get("lightscreen", 0.0))
        veil = b.moves.get("auroraveil", 0.0)
        role("screens", s, max(dual, veil), "Aurora Veil" if veil > dual else "Reflect + Light Screen")
        role("Sticky Web", s, b.moves.get("stickyweb", 0.0), "Sticky Web")
        for m, label in HAZARD_MOVES.items():
            if m != "stickyweb":
                role("hazards", s, b.moves.get(m, 0.0), label)
        for m, label in REMOVAL.items():
            role("hazard removal", s, b.moves.get(m, 0.0), label)
        piv = _any(b.moves, PIVOTS) + (1 - _any(b.moves, PIVOTS)) * b.items.get("ejectbutton", 0.0)
        role("pivots", s, piv, "/".join(sorted(m for m in PIVOTS if b.moves.get(m, 0.0) >= 0.2)) or "Eject Button")
        pri = [m for m in PRIORITY if b.moves.get(m, 0.0) >= 0.3]
        if pri:
            role("priority", s, max(b.moves[m] for m in pri), "/".join(sorted(pri)))
        stp = [m for m in SETUP if b.moves.get(m, 0.0) >= 0.3]
        if stp:
            role("setup", s, max(b.moves[m] for m in stp), "/".join(sorted(stp)))
    scr = [max(min(b.moves.get("reflect", 0.0), b.moves.get("lightscreen", 0.0)), b.moves.get("auroraveil", 0.0))
           for b in beliefs.values()]
    tags["screens"] = round(1 - math.prod(1 - p for p in scr), 3)
    tags["webs"] = round(1 - math.prod(1 - b.moves.get("stickyweb", 0.0) for b in beliefs.values()), 3)
    rocks = 1 - math.prod(1 - max(b.moves.get("stealthrock", 0.0), b.moves.get("stoneaxe", 0.0))
                          for b in beliefs.values())
    spikes = 1 - math.prod(1 - max(b.moves.get("spikes", 0.0), b.moves.get("ceaselessedge", 0.0),
                                   b.moves.get("toxicspikes", 0.0)) for b in beliefs.values())
    tags["hazardstack"] = round(rocks * spikes, 3)
    tags["removal"] = round(1 - math.prod(1 - _any(b.moves, REMOVAL) for b in beliefs.values()), 3)
    piv = [_any(b.moves, PIVOTS) for b in beliefs.values()]
    tags["voltturn"] = round(_at_least(piv, 3), 3)
    prof.tags, prof.roles = tags, roles
    st = [stalliness(b) for b in beliefs.values()]
    prof.stalliness = round(sum(st) / len(st), 3)
    prof.style = style_of(prof.stalliness)
    leads = {s: max((kb["species"][s].get("lead") or 0.0), 1e-3) for s in beliefs}
    z = sum(leads.values())
    prof.leads = sorted(((s, round(v / z, 3)) for s, v in leads.items()), key=lambda kv: -kv[1])[:3]
    for s, b in beliefs.items():
        why = []
        stp = [m for m in SETUP if b.moves.get(m, 0.0) >= 0.4]
        if stp:
            why.append("sets up with " + "/".join(_move_name(m) for m in sorted(stp)))
        if b.items.get("boosterenergy", 0.0) >= 0.4:
            why.append("Booster Energy")
        ch = max((b.items.get(c, 0.0), c) for c in ("choiceband", "choicespecs", "choicescarf"))
        if ch[0] >= 0.4:
            why.append(_item_name(ch[1]))
        if b.abilities.get("supremeoverlord", 0.0) >= 0.5:
            why.append("late-game cleaner (Supreme Overlord)")
        if why:
            prof.wincons.append((s, ", ".join(why)))
    return prof


def _move_name(m: str) -> str:
    return ((store.dex().get("moves") or {}).get(m) or {}).get("name", m)


def _item_name(i: str) -> str:
    return ((store.dex().get("items") or {}).get(i) or {}).get("name", i)


def threats(battle, prof: TeamProfile, top: int = 3) -> list[str]:
    """Opponent Pokemon that hit most of our team hard, and our best answer to each (estimated calcs)."""
    from arena.agents.team_preview import _best_damage, _previewed
    from arena.augment.speed import speed_band
    ours = list(battle.team.values())
    theirs = list(getattr(battle, "teampreview_opponent_team", None) or [])
    if not ours or not theirs:
        return []
    rows = []
    with _previewed(battle, theirs):
        for o in theirs:
            hits = [(m, _best_damage(battle, o, m), _best_damage(battle, m, o)) for m in ours]
            spe_o = speed_band(battle, o)[1]
            faster = sum(1 for m in ours if spe_o is not None and (speed_band(battle, m)[1] or 0) < spe_o)
            heavy = sum(1 for _m, d_in, _d_out in hits if d_in >= 50)
            answer = max(hits, key=lambda h: h[2] - h[1])
            rows.append((heavy, faster, o, answer))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    kb = kb_for(getattr(battle, "format", "") or "")
    name = lambda sp: ((kb.get("species") or {}).get(resolve_species(kb, sp) or "", {}) or {}).get("name", sp)  # noqa: E731
    out = []
    for heavy, faster, o, (m, d_in, d_out) in rows[:top]:
        if heavy == 0:
            continue
        out.append(f"{name(o.species)} hits {heavy} of yours for 50%+ and outspeeds {faster}; best answer: your "
                   f"{name(m.species)} (takes ~{d_in:.0f}%, deals ~{d_out:.0f}%)")
    return out


def render(prof: TeamProfile, max_roles: int = 3) -> str:
    n = prof.names
    tags = [t for t in ("rain", "sun", "sand", "snow", "trickroom", "screens", "webs", "hazardstack", "voltturn")
            if prof.tags.get(t, 0.0) >= 0.5]
    lines = [f"Opponent profile (from team preview; usage stats + teammates): {prof.style} "
             f"(stalliness {prof.stalliness:+.2f})" + (f"; {', '.join(f'{t} {prof.tags[t]:.0%}' for t in tags)}"
                                                         if tags else "; no weather / Trick Room / screens / webs")]
    if prof.leads:
        lines.append("Likely leads: " + ", ".join(f"{n[s]} {p:.0%}" for s, p in prof.leads))
    for r in ("hazards", "hazard removal", "pivots", "screens", "Trick Room setter", "priority"):
        by_mon: dict = {}
        for s, p, w in sorted(prof.roles.get(r) or [], key=lambda x: -x[1]):
            by_mon.setdefault(s, []).append(f"{w} {p:.0%}")
        if by_mon:
            lines.append(f"{r[0].upper() + r[1:]}: " + "; ".join(f"{n[s]} ({', '.join(ws)})"
                                                                  for s, ws in list(by_mon.items())[:max_roles]))
    for tag in ("rain", "sun", "sand", "snow"):
        rows = sorted(prof.roles.get(f"{tag} setter") or [], key=lambda x: -x[1])[:1]
        if prof.tags.get(tag, 0.0) >= 0.5 and rows:
            lines.append(f"{tag.title()} setter: {n[rows[0][0]]} ({rows[0][1]:.0%})")
    if prof.wincons:
        lines.append("Win conditions: " + "; ".join(f"{n[s]} ({why})" for s, why in prof.wincons[:4]))
    if prof.threats:
        lines.append("Threats to your team: " + " | ".join(prof.threats))
    for t in tags + [prof.style]:
        if t in TIPS:
            lines.append("Note: " + TIPS[t])
    return "\n".join(lines)


def profile_for(battle, with_threats: bool = True) -> Optional[TeamProfile]:
    """One profile per battle from its team preview (None: no preview, no knowledge, or not gen9 OU)."""
    cached = getattr(battle, "_arena_opp_profile", None)
    if cached is not None:
        return cached or None
    preview = [p.species for p in (getattr(battle, "teampreview_opponent_team", None) or [])]
    kb = kb_for(getattr(battle, "format", "") or "")
    prof = build_profile(kb, preview) if (kb and len(preview) >= 2) else None
    if prof is not None and with_threats:
        try:
            prof.threats = threats(battle, prof)
        except Exception:  # noqa: BLE001  (a calc corner case must not lose the profile)
            prof.threats = []
    try:
        battle._arena_opp_profile = prof if prof is not None else False
    except AttributeError:
        pass
    return prof


def species_tags(prof: TeamProfile, threshold: float = 0.5) -> str:
    """One line for compact views: 'balance; hazardstack 71%, voltturn 64%'."""
    tags = [f"{t} {prof.tags[t]:.0%}" for t in ("rain", "sun", "sand", "snow", "trickroom", "screens", "webs",
                                                 "hazardstack", "voltturn") if prof.tags.get(t, 0.0) >= threshold]
    return prof.style + (("; " + ", ".join(tags)) if tags else "")


def to_id(s: str) -> str:
    return to_id_str(s)
