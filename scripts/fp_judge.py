"""Foul Play as an offline judge of our decisions: prepare the inputs, then report.

    python scripts/fp_judge.py prepare --runs /data/wangxinqi/game_agent_runs/strong_kakuna \
        /data/wangxinqi/game_agent_runs/strong_foulplay --out /data/wangxinqi/fp_judge/streams.jsonl
    external/foulplay/venv/bin/python external/foulplay/analyze.py --streams .../streams.jsonl --out .../judged.jsonl
    python scripts/fp_judge.py report --streams .../streams.jsonl --judged .../judged.jsonl --out reports/data/fp_judge.json

prepare: for every battle our agents played in the given runs, the Showdown server log (logchallenges) is
replayed through the simulator (``scripts/node/replay_battle.js``) to rebuild exactly what our player received;
the lines are cut into websocket-like messages (each ``|request|`` on its own, as the server sends it) for Foul
Play's battle tracker. With them: our team as a Showdown export, and our recorded decisions keyed by rqid (route,
the student's choice, System 2's choice, the move played) from ``traj_*.jsonl`` and ``s1exp_*.jsonl``.

report: joins the judge's per-option policy (visit share over sampled opponent sets, mean MCTS score) with our
decisions: agreement with Foul Play's top choice, regret in its score, and -- where System 2 replaced the
student's move -- whether Foul Play prefers the replacement (override precision).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SHOWDOWN = "/home/wangxinqi/project/pokemon-showdown"
REPLAY_JS = Path(__file__).resolve().parent / "node" / "replay_battle.js"


def _alnum(s: str) -> str:
    return "".join(c for c in s if c.isalnum())


def export_team(sets: list[dict]) -> str:
    """Showdown export text of a server-log team (list of set dicts)."""
    stat = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
    out = []
    for s in sets:
        head = s.get("species") or s.get("name")
        if s.get("name") and s.get("name") != s.get("species"):
            head = f"{s['name']} ({s['species']})"
        lines = [head + (f" @ {s['item']}" if s.get("item") else "")]
        if s.get("ability"):
            lines.append(f"Ability: {s['ability']}")
        if s.get("level") and s["level"] != 100:
            lines.append(f"Level: {s['level']}")
        if s.get("teraType"):
            lines.append(f"Tera Type: {s['teraType']}")
        evs = [f"{v} {stat[k]}" for k, v in (s.get("evs") or {}).items() if v]
        if evs:
            lines.append("EVs: " + " / ".join(evs))
        if s.get("nature"):
            lines.append(f"{s['nature']} Nature")
        ivs = [f"{v} {stat[k]}" for k, v in (s.get("ivs") or {}).items() if v != 31]
        if ivs:
            lines.append("IVs: " + " / ".join(ivs))
        lines += [f"- {m}" for m in s.get("moves") or []]
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def our_decisions(shard: Path, agent: str) -> dict[str, dict[int, dict]]:
    """battle_tag -> rqid -> recorded decision."""
    dec = {}
    for p in shard.glob("s1exp_*.jsonl"):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "decision":
                dec[r["decision_id"]] = r
    out: dict = defaultdict(dict)
    for line in open(shard / f"traj_{agent}.jsonl"):
        t = json.loads(line)
        e = t.get("extra") or {}
        if t.get("rqid") is None:
            continue
        did = e.get("decision_id") or f"{t['battle_tag']}:{e.get('decision')}"      # jev_fast / HyMem layouts
        d = dec.get(did, {})
        pre = "student_s1_" if "student_s1_choice" in e else "s1_"
        out[t["battle_tag"]][int(t["rqid"])] = {
            "rqid": int(t["rqid"]), "turn": t.get("turn"), "decision": did.rsplit(":", 1)[-1],
            "route": e.get("route") or t.get("note"), "student": e.get(f"{pre}choice"),
            "student_top": [k for k, _ in (e.get(f"{pre}top") or [])],
            "student_conf": e.get(f"{pre}confidence"), "gate": e.get(f"{pre}reason"),
            "s2": d.get("s2_key"), "final": d.get("final_key") or order_key(t.get("parsed_order") or ""),
            "n_options": d.get("n_options")}
    return out


def order_key(order: str) -> str | None:
    """'/choose move knockoff terastallize' -> move_knockoff_tera; '/choose switch kingambit' -> switch_kingambit."""
    m = re.match(r"/choose (move|switch) ([A-Za-z0-9-]+)( terastallize)?", order)
    if not m or m[2].isdigit():
        return None
    ident = re.sub(r"[^a-z0-9]", "", m[2].lower())
    return f"move_{ident}" + ("_tera" if m[3] else "") if m[1] == "move" else f"switch_{ident}"


def find_log(battle_tag: str) -> str | None:
    n = battle_tag.rsplit("-", 1)[-1]
    hits = glob.glob(f"{SHOWDOWN}/logs/*/gen9ou/*/gen9ou-{n}.log.json")
    return hits[0] if hits else None


def messages(tag: str, title: str, lines: list[str]) -> list[str]:
    """Our player's lines as websocket messages: the room header first, every |request| and |player| line alone
    (Foul Play reads the player slot from the start of the message that names its opponent)."""
    out = [f">{tag}\n|init|battle\n|title|{title}"]
    buf: list[str] = []
    for ln in lines:
        if ln.startswith(("|request|", "|player|")):
            if buf:
                out.append(f">{tag}\n" + "\n".join(buf))
                buf = []
            out.append(f">{tag}\n{ln}")
        else:
            buf.append(ln)
    if buf:
        out.append(f">{tag}\n" + "\n".join(buf))
    return out


def prepare(args) -> None:
    jobs = []                                            # (run, agent, battle_tag, decisions)
    for run in args.runs:
        for agent_dir in sorted(p for p in Path(run).iterdir() if p.is_dir()):
            for shard in sorted(agent_dir.glob("s*")):
                if not (shard / f"traj_{agent_dir.name}.jsonl").exists():
                    continue
                for tag, decs in our_decisions(shard, agent_dir.name).items():
                    jobs.append((Path(run).name, agent_dir.name, tag, decs))
    logs = {tag: find_log(tag) for _, _, tag, _ in jobs}
    files = sorted({f for f in logs.values() if f})
    replays = {}
    for i in range(0, len(files), 40):
        res = subprocess.run(["node", str(REPLAY_JS), SHOWDOWN, *files[i:i + 40]], capture_output=True, text=True,
                             check=True)
        for line in res.stdout.splitlines():
            r = json.loads(line)
            replays[r["file"]] = r
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = missing = 0
    with open(out, "w") as f:
        for run, agent, tag, decs in jobs:
            path = logs.get(tag)
            if not path or path not in replays:
                missing += 1
                continue
            log = json.load(open(path))
            prefix = _alnum(agent)[:7].lower()
            side = next((s for s in ("p1", "p2") if _alnum(log[s]).lower().startswith(prefix)), None)
            if side is None:
                missing += 1
                continue
            rec = {"run": run, "agent": agent, "battle_tag": tag, "log": path, "our_side": side,
                   "our_name": log[side], "opp_name": log["p2" if side == "p1" else "p1"],
                   "winner": log.get("winner"), "team": export_team(log[f"{side}team"]),
                   "messages": messages(tag, f"{log['p1']} vs. {log['p2']}", replays[path][side]),
                   "decisions": sorted(decs.values(), key=lambda d: d["rqid"])}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(json.dumps({"battles": n, "missing_logs": missing, "decisions": sum(len(d) for *_, d in jobs)}))


# --- report ------------------------------------------------------------------------------------------------
def fp_to_key(name: str) -> str:
    if name.startswith("switch "):
        return "switch_" + re.sub(r"[^a-z0-9]", "", name[7:].lower())
    if name.endswith("-tera"):
        return "move_" + name[:-5] + "_tera"
    return "move_" + name


def report(args) -> None:
    decs = {}
    for line in (ln for path in args.streams for ln in open(path)):
        r = json.loads(line)
        for d in r["decisions"]:
            decs[(r["battle_tag"], d["rqid"])] = {**d, "run": r["run"], "agent": r["agent"],
                                                 "won": r["winner"] == r["our_name"]}
    rows = []
    for (tag, rqid), pol in load_policies(args.judged).items():           # key -> {share, score}
        d = decs.get((tag, rqid))
        if d is None:
            continue
        best = max(pol, key=lambda k: pol[k]["share"])
        best_score = max(v["score"] for v in pol.values() if v["score"] is not None)
        rows.append({**d, "pol": pol, "fp_top": best, "best_score": best_score})

    def q(row, key):
        v = row["pol"].get(key)
        return (v["share"], v["score"]) if v else (0.0, None)

    def summarise(rs: list[dict], choice: str) -> dict:
        rs = [r for r in rs if r.get(choice)]
        if not rs:
            return {"n": 0}
        agree = sum(r[choice] == r["fp_top"] for r in rs)
        share = [q(r, r[choice])[0] for r in rs]
        regret = [r["best_score"] - q(r, r[choice])[1] for r in rs if q(r, r[choice])[1] is not None]
        return {"n": len(rs), "agree_fp_top": round(agree / len(rs), 3),
                "fp_share_of_choice": round(sum(share) / len(share), 3),
                "regret_mean": round(sum(regret) / len(regret), 4) if regret else None,
                "unscored": len(rs) - len(regret)}

    res: dict = {"decisions_judged": len(rows), "by_agent": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["run"], r["agent"])].append(r)
    for (run, agent), rs in sorted(groups.items()):
        g = {"all_played": summarise(rs, "final"), "student_top1": summarise(rs, "student")}
        by_route = defaultdict(list)
        for r in rs:
            by_route[r["route"]].append(r)
        g["by_route"] = {k: {"played": summarise(v, "final"), "student": summarise(v, "student")}
                         for k, v in sorted(by_route.items())}
        ov = [r for r in rs if r.get("s2") and r.get("student") and r["s2"] != r["student"]]
        if ov:
            better = sum(q(r, r["s2"])[0] > q(r, r["student"])[0] for r in ov)
            worse = sum(q(r, r["s2"])[0] < q(r, r["student"])[0] for r in ov)
            dq = [q(r, r["s2"])[1] - q(r, r["student"])[1] for r in ov
                  if q(r, r["s2"])[1] is not None and q(r, r["student"])[1] is not None]
            boot = []
            rng = random.Random(0)
            for _ in range(1000):
                smp = [rng.choice(dq) for _ in dq]
                boot.append(sum(smp) / len(smp))
            boot.sort()
            g["overrides"] = {"n": len(ov), "fp_prefers_override": round(better / len(ov), 3),
                              "fp_prefers_student": round(worse / len(ov), 3),
                              "mean_dscore": round(sum(dq) / len(dq), 4) if dq else None,
                              "mean_dscore_ci95": [round(boot[25], 4), round(boot[974], 4)] if dq else None,
                              "s2_agree_fp_top": round(sum(r["s2"] == r["fp_top"] for r in ov) / len(ov), 3),
                              "student_agree_fp_top": round(sum(r["student"] == r["fp_top"] for r in ov) / len(ov), 3)}
        kinds = defaultdict(list)
        for r in rs:
            kinds["switch" if (r.get("final") or "").startswith("switch_") else "move"].append(r)
        g["by_kind_played"] = {k: summarise(v, "final") for k, v in kinds.items()}
        res["by_agent"][f"{run}/{agent}"] = g
    text = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def load_policies(paths: list[str]) -> dict:
    pol = {}
    for path in paths:
        for line in open(path):
            j = json.loads(line)
            if not j.get("error") and j.get("policy") and j.get("repeat", 0) == 0:
                pol[(j["battle_tag"], j["rqid"])] = {fp_to_key(k): v for k, v in j["policy"].items()}
    return pol


def variants_report(args) -> None:
    """Planner prompt variants (scripts/replay_planner_prompts.py) judged by Foul Play."""
    pol = load_policies(args.judged)
    recs = [json.loads(line) for line in open(args.variants)]
    recs = [r for r in recs if (r["battle_tag"], r["rqid"]) in pol]

    def stat(choices):
        rs = [(pol[(r["battle_tag"], r["rqid"])], c) for r, c in choices if c]
        if not rs:
            return {"n": 0}
        top = lambda p: max(p, key=lambda k: p[k]["share"])
        best = lambda p: max(v["score"] for v in p.values() if v["score"] is not None)
        sc = [(best(p) - p[c]["score"]) for p, c in rs if c in p and p[c]["score"] is not None]
        return {"n": len(rs), "agree_fp_top": round(sum(c == top(p) for p, c in rs) / len(rs), 3),
                "fp_share_of_choice": round(sum(p.get(c, {}).get("share", 0) for p, c in rs) / len(rs), 3),
                "regret_mean": round(sum(sc) / len(sc), 4) if sc else None}

    res = {"decisions": len(recs),
           "student_top1": stat([(r, r["student"]) for r in recs]),
           "planner_final_recorded": stat([(r, r["s2"]) for r in recs])}
    for v in ("contract", "no_hint", "answer_only"):
        res[v] = stat([(r, r[v]["answer"]) for r in recs])
        res[v]["directive"] = dict(Counter(r[v]["directive"] for r in recs))
    rk = [(pol[(r["battle_tag"], r["rqid"])], r["contract"]["ranking"]) for r in recs if r["contract"]["ranking"]]
    if rk:
        top = lambda p: max(p, key=lambda k: p[k]["share"])
        res["contract_ranking"] = {"n": len(rk), "fp_top_in_ranking_top1": round(sum(top(p) in r[:1] for p, r in rk) / len(rk), 3),
                                   "fp_top_in_ranking_top3": round(sum(top(p) in r[:3] for p, r in rk) / len(rk), 3)}
    av = [(pol[(r["battle_tag"], r["rqid"])], r["contract"]["avoid"], r["student"]) for r in recs if r["contract"]["avoid"]]
    if av:
        shares = [p.get(k, {}).get("share", 0) for p, ks, _ in av for k in ks]
        n_opts = [len(p) for p, _, _ in av]
        stu = [(p, s) for p, ks, s in av if s in ks]
        res["contract_avoid"] = {"n": len(av), "mean_fp_share_of_avoided": round(sum(shares) / len(shares), 4),
                                 "uniform_share": round(sum(1 / n for n in n_opts) / len(n_opts), 4),
                                 "student_top_avoided": len(stu),
                                 "fp_share_of_avoided_student_top": round(sum(p.get(s, {}).get("share", 0) for p, s in stu) / max(1, len(stu)), 4)}
    text = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--streams", nargs="+", required=True)
    r.add_argument("--judged", nargs="+", required=True)
    r.add_argument("--out", default=None)
    v = sub.add_parser("variants")
    v.add_argument("--variants", required=True)
    v.add_argument("--judged", nargs="+", required=True)
    v.add_argument("--out", default=None)
    args = ap.parse_args()
    {"prepare": prepare, "report": report, "variants": variants_report}[args.cmd](args)


if __name__ == "__main__":
    main()
