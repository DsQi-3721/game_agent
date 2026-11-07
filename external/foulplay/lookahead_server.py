"""Look-ahead service: Foul Play's battle tracking + poke-engine search as a tool for our agent's deep thinking.

    venv/bin/python lookahead_server.py [--port 8104] [--workers 16]

Runs INSIDE external/foulplay/venv (never on the arena import path). Our agent forwards every websocket message of
a battle as it receives it; the service feeds them, in order, to Foul Play's own battle tracker (the same path as a
live Foul Play game and as ``analyze.py``), so it always knows the position as our player sees it. On request it
searches a decision: ``samples`` opponent sets are drawn from Foul Play's datasets and each is searched with
poke-engine MCTS for ``search_ms`` in a process pool; the answer is the per-option policy for us (visit share and
mean score) and the predicted policy of the opponent (its side of the same searches).

    POST /start   {tag, our_name, team}             team: Showdown export of our team
    POST /msg     {tag, text}                       one websocket message, in arrival order
    POST /action  {tag, rqid, key}                  what we played (move_<id>[_tera] / switch_<species>)
    POST /analyze {tag, rqid, search_ms, samples, wait_s, beliefs?}
                  -> {rqid, policy: {key: {share, score}}, opp_policy: {name: share}, samples, ms}

``beliefs`` (optional): our opponent model's view of each opposing Pokemon, {species id: {items, abilities, tera:
{id: p}, moves: {id: P(in set)}, spreads, spreads_by_item}}. ``belief_mode`` says how the search uses it:
``reweight`` (default) re-weights the sets Foul Play samples so their item / ability / Tera marginals match ours
(iterative proportional fitting over the candidates Foul Play still considers consistent), tilted towards our likely
moves; ``direct`` draws each set from our distributions (``belief_sampling.direct_set``, kept only if Foul Play's
consistency checks accept it, else re-weighted); ``mix`` does either for half of the samples. Without beliefs the
sampling is Foul Play's own.
    POST /close   {tag}
    GET  /health

With ``progressive: true``, /analyze streams initial and complete snapshots of the
same native tree. ``strategic_plan`` supplies conditional routes and phase goals.
POST /guide {tag, rqid, student} supplies the current S1 distribution to remaining
search slices. The persistent binding is built by scripts/build_strategic_engine.py.
"""

import argparse
import asyncio
import contextvars
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo"))
os.chdir(os.path.join(HERE, "repo"))

from aiohttp import web  # noqa: E402

from fp.data import pokedex  # noqa: E402

for _entry in pokedex.values():            # as in bridge_run.py: formes set at team validation (Keldeo-Resolute)
    if _entry.get("changesFrom") and not _entry.get("otherFormes"):
        _entry["otherFormes"] = [_entry["changesFrom"]]

import fp.modes.standard_battle as standard_battle  # noqa: E402
from fp.battle.protocol import update_battle  # noqa: E402
from fp.battle.state import LastUsedMove  # noqa: E402
from fp.config import FoulPlayConfig  # noqa: E402
from fp.search.main import get_result_from_mcts  # noqa: E402
from fp.search.poke_engine_helpers import battle_to_poke_engine_state  # noqa: E402
from fp.teams.team_converter import export_to_dict  # noqa: E402
from poke_engine import State  # noqa: E402

log = logging.getLogger("lookahead")


import fp.modes.base as fp_base  # noqa: E402

_USERNAME = contextvars.ContextVar("our_name", default="")     # one per session task (FoulPlayConfig is global)


async def _no_team_preview(self, battle, ps_websocket_client):
    return None


async def _battle_tag_and_opponent(ps_websocket_client):
    while True:
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split("|")
        if "battle" in split_msg[0]:
            tag = split_msg[0].replace(">", "").strip()
            return tag, split_msg[4].replace(_USERNAME.get(), "").replace("vs.", "").strip()


standard_battle.StandardBattleMode.handle_team_preview = _no_team_preview
fp_base.get_battle_tag_and_opponent = _battle_tag_and_opponent

from belief_search import _BELIEFS, SAMPLING  # noqa: E402  (patches Foul Play's set sampler on import)


def to_key(name):
    if name.startswith("switch "):
        return "switch_" + "".join(c for c in name[7:].lower() if c.isalnum())
    if name.endswith("-tera"):
        return "move_" + name[:-5] + "_tera"
    return "move_" + name


def fp_name(key):
    return "switch " + key[7:] if key.startswith("switch_") else key[5:].removesuffix("_tera")


class Feed:
    """PSWebsocketClient stand-in fed by /msg."""

    def __init__(self):
        self.q = asyncio.Queue()

    async def receive_message(self):
        return await self.q.get()

    async def send_message(self, room, message_list):
        return None


class Session:
    def __init__(self, tag, our_name, team):
        self.tag, self.our_name, self.team = tag, our_name, team
        self.feed = Feed()
        self.battle = None
        self.mode = standard_battle.StandardBattleMode()
        self.snapshots = {}                  # rqid -> battle copy at that request (ready to search)
        self.ready = {}                      # rqid -> asyncio.Event
        self.pending_actions = {}            # rqid -> our key, applied once that request has been processed
        self.guidance = {}
        self.latest = 0                      # newest rqid asked about: older analyses still queued are dropped
        self.error = None
        self.task = asyncio.ensure_future(self.run())

    def event(self, rqid):
        return self.ready.setdefault(rqid, asyncio.Event())

    async def run(self):
        try:
            _USERNAME.set(self.our_name)
            self.battle = await self.mode.start_battle(self.feed, "gen9ou", export_to_dict(self.team))
            while True:
                msg = await self.feed.receive_message()
                if msg is None:
                    return
                prev = self.battle.rqid
                if update_battle(self.battle, msg):
                    rq = self.battle.rqid
                    snap = deepcopy(self.battle)
                    snap.user.update_from_request_json(snap.request_json)
                    self.snapshots[rq] = snap
                    for old in [r for r in self.snapshots if r < rq - 12]:
                        self.snapshots.pop(old, None)
                    self.event(rq).set()
                elif self.battle.rqid != prev:
                    self.event(self.battle.rqid).set()          # a wait request: nothing to search
                key = self.pending_actions.pop(self.battle.rqid, None)
                if key:
                    self.set_action(key)
        except Exception as e:  # noqa: BLE001  (the session stops answering; the agent falls back)
            self.error = f"{type(e).__name__}: {e}"
            log.warning("session %s failed: %s", self.tag, self.error)
            for ev in self.ready.values():
                ev.set()

    def set_action(self, key):
        if self.battle is not None and self.battle.user.active is not None:
            self.battle.user.last_selected_move = LastUsedMove(self.battle.user.active.name, fp_name(key),
                                                               self.battle.turn)


class Superseded(Exception):
    pass


def timed_search(state, milliseconds, index, threads=1):
    wall, cpu = time.perf_counter(), time.process_time()
    result = get_result_from_mcts(state, milliseconds, index, threads)
    return result, time.perf_counter() - wall, time.process_time() - cpu


def comparison_states(snap, mode, samples, beliefs, belief_mode):
    started = time.thread_time()
    _BELIEFS.value, _BELIEFS.cache, _BELIEFS.mode = beliefs or None, {}, belief_mode
    try:
        sampled = mode.prepare_battles(snap, samples)
    finally:
        _BELIEFS.value, _BELIEFS.cache, _BELIEFS.mode = None, None, "reweight"
    states = [(battle_to_poke_engine_state(b).to_string(), weight) for b, weight in sampled]
    return states, time.thread_time() - started


def search(snap, mode, pool, samples, search_ms, stale=lambda: False, beliefs=None, belief_mode="reweight",
           evidence=False, threads=1):
    caller_cpu = time.thread_time()
    _BELIEFS.value, _BELIEFS.cache, _BELIEFS.mode = beliefs or None, {}, belief_mode
    try:
        sampled = mode.prepare_battles(snap, samples)
    finally:
        _BELIEFS.value, _BELIEFS.cache, _BELIEFS.mode = None, None, "reweight"
    states = [(battle_to_poke_engine_state(b).to_string(), chance) for b, chance in sampled]
    futs = [(pool.submit(timed_search, state, search_ms, i, threads), chance)
            for i, (state, chance) in enumerate(states)]
    ours, theirs, total_c, iters = {}, {}, 0.0, []
    details, worker_wall, worker_cpu = [], 0., 0.
    for i, (fut, chance) in enumerate(futs):
        if stale():                          # a newer decision of this battle was asked about: free the workers
            for f, _c in futs:
                f.cancel()
            raise Superseded()
        res, seconds, cpu_seconds = fut.result()
        worker_wall += seconds
        worker_cpu += cpu_seconds
        tv = res.total_visits or 1
        iters.append(res.total_visits)
        total_c += chance
        if evidence:
            native = State.from_string(states[i][0])
            active = native.side_two.pokemon[int(native.side_two.active_index)]
            details.append({"sample": i, "weight": chance, "search_cpu_s": round(cpu_seconds, 4),
                            "search_wall_s": round(seconds, 4),
                            "assumed_active_set": {"species": active.id, "item": active.item,
                                                   "moves": [m.id for m in active.moves if m.id != "none"]},
                            "policy": {to_key(o.move_choice): {"share": round(o.visits / tv, 4),
                                                               "score": round(o.total_score / o.visits, 4) if o.visits else None,
                                                               "visits": o.visits}
                                       for o in res.side_one}})
        for o in res.side_one:
            p = ours.setdefault(o.move_choice, [0.0, 0.0, 0.0])
            p[0] += chance * o.visits / tv
            if o.visits:
                p[1] += chance * o.total_score / o.visits
                p[2] += chance
        for o in res.side_two:
            theirs[o.move_choice] = theirs.get(o.move_choice, 0.0) + chance * o.visits / tv
    policy = {to_key(k): {"share": round(v[0] / total_c, 4), "score": round(v[1] / v[2], 4) if v[2] else None}
              for k, v in ours.items()}
    opp = {k: round(v / total_c, 4) for k, v in sorted(theirs.items(), key=lambda kv: -kv[1])}
    compute = {"samples": len(iters), "search_ms": search_ms, "threads": threads,
               "budget_ms": len(iters) * search_ms * threads,
               "worker_wall_s": worker_wall, "worker_cpu_s": worker_cpu,
               "preparation_and_aggregation_cpu_s": time.thread_time() - caller_cpu}
    return policy, opp, iters, details, compute


def make_app(workers):
    sessions = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    tree_pool = ThreadPoolExecutor(max_workers=workers)
    stats = {"started": 0, "analyses": 0, "failed": 0}

    async def start(req):
        b = await req.json()
        old = sessions.pop(b["tag"], None)
        if old is not None:
            old.feed.q.put_nowait(None)
        sessions[b["tag"]] = Session(b["tag"], b["our_name"], b["team"])
        stats["started"] += 1
        return web.json_response({"ok": True})

    async def msg(req):
        b = await req.json()
        s = sessions.get(b["tag"])
        if s is not None:
            s.feed.q.put_nowait(b["text"])
        return web.json_response({"ok": s is not None})

    async def action(req):
        b = await req.json()
        s = sessions.get(b["tag"])
        if s is not None:
            if s.battle is not None and s.battle.rqid == b["rqid"]:
                s.set_action(b["key"])
            else:
                s.pending_actions[b["rqid"]] = b["key"]
        return web.json_response({"ok": s is not None})

    async def progressive(req, body, session, snap, rqid, started):
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(req)
        def stale():
            return session.latest > rqid or sessions.get(body["tag"]) is not session
        async def emit(result):
            result.update(rqid=rqid, ms=round(1000 * (time.time() - started)))
            await response.write((json.dumps(result) + "\n").encode())
        try:
            states, preparation = await asyncio.to_thread(
                comparison_states, deepcopy(snap), session.mode, int(body.get("samples", 8)),
                body.get("beliefs"), body.get("belief_mode", "reweight"))
            milliseconds = int(body.get("search_ms", 700))
            from persistent_search import progressive_search
            guidance = session.guidance.setdefault(rqid, {})
            await progressive_search(states, milliseconds, body.get("strategic_plan"), guidance,
                                     tree_pool, emit, stale, preparation, rqid * 100, body.get("task"))
            stats["analyses"] += 1
        except Superseded:
            stats["superseded"] = stats.get("superseded", 0) + 1
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:
            stats["failed"] += 1
            await emit({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            session.guidance.pop(rqid, None)
        return response

    async def analyze(req):
        b = await req.json()
        s = sessions.get(b["tag"])
        if s is None:
            return web.json_response({"error": "no session"}, status=404)
        rqid, t0 = int(b["rqid"]), time.time()
        s.latest = max(s.latest, rqid)
        try:
            await asyncio.wait_for(s.event(rqid).wait(), timeout=float(b.get("wait_s", 5.0)))
        except asyncio.TimeoutError:
            return web.json_response({"error": "request not seen", "rqid": rqid}, status=409)
        snap = s.snapshots.get(rqid)
        if rqid < s.latest:                  # the battle moved on: a search of a past decision only costs CPU
            stats["superseded"] = stats.get("superseded", 0) + 1
            return web.json_response({"error": "superseded", "rqid": rqid}, status=409)
        if s.error or snap is None:
            stats["failed"] += 1
            return web.json_response({"error": s.error or "no decision at this request", "rqid": rqid}, status=409)
        if b.get("progressive"):
            return await progressive(req, b, s, snap, rqid, t0)
        try:
            policy, opp, iters, details, compute = await asyncio.get_running_loop().run_in_executor(
                None, search, deepcopy(snap), s.mode, pool, int(b.get("samples", 8)), int(b.get("search_ms", 1000)),
                lambda: s.latest > rqid, b.get("beliefs"), b.get("belief_mode", "reweight"),
                b.get("evidence", False), int(b.get("threads", 1)))
        except Superseded:
            stats["superseded"] = stats.get("superseded", 0) + 1
            return web.json_response({"error": "superseded", "rqid": rqid}, status=409)
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            return web.json_response({"error": f"{type(e).__name__}: {e}", "rqid": rqid}, status=500)
        stats["analyses"] += 1
        return web.json_response({"rqid": rqid, "policy": policy, "opp_policy": opp, "iterations": iters,
                                  "ms": round(1000 * (time.time() - t0)), "evidence": details, "task": b.get("task"),
                                  "compute": compute})

    async def guide(req):
        body = await req.json()
        session = sessions.get(body["tag"])
        if session is None or int(body["rqid"]) < session.latest:
            return web.json_response({"accepted": False})
        session.guidance.setdefault(int(body["rqid"]), {})["student"] = body["student"]
        return web.json_response({"accepted": True})

    async def close(req):
        b = await req.json()
        s = sessions.pop(b["tag"], None)
        if s is not None:
            s.feed.q.put_nowait(None)
        return web.json_response({"ok": s is not None})

    async def health(_req):
        return web.json_response({"ok": True, "sessions": len(sessions), **stats, "sampling": SAMPLING})

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.add_routes([web.post("/start", start), web.post("/msg", msg), web.post("/action", action),
                    web.post("/analyze", analyze), web.post("/guide", guide), web.post("/close", close), web.get("/health", health)])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8104)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    FoulPlayConfig.pokemon_format = "gen9ou"
    FoulPlayConfig.smogon_stats = None
    FoulPlayConfig.log_to_file = False
    FoulPlayConfig.search_threads = 1
    FoulPlayConfig.parallelism = 8
    FoulPlayConfig.search_time_ms = 1000
    print(f"[lookahead] listening on {args.host}:{args.port} with {args.workers} search workers", flush=True)
    web.run_app(make_app(args.workers), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
