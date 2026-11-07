"""Client of the look-ahead service (``external/foulplay/lookahead_server.py``, a separate process in Foul Play's
own environment): it mirrors each battle from the messages we forward and searches our decisions on request.

Messages, our actions and the battle's start/close go through one ordered queue per battle (the service must see
them in arrival order); analyses are separate requests that wait on the service side until that request has been
processed. Every call is best-effort: a failure only means "no look-ahead for this decision".
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import httpx

STAT = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def team_export(teambuilder) -> Optional[str]:
    """Showdown export text of a poke-env Teambuilder's team (ConstantTeambuilder), or None."""
    mons = getattr(teambuilder, "team", None)
    if not mons:
        return None
    out = []
    for m in mons:
        head = m.species or m.nickname
        if m.nickname and m.species and m.nickname != m.species:
            head = f"{m.nickname} ({m.species})"
        lines = [head + (f" @ {m.item}" if m.item else "")]
        if m.ability:
            lines.append(f"Ability: {m.ability}")
        if getattr(m, "tera_type", None):
            lines.append(f"Tera Type: {m.tera_type}")
        evs = [f"{v} {STAT[k]}" for k, v in zip(STAT, m.evs or []) if v]
        if evs:
            lines.append("EVs: " + " / ".join(evs))
        if m.nature:
            lines.append(f"{m.nature} Nature")
        ivs = [f"{v} {STAT[k]}" for k, v in zip(STAT, m.ivs or []) if v != 31]
        if ivs:
            lines.append("IVs: " + " / ".join(ivs))
        lines += [f"- {mv}" for mv in m.moves or []]
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


class LookaheadClient:
    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self._http: Optional[httpx.AsyncClient] = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._senders: dict[str, asyncio.Task] = {}
        self.errors = 0
        self._partials: dict[str, tuple[int, Optional[dict]]] = {}

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.url, timeout=self.timeout_s, trust_env=False)
        return self._http

    def _send(self, tag: str, path: str, payload: dict) -> None:
        q = self._queues.get(tag)
        if q is None:
            q = self._queues[tag] = asyncio.Queue()
            self._senders[tag] = asyncio.ensure_future(self._drain(q))
        q.put_nowait((path, payload))

    async def _drain(self, q: asyncio.Queue) -> None:
        while True:
            path, payload = await q.get()
            if path is None:
                return
            try:
                await self._client().post(path, json=payload)
            except Exception:  # noqa: BLE001  (best effort; the decision just goes without look-ahead)
                self.errors += 1

    # --- the ordered stream --------------------------------------------------------------------------------
    def start(self, tag: str, our_name: str, team: str) -> None:
        self._send(tag, "/start", {"tag": tag, "our_name": our_name, "team": team})

    def message(self, tag: str, text: str) -> None:
        if tag in self._queues:
            self._send(tag, "/msg", {"tag": tag, "text": text})

    def action(self, tag: str, rqid: int, key: str) -> None:
        if tag in self._queues:
            self._send(tag, "/action", {"tag": tag, "rqid": rqid, "key": key})

    def guide(self, tag: str, rqid: int, student: dict) -> None:
        self._send(tag, "/guide", {"tag": tag, "rqid": rqid, "student": dict(student)})

    def close(self, tag: str) -> None:
        self._partials.pop(tag, None)
        if tag in self._queues:
            self._send(tag, "/close", {"tag": tag})
            self._queues.pop(tag).put_nowait((None, None))
            self._senders.pop(tag, None)

    # --- analyses --------------------------------------------------------------------------------------------
    async def analyze(self, tag: str, rqid: int, search_ms: int, samples: int, wait_s: float = 5.0,
                      beliefs: Optional[dict] = None, belief_mode: str = "reweight", *,
                      evidence: bool = False, task: Optional[dict] = None, threads: int = 1,
                      progressive: bool = False, strategic_plan: Optional[dict] = None) -> Optional[dict]:
        """-> {policy: {key: {share, score}}, opp_policy, ms, ...} or None. ``beliefs``: our opponent model's view
        of the opposing Pokemon (``arena.opponent.model.export_beliefs``) to weight the sets the search samples."""
        if tag not in self._queues:
            return None
        t0 = time.perf_counter()
        body = {"tag": tag, "rqid": rqid, "search_ms": search_ms, "samples": samples,
                "wait_s": wait_s, "threads": threads}
        if strategic_plan is not None:
            body["strategic_plan"] = strategic_plan
        if evidence:
            body["evidence"] = True
        if task is not None:
            body["task"] = task
        if beliefs:
            body["beliefs"] = beliefs
            body["belief_mode"] = belief_mode
        if progressive:
            body["progressive"] = True
            self._partials[tag] = (rqid, None)
            latest = None
            try:
                async with self._client().stream("POST", "/analyze", json=body) as response:
                    if response.status_code != 200:
                        return None
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        result = json.loads(line)
                        if "error" in result:
                            self.errors += 1
                            continue
                        result["latency_s"] = round(time.perf_counter() - t0, 3)
                        latest = result
                        if self._partials.get(tag, (None,))[0] == rqid:
                            self._partials[tag] = (rqid, result)
            except Exception:
                self.errors += 1
            return latest
        try:
            r = await self._client().post("/analyze", json=body)
        except Exception:  # noqa: BLE001
            self.errors += 1
            return None
        if r.status_code != 200:
            return None
        out = r.json()
        out["latency_s"] = round(time.perf_counter() - t0, 3)
        return out

    def preview(self, tag: str, rqid: int) -> Optional[dict]:
        current, result = self._partials.get(tag, (None, None))
        return result if current == rqid else None

    async def aclose(self) -> None:
        for tag in list(self._queues):
            self.close(tag)
        if self._http is not None:
            await self._http.aclose()
            self._http = None
