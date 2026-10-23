"""TypeSafe-compatible HTTP server around a local scorer.

    POST /v1/systemone   {"model", "state", "questions"} -> {"model", "answers", "usage"}
    GET  /v1/models      {"models": [{"name", "description", "release_date"}]}
    GET  /health

Point any agent at it with ``TYPESAFE_BASE_URL=http://127.0.0.1:<port>`` (jev_fast /
llm_hymem_dual) or ``student_url`` (the cascade's first tier); no code changes. Requests are
served one at a time (the model is not shared across threads).
"""

from __future__ import annotations

import asyncio
import datetime as dt

from aiohttp import web

from arena.s1.prompts import LABELS, options


def validate(body: dict) -> str:
    if not isinstance(body, dict) or "questions" not in body or "state" not in body:
        return "body needs 'state' and 'questions'"
    qs = body["questions"]
    if not isinstance(qs, dict) or not qs:
        return "'questions' must be a non-empty object"
    for qid, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            return f"question {qid!r}: type must be choice|noul|score"
        try:
            n = len(options(q))
        except Exception as e:  # noqa: BLE001
            return f"question {qid!r}: {e}"
        if not 1 <= n <= len(LABELS):
            return f"question {qid!r}: 1..{len(LABELS)} options supported, got {n}"
    return ""


def make_app(scorer, *, description: str = "local System One model") -> web.Application:
    lock = asyncio.Lock()
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    async def system_one(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"detail": "invalid JSON"}, status=400)
        err = validate(body)
        if err:
            return web.json_response({"detail": err}, status=422)
        async with lock:
            answers, n_tok = await asyncio.to_thread(scorer.answer, body["state"], body["questions"])
        return web.json_response({"model": scorer.model_id, "answers": answers,
                                  "usage": {"input_tokens": n_tok, "output_tokens": 0}})

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": scorer.model_id, "description": description,
                                              "release_date": started}]})

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "model": scorer.model_id})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.post("/v1/systemone", system_one), web.get("/v1/models", models), web.get("/health", health)])
    return app
