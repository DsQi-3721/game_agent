"""TypeSafe-compatible HTTP server around a local scorer.

    POST /v1/systemone   {"model", "state", "questions"} -> {"model", "answers", "usage"}
    GET  /v1/models      {"models": [{"name", "description", "release_date"}]}
    GET  /health

Point any agent at it with ``TYPESAFE_BASE_URL=http://127.0.0.1:<port>`` (jev_fast /
llm_hymem_dual) or ``student_url`` (the cascade's first tier); no code changes. Requests that arrive
together (several battles at once) are scored as one batch when the scorer has ``answer_many`` (HF):
the worker takes everything queued within ``batch_window_ms`` (up to ``max_batch`` requests); otherwise
requests are served one at a time. The model runs on one worker thread only.

Two priority lanes (HF scorer): a request is ``high`` or ``low`` by its ``X-S1-Priority`` header (without one,
requests with more than ``low_threshold`` questions are low). High requests -- the decision being made -- are
always served first and batched as above; a low request -- background work such as memory routing -- is scored
in slices of ``low_slice`` questions, and any high request that arrives meanwhile is served between two slices.
Without lanes a 40-question routing request batched with a 6-question action request made the action wait for
both (action p50 0.73 s with routing vs 0.30 s without, v7 HyMem runs). Rows are scored independently, so slicing
does not change any answer.
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


def make_app(scorer, *, description: str = "local System One model", batch_window_ms: float = 3.0,
             max_batch: int = 16, low_slice: int = 16, low_threshold: int = 32) -> web.Application:
    lock = asyncio.Lock()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    high: asyncio.Queue = asyncio.Queue()
    low: asyncio.Queue = asyncio.Queue()
    wake = asyncio.Event()
    stats = {"requests": 0, "batches": 0, "high_requests": 0, "low_requests": 0, "low_slices": 0,
             "high_wait_s": 0.0, "low_wait_s": 0.0}

    def finish(item, res, lane):
        _st, _qs, fut, t_in = item
        stats[f"{lane}_wait_s"] = round(stats[f"{lane}_wait_s"] + asyncio.get_running_loop().time() - t_in, 4)
        if not fut.done():
            fut.set_result(res)

    async def serve_high() -> None:
        batch = [high.get_nowait()]
        deadline = asyncio.get_running_loop().time() + batch_window_ms / 1000
        while len(batch) < max_batch:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(high.get(), timeout))
            except asyncio.TimeoutError:
                break
        try:
            results = await asyncio.to_thread(scorer.answer_many, [(st, qs) for st, qs, *_x in batch])
        except Exception as e:  # noqa: BLE001
            for _st, _qs, fut, _t in batch:
                if not fut.done():
                    fut.set_exception(e)
            return
        stats["batches"] += 1
        stats["requests"] += len(batch)
        stats["high_requests"] += len(batch)
        for item, res in zip(batch, results):
            finish(item, res, "high")

    async def serve_low() -> None:
        item = low.get_nowait()
        state, questions, fut, _t = item
        items = list(questions.items())
        answers, n_tok = {}, 0
        try:
            for i in range(0, len(items), low_slice):
                while not high.empty():                     # the decision being made goes first
                    await serve_high()
                [(a, t)] = await asyncio.to_thread(scorer.answer_many, [(state, dict(items[i:i + low_slice]))])
                answers.update(a)
                n_tok += t
                stats["low_slices"] += 1
        except Exception as e:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(e)
            return
        stats["requests"] += 1
        stats["low_requests"] += 1
        finish(item, ({qid: answers[qid] for qid in questions}, n_tok), "low")

    async def worker() -> None:
        while True:
            if high.empty() and low.empty():
                wake.clear()
                await wake.wait()
                continue
            if not high.empty():
                await serve_high()
            else:
                await serve_low()

    async def score(state, questions, priority=None):
        if not hasattr(scorer, "answer_many"):
            async with lock:
                return await asyncio.to_thread(scorer.answer, state, questions)
        if priority not in ("high", "low"):
            priority = "low" if len(questions) > low_threshold else "high"
        fut = asyncio.get_running_loop().create_future()
        await (high if priority == "high" else low).put((state, questions, fut, asyncio.get_running_loop().time()))
        wake.set()
        return await fut

    async def start_worker(app):
        if hasattr(scorer, "answer_many"):
            app["batch_worker"] = asyncio.create_task(worker())

    async def stop_worker(app):
        task = app.get("batch_worker")
        if task is not None:
            task.cancel()

    async def system_one(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"detail": "invalid JSON"}, status=400)
        err = validate(body)
        if err:
            return web.json_response({"detail": err}, status=422)
        answers, n_tok = await score(body["state"], body["questions"], request.headers.get("X-S1-Priority"))
        return web.json_response({"model": scorer.model_id, "answers": answers,
                                  "usage": {"input_tokens": n_tok, "output_tokens": 0}})

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": scorer.model_id, "description": description,
                                              "release_date": started}]})

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "model": scorer.model_id, **stats})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.post("/v1/systemone", system_one), web.get("/v1/models", models), web.get("/health", health)])
    app.on_startup.append(start_worker)
    app.on_cleanup.append(stop_worker)
    return app
