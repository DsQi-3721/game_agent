"""Preflight a registry model before spending a battle on it.

    python scripts/check_llm.py --model qwen3.5-4b-mlx

Four checks through the same client the agents use (LiteLLMClient):
  1. chat        -- plain completion round-trip
  2. thinking    -- enable_thinking on puts deliberation in `reasoning`, off leaves it empty
  3. tool call   -- native function calling is parsed into tool_calls
  4. json        -- a JSON-only answer parses
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

_TOOL = {
    "type": "function",
    "function": {
        "name": "pokedex_lookup",
        "description": "Look up a Pokemon's typing and base stats.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "species name"}},
            "required": ["query"],
        },
    },
}


def _thinking(flag: bool) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": flag}}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b-mlx")
    ap.add_argument("--skip-thinking", action="store_true", help="skip the slow thinking-on check")
    args = ap.parse_args()

    for v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1," + os.environ.get("NO_PROXY", "")

    from arena.llm.client import LiteLLMClient
    from arena.llm.model_registry import ModelRegistry

    client = LiteLLMClient(ModelRegistry.load())
    results: list[tuple[str, bool, str]] = []

    async def run(name, coro_fn):
        try:
            ok, info = await coro_fn()
        except Exception as e:  # noqa: BLE001
            ok, info = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, info))
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {info}", flush=True)

    async def chat():
        r = await client.complete(args.model, [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Reply with exactly the word: pong"}],
            temperature=0.0, max_tokens=16, extra_body=_thinking(False))
        return "pong" in r.text.lower(), f"{r.text!r} ({r.latency_s:.1f}s, {r.prompt_tokens}+{r.completion_tokens} tok)"

    async def thinking():
        off = await client.complete(args.model, [{"role": "user", "content": "What is 17*3? Answer with the number."}],
                                    temperature=0.0, max_tokens=64, extra_body=_thinking(False))
        if args.skip_thinking:
            return not off.reasoning, f"off: reasoning={len(off.reasoning)} chars (on-check skipped)"
        on = await client.complete(args.model, [{"role": "user", "content": "What is 17*3? Answer with the number."}],
                                   temperature=0.6, max_tokens=768, extra_body=_thinking(True))
        ok = (not off.reasoning) and bool(on.reasoning) and "</think>" not in on.text
        return ok, (f"off: reasoning={len(off.reasoning)}c text={off.text.strip()[:20]!r}; "
                    f"on: reasoning={len(on.reasoning)}c text={on.text.strip()[:20]!r} ({on.latency_s:.1f}s)")

    async def tool_call():
        r = await client.complete(args.model, [
            {"role": "system", "content": "Use tools when they help."},
            {"role": "user", "content": "Look up Garchomp with the pokedex_lookup tool."}],
            tools=[_TOOL], temperature=0.0, max_tokens=256, extra_body=_thinking(False))
        calls = [(c.name, c.arguments) for c in r.tool_calls]
        return bool(calls) and calls[0][0] == "pokedex_lookup", f"tool_calls={calls} text={r.text[:60]!r}"

    async def json_only():
        r = await client.complete(args.model, [
            {"role": "user", "content": 'Output only JSON, no prose: {"goal": "<a short goal>", "threats": ["<one>"]}'}],
            temperature=0.3, max_tokens=128, presence_penalty=0.0, extra_body=_thinking(False))
        text = r.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        return isinstance(data, dict) and "goal" in data, f"{data}"

    await run("chat", chat)
    await run("thinking", thinking)
    await run("tool_call", tool_call)
    await run("json", json_only)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed for {args.model}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
