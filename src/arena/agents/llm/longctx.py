"""Persistent-context (flat) agents: the paper's long-horizon baselines.

One conversation per battle, everything appended in order -- the flat context
C_t = (P, q, (a_1, o_1), ..., (a_t, o_t)) of paper Eq. 1: each decision adds the
new battle events (raw protocol lines by default), the current state, the legal
options, the model's replies and every RAW tool output. No isolation, no memory.
When the conversation would exceed the context budget, the oldest decision blocks
are dropped (sliding window); ``manage_context`` is the hook subclasses override
(the summary baseline compresses instead).

Same state view, options, answer format, tools and sampling path as HyMem, so a
comparison isolates the context-management strategy.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from arena.agents.base import AgentContext
from arena.agents.hymem.config import DEFAULT_TOOLS
from arena.agents.hymem.messages import clip, parse_planner, strip_think
from arena.agents.llm.base import LLMAgent
from arena.llm.client import LLMResponse
from arena.llm.spaces import SpaceConfig, complete_in_space, model_spec
from arena.llm.tool_calling import make_protocol
from arena.logging.calls import CallLogger
from arena.parse.choice import parse_choice
from arena.parse.validate import validate_or_fallback
from arena.state.battle_log import NOISE_KINDS, LogDigester, events_since
from arena.state.views import legal_options, option_for, render_brief, render_options, render_state

FLAT_SYSTEM = """You are a competitive Pokemon battle agent on Pokemon Showdown. The whole battle so far is in this conversation.

## Each time you must act you get
the new battle events, the current state and the numbered legal actions.

## Reply
Start with <analysis>at most 3 short sentences</analysis>. Then either call a tool to get facts (damage, speed, the opponent's likely sets, switch-ins, type chart, Pokedex, battle log), or commit with <answer>N</answer><confidence>0-1</confidence> where N is a legal action number (write <answer>N tera</answer> to terastallize).

## Battle brief
{brief}{tool_docs}{summary}"""

FLAT_USER = """## New battle events
{events}

## Current state
{state}
{passive}
## Legal actions (answer with the number)
{options}

Decide now."""

FINAL_HINT = "When you have enough facts, stop calling tools and reply with <answer>N</answer>."
CONTINUE_HINT = "Call another tool if needed, otherwise reply with <answer>N</answer>."
FINAL_STEP = "FINAL STEP for this decision: reply with <analysis> and <answer>N</answer><confidence>0-1</confidence> only."
NUDGE = "Call a tool or reply with <answer>N</answer>."


def estimate_tokens(messages: list[dict]) -> int:
    return int(sum(len(str(m.get("content") or "")) for m in messages) / 3.5)


@dataclass
class FlatSession:
    tag: str
    brief: str
    messages: list = field(default_factory=list)
    blocks: list = field(default_factory=list)     # index where each decision block starts
    cursor: int = 0
    digester: LogDigester = field(default_factory=LogDigester)
    summary: str = ""
    decision_idx: int = 0
    last_key: Optional[tuple] = None
    last_order: Optional[str] = None
    truncated_blocks: int = 0
    summaries: int = 0
    closed: bool = False
    in_flight: int = 0


class PersistentContextAgent(LLMAgent):
    name = "llm_flat"
    space = "flat"

    def __init__(self, ctx: Optional[AgentContext] = None, *, model: str = "qwen3.5-4b-mlx",
                 max_steps: int = 4, obs_format: str = "raw", max_context_tokens: Optional[int] = None,
                 tools: str = "", supports_tools: Optional[bool] = None, temperature: float = 0.7,
                 top_p: float = 0.8, max_tokens: int = 320, thinking: bool = False,
                 decision_budget_s: float = 900.0, log_calls: str = "full"):
        super().__init__(ctx, model=model, temperature=temperature)
        if obs_format not in ("raw", "digest"):
            raise ValueError("obs_format must be raw|digest")
        self.max_steps = max(1, int(max_steps))
        self.obs_format = obs_format
        self.max_context_tokens = max_context_tokens
        self.tool_names = tuple(t.strip() for t in (tools or "").split(",") if t.strip()) or None
        self._supports_tools = supports_tools
        self.sc = SpaceConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                              presence_penalty=0.0, thinking=thinking)
        self.decision_budget_s = decision_budget_s
        self.call_log = CallLogger.beside(self.ctx.logger, level=log_calls)
        self.sessions: "OrderedDict[str, FlatSession]" = OrderedDict()
        self._tools = None
        self.n_tool_calls = 0
        self.metrics: dict = {}

    # --- plumbing -------------------------------------------------------------------
    def _metric(self, k: str, v: float = 1.0, mode: str = "add") -> None:
        if mode == "max":
            self.metrics[k] = max(self.metrics.get(k, 0), v)
        else:
            self.metrics[k] = self.metrics.get(k, 0) + v

    def supports_tools(self) -> bool:
        if self._supports_tools is not None:
            return self._supports_tools
        spec = model_spec(self.ctx.llm, self.model)
        return spec.supports_tools if spec is not None else True

    def context_budget(self) -> int:
        if self.max_context_tokens:
            return int(self.max_context_tokens)
        spec = model_spec(self.ctx.llm, self.model)
        window = spec.context_window if spec is not None and spec.context_window else 32768
        return int(window * 0.9)

    def tools(self) -> list:
        if self._tools is None:
            tools: list = []
            if self.tool_names is None and self.ctx.augmentations is not None:
                tools = self.ctx.augmentations.tools()
            if not tools:
                from arena.registry.augments import entries
                reg = entries()
                for n in self.tool_names or DEFAULT_TOOLS:
                    if n in reg:
                        t = reg[n]().as_tool()
                        if t is not None:
                            tools.append(t)
            self._tools = tools
        return self._tools

    async def _call(self, sess: FlatSession, messages: list[dict], *, tools=None,
                    space: Optional[str] = None, sc: Optional[SpaceConfig] = None) -> LLMResponse:
        t_wall = time.time()
        resp, latency = await complete_in_space(self.ctx.llm, self.model, messages, sc or self.sc, tools=tools)
        self.account(resp)
        tokens = resp.prompt_tokens or estimate_tokens(messages)
        self._metric("llm_calls")
        if (space or self.space) == self.space:
            self._metric("ctx_tokens_sum", tokens)
            self._metric("ctx_tokens_max", tokens, "max")
            self._metric("main_calls")
        if self.call_log is not None:
            self.call_log.log(battle_tag=sess.tag, decision=sess.decision_idx, space=space or self.space,
                              messages=messages, output=resp.text, reasoning=resp.reasoning,
                              prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                              latency_s=latency, t_start=t_wall, cached_tokens=resp.cached_tokens,
                              finish_reason=resp.finish_reason)
        return resp

    def _system(self, sess: FlatSession) -> str:
        docs = make_protocol(self.supports_tools(), final_hint=FINAL_HINT).tool_docs(self.tools())
        summary = f"\n\n## Summary of earlier turns\n{sess.summary}" if sess.summary else ""
        return FLAT_SYSTEM.format(brief=sess.brief, tool_docs=f"\n\n{docs}" if docs else "", summary=summary)

    def _session(self, battle: AbstractBattle) -> FlatSession:
        sess = self.sessions.get(battle.battle_tag)
        if sess is None:
            sess = FlatSession(battle.battle_tag, render_brief(battle))
            sess.messages = [{"role": "system", "content": self._system(sess)}]
            self.sessions[battle.battle_tag] = sess
            while len(self.sessions) > 8:
                old = next((t for t, s in self.sessions.items() if s.in_flight == 0), None)
                if old is None:
                    break
                self.sessions.pop(old)
        return sess

    def _observation(self, sess: FlatSession, battle, events: list) -> str:
        if self.obs_format == "digest":
            results, _ = sess.digester.ingest(events, battle)
            return "\n".join(r.render() for r in results)
        return "\n".join("|".join(e) for e in events if len(e) > 1 and e[1] not in NOISE_KINDS)

    # --- context management hook ------------------------------------------------------
    async def manage_context(self, sess: FlatSession, battle) -> None:
        """Sliding window: drop the oldest decision blocks beyond the budget."""
        budget = self.context_budget()
        while estimate_tokens(sess.messages) > budget and len(sess.blocks) > 1:
            start, end = sess.blocks[0], sess.blocks[1]
            del sess.messages[start:end]
            shift = end - start
            sess.blocks = [b - shift for b in sess.blocks[1:]]
            sess.truncated_blocks += 1
            self._metric("truncated_blocks")

    # --- decision ----------------------------------------------------------------------
    async def act(self, battle: AbstractBattle) -> BattleOrder:
        sess = self._session(battle)
        sess.in_flight += 1
        try:
            return await self._act(sess, battle)
        finally:
            sess.in_flight -= 1
            if sess.closed and sess.in_flight == 0:
                self.sessions.pop(sess.tag, None)

    async def _act(self, sess: FlatSession, battle: AbstractBattle) -> BattleOrder:
        t0 = time.monotonic()
        options = legal_options(battle)
        req = getattr(battle, "_last_request", None) or {}
        key = (req.get("rqid") if isinstance(req, dict) else None, len(getattr(battle, "_replay_data", []) or []))
        if sess.last_key == key and sess.last_order is not None:
            alt = next((o for o in options if o.order.message != sess.last_order), None)
            final, fallback = validate_or_fallback(alt.order if alt else None, battle)
            sess.last_order = final.message
            self._metric("recalls")
            return self._finish(sess, battle, final, fallback, None, t0, "", note="recall")

        sess.decision_idx += 1
        events, sess.cursor = events_since(battle, sess.cursor)
        passive = ""
        if self.ctx.augmentations is not None:
            passive = "\n".join(f"[{r.name}]\n{r.text}" for r in self.ctx.augmentations.passive(battle))
        user = FLAT_USER.format(events=self._observation(sess, battle, events) or "(none)",
                                state=render_state(battle), passive=f"\n{passive}\n" if passive else "",
                                options=render_options(options))
        sess.blocks.append(len(sess.messages))
        sess.messages.append({"role": "user", "content": user})

        if len(options) <= 1:
            final, fallback = validate_or_fallback(options[0].order if options else None, battle)
            sess.messages.append({"role": "assistant", "content": "<answer>1</answer>"})
            sess.last_key, sess.last_order = key, final.message
            self._metric("single_option")
            return self._finish(sess, battle, final, fallback, None, t0, "", note="single_option")

        opt, conf, raw, error = None, None, "", ""
        try:
            await self.manage_context(sess, battle)
            opt, conf, raw = await asyncio.wait_for(self._loop(sess, battle, options),
                                                    timeout=self.decision_budget_s * 2)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            error = "decision_timeout"
            self._metric("timeouts")
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {clip(str(e), 200)}"
            self._metric("errors")
        final, fallback = validate_or_fallback(opt.order if opt else None, battle)
        if sess.messages[-1]["role"] != "assistant":
            chosen = option_for(options, final)
            sess.messages.append({"role": "assistant", "content": f"<answer>{chosen.idx if chosen else 1}</answer>"})
        sess.last_key, sess.last_order = key, final.message
        return self._finish(sess, battle, final, fallback, conf, t0, raw, error=error)

    async def _loop(self, sess: FlatSession, battle, options):
        tools = self.tools()
        tool_map = {t.name: t for t in tools}
        proto = make_protocol(self.supports_tools(), final_hint=FINAL_HINT, continue_hint=CONTINUE_HINT)
        complete_tools = proto.complete_kwargs(tools).get("tools")
        last_text = ""
        for step in range(self.max_steps):
            final_step = step == self.max_steps - 1
            if final_step:
                if sess.messages[-1]["role"] == "user":
                    sess.messages[-1]["content"] += "\n\n" + FINAL_STEP
                else:
                    sess.messages.append({"role": "user", "content": FINAL_STEP})
            resp = await self._call(sess, sess.messages, tools=None if final_step else complete_tools)
            last_text = resp.text
            calls = proto.extract_calls(resp) if (tools and not final_step) else []
            answer = next((d for d in parse_planner(resp.text)[0] if d.kind == "answer"), None)
            if answer is not None and not calls:
                sess.messages.append({"role": "assistant", "content": strip_think(resp.text) or "(no output)"})
                opt = parse_choice(answer.text, options)
                if opt is not None:
                    return opt, answer.confidence, last_text
                self._metric("parse_failures")
                sess.messages.append({"role": "user", "content":
                                      f"'{clip(answer.text, 40)}' is not a legal action number. Answer with a number from the list."})
                continue
            if calls:
                outputs = []
                for c in calls[:3]:
                    tool = tool_map.get(c.name)
                    try:
                        out = await tool.call(battle, c.arguments or {}) if tool else f"{c.name}: unknown tool"
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        out = f"error: {type(e).__name__}: {e}"
                    outputs.append(out)          # raw: flat agents keep every tool output
                    self.n_tool_calls += 1
                proto.append_turn(sess.messages, resp, calls[:3], outputs)
                continue
            self._metric("parse_failures")
            sess.messages.append({"role": "assistant", "content": strip_think(resp.text) or "(no output)"})
            sess.messages.append({"role": "user", "content": NUDGE})
        return parse_choice(strip_think(last_text), options), None, last_text

    def _finish(self, sess, battle, final, fallback, conf, t0, raw, *, error: str = "", note: str = ""):
        self.n_decisions += 1
        self.n_fallbacks += int(fallback)
        latency = time.monotonic() - t0
        ctx_tokens = estimate_tokens(sess.messages)
        self._metric("decisions")
        self._metric("fallbacks", int(fallback))
        self._metric("decision_latency_s_sum", latency)
        if self.ctx.logger is not None:
            self.ctx.logger.log_turn(
                battle_tag=battle.battle_tag, turn=battle.turn, state_text=render_state(battle),
                prompt=sess.messages[-2]["content"][-4000:] if len(sess.messages) > 1 else "",
                raw_llm_output=raw, tool_calls=[], parsed_order=final.message, fallback_used=fallback,
                note=note or (error and f"error={error}") or "",
                extra={"decision": sess.decision_idx, "context_tokens_est": ctx_tokens,
                       "n_messages": len(sess.messages), "blocks": len(sess.blocks),
                       "truncated_blocks": sess.truncated_blocks, "summaries": sess.summaries,
                       "confidence": conf if conf is not None else -1.0, "latency_s": round(latency, 2),
                       "fallback": fallback, "error": error})
        return final

    def on_battle_end(self, battle: AbstractBattle) -> None:
        sess = self.sessions.get(battle.battle_tag)
        if sess is None:
            return
        sess.closed = True
        self._metric("battles")
        if self.call_log is not None:
            facts = []
            try:
                events, _ = events_since(battle, 0)
                _, kfs = LogDigester().ingest(events, battle)
                facts = [{"turn": kf.turn, "kind": kf.kind, "text": kf.text} for kf in kfs]
            except Exception:  # noqa: BLE001
                pass
            self.call_log.log(battle_tag=sess.tag, decision=sess.decision_idx, space="event:battle_end",
                              messages=[], output="", extra={"won": battle.won, "turns": battle.turn,
                                                             "truncated_blocks": sess.truncated_blocks,
                                                             "summaries": sess.summaries, "key_facts": facts})
        if sess.in_flight == 0:
            self.sessions.pop(sess.tag, None)

    def extra_metrics(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.metrics.items()}
