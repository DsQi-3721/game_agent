"""System-1 layer offline: TypeSafe client (mock transport), battle facts, the fast decider and
its gate, the jev_fast agent, the memory router and the HyMem dual-process agent."""

import asyncio
import copy
import json
import pickle
from pathlib import Path

import httpx
import pytest

from arena.agents.base import AgentContext
from arena.agents.hymem.memory import HyMemStore
from arena.agents.system1.agents import HyMemDualAgent, JevFastAgent
from arena.agents.system1.decider import FastDecider, RouteConfig, choice_confidence
from arena.agents.system1.facts import build_option_facts, build_state, ko_label, option_key
from arena.agents.system1.memory_router import MemoryRouter, RoutedMemory, collect_items
from arena.knowledge import store
from arena.llm import typesafe as ts
from arena.llm.client import MockLLMClient
from arena.logging.trajectory import TrajectoryLogger
from arena.state.battle_log import KeyFact
from arena.state.views import legal_options

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"
KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge_mini"


@pytest.fixture(autouse=True)
def mini_knowledge(monkeypatch):
    monkeypatch.setenv("ARENA_KNOWLEDGE_DIR", str(KNOWLEDGE))
    # offline means offline: never read the project .env (it may hold a real key) and never open a
    # real connection -- every client in these tests must use an httpx.MockTransport or a mock
    monkeypatch.setattr(ts, "_load_env", lambda: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    real_client = ts.TypeSafeClient._client

    def guarded(self):
        if self._transport is None:
            raise AssertionError("offline test tried to open a real connection to the TypeSafe API")
        return real_client(self)
    monkeypatch.setattr(ts.TypeSafeClient, "_client", guarded)
    store.clear_cache()
    yield
    store.clear_cache()


@pytest.fixture
def battle():
    with open(FIXTURE, "rb") as f:
        return copy.deepcopy(pickle.load(f))


def responder(pick=None, p=0.8, rev_pick=None, p_hard=0.1, extra=None, p_stay=None):
    """Mock Jev: favour ``pick`` in every best* Choice that offers it (``rev_pick`` in the reversed
    ones); the attack-or-switch Noul follows the kind of ``pick`` unless ``p_stay`` is given."""
    def reply(state, questions):
        out = {}
        stay = p_stay if p_stay is not None else (0.1 if (pick or "").startswith("switch_") else 0.9)
        for qid, q in questions.items():
            if q["type"] == "choice" and qid.startswith("best"):
                want = rev_pick if qid.endswith("_rev") and rev_pick else pick
                if want not in q["criteria"]:
                    want = next(iter(q["criteria"]))
                out[qid] = ts.uniform_choice(q["criteria"], want, p)
            elif q["type"] == "noul" and qid == "stay":
                out[qid] = {"type": "noul", "noul": stay}
            elif q["type"] == "noul" and qid == "hard":
                out[qid] = {"type": "noul", "noul": p_hard}
            elif extra is not None:
                out[qid] = extra(qid, q)
        return out
    return reply


# --- client ---------------------------------------------------------------------------
async def test_client_request_shape_retry_and_errors():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        body = json.loads(request.content)
        assert body["model"] == "jev-1.13.0" and body["state"] == {"x": 1}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "q": {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.75, "b": 0.25}}},
            "usage": {"input_tokens": 42, "output_tokens": 3}})

    c = ts.TypeSafeClient(api_key="k", model="jev-1.13.0", transport=httpx.MockTransport(handler))
    res = await c.system_one({"x": 1}, {"q": ts.choice("pick", {"a": None, "b": None})})
    assert len(seen) == 2 and seen[-1].headers["authorization"] == "Bearer k"
    assert str(seen[-1].url) == "https://api.typesafe.ai/v1/systemone"
    a = res.answers["q"]
    assert a.choice == "a" and a.p_top == 0.75 and a.margin == 0.5 and res.input_tokens == 42
    assert c.n_calls == 1 and c.input_tokens == 42

    bad = ts.TypeSafeClient(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    with pytest.raises(ts.TypeSafeError) as e:
        await bad.system_one("s", {"q": ts.noul("?")})
    assert e.value.status == 401
    with pytest.raises(ts.TypeSafeError):
        await ts.TypeSafeClient().system_one("s", {"q": ts.noul("?")})     # no key configured


def test_question_builders_validate():
    assert ts.noul("x", true="yes", false="no")["criteria"] == {"true": "yes", "false": "no"}
    with pytest.raises(ValueError):
        ts.choice("x", {})
    with pytest.raises(ValueError):
        ts.score("x", [])
    assert abs(choice_confidence({"a": 0.6, "b": 0.3, "c": 0.1}) - 0.4) < 1e-9


# --- facts ----------------------------------------------------------------------------
def test_facts_cover_every_option(battle):
    options = legal_options(battle)
    facts = build_option_facts(battle, options)
    keys = [f.key for f in facts]
    assert len(keys) == len(set(keys)) == len(options)
    assert keys == [option_key(o) for o in options]
    assert all(f.fields["action"] for f in facts)
    assert any(k.endswith("_tera") for k in keys) and any(k.startswith("switch_") for k in keys)
    state = build_state(battle, extra_facts=["opp garganacl has Salt Cure"])
    json.dumps(state)
    assert state["you"]["active"]["species"] == "passimian" and state["known_facts"]


def test_ko_label():
    assert ko_label(110, 130, 100) == ("guaranteed KO this turn", True)
    assert ko_label(80, 120, 100)[1] is True
    assert ko_label(30, 40, 100) == ("3-4 hits to KO", False)
    assert ko_label(0, 0, 100)[1] is False


# --- decider ----------------------------------------------------------------------------
async def test_decider_accepts_confident_consistent_answer(battle):
    options = legal_options(battle)
    target = option_key(options[1])
    client = ts.MockTypeSafeClient(responder(pick=target, p=0.8))
    d = FastDecider(client, RouteConfig())
    fd = await d.decide(battle, options)
    assert fd.key == target and fd.option is options[1] and fd.consistent and fd.accept
    q = client.calls[0]["questions"]
    assert set(q) == {"best_move", "best_move_rev", "best_switch", "best_switch_rev", "stay", "hard"}
    assert list(q["best_move_rev"]["criteria"]) == list(reversed(list(q["best_move"]["criteria"])))
    assert all(k.startswith("switch_") for k in q["best_switch"]["criteria"])
    assert fd.branch == "move" and abs(fd.p_stay - 0.9) < 1e-9 and abs(sum(fd.probabilities.values()) - 1) < 1e-9
    assert "[system1_suggestion]" in fd.hint(options) and fd.telemetry()["s1_accept"] is True


async def test_decider_gate_rejections(battle):
    options = legal_options(battle)
    a, b = option_key(options[0]), option_key(options[1])
    fd = await FastDecider(ts.MockTypeSafeClient(responder(pick=a, rev_pick=b, p=0.9))).decide(battle, options)
    assert not fd.consistent and not fd.accept and fd.reason_code == "order_inconsistent"
    fd = await FastDecider(ts.MockTypeSafeClient(responder(pick=a, p=0.9, p_hard=0.9))).decide(battle, options)
    assert not fd.accept and fd.reason_code == "hard_turn"
    fd = await FastDecider(ts.MockTypeSafeClient(responder(pick=a, p=0.12))).decide(battle, options)
    assert not fd.accept and fd.reason_code == "low_confidence"


async def test_decider_piggybacks_extra_questions(battle):
    options = legal_options(battle)
    client = ts.MockTypeSafeClient(responder(extra=lambda qid, q: {"type": "noul", "noul": 0.7}))
    fd = await FastDecider(client).decide(battle, options, extra_questions={"useful_m0": ts.noul("?")})
    assert fd.extra_answers["useful_m0"].noul == 0.7 and len(client.calls) == 1
    with pytest.raises(ValueError):
        await FastDecider(client).decide(battle, options, extra_questions={"best_move": ts.noul("?")})


# --- jev_fast agent -------------------------------------------------------------------------
async def test_jev_fast_agent_plays_and_logs(battle, tmp_path):
    options = legal_options(battle)
    target = option_key(options[2])
    logger = TrajectoryLogger(tmp_path / "traj.jsonl", "jev")
    agent = JevFastAgent(AgentContext(logger=logger), client=ts.MockTypeSafeClient(responder(pick=target)))
    order = await agent.act(battle)
    assert order.message == options[2].order.message and agent.n_fallbacks == 0
    rec = json.loads((tmp_path / "traj.jsonl").read_text().splitlines()[0])
    assert rec["extra"]["s1_choice"] == target and rec["extra"]["s1_accept"] is True
    # the same request again (order rejected by the server) -> next-ranked option, no new call
    again = await agent.act(battle)
    assert again.message != order.message and agent.metrics["recalls"] == 1


async def test_jev_fast_agent_falls_back_on_api_error(battle):
    async def boom(state, questions):
        raise ts.TypeSafeError(529, "overloaded")
    agent = JevFastAgent(client=ts.MockTypeSafeClient(boom), fallback="max_damage")
    order = await agent.act(battle)
    assert order.message in {o.message for o in battle.valid_orders}
    assert agent.n_fallbacks == 1 and agent.metrics["errors"] == 1


# --- memory router ------------------------------------------------------------------------
def sample_store():
    st = HyMemStore()
    st.key_facts = [KeyFact(1, "opp_move", "opp Garganacl used Salt Cure"),
                    KeyFact(2, "faint", "opp Luvdisc fainted")]
    st.episode = {"task_description": "win", "key_events": [{"turn": 1, "description": "Garganacl used Salt Cure",
                                                             "outcome": "salted"}],
                  "verified_facts": [{"fact": "Garganacl is at 57% HP", "turn": 1, "confidence": "high"},
                                     {"fact": "opp Garganacl holds Leftovers", "turn": 1, "confidence": "high"}],
                  "current_progress": "6 vs 5"}
    st.working = {"immediate_goal": "KO Garganacl with Close Combat", "current_challenges": "",
                  "active_hypotheses": [{"hypothesis": "Garganacl will Recover next turn"}],
                  "failed_approaches": ["Rock Slide barely scratches Garganacl"], "next_actions": []}
    st.tool = {"derived_rules": ["Always check type matchups first"], "effective_strategies": [],
               "dead_ends": ["damage_calc fails without the exact move name"], "useful_sources": []}
    st.summary = "We lead 6-5. Garganacl walls our physical attackers."
    return st


KIND_OF = {"57% HP": "current_numbers", "Leftovers": "opponent_set", "used Salt Cure ->": "battle_event",
           "KO Garganacl": "plan", "Recover next": "hypothesis", "barely scratches": "lesson",
           "type matchups": "generic", "damage_calc": "tool_tip", "6-5": "current_numbers", "walls": "lesson"}


def memory_answers(qid, q):
    note = q["instructions"]["note"]
    kind = next((k for frag, k in KIND_OF.items() if frag in note), "generic")
    if qid.startswith("kind_"):
        return ts.uniform_choice(q["criteria"], kind, 0.9)
    if qid.startswith("valid_"):
        return {"type": "noul", "noul": 0.1 if "57% HP" in note else 0.9}
    return {"type": "noul", "noul": 0.8}


async def test_memory_router_classifies_validates_and_places(battle):
    items = collect_items(sample_store())
    texts = [m.text for m in items]
    assert items[0].trusted and items[0].kind_hint == "opponent_set" and len(set(texts)) == len(texts)
    client = ts.MockTypeSafeClient(lambda s, q: {qid: memory_answers(qid, qq) for qid, qq in q.items()})
    router = MemoryRouter(client)
    qs = router.questions(items)
    trusted = [m for m in items if m.trusted]
    assert len(qs) == 3 * len(items) - 2 * len(trusted)
    routed = await router.route(build_state(battle), items)
    by_text = {r.item.text: r for r in routed.items}
    assert by_text["Garganacl is at 57% HP"].dropped == "current_numbers"
    assert by_text["Always check type matchups first"].dropped == "generic"
    assert "executor" in by_text["damage_calc fails without the exact move name"].dest
    planner, executor = routed.render("planner"), routed.render("executor")
    assert "Leftovers" in planner and "57% HP" not in planner and "damage_calc fails" not in planner
    assert "damage_calc fails" in executor and "Recover next" not in executor
    assert "Leftovers" in " ".join(routed.fast_facts())
    s = routed.stats()
    assert s["mem_items"] == len(items) and s["mem_kept"] < len(items)
    # kinds are cached: the next routing only re-checks validity / usefulness
    again = router.questions(items)
    assert not any(k.startswith("kind_") for k in again) and len(again) == len(qs) - (len(items) - len(trusted))
    routed2 = await router.route(build_state(battle), items)
    assert [r.kind for r in routed2.items] == [r.kind for r in routed.items]


# --- dual-process HyMem --------------------------------------------------------------------------
PLANNER_SYS = "You are the PLANNER"


def planner_script(answer="<analysis>ok</analysis><answer>3</answer><confidence>0.7</confidence>", delay=0.0):
    seen = []

    async def reply(messages, tools, idx):
        seen.append(messages)
        if delay:
            await asyncio.sleep(delay)
        sys = messages[0]["content"]
        if sys.startswith(PLANNER_SYS):
            return answer
        if sys.startswith("You are the MEMORY"):
            return "{}"
        return "<results>\nFINDINGS: none\nSTATUS: partial\n</results>"
    return reply, seen


def make_dual(script, jev_reply, tmp_path=None, **params):
    logger = TrajectoryLogger(tmp_path / "traj_dual.jsonl", "dual") if tmp_path else None
    ctx = AgentContext(llm=MockLLMClient(script), logger=logger)
    return HyMemDualAgent(ctx, model="mock", supports_tools=False, jev_client=ts.MockTypeSafeClient(jev_reply),
                          **params)


async def test_dual_fast_route_skips_the_planner(battle, tmp_path):
    options = legal_options(battle)
    script, seen = planner_script()
    agent = make_dual(script, responder(pick=option_key(options[1]), p=0.85), tmp_path)
    order = await agent.act(battle)
    assert order.message == options[1].order.message and not seen
    rec = json.loads((tmp_path / "traj_dual.jsonl").read_text().splitlines()[0])
    assert rec["extra"]["route"] == "fast" and rec["extra"]["llm_calls"] == 0
    sess = next(iter(agent.sessions._s.values()), None)
    assert sess is None or any(e.kind == "decision" for e in sess.entries)


async def test_dual_slow_route_gets_the_hint(battle, tmp_path):
    options = legal_options(battle)
    script, seen = planner_script()
    agent = make_dual(script, responder(pick=option_key(options[1]), p=0.3, p_hard=0.9), tmp_path)
    order = await agent.act(battle)
    assert order.message == options[2].order.message            # the planner's answer (3)
    planner_prompts = [m[1]["content"] for m in seen if m[0]["content"].startswith(PLANNER_SYS)]
    assert planner_prompts and "[system1_suggestion]" in planner_prompts[0]
    rec = json.loads((tmp_path / "traj_dual.jsonl").read_text().splitlines()[0])
    assert rec["extra"]["route"] == "slow" and rec["extra"]["s1_accept"] is False
    assert not agent._hints                                      # the hint never leaks into a later turn


async def test_dual_slow_timeout_plays_the_fast_answer(battle):
    options = legal_options(battle)
    script, _ = planner_script(delay=1.0)
    agent = make_dual(script, responder(pick=option_key(options[4]), p=0.3, p_hard=0.9), slow_budget_s=0.2)
    order = await agent.act(battle)
    assert order.message == options[4].order.message and agent.n_fallbacks == 0
    assert agent.metrics["timeouts"] == 1


async def test_dual_routes_memory_in_a_parallel_request(battle):
    options = legal_options(battle)
    script, seen = planner_script()

    def jev(state, questions):
        out = responder(pick=option_key(options[1]), p=0.3, p_hard=0.9)(state, questions)
        out.update({qid: memory_answers(qid, q) for qid, q in questions.items() if qid not in out})
        return out
    agent = make_dual(script, jev, memory_router="jev")
    sess = agent.sessions.get_or_create(battle.battle_tag, lambda: "brief")
    sess.store = sample_store()
    await agent.act(battle)
    assert len(agent.jev.calls) == 2                     # the decision and the memory routing run side by side
    decision_q, memory_q = sorted((c["questions"] for c in agent.jev.calls), key=lambda q: "stay" not in q)
    assert "stay" in decision_q and not any(k.startswith("kind_") for k in decision_q)
    assert memory_q and all(k.split("_")[0] in ("kind", "valid", "useful") for k in memory_q)
    planner_prompt = next(m[1]["content"] for m in seen if m[0]["content"].startswith(PLANNER_SYS))
    assert "Opponent info" in planner_prompt and "57% HP" not in planner_prompt
    assert isinstance(sess.routed, RoutedMemory)


# --- Tier 0 ---------------------------------------------------------------------------------
def damaged_opponent(battle, hp=30):
    opp = battle.opponent_active_pokemon
    opp._current_hp, opp._max_hp = hp, 100
    return battle


async def test_tier0_plays_a_sure_first_strike_ko_without_any_model(battle):
    from arena.agents.system1.decider import obvious_choice
    assert obvious_choice(battle, build_option_facts(battle, legal_options(battle))) is None   # full HP: Sash/Sturdy
    damaged_opponent(battle)
    options = legal_options(battle)
    sure = obvious_choice(battle, build_option_facts(battle, options))
    assert sure is not None and sure.key == "move_closecombat"          # no tera spent, Rock Slide can miss
    script, seen = planner_script()
    agent = make_dual(script, responder(), memory_router="jev")
    order = await agent.act(battle)
    assert order.message == sure.option.order.message and not agent.jev.calls and not seen
    pure = JevFastAgent(client=ts.MockTypeSafeClient(responder(pick="switch_mewtwo")))   # tier0 off: Jev decides
    assert (await pure.act(battle)).message != sure.option.order.message


async def test_split_mode_switches_only_when_the_attack_or_switch_call_says_so(battle):
    options = legal_options(battle)
    client = ts.MockTypeSafeClient(responder(pick="switch_mewtwo", p=0.8))
    fd = await FastDecider(client).decide(battle, options)
    assert fd.key == "switch_mewtwo" and fd.branch == "switch" and fd.accept
    unclear = ts.MockTypeSafeClient(responder(pick="switch_mewtwo", p=0.8, p_stay=0.45))
    fd = await FastDecider(unclear).decide(battle, options)
    assert fd.key == "switch_mewtwo" and not fd.accept and fd.reason_code == "unclear_switch"
    flat = await FastDecider(ts.MockTypeSafeClient(responder(pick="switch_mewtwo")), RouteConfig(mode="flat")).decide(
        battle, options)
    assert flat.key == "switch_mewtwo" and flat.branch == ""


async def test_jev_fast_escalates_unsure_turns_to_the_heuristic(battle):
    from poke_env.player.baselines import SimpleHeuristicsPlayer
    heur = SimpleHeuristicsPlayer.choose_singles_move(battle)[0].message
    unsure = JevFastAgent(client=ts.MockTypeSafeClient(responder(pick="switch_mewtwo", p_stay=0.5)),
                          escalate="heuristic")
    assert (await unsure.act(battle)).message == heur and unsure.metrics["escalated"] == 1
    sure = JevFastAgent(client=ts.MockTypeSafeClient(responder(pick="switch_mewtwo")), escalate="heuristic")
    assert (await sure.act(battle)).message.lower() == "/choose switch mewtwo"


async def test_dual_fast_turn_does_not_wait_for_memory_routing(battle):
    import time as _time
    options = legal_options(battle)
    script, seen = planner_script()
    decide = responder(pick=option_key(options[1]), p=0.85)

    async def jev(state, questions):
        if any(k.startswith("kind_") for k in questions):          # the slow, large routing request
            await asyncio.sleep(0.5)
            return {qid: memory_answers(qid, q) for qid, q in questions.items()}
        return decide(state, questions)
    agent = make_dual(script, jev, memory_router="jev")
    sess = agent.sessions.get_or_create(battle.battle_tag, lambda: "brief")
    sess.store = sample_store()
    t0 = _time.perf_counter()
    order = await agent.act(battle)
    assert order.message == options[1].order.message and _time.perf_counter() - t0 < 0.3 and not seen
    await sess.route_task                                           # finishes in the background
    assert isinstance(sess.routed, RoutedMemory) and sess.routed.items


async def test_dual_defers_folds_to_the_background(battle):
    options = legal_options(battle)
    seen = []

    async def llm(messages, tools, idx):
        seen.append(messages[0]["content"][:30])
        sys = messages[0]["content"]
        if sys.startswith("You are the MEMORY"):
            return "{}"
        if sys.startswith(PLANNER_SYS) and "state handover" in messages[-1]["content"]:
            return "<summary>## 1. Objective: win</summary>"
        if sys.startswith(PLANNER_SYS):
            return "<analysis>ok</analysis><answer>3</answer><confidence>0.7</confidence>"
        return "<results>\nFINDINGS: none\nSTATUS: partial\n</results>"
    agent = make_dual(llm, responder(pick=option_key(options[1]), p=0.3, p_hard=0.9))
    sess = agent.sessions.get_or_create(battle.battle_tag, lambda: "brief")
    sess.trigger.should_fold = lambda **kw: "tool_calls_threshold"       # a fold is due right now
    order = await agent.act(battle)
    assert order.message == options[2].order.message                  # the planner decided without folding
    assert sess.store.folds == 0 and agent.metrics["background_folds"] == 1
    await sess.fold_task                                               # ... and the fold ran right after
    assert sess.store.folds == 1 and not sess.deciding and not sess.pending_fold
