"""System-1 learning loop and deep fusion, offline: experience recording, dataset, cascade,
controller (step control / router / executor), typed folds, the local server protocol,
evaluation and the model registry. No model weights, no network beyond a localhost test server."""

import copy
import json
import pickle
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer

from arena.agents.base import AgentContext
from arena.agents.hymem.memory import HyMemStore, MemoryManager
from arena.agents.hymem.config import HyMemConfig
from arena.agents.system1.agents import HyMemDualAgent, JevFastAgent
from arena.agents.system1.facts import option_key
from arena.knowledge import store
from arena.llm import typesafe as ts
from arena.llm.client import MockLLMClient
from arena.logging.trajectory import TrajectoryLogger
from arena.s1 import dataset as ds
from arena.s1.evaluate import evaluate
from arena.s1.prompts import options, render, target_distribution, to_answer
from arena.s1.recorder import load_records
from arena.s1.registry import Registry, decide_promotion
from arena.s1.server import make_app, validate
from arena.state.battle_log import KeyFact
from arena.state.views import legal_options

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"
KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge_mini"
PLANNER_SYS = "You are the PLANNER"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("ARENA_KNOWLEDGE_DIR", str(KNOWLEDGE))
    monkeypatch.setattr(ts, "_load_env", lambda: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    store.clear_cache()
    yield
    store.clear_cache()


@pytest.fixture
def battle():
    with open(FIXTURE, "rb") as f:
        return copy.deepcopy(pickle.load(f))


def favour(pick=None, p=0.85, p_stay=None, extra=None):
    """Mock System One: favour ``pick`` in every best* Choice that offers it."""
    def reply(state, questions):
        out = {}
        stay = p_stay if p_stay is not None else (0.1 if (pick or "").startswith("switch_") else 0.9)
        for qid, q in questions.items():
            if q["type"] == "choice" and qid.startswith("best"):
                want = pick if pick in q["criteria"] else next(iter(q["criteria"]))
                out[qid] = ts.uniform_choice(q["criteria"], want, p)
            elif qid == "stay":
                out[qid] = {"type": "noul", "noul": stay}
            elif qid == "hard":
                out[qid] = {"type": "noul", "noul": 0.1}
            elif extra is not None:
                out[qid] = extra(qid, q)
        return out
    return reply


# --- prompts -----------------------------------------------------------------------------------
def test_render_and_answer_shapes():
    q = ts.choice("pick one", {"a": "first", "b": {"why": "structured"}})
    msgs, keys = render({"x": 1}, q)
    assert keys == ["a", "b"] and "A) a: first" in msgs[1]["content"] and msgs[1]["content"].startswith("STATE:")
    ans = to_answer(q, keys, [0.75, 0.25])
    assert ans["choice"] == "a" and abs(ans["confidence"] - 0.5) < 1e-9
    n = ts.noul("is it?", true="yes it is")
    assert [k for k, _ in options(n)] == ["true", "false"] and to_answer(n, ["true", "false"], [0.3, 0.7])["noul"] == 0.3
    assert target_distribution(n, {"noul": 0.8}, ["true", "false"]) == [0.8, pytest.approx(0.2)]
    s = ts.score("rate", ["low", "mid", "high"])
    assert to_answer(s, ["0", "1", "2"], [0.1, 0.2, 0.7])["score"] == 2


# --- experience recording + cascade ---------------------------------------------------------------
async def test_jev_fast_records_requests_decisions_and_outcomes(battle, tmp_path):
    options_ = legal_options(battle)
    logger = TrajectoryLogger(tmp_path / "traj_j.jsonl", "j")
    agent = JevFastAgent(AgentContext(logger=logger), client=ts.MockTypeSafeClient(favour(option_key(options_[1]))))
    await agent.act(battle)
    battle._won = True
    agent.on_battle_end(battle)
    recs = load_records([tmp_path / "s1exp_j.jsonl"])
    kinds = [r["type"] for r in recs]
    assert kinds == ["s1", "decision", "outcome"]
    s1, dec, out = recs
    assert s1["purpose"] == "action" and s1["tier"] == "teacher" and s1["backend"].startswith("typesafe:")
    assert s1["decision_id"] == dec["decision_id"] and set(s1["answers"]) >= {"stay", "best_move"}
    assert dec["final_key"] == option_key(options_[1]) and dec["route"] == "fast_teacher"
    assert out["won"] is True and out["battle_tag"] == battle.battle_tag


async def test_cascade_asks_the_teacher_only_when_the_student_is_unsure(battle):
    options_ = legal_options(battle)
    sure_student = ts.MockTypeSafeClient(favour(option_key(options_[0]), p=0.9))
    teacher = ts.MockTypeSafeClient(favour(option_key(options_[1]), p=0.9))
    agent = JevFastAgent(client=teacher, student_client=sure_student)
    assert (await agent.act(battle)).message == options_[0].order.message
    assert len(sure_student.calls) == 1 and not teacher.calls
    unsure_student = ts.MockTypeSafeClient(favour(option_key(options_[0]), p=0.15))
    agent = JevFastAgent(client=teacher, student_client=unsure_student)
    battle._replay_data.append(["", "turn", "99"])                 # a new request key
    assert (await agent.act(battle)).message == options_[1].order.message
    assert len(unsure_student.calls) == 1 and len(teacher.calls) == 1
    assert agent.metrics["student_requests"] == 1 and agent.metrics["teacher_accepted"] == 1


# --- dataset ------------------------------------------------------------------------------------
def test_dataset_soft_labels_s2_mixing_weights_and_split():
    q_stay = ts.noul("stay?")
    q_move = ts.choice("move?", {"move_a": None, "move_b": None})
    recs = [
        {"type": "s1", "backend": "typesafe:jev-1.13.0", "role": "teacher", "purpose": "action", "agent": "x",
         "battle_tag": "b1", "decision_id": "b1:1", "state": {"s": 1}, "model": "jev-1.13.0",
         "questions": {"stay": q_stay, "best_move": q_move},
         "answers": {"stay": {"type": "noul", "noul": 0.7},
                     "best_move": {"type": "choice", "probabilities": {"move_a": 0.8, "move_b": 0.2}}}},
        {"type": "s1", "backend": "local:student", "role": "student", "purpose": "action", "agent": "x",
         "battle_tag": "b1", "decision_id": "b1:1", "state": {}, "questions": {"stay": q_stay},
         "answers": {"stay": {"type": "noul", "noul": 0.1}}},
        {"type": "decision", "agent": "x", "decision_id": "b1:1", "route": "slow", "s2_key": "move_b"},
        {"type": "outcome", "battle_tag": "b1", "won": False},
    ]
    ex = ds.build(recs, ds.BuildConfig(s2_weight=0.5))
    assert len(ex) == 2 and {e["qid"] for e in ex} == {"stay", "best_move"}       # student answers are no labels
    move = next(e for e in ex if e["qid"] == "best_move")
    assert move["target"] == [pytest.approx(0.4), pytest.approx(0.6)] and move["source"] == "teacher+s2"
    stay = next(e for e in ex if e["qid"] == "stay")
    assert stay["target"] == [pytest.approx(0.85), pytest.approx(0.15)]           # S2 played a move: stay = true
    assert all(e["weight"] == 0.5 for e in ex) and len({e["split"] for e in ex}) == 1
    assert ds.describe(ex)["by_source"] == {"teacher+s2": 2}


# --- evaluation + registry --------------------------------------------------------------------------
async def test_evaluate_against_teacher_labels():
    q = ts.choice("move?", {"a": None, "b": None, "c": None})
    exs = [{"group": "g", "request": 0, "qid": "q", "question": q, "keys": ["a", "b", "c"], "state": {},
            "purpose": "action", "target": [0.7, 0.2, 0.1]}]
    good = ts.MockTypeSafeClient(lambda s, qs: {"q": ts.uniform_choice(q["criteria"], "a", 0.7)})
    bad = ts.MockTypeSafeClient(lambda s, qs: {"q": ts.uniform_choice(q["criteria"], "c", 0.9)})
    g, b = await evaluate(good, exs), await evaluate(bad, exs)
    assert g["overall"]["agree"] == 1.0 and b["overall"]["agree"] == 0.0 and g["overall"]["kl"] < b["overall"]["kl"]
    assert g["by_purpose"]["action"]["n"] == 1


def test_registry_promotion(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.add({"id": "s1-v1", "metrics": {"valid": {"overall": {"agree": 0.6, "ece": 0.1}},
                                        "base_valid": {"overall": {"agree": 0.3}}}})
    ok, why = decide_promotion(reg.get("s1-v1"), reg.champion())
    assert ok, why
    reg.promote("s1-v1", why)
    reg.add({"id": "s1-v2", "metrics": {"valid": {"overall": {"agree": 0.605, "ece": 0.1}}}})
    assert not decide_promotion(reg.get("s1-v2"), reg.champion())[0]          # below min gain
    reg.add({"id": "s1-v3", "metrics": {"valid": {"overall": {"agree": 0.7, "ece": 0.09}}}})
    ok, _ = decide_promotion(reg.get("s1-v3"), reg.champion())
    reg.promote("s1-v3", "better")
    assert ok and Registry(tmp_path / "registry.json").champion()["id"] == "s1-v3"
    assert reg.get("s1-v1")["status"] == "retired"


# --- the local server speaks the TypeSafe protocol ------------------------------------------------------
class FakeScorer:
    model_id = "local:fake"

    def answer(self, state, questions):
        return ({qid: to_answer(q, [k for k, _ in options(q)], [1.0] + [0.0] * (len(options(q)) - 1))
                 for qid, q in questions.items()}, 7)


async def test_server_is_typesafe_compatible(battle):
    server = TestServer(make_app(FakeScorer()))
    await server.start_server()
    try:
        client = ts.TypeSafeClient(base_url=str(server.make_url("")).rstrip("/"), api_key="local")
        res = await client.system_one({"s": 1}, {"q": ts.choice("?", {"x": None, "y": None}), "n": ts.noul("?")})
        assert res.model == "local:fake" and res.answers["q"].choice == "x" and res.answers["n"].noul == 1.0
        assert res.input_tokens == 7
        models = await client.list_models()
        assert models[0]["name"] == "local:fake"
        # the same client drives the decider: a student is a drop-in System One
        from arena.agents.system1.decider import FastDecider
        fd = await FastDecider(client).decide(battle, legal_options(battle))
        assert fd.key in {f.key for f in fd.facts}
        await client.aclose()
    finally:
        await server.close()
    assert "type" in validate({"state": {}, "questions": {"q": {"type": "box"}}})


# --- deep fusion ------------------------------------------------------------------------------------
def scripted_llm(seen, answer="<analysis>ok</analysis><answer>3</answer><confidence>0.7</confidence>"):
    async def reply(messages, tools, idx):
        seen.append(messages)
        sys = messages[0]["content"]
        if sys.startswith(PLANNER_SYS) and "state handover" in messages[-1]["content"]:
            return "<summary>## 1. Objective: win</summary>"
        if sys.startswith(PLANNER_SYS):
            last = messages[-1]["content"]
            if "FINAL STEP" in last:
                return answer
            return "<analysis>need data</analysis><task>damage of my moves against the opponent</task>"
        if sys.startswith("You are the MEMORY"):
            return "{}"
        return "<results>\nFINDINGS: llm executor\nSTATUS: partial\n</results>"
    return reply


def make_deep(seen, jev_reply, **params):
    ctx = AgentContext(llm=MockLLMClient(scripted_llm(seen)))
    params.setdefault("min_confidence", 0.99)            # always escalate to System 2
    return HyMemDualAgent(ctx, model="mock", supports_tools=False, jev_client=ts.MockTypeSafeClient(jev_reply),
                          **params)


def controller_answers(step="answer_now", route="lookup", use=("damage_calc",)):
    def extra(qid, q):
        if qid == "step":
            return ts.uniform_choice(q["criteria"], step, 0.8)
        if qid == "route":
            return ts.uniform_choice(q["criteria"], route, 0.9)
        if qid.startswith("use_"):
            return {"type": "noul", "noul": 0.9 if qid[4:] in use else 0.05}
        if qid.startswith("arg_"):
            return ts.uniform_choice(q["criteria"], "default", 0.9)
        return {"type": "noul", "noul": 0.5}
    return extra


def planner_prompts(seen):
    return [m[-1]["content"] for m in seen if m[0]["content"].startswith(PLANNER_SYS)]


async def test_step_control_makes_the_first_planner_step_answer_only(battle):
    seen = []
    agent = make_deep(seen, favour(p=0.3, extra=controller_answers("answer_now")), step_control=True,
                      shared_facts=True)
    order = await agent.act(battle)
    prompts = planner_prompts(seen)
    assert len(prompts) == 1 and "FINAL STEP" in prompts[0]                   # one short planner call
    assert "[system1_facts]" in prompts[0] and "[system1_suggestion]" in prompts[0]
    assert order.message == legal_options(battle)[2].order.message
    assert agent.metrics["step_answer_now"] == 1


async def test_s1_router_and_executor_replace_the_executor_llm(battle):
    seen = []
    agent = make_deep(seen, favour(p=0.3, extra=controller_answers("research", "lookup", ("damage_calc",))),
                      step_control=True, s1_router=True, s1_executor=True, planner_steps=2)
    await agent.act(battle)
    spaces = [m[0]["content"][:22] for m in seen]
    assert not any(s.startswith("You are the ANALYST") for s in spaces)     # no executor LLM call
    prompts = planner_prompts(seen)
    assert len(prompts) == 2 and "<RESULTS" in prompts[1] and "[damage_calc" in prompts[1]
    tools = [c["tool"] for c in next(iter(agent.sessions._s.values())).store.tool_calls]
    assert tools == ["damage_calc"]
    calls = agent.jev.calls
    assert any("route" in c["questions"] for c in calls) and any("use_damage_calc" in c["questions"] for c in calls)


async def test_typed_folds_skip_generative_extraction():
    calls = []

    async def call(space, messages):
        calls.append(space)
        from arena.llm.client import LLMResponse
        return LLMResponse(text="<summary>## 1. Objective: KO Garganacl</summary>")
    st = HyMemStore(key_facts=[KeyFact(2, "faint", "opp Luvdisc fainted")])
    light = MemoryManager(HyMemConfig.from_params(fold_mode="typed_light"))
    out = await light.fold(st, call=call, reason="r", task="t", summary_messages=[{"role": "user", "content": "x"}],
                           history_text="")
    assert calls == [] and st.folds == 1 and st.episode and out.mode == "typed_light"
    typed = MemoryManager(HyMemConfig.from_params(fold_mode="typed"))
    await typed.fold(st, call=call, reason="r", task="t", summary_messages=[{"role": "user", "content": "x"}],
                     history_text="")
    assert calls == ["fold_summary"] and "KO Garganacl" in st.summary and st.folds == 2
