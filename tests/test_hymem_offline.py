"""HyMem agent (offline, scripted mock LLM routed by context space).

The central invariant under test is the paper's information boundary (Eq. 4):
raw tool output and isolated-reasoning deliberation never reach the planner
context; only typed returns (<TURN_RESULT>, <RESULTS>, <RETURN>, <DECISION>) do.
"""

import asyncio
import copy
import json
import pickle
from pathlib import Path

import pytest

from arena.agents.base import AgentContext
from arena.agents.hymem.agent import HyMemAgent
from arena.knowledge import store
from arena.llm.client import LLMResponse, MockLLMClient, ToolCall
from arena.logging.trajectory import TrajectoryLogger
from arena.state.views import legal_options

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"
KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge_mini"

MEMORY_EPISODE = json.dumps({"task_description": "beat the opponent", "key_events": [
    {"turn": 1, "description": "Garganacl used Salt Cure", "outcome": "Passimian is salted"}],
    "verified_facts": [{"fact": "opp Garganacl has Salt Cure", "turn": 1, "confidence": "high"}],
    "current_progress": "6 vs 6"})
MEMORY_WORKING = json.dumps({"immediate_goal": "KO Garganacl", "current_challenges": "Salt Cure chip",
                             "active_hypotheses": ["Garganacl is Bulky Support"], "failed_approaches": [],
                             "next_actions": [{"type": "attack", "description": "Close Combat", "rationale": "2HKO"}]})
MEMORY_TOOL = json.dumps({"effective_strategies": ["matchup first"], "dead_ends": [],
                          "derived_rules": ["use damage_calc for exact numbers"], "useful_sources": ["matchup"]})


def memory_by_module(episode="{}", working="{}", tool="{}"):
    """Memory replies keyed by which extraction prompt is being answered."""
    def reply(messages):
        p = messages[-1]["content"]
        return episode if "EPISODE memory" in p else working if "WORKING memory" in p else tool
    return reply


class Script:
    """Responder: per-space reply queues (the last reply repeats)."""

    def __init__(self, **queues):
        self.q = {k: list(v) for k, v in queues.items()}
        self.log: list[tuple[str, list]] = []

    @staticmethod
    def space(messages):
        sys = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user = messages[-1]["content"] if messages else ""
        if sys.startswith("You are the PLANNER"):
            return "summary" if "state handover" in messages[1]["content"] else "planner"
        if sys.startswith("You are the ANALYST"):
            return "executor"
        if sys.startswith("You are an ISOLATED"):
            return "iso"
        if sys.startswith("You are the MEMORY"):
            return "memory"
        if user.startswith("Classify one subtask"):
            return "router"
        if user.startswith("Compress the following"):
            return "distill"
        return "unknown"

    DEFAULTS = {"planner": "<analysis>ok</analysis><answer>1</answer><confidence>0.9</confidence>",
                "executor": "<results>\nFINDINGS: nothing\nSTATUS: partial\n</results>",
                "iso": "<return>\nCONCLUSION: attack\nCONFIDENCE: 0.6\n</return>",
                "memory": "{}", "summary": "<summary>## 1. Objective: win</summary>",
                "router": '{"complexity": "search", "confidence": 0.9}', "distill": "short"}

    def __call__(self, messages, tools, idx):
        sp = self.space(messages)
        self.log.append((sp, [dict(m) for m in messages]))
        q = self.q.get(sp)
        if q:
            item = q.pop(0) if len(q) > 1 else q[0]
        else:
            item = self.DEFAULTS.get(sp, "")
        return item(messages) if callable(item) else item

    def prompts(self, space):
        return [msgs[-1]["content"] for sp, msgs in self.log if sp == space]

    def count(self, space):
        return sum(1 for sp, _ in self.log if sp == space)


@pytest.fixture(autouse=True)
def mini_knowledge(monkeypatch):
    monkeypatch.setenv("ARENA_KNOWLEDGE_DIR", str(KNOWLEDGE))
    store.clear_cache()
    yield
    store.clear_cache()


@pytest.fixture
def battle():
    with open(FIXTURE, "rb") as f:
        return copy.deepcopy(pickle.load(f))


def make(script, tmp_path=None, **params) -> HyMemAgent:
    logger = TrajectoryLogger(tmp_path / "traj_hymem.jsonl", "hymem") if tmp_path else None
    ctx = AgentContext(llm=MockLLMClient(script), logger=logger)
    params.setdefault("supports_tools", False)
    return HyMemAgent(ctx, model="mock", **params)


def next_turn(battle, n, *events):
    """Advance the fixture battle by one turn (new protocol events -> a new decision)."""
    battle._replay_data.extend([list(e) for e in events] + [["", "turn", str(n)]])
    battle.turn = n


def legal(battle):
    return {o.message for o in battle.valid_orders}


# --- basic flow -----------------------------------------------------------------------
async def test_direct_answer_and_logging(battle, tmp_path):
    s = Script(planner=["<analysis>CC is strong</analysis><answer>2</answer><confidence>0.8</confidence>"])
    agent = make(s, tmp_path)
    order = await agent.act(battle)
    assert order.message == legal_options(battle)[1].order.message
    assert s.count("planner") == 1 and agent.n_fallbacks == 0
    rec = json.loads((tmp_path / "traj_hymem.jsonl").read_text().splitlines()[0])
    assert rec["extra"]["planner_calls"] == 1 and rec["extra"]["confidence"] == 0.8
    calls = [json.loads(x) for x in (tmp_path / "calls_hymem.jsonl").read_text().splitlines()]
    assert calls[0]["space"] == "planner" and "Legal actions" in calls[0]["messages"][1]["content"]
    first_prompt = s.prompts("planner")[0]
    assert "<TURN_RESULT" in first_prompt and "your sent out Passimian" in first_prompt
    assert "Battle brief" in s.log[0][1][0]["content"]          # brief lives in the system prompt


async def test_task_results_but_no_raw_tool_output_in_planner(battle):
    s = Script(planner=["<analysis>check</analysis><task>damage of closecombat on garganacl</task>",
                        "<analysis>2HKO</analysis><answer>2</answer><confidence>0.9</confidence>"],
               executor=['Tool: damage_calc {"move": "close combat"}',
                         "<results>\nFINDINGS: closecombat does 72-85% (2HKO)\nSTATUS: success\n</results>"])
    agent = make(s)
    order = await agent.act(battle)
    assert order.message in legal(battle) and agent.n_fallbacks == 0
    second = s.prompts("planner")[1]
    assert "<RESULTS" in second and "72-85% (2HKO)" in second
    raw_marker = "assumes Leftovers"                              # only in the raw damage_calc output
    executor_msgs = [m["content"] for sp, msgs in s.log if sp == "executor" for m in msgs]
    assert any(raw_marker in c for c in executor_msgs)
    assert all(raw_marker not in p for p in s.prompts("planner"))
    assert agent.n_tool_calls == 1 and agent.metrics["exec_calls"] == 1


async def test_native_tool_protocol(battle):
    s = Script(planner=["<task>full matchup vs garganacl</task>", "<answer>2</answer>"],
               executor=[LLMResponse(text="", tool_calls=[ToolCall("matchup", {}, "c1")]),
                         "<results>\nFINDINGS: I move first\nSTATUS: success\n</results>"])
    agent = make(s, supports_tools=True)
    assert (await agent.act(battle)).message in legal(battle)
    native_call = next(msgs for sp, msgs in s.log if sp == "executor" and any(m["role"] == "tool" for m in msgs))
    assert any("== your passimian" in str(m.get("content")) for m in native_call)


async def test_iso_private_reasoning_never_reaches_planner(battle):
    s = Script(planner=["<isoReason><description>predict opp</description>"
                        "<prompt>Will Garganacl Recover this turn?</prompt></isoReason>",
                        "<answer>2</answer><confidence>0.7</confidence>"],
               iso=["<analysis>SECRET-XYZZY deliberation</analysis><task>likely set of garganacl</task>",
                    "<return>\nCONCLUSION: attack now with Close Combat\nKEY EVIDENCE: 2HKO\nCONFIDENCE: 0.7\n</return>"],
               executor=["Tool: opponent_sets {}", "<results>\nFINDINGS: Recover 100%\nSTATUS: success\n</results>"])
    agent = make(s)
    assert (await agent.act(battle)).message in legal(battle)
    assert s.count("iso") == 2 and s.count("executor") == 2 and agent.metrics["iso_calls"] == 1
    planner_after = s.prompts("planner")[1]
    assert "<RETURN" in planner_after and "attack now with Close Combat" in planner_after
    assert all("SECRET-XYZZY" not in p for p in s.prompts("planner"))
    iso_second = [msgs for sp, msgs in s.log if sp == "iso"][1]
    assert any("Recover 100%" in m["content"] for m in iso_second)  # executor result went to iso only


async def test_router_promotes_complex_task(battle):
    s = Script(planner=["<task>should I set up or attack? evaluate the risk of Body Press</task>",
                        "<answer>2</answer>"])
    agent = make(s)
    await agent.act(battle)
    assert s.count("iso") >= 1 and s.count("executor") == 0
    assert agent.metrics["routed_to_iso"] == 1


async def test_parallel_tasks(battle):
    s = Script(planner=["<task>speed of passimian vs garganacl</task><task>likely set of garganacl</task>",
                        "<answer>2</answer>"])
    agent = make(s)
    await agent.act(battle)
    assert s.count("executor") >= 2
    assert s.prompts("planner")[1].count("<RESULTS") == 2


# --- folds & memory ---------------------------------------------------------------------
async def test_two_stage_fold_memory_and_recent_buffer(battle):
    s = Script(memory=[memory_by_module(MEMORY_EPISODE, MEMORY_WORKING, MEMORY_TOOL)],
               summary=["<summary>## 1. Objective: KO Garganacl\n## 2. Verified facts: opp Garganacl has Salt Cure (T1)</summary>"])
    agent = make(s, n_turn=2, min_turns_between_folds=0, recent_buffer=2)
    for n in range(2, 6):
        await agent.act(battle)
        next_turn(battle, n, ["", "move", "p2a: Garganacl", "Salt Cure", "p1a: Passimian"],
                  ["", "-damage", "p1a: Passimian", f"{300 - 20 * n}/302"])
    sess = agent.sessions.get(battle.battle_tag)
    assert sess.store.folds >= 1 and s.count("summary") >= 1 and s.count("memory") >= 3
    ev = sess.store.fold_events[0]
    assert ev["reason"] == "planner_turns_threshold" and ev["summary_ok"] and ev["extracted"]["episode"]
    assert sess.store.working["immediate_goal"] == "KO Garganacl"
    assert "derived_rules" in sess.store.tool
    last_planner = s.prompts("planner")[-1]
    assert "### Battle state summary" in last_planner and "KO Garganacl" in last_planner
    assert ev["entries_after"] <= 2 + 3                      # recent buffer + this decision's entries


async def test_asymmetric_injection(battle):
    s = Script(memory=[memory_by_module(MEMORY_EPISODE, "{}", '{"derived_rules": ["call matchup first"]}')],
               planner=["<to_memory/>", "<task>likely set of garganacl</task>", "<answer>1</answer>"])
    agent = make(s, min_turns_between_folds=0)
    await agent.act(battle)
    sess = agent.sessions.get(battle.battle_tag)
    assert sess.store.fold_events[0]["reason"] == "active_fold"
    exec_user = next(msgs for sp, msgs in s.log if sp == "executor")[1]["content"]
    assert "Tool experience" in exec_user and "call matchup first" in exec_user
    assert "Key findings" not in exec_user and "Battle state summary" not in exec_user


async def test_bad_memory_json_falls_back(battle):
    s = Script(memory=["not json at all"], planner=["<to_memory/>", "<answer>1</answer>"])
    agent = make(s, min_turns_between_folds=0)
    battle._replay_data.extend([["", "move", "p2a: Garganacl", "Salt Cure", "p1a: Passimian"],
                                ["", "-damage", "p1a: Passimian", "250/302"]])
    order = await agent.act(battle)
    sess = agent.sessions.get(battle.battle_tag)
    assert order.message in legal(battle)
    ev = sess.store.fold_events[0]["extracted"]
    assert ev == {"episode": False, "working": False, "tool": False}
    facts = [f["fact"] for f in sess.store.episode["verified_facts"]]
    assert "opp Garganacl has Salt Cure" in facts               # rule-based episode from the battle log
    assert sess.store.working is None


# --- robustness ----------------------------------------------------------------------------
async def test_rerequest_makes_no_llm_call(battle):
    s = Script(planner=["<answer>2</answer>"])
    agent = make(s)
    first = await agent.act(battle)
    n = len(s.log)
    second = await agent.act(battle)            # same rqid + same log length = server re-request
    assert len(s.log) == n and second.message != first.message and second.message in legal(battle)
    assert agent.metrics["recalls"] == 1


async def test_single_option_skips_llm(battle):
    b = copy.deepcopy(battle)
    b._force_switch = True
    b._available_moves = []
    only = list(b.available_switches)[0]
    b._available_switches = [only]
    s = Script()
    agent = make(s)
    order = await agent.act(b)
    assert s.log == [] and order.message == f"/choose switch {only.name}"


async def test_llm_error_falls_back(battle):
    def boom(messages, tools, idx):
        raise RuntimeError("provider down")

    agent = make(boom)
    order = await agent.act(battle)
    assert order.message in legal(battle) and agent.n_fallbacks == 1 and agent.metrics["errors"] == 1


async def test_decision_timeout_falls_back(battle):
    async def slow(messages, tools, idx):
        await asyncio.sleep(5)
        return "<answer>1</answer>"

    agent = make(slow, decision_budget_s=0.05)
    order = await agent.act(battle)
    assert order.message in legal(battle) and agent.metrics["timeouts"] == 1


async def test_battle_end_during_decision(battle):
    gate = asyncio.Event()

    async def waiting(messages, tools, idx):
        await gate.wait()
        return "<answer>1</answer>"

    agent = make(waiting)
    task = asyncio.ensure_future(agent.act(battle))
    await asyncio.sleep(0.01)
    agent.on_battle_end(battle)                   # battle ends while act() awaits the LLM
    assert agent.sessions.get(battle.battle_tag) is not None
    gate.set()
    order = await task
    assert order.message in legal(battle) and agent.sessions.get(battle.battle_tag) is None


async def test_invalid_answer_then_valid(battle):
    s = Script(planner=["<answer>99</answer>", "<answer>3</answer>"])
    agent = make(s)
    order = await agent.act(battle)
    assert order.message == legal_options(battle)[2].order.message
    assert "is not a legal action" in s.prompts("planner")[1]


async def test_last_step_is_answer_only(battle):
    s = Script(planner=["<task>speed check</task>"])
    agent = make(s, planner_steps=2)
    order = await agent.act(battle)
    assert "FINAL STEP" in s.prompts("planner")[-1] and order.message in legal(battle)


async def test_tera_answer(battle):
    s = Script(planner=["<answer>2 tera</answer>"])
    order = await make(s).act(battle)
    assert order.message.endswith("terastallize")


async def test_replan_low_confidence(battle):
    s = Script(planner=["<answer>1</answer><confidence>0.2</confidence>",
                        "<answer>2</answer><confidence>0.9</confidence>"])
    agent = make(s, replan=True)
    order = await agent.act(battle)
    assert "is low" in s.prompts("planner")[1] and order.message == legal_options(battle)[1].order.message
    assert agent.metrics["replans"] == 1


# --- ablations ------------------------------------------------------------------------------
async def test_ablation_no_iso_demotes_to_executor(battle):
    s = Script(planner=["<isoReason><description>x</description><prompt>predict the opponent</prompt></isoReason>",
                        "<answer>1</answer>"])
    agent = make(s, ablation="no_iso")
    await agent.act(battle)
    assert s.count("iso") == 0 and s.count("executor") >= 1
    assert "<isoReason>" not in s.prompts("planner")[0].split("## Legal actions")[1]  # not offered


async def test_ablation_no_mm_never_folds(battle):
    s = Script(planner=["<to_memory/>", "<answer>1</answer>"])
    agent = make(s, ablation="no_mm", n_turn=1, min_turns_between_folds=0)
    for n in range(2, 5):
        await agent.act(battle)
        next_turn(battle, n, ["", "move", "p2a: Garganacl", "Recover", "p2a: Garganacl"])
    assert s.count("memory") == 0 and s.count("summary") == 0
    assert agent.sessions.get(battle.battle_tag).store.folds == 0


async def test_ablation_no_typed_isolation_leaks_raw_output(battle):
    s = Script(planner=["<task>damage of closecombat</task>", "<answer>2</answer>"],
               executor=['Tool: damage_calc {"move": "close combat"}',
                         "<results>\nFINDINGS: 2HKO\nSTATUS: success\n</results>"])
    agent = make(s, ablation="no_typed_isolation")
    await agent.act(battle)
    assert "assumes Leftovers" in s.prompts("planner")[1]


def test_extra_metrics_and_config_validation():
    with pytest.raises(ValueError):
        make(Script(), ablation="nope")
    agent = make(Script())
    assert agent.extra_metrics() == {}


def test_truncated_memory_json_is_salvaged():
    from arena.agents.hymem.memory import norm_episode, parse_json

    cut = ('```json\n{"task_description": "win", "key_events": [{"turn": 1, "description": "KO Gyarados", '
           '"outcome": "lead"}, {"turn": 2, "descr')
    ep = norm_episode(parse_json(cut))
    assert ep["key_events"] == [{"turn": 1, "description": "KO Gyarados", "outcome": "lead"}]
    assert parse_json("no json here") is None


async def test_latency_telemetry(battle, tmp_path):
    s = Script(planner=["<task>damage of closecombat on garganacl</task>", "<answer>2</answer>"],
               executor=['Tool: damage_calc {"move": "close combat"}',
                         "<results>\nFINDINGS: 2HKO\nSTATUS: success\n</results>"])
    agent = make(s, tmp_path)
    await agent.act(battle)
    rec = json.loads((tmp_path / "traj_hymem.jsonl").read_text().splitlines()[0])["extra"]
    assert rec["t_end"] >= rec["t_start"] > 0
    assert set(rec["llm_time_by_space"]) == {"planner", "executor"} and rec["tool_time_s"] > 0
    calls = [json.loads(x) for x in (tmp_path / "calls_hymem.jsonl").read_text().splitlines()]
    llm = [c for c in calls if not c["space"].startswith("event:")]
    assert all("t_start" in c and c["t_start"] <= c["ts"] for c in llm)
    tools = [c for c in calls if c["space"] == "event:tool"]
    assert len(tools) == 1 and tools[0]["extra"]["tool"] == "damage_calc"
    assert tools[0]["extra"]["latency_s"] >= 0 and tools[0]["extra"]["parent"] == "planner"
