"""Long-context baselines (offline): flat ReAct grows / slides, Summary compresses."""

import copy
import pickle
from pathlib import Path

import pytest

from arena.agents.base import AgentContext
from arena.agents.llm.flat_react import FlatReActAgent
from arena.agents.llm.summary_ctx import SummaryAgent
from arena.knowledge import store
from arena.llm.client import MockLLMClient
from arena.registry.agents import entries

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"
KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge_mini"


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


def advance(battle, n):
    battle._replay_data.extend([["", "move", "p2a: Garganacl", "Salt Cure", "p1a: Passimian"],
                                ["", "-damage", "p1a: Passimian", f"{300 - 10 * n}/302"], ["", "turn", str(n)]])
    battle.turn = n


def legal(b):
    return {o.message for o in b.valid_orders}


async def test_flat_context_grows_with_raw_tool_output(battle):
    calls = {"n": 0}

    def responder(messages, tools, idx):
        calls["n"] += 1
        if calls["n"] == 1:
            return 'checking\nTool: damage_calc {"move": "close combat"}'
        return "<analysis>go</analysis><answer>2</answer><confidence>0.7</confidence>"

    agent = FlatReActAgent(AgentContext(llm=MockLLMClient(responder)), model="mock", supports_tools=False)
    sizes = []
    for n in range(2, 6):
        order = await agent.act(battle)
        assert order.message in legal(battle)
        sizes.append(len(agent.sessions[battle.battle_tag].messages))
        advance(battle, n)
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]          # monotone growth
    convo = "\n".join(str(m["content"]) for m in agent.sessions[battle.battle_tag].messages)
    assert "assumes Leftovers" in convo                              # raw tool output kept verbatim
    assert "|move|p2a: Garganacl|Salt Cure|p1a: Passimian" in convo  # raw protocol events kept
    assert agent.extra_metrics()["ctx_tokens_max"] > 0


async def test_flat_sliding_window(battle):
    agent = FlatReActAgent(AgentContext(llm=MockLLMClient(lambda m, t, i: "<answer>1</answer>")),
                           model="mock", supports_tools=False, max_context_tokens=900)
    for n in range(2, 9):
        await agent.act(battle)
        advance(battle, n)
    sess = agent.sessions[battle.battle_tag]
    assert sess.truncated_blocks > 0 and sess.messages[0]["role"] == "system"
    assert sess.messages[1]["role"] == "user"


async def test_summary_agent_compresses_old_blocks(battle):
    def responder(messages, tools, idx):
        if messages[-1]["content"].startswith("Summarize this Pokemon battle"):
            return "- opp Garganacl has Salt Cure\n- plan: Close Combat"
        return "<answer>2</answer>"

    agent = SummaryAgent(AgentContext(llm=MockLLMClient(responder)), model="mock", supports_tools=False,
                         summary_threshold=1200, keep_last=2)
    for n in range(2, 8):
        assert (await agent.act(battle)).message in legal(battle)
        advance(battle, n)
    sess = agent.sessions[battle.battle_tag]
    assert sess.summaries >= 1 and "Summary of earlier turns" in sess.messages[0]["content"]
    assert "opp Garganacl has Salt Cure" in sess.messages[0]["content"]
    assert len(sess.blocks) <= 2 + sess.summaries + 1


def test_baselines_registered():
    reg = entries()
    assert {"llm_flat_react", "llm_summary", "llm_hymem"} <= set(reg)
