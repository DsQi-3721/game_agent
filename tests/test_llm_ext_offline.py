"""Platform plumbing (offline): LLM client knobs for local servers, reasoning
handling, cache keys, tool-protocol hints, logging extras, the battle-end hook."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arena.agents.adapter import AgentPlayer
from arena.agents.base import Agent
from arena.eval.tournament import merge_extra_metrics
from arena.llm.cache import ResponseCache
from arena.llm.client import LiteLLMClient, LLMResponse, MockLLMClient, deep_merge, split_think
from arena.llm.model_registry import ModelRegistry, ModelSpec
from arena.llm.tool_calling import TextToolProtocol, make_protocol
from arena.logging.calls import CallLogger
from arena.logging.trajectory import TrajectoryLogger

FIXTURE = Path(__file__).parent / "fixtures" / "battles" / "gen9random_midbattle.pkl"


@pytest.fixture(scope="module")
def battle():
    with open(FIXTURE, "rb") as f:
        return pickle.load(f)


def _fake_resp(content="ok", reasoning=None, usage=(11, 7)):
    msg = NS(content=content, tool_calls=[], reasoning_content=reasoning)
    return NS(choices=[NS(message=msg, finish_reason="stop")],
              usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1], prompt_tokens_details=None))


@pytest.fixture
def captured(monkeypatch):
    import litellm

    calls = []
    replies = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return replies.pop(0) if replies else _fake_resp()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls, replies


def _registry(monkeypatch):
    monkeypatch.setenv("X_TEST_REMOTE_KEY", "sk-test")
    monkeypatch.delenv("X_TEST_LOCAL_KEY", raising=False)
    return ModelRegistry({
        "local": ModelSpec(name="local", litellm_id="openai/default_model",
                           api_base="http://127.0.0.1:8080/v1", api_key_env="X_TEST_LOCAL_KEY",
                           extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                           max_tokens=512, timeout_s=99.0, max_retries=0),
        "remote": ModelSpec(name="remote", litellm_id="deepseek/deepseek-chat",
                            api_key_env="X_TEST_REMOTE_KEY", price_in=0.27, price_out=1.10),
    })


async def test_local_server_kwargs(monkeypatch, captured):
    calls, _ = captured
    client = LiteLLMClient(_registry(monkeypatch))
    resp = await client.complete("local", [{"role": "user", "content": "hi"}], top_p=0.9,
                                 extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    kw = calls[0]
    assert kw["api_key"] == "EMPTY"                     # dummy key for a local api_base
    assert kw["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    assert kw["max_tokens"] == 512 and kw["timeout"] == 99.0 and kw["max_retries"] == 0
    assert kw["top_p"] == 0.9
    assert resp.cost_usd == 0.0 and resp.prompt_tokens == 11


async def test_remote_spec_unchanged(monkeypatch, captured):
    calls, _ = captured
    client = LiteLLMClient(_registry(monkeypatch))
    await client.complete("remote", [{"role": "user", "content": "hi"}])
    assert set(calls[0]) == {"model", "messages", "temperature", "api_key"}


async def test_reasoning_field_and_inline_think(monkeypatch, captured):
    _, replies = captured
    client = LiteLLMClient(_registry(monkeypatch))
    replies.append(_fake_resp(content="Final.", reasoning="because"))
    r1 = await client.complete("local", [{"role": "user", "content": "q"}])
    assert (r1.text, r1.reasoning) == ("Final.", "because")
    replies.append(_fake_resp(content="mulling it over</think>\n\n<answer>1</answer>"))
    r2 = await client.complete("local", [{"role": "user", "content": "q"}])
    assert r2.text == "<answer>1</answer>" and r2.reasoning == "mulling it over"


def test_split_think_and_deep_merge():
    assert split_think("plain") == ("plain", "")
    assert split_think("<think>a</think>b") == ("b", "a")
    base = {"a": {"x": 1, "y": 2}, "k": 1}
    assert deep_merge(base, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}, "k": 1}
    assert base == {"a": {"x": 1, "y": 2}, "k": 1}  # not mutated


def test_cache_key_backward_compatible():
    msgs = [{"role": "user", "content": "x"}]
    old = ResponseCache.key("m", msgs, None, 0)
    assert old == ResponseCache.key("m", msgs, None, 0, extra=None)
    assert old != ResponseCache.key("m", msgs, None, 0, extra={"extra_body": {"t": 1}})


async def test_mock_records_kwargs_and_async_responder():
    async def responder(messages, tools, idx):
        return "thought</think>Action: x"

    mock = MockLLMClient(responder)
    r = await mock.complete("m", [{"role": "user", "content": "hi"}], max_tokens=5,
                            extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    assert r.text == "Action: x" and r.reasoning == "thought"
    assert mock.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert mock.calls[0]["max_tokens"] == 5


def test_text_protocol_hints():
    default = make_protocol(False)
    assert isinstance(default, TextToolProtocol)
    msgs: list[dict] = []
    default.append_turn(msgs, LLMResponse(text="t"), [], [])
    assert "give your Action" in msgs[-1]["content"]
    custom = make_protocol(False, final_hint="FINISH WITH <results>", continue_hint="GO ON")
    msgs = []
    custom.append_turn(msgs, LLMResponse(text="t"), [], [])
    assert msgs[-1]["content"].endswith("GO ON")


def test_trajectory_extra_and_call_logger(tmp_path):
    traj = TrajectoryLogger(tmp_path / "traj_a.jsonl", "a")
    traj.log_turn(battle_tag="b", turn=1, state_text="", prompt="", raw_llm_output="",
                  extra={"planner_prompt_tokens": 1234})
    traj.log_turn(battle_tag="b", turn=2, state_text="", prompt="", raw_llm_output="")
    traj.close()
    recs = [json.loads(x) for x in (tmp_path / "traj_a.jsonl").read_text().splitlines()]
    assert recs[0]["extra"] == {"planner_prompt_tokens": 1234} and "extra" not in recs[1]

    full = CallLogger.beside(traj)
    assert full.path.name == "calls_a.jsonl"
    full.log(battle_tag="b", decision=1, space="planner", messages=[{"role": "user", "content": "hey"}],
             output="<answer>1</answer>", prompt_tokens=3)
    meta = CallLogger(tmp_path / "meta.jsonl", "a", level="meta")
    meta.log(battle_tag="b", decision=1, space="executor", messages=[{"role": "user", "content": "x"}], output="y")
    assert CallLogger.beside(traj, level="off") is None
    rec_full = json.loads(full.path.read_text().splitlines()[0])
    rec_meta = json.loads((tmp_path / "meta.jsonl").read_text().splitlines()[0])
    assert rec_full["messages"][0]["content"] == "hey" and rec_full["space"] == "planner"
    assert "messages" not in rec_meta and rec_meta["prompt_chars"] == 1


def test_battle_end_hook_forwarded_and_isolated(battle):
    class Hooked(Agent):
        def __init__(self):
            super().__init__()
            self.ended = []

        async def act(self, b):  # pragma: no cover - not exercised
            raise NotImplementedError

        def on_battle_end(self, b):
            self.ended.append(b.battle_tag)
            raise RuntimeError("hook bug must not reach poke-env")

    player = AgentPlayer.__new__(AgentPlayer)  # no network: only the hook path is tested
    player.agent = Hooked()
    player._battle_finished_callback(battle)
    assert player.agent.ended == [battle.battle_tag]


def test_registry_yaml_has_local_qwen():
    reg = ModelRegistry.load()
    spec = reg.get("qwen3.5-4b-mlx")
    assert spec.litellm_id == "openai/default_model" and spec.max_retries == 0
    assert spec.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert spec.context_window > 0 and reg.get("qwen3.5-4b-vllm").supports_tools


def test_merge_extra_metrics():
    tot: dict = {}
    merge_extra_metrics(tot, {"n_folds": 2, "planner_ctx_tokens_max": 900, "flag": True, "s": "x"})
    merge_extra_metrics(tot, {"n_folds": 3, "planner_ctx_tokens_max": 700})
    assert tot == {"n_folds": 5, "planner_ctx_tokens_max": 900}
