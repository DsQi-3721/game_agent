"""scripts/analyze_latency.py helpers and an end-to-end run on synthetic logs (offline)."""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_latency.py"
spec = importlib.util.spec_from_file_location("analyze_latency", SCRIPT)
al = importlib.util.module_from_spec(spec)
spec.loader.exec_module(al)


def test_sweep_equal_share_and_rounds():
    segs = [(0.0, 10.0, "a"), (5.0, 15.0, "b"), (20.0, 22.0, "c")]
    attr, covered = al.sweep(segs, 0.0, 25.0)
    assert covered == 17.0
    assert attr == {"a": 7.5, "b": 7.5, "c": 2.0}
    assert al.rounds([(0, 10), (5, 15), (20, 22)]) == 2
    assert al.rounds([(0, 1), (1, 2), (2, 3)]) == 3


def test_timer_replay_ladder_vs_challenge():
    rows = [{"idx": i + 1, "turn": i + 1, "latency_s": 60.0} for i in range(6)]
    lad = al.simulate_timer(rows, al.LADDER)
    # bank 210 -> 150 (cap) after turn 1 ... drains 60 - 10 per turn
    assert lad["timeout_at_decision"] is not None and lad["timeout_at_turn"] <= 4
    ch = al.simulate_timer([{"idx": 1, "turn": 1, "latency_s": 290.0}], al.CHALLENGE)
    assert ch["timeout_at_decision"] is None
    assert al.simulate_timer([{"idx": 1, "turn": 1, "latency_s": 151.0}], al.LADDER)["timeout_at_decision"] == 1
    assert al.simulate_timer([{"idx": 1, "turn": 1, "latency_s": 149.0}], al.LADDER)["timeout_at_decision"] is None


def _ts(epoch: float) -> str:
    d = datetime.fromtimestamp(epoch)
    return d.strftime("%Y-%m-%d %H:%M:%S") + f",{int((epoch % 1) * 1000):03d}"


def test_end_to_end_with_server_log(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    t0 = 1_800_000_000.0
    calls = [
        {"battle_tag": "b", "decision": 1, "space": "planner", "t_start": t0, "ts": t0 + 10, "latency_s": 10.0,
         "prompt_tokens": 1500, "cached_tokens": 0, "completion_tokens": 80, "finish_reason": "stop",
         "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]},
        {"battle_tag": "b", "decision": 1, "space": "executor", "t_start": t0 + 10, "ts": t0 + 20, "latency_s": 10.0,
         "prompt_tokens": 1600, "cached_tokens": 1000, "completion_tokens": 90, "finish_reason": "tool_calls",
         "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Task: damage\nx"}]},
        {"battle_tag": "b", "decision": 1, "space": "event:tool", "ts": t0 + 20, "latency_s": 0,
         "extra": {"tool": "damage_calc", "t_start": t0 + 20, "latency_s": 0.01}},
        {"battle_tag": "b", "decision": 1, "space": "executor", "t_start": t0 + 20.01, "ts": t0 + 30, "latency_s": 9.99,
         "prompt_tokens": 900, "cached_tokens": 0, "completion_tokens": 100, "finish_reason": "stop",
         "output": "<results>\nFINDINGS: 2HKO\n</results>",
         "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Task: damage\nx"}]},
    ]
    (run / "calls_hymem.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    traj = [{"battle_tag": "b", "turn": 1, "parsed_order": "/choose move x", "fallback": False,
             "extra": {"decision": 1, "latency_s": 30.5, "t_start": t0, "t_end": t0 + 30.5, "note": ""}}]
    (run / "traj_hymem.jsonl").write_text(json.dumps(traj[0]) + "\n")
    log = tmp_path / "server.log"
    log.write_text("\n".join([
        f"{_ts(t0 + 4.0)} - INFO - Prompt processing progress: 1500/1500",
        f"{_ts(t0 + 11.5)} - INFO - Prompt processing progress: 600/600",
        f"{_ts(t0 + 22.5)} - INFO - Prompt processing progress: 900/900",
    ]) + "\n")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["analyze_latency.py", str(run), "--server-log", str(log), "--out", str(out)])
    assert al.main() == 0
    rep = json.loads((out / "latency_summary.json").read_text())
    assert rep["n_llm_decisions"] == 1 and abs(rep["decision_latency"]["mean"] - 30.5) < 1e-6
    attr = rep["attribution_s"]
    assert abs(attr["planner:prefill"] - 4.0) < 0.01 and abs(attr["planner:decode"] - 6.0) < 0.01
    assert abs(attr["executor:prefill"] - (1.5 + 2.49)) < 0.02
    assert abs(sum(attr.values()) - 30.5) < 1e-6                      # attribution sums to the wall time
    assert rep["per_space"]["executor"]["prefill_from_server_log"] == 1.0
    assert rep["executor_sessions"]["n"] == 1 and rep["executor_sessions"]["tool_turns_mean"] == 1
    assert rep["structure"]["sequential_rounds_per_decision"] == 3
    assert rep["counterfactual_mean_s"]["rule_results"] < 30.5
    assert rep["executor_calls"]["tool_turns"] == 1 and rep["executor_calls"]["results_calls"] == 1
    assert (out / "latency_tables.md").exists()
