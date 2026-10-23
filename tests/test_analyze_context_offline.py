"""scripts/analyze_context.py on a synthetic run (offline)."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_context.py"
spec = importlib.util.spec_from_file_location("analyze_context", SCRIPT)
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_fact_signature():
    assert ac.fact_signature("opp Garganacl has Salt Cure") == ["garganacl", "saltcure"]
    assert ac.fact_signature("opp Gholdengo holds Choice Scarf") == ["gholdengo", "choicescarf"]
    assert ac.fact_signature("your Hitmontop fainted (turn 4)") == ["hitmontop", "fainted"]
    assert ac.fact_signature("opp side has Stealth Rock") == ["stealthrock"]


def test_summary_and_dilution_export(tmp_path, monkeypatch):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"agents": {"hymem": {"win_rate": 1.0, "fallback_rate": 0.0}}}))
    ctx = "opp Garganacl has Salt Cure ... rest of the planner context"
    _write(run / "calls_hymem.jsonl", [
        {"battle_tag": "b1", "decision": 1, "space": "planner", "prompt_tokens": 1500,
         "messages": [{"role": "user", "content": ctx}]},
        {"battle_tag": "b1", "decision": 1, "space": "executor", "prompt_tokens": 1200},
        {"battle_tag": "b1", "decision": 2, "space": "planner", "prompt_tokens": 1700,
         "messages": [{"role": "user", "content": "nothing relevant"}]},
        {"battle_tag": "b1", "decision": 2, "space": "event:fold", "extra": {"reason": "planner_turns_threshold"}},
        {"battle_tag": "b1", "decision": 2, "space": "event:battle_end",
         "extra": {"won": True, "turns": 2, "key_facts": [{"turn": 1, "kind": "opp_move",
                                                          "text": "opp Garganacl has Salt Cure"}]}},
    ])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["analyze_context.py", str(run), "--out", str(out), "--export-dilution"])
    assert ac.main() == 0
    rows = json.loads((out / "summary.json").read_text())
    row = rows[0]
    assert row["agent"] == "hymem" and row["decisions"] == 2 and row["main_ctx_max"] == 1700
    assert row["total_prompt_per_decision"] == round((2700 + 1700) / 2) and row["folds_per_battle"] == 1
    assert row["kfsr_mean"] == 0.5        # fact present at decision 1, gone at decision 2
    trace = json.loads((out / "dilution" / "run1" / "hymem" / "b1.json").read_text())
    assert len(trace["stats"]["planner_context_per_turn"]) == 2
    assert (out / "dilution" / "key_facts.jsonl").exists()
