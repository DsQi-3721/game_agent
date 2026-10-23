# arena — Pokémon Showdown adversarial-game agent evaluation platform

A clean, pluggable platform for evaluating game-playing agents (LLM, classical
search/heuristic, and hybrids) on Pokémon Showdown. Built on
[poke-env](https://github.com/hsahovic/poke-env). The LLM agent's reasoning is the
primary research subject; the harness makes models and methods directly comparable.

## Setup

```bash
conda activate SystemofML            # has poke-env, litellm, pydantic
cd /Users/hoi/project/game/pokemonbot
pip install -e ".[dev]"              # once

# local Showdown server (separate terminal, keep running)
cd ../pokemon-showdown && node pokemon-showdown start --no-security
```

LLM keys: copy `.env.example` to `.env` and fill in (e.g. `DEEPSEEK_API_KEY`).

## Interactive TUI (POKEBOTMON)

```bash
python -m arena.cli tui     # configure → live monitor → results → trajectory/history
```
The TUI runs the eval as a subprocess and tails `runs/<name>/events.jsonl`, so it
stays smooth and the eval survives closing the UI. Charts: in-terminal always, an
interactive `runs/<name>/charts.html`, and optional W&B (`arena run ... --wandb`).
New agents auto-appear in the configurator via `@register_agent` (see below).

## Run an evaluation (CLI)

```bash
# baselines only — fast, no API cost
python -m arena.cli run configs/experiments/quick_baselines.yaml

# LLM methods (DeepSeek) with grounding — estimate cost first
python -m arena.cli run configs/experiments/grounded_deepseek.yaml --dry-run
python -m arena.cli run configs/experiments/grounded_deepseek.yaml
```

Output: a win-rate matrix + per-agent win rate (Wilson CI), Elo, fallback rate and
cost, written to `runs/<name>/report.json`, plus per-LLM-agent decision
trajectories in `runs/<name>/traj_<agent>.jsonl`.

## Concepts

- **Agent** (`agents/base.py`): `async act(battle) -> BattleOrder`, network-unaware,
  plus an optional `on_battle_end(battle)` hook. `AgentPlayer` adapts any Agent onto
  poke-env. Built-ins: baselines + `llm_io`, `llm_cot`, `llm_self_consistency`,
  `llm_react`, `llm_hymem` (below), and the long-context baselines `llm_flat_react` /
  `llm_summary`.
- **State** (`state/`): partial-observability renderer (default), `full_info` toggle;
  `views.py` (numbered legal options, battle brief, snapshot) and `battle_log.py`
  (protocol log -> per-turn digests + key facts) shared by the persistent-context agents.
- **Augmentations** (`augment/`): damage calc (both directions; hidden opponent stats
  estimated from set priors), `matchup`, `speed_check`, `opponent_sets`, `switch_eval`,
  type chart, Pokedex (with effect text), raw `battle_log`, base stats, ability -- each
  `off | passive | tool` per config. `tool` mode feeds the ReAct / HyMem tool loops.
- **Knowledge** (`knowledge/`, `data/knowledge/`): Showdown Dex text, randbats set
  distributions sampled from the server's own generator, gen9ou usage sets filtered by
  the server's rules. Rebuild with `python scripts/build_knowledge.py` (needs node).
- **Teams** (`teams/`): `scripts/ingest_teams.py` downloads Smogon sample teams;
  `scripts/prune_teams.py` validates them against the server. Modes: mirror / pool_random.
- **Eval** (`eval/`): round-robin or anchored tournament, metrics, guardrails.

## Config (YAML)

```yaml
name: my_exp
format: gen9ou                 # or gen9randombattle (teamless)
agents:
  - {name: maxpow, type: max_base_power}
  - {name: react, type: llm_react, model: deepseek-chat, params: {max_steps: 3}}
augmentations: {damage_calc: passive, type_matchup: passive, pokedex: tool}
teams: {mode: mirror}
evaluation: {mode: anchored, n_battles: 5, anchors: [maxpow]}
guardrails: {max_cost_usd: 1.0}
```

## HyMem agent (`llm_hymem`)

HyMem ("Hierarchical Context Management for Long-Horizon Agents via Information
Isolation", arXiv 2608.15703) ported to battles, following the reference
implementation (HyMem2026). A battle is the long-horizon task: the **planner** is the
only persistent context and sees only typed messages -- `<TURN_RESULT>` digests of the
protocol log, `<RESULTS>` from a fresh tool-using **executor**, `<RETURN>` from
**isolated reasoning** sessions, its own `<DECISION>`s -- plus the current state and
numbered legal actions. **Folds** (two-stage: planner `<summary>` -> three parallel
JSON extractions) rewrite episode / working / tool memory when a paper Eq. 17 trigger
fires. Raw tool output and isolated deliberation never reach the planner.

```bash
bash scripts/serve_qwen35_mlx.sh                       # local Qwen3.5-4B (Apple silicon)
python scripts/check_llm.py --model qwen3.5-4b-mlx     # 4 preflight checks
python scripts/build_knowledge.py                      # once (data/knowledge/)
python -m arena.cli run configs/experiments/hymem_smoke_randbats.yaml --emit-events
python -m arena.cli run configs/experiments/longctx_smoke.yaml          # flat / summary baselines
python scripts/analyze_context.py runs/hymem_smoke_randbats runs/longctx_smoke --export-dilution
```

- Parameters: `configs/hymem/{mac,gpu}.yaml` via `params: {config_path: ...}`, or inline
  (`planner_steps`, `execute_turns`, `ablation: full|no_iso|no_mm|no_typed_isolation`,
  `fold_mode: two_stage|direct`, `router: off|rule|llm`, `think_iso`, `replan`, `aux_model`,
  per-space knobs like `iso_max_tokens`). See `agents/hymem/config.py`.
- Logs: `traj_<agent>.jsonl` (per decision, with an `extra` block of context telemetry) and
  `calls_<agent>.jsonl` (every LLM call: space, messages, output, tokens) -- the input of
  `scripts/analyze_context.py` and a ready-made source of SFT/RL data.
- GPU server: `bash scripts/serve_qwen35_vllm.sh`, set `qwen3.5-4b-vllm.api_base`, then
  `configs/experiments/hymem_{main,ablation}_gpu_{randbats,ou}.yaml`.
- Extension points: new context spaces (planner directive registry), the
  `knowledge.opponent_model.OpponentModel` protocol, `FoldTrigger` / `Router` policies.
- Real-time analysis: `scripts/bench_llm_latency.py` (server prefill/decode/cache/concurrency),
  `configs/experiments/hymem_latency_randbats.yaml` (one battle, server log kept),
  `scripts/analyze_latency.py` (per-decision wall-clock decomposition, Showdown timer replay,
  counterfactuals) and `scripts/plot_latency_report.py`. Findings: `reports/hymem_realtime_report.md`.

## System 1: fast typed decisions with Jev (`jev_fast`, `llm_hymem_dual`)

A System One model ([TypeSafe Jev](https://docs.typesafe.ai)) answers typed questions with
calibrated probabilities in one ~0.3–0.8 s request; code computes every number first (damage
ranges and KO verdicts, move order, hazards, opponent set priors -> `agents/system1/facts.py`).

- `jev_fast`: every decision is one Jev request (best action asked in both option orders to
  cancel position bias, plus a "hard turn" Noul); routing signals are logged per decision.
- `llm_hymem_dual`: Tier 0 rule (sure first-strike KO) -> Tier 1 Jev when confident /
  order-consistent / not hard -> Tier 2 HyMem planner with the Jev ranking as a hint and a time
  cap (on timeout the Jev answer is played). `memory_router: jev` classifies, validates and
  routes HyMem's memory items per context space in the same request.
- Key: `TYPESAFE_API_KEY` in `.env` (`TYPESAFE_BASE_URL` can point at a compatible local server).
  Validation: `bash scripts/run_jev_validation.sh` -> `reports/jev_validation_results.md`;
  design and research: `reports/jev_system1_design.md`.

## System-1 learning loop (`arena.s1`)

Every typed System-One request (teacher Jev or a local student), the decision it fed and the
battle outcome are recorded (`s1exp_<agent>.jsonl`). From that experience a local student
(Qwen3.5 + LoRA) is distilled, evaluated against held-out teacher labels, registered and, when
better than the champion, promoted. The student is served over the same TypeSafe protocol and
plugged in front of the teacher (`student_url`): the cascade asks the teacher only when the
student is unsure, so every deployment collects exactly the labels the next student needs.

```bash
python -m arena.cli run configs/experiments/s1_collect_randbats.yaml          # collect (teacher plays)
python scripts/s1_loop.py --id s1-v1 --base <models>/Qwen3.5-0.8B --experience "data/s1/experience/*.jsonl"
python scripts/s1_serve.py --model <models>/Qwen3.5-0.8B --adapter runs/s1_models/s1-v1 --port 8090
python -m arena.cli run configs/experiments/s1_cascade_randbats.yaml          # deploy + collect (cascade)
```

Backends: `--backend mlx` (Apple silicon; `--mode head` trains the last attention block of the
hybrid Qwen3.5) or `--backend hf` (torch/peft, GPU servers, full LoRA). Deep fusion inside HyMem
(`llm_hymem_dual` params `shared_facts`, `step_control`, `s1_router`, `s1_executor`, `fold_mode: typed`,
`memory_router: jev`, `fast_memory`): `configs/experiments/hymem_deep_latency_randbats.yaml`,
GPU ablation `configs/experiments/hymem_s1_ablation_gpu_randbats.yaml`. Handover: `HANDOVER.md`.

## Tests

```bash
python -m pytest -q     # offline, no server/keys/node needed
```

## Status / roadmap

v1 complete (M0–M6). HyMem agent + long-context baselines + rebuilt game tools added
(offline-tested, smoke-tested live with local Qwen3.5-4B). Next: GPU-scale evaluations,
cross-battle memory, search-augmented spaces, SFT/RL on the call logs. Deferred: agent
self-built teams and tournament-level concurrency across pairings.
