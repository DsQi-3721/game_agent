# PokeLLMon bridge (official, faithful)

[git-disl/PokeLLMon](https://github.com/git-disl/PokeLLMon) — "A Human-Parity
Agent for Pokémon Battles" (Hu et al., 2024). We run its **real code** as an
external opponent (like PokeChamp) for true fidelity.

## Native variant vs this bridge

There are two `PokeLLMon` agents in this repo — pick by fidelity need:

| | agent | what it is |
|---|---|---|
| **Faithful** | `pokellmon` (external, this bridge) | runs the official `GPTPlayer` code: its prompts, knowledge injection, self-consistency, and accumulated text-history feedback |
| Framework-integrated | `llm_pokellmon` (native, `src/arena/agents/llm/pokellmon.py`) | a reimplementation in the *spirit* of the 3 strategies, using our renderer/cache/cost. **Not a faithful repro** (different knowledge content, an invented anti-switch rule, coarser feedback) — kept for cache/cost integration, not paper comparison |

## Full-PokeLLMon mode (the bridge enables all three strategies)

Upstream defaults ship the headline strategies OFF (`knowledge:false,
reason_algo:io, T:0`). `bridge_accept.py` forces the paper's full agent:
`knowledge=True` (move-effect text + team-constrained opponent type buckets),
`reason_algo="sc"` (self-consistency over 3 CoT samples), `T=2` (render last 3
turns of accumulated natural-language battle history), `n_shot=1`, `temperature=0.7`.

## Reproduce

```bash
bash external/pokellmon/setup.sh    # clone + venv + deps + DeepSeek base_url patch
```

No upstream `requirements.txt`; setup installs the inferred deps. PokeLLMon vendors
a modified poke-env under `src/` (isolated in the venv). LLM routes to DeepSeek
(`OPENAI_BASE_URL=https://api.deepseek.com`, `model=deepseek-chat`) via the existing
`DEEPSEEK_API_KEY`.

## Format

PokeLLMon's native/tested format is **random battle** (no team building) — the
bridge defaults to `gen9randombattle`. Its data supports gen9 (`gen9moves.json`,
type chart via `GenData.from_format`). For a same-format tournament vs PokeChamp
(gen9ou), PokeLLMon would need a team set provided — not wired yet.

## Bridge contract

`run.sh` logs in as `ARENA_USERNAME` and `accept_challenges(ARENA_OPPONENT, n)` in
`ARENA_FORMAT`. Grey-box: we capture its per-turn moves into `traj_<name>.jsonl`;
win/Elo are exact. Cost is not metered through our guardrail (it makes its own
OpenAI-compatible calls).
