# PokeChamp bridge

[sethkarten/pokechamp](https://github.com/sethkarten/pokechamp) — ICML 2025
spotlight. LLM-driven **minimax** (LLM does action sampling, opponent modeling,
value estimation over a forward-simulating, *modified* `poke_env`). We run it as
an external opponent rather than reimplementing its simulator.

## Reproduce (M2)

```bash
bash external/pokechamp/setup.sh          # clone + uv sync (isolated venv)
# smoke test against a running Showdown server:
cd external/pokechamp/repo && uv run python -m pokechamp.play --help
```

Needs an LLM key (PokeChamp uses OpenRouter by default: `OPENROUTER_API_KEY`).
Set `POKECHAMP_PIN=<sha>` before setup to lock a known-good commit.

## Bridge contract

PokeChamp ships **no "accept challenges from a username" CLI** — only
`local_1v1.py` (runs both players in one process) and ladder scripts. So we drive
its own minimax `LLMPlayer` (built by the repo's `get_llm_player(..., prompt_algo=
"minimax")`, which is a poke-env `Player`) through our shim **`bridge_accept.py`**,
which parks it on `accept_challenges(ARENA_OPPONENT, n)`.

- `run.sh` → `uv run --project repo python bridge_accept.py`
- `bridge_accept.py` adds `repo/` to `sys.path`, imports PokeChamp's vendored
  `poke_env` (not ours), logs in as `ARENA_USERNAME`, and accepts our challenges.
- LLM backend: set `POKECHAMP_MODEL` (default `gpt-4o`) + `OPENAI_API_KEY` /
  `OPENROUTER_API_KEY`. Point at DeepSeek via an OpenAI-compatible base URL if desired.

## Reproduction patches (auto-applied by setup.sh, idempotent)

Reproducing an older research repo against a current Showdown server needed a few
fixes, all applied to the cloned `repo/` by `setup.sh`:

1. **bitsandbytes → Linux-only** — no macOS arm64 wheel; it's local-quant-only and
   unused for API models, so `uv sync` was failing on it.
2. **Tolerate `|player|pX|`** (empty/player-left) — the vendored poke_env raised
   `RuntimeError: Invalid player message` on a message form modern Showdown sends.
3. **`gpt_player.py`**: honor `OPENAI_BASE_URL` (so the OpenAI client hits DeepSeek)
   and stop hardcoding `model='gpt-4o'` in the json branch (use the requested model).

The shim also `chdir`s into `repo/` (PokeChamp loads `pokedex`/`sets_*.json` via
cwd-relative paths). Requires `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

**Known quirk:** in `gen9randombattle` PokeChamp emits a team-preview order where
there is none → one harmless `[Invalid choice] There's nothing to choose` at battle
start. Its opponent-set priors are gen9ou-only (`[WARN] ...moves set not found`),
so randombattle play is somewhat degraded; gen9ou is the faithful format.

## Caveat

PokeChamp makes its own LLM calls, so its token cost does **not** flow through our
cost guardrail; cost shows as 0 for this agent in reports. Win/Elo/win-rate and
its per-turn moves (grey-box `traj_<name>.jsonl`) are captured normally.
