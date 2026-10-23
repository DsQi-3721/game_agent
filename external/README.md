# External opponents (out-of-process bridges)

This directory holds **reproductions of published agents** that we run as
*opponents* without importing their code into our framework. Each lives in its
own venv/container so a vendored `poke_env` (e.g. PokeChamp's modified fork) never
collides with ours. The rendezvous is the shared local Showdown server.

Nothing here is on the `arena` Python import path. `arena.external.launcher`
only spawns `run.sh` (or `docker run`) and watches its stdout.

## Handshake contract

Our framework launches `run.sh` with these env vars and then **our** in-process
player sends N challenges to the external bot:

| env var          | meaning                                                        |
|------------------|----------------------------------------------------------------|
| `ARENA_SERVER`   | websocket URL, e.g. `ws://localhost:8000/showdown/websocket`   |
| `ARENA_USERNAME` | the username the external bot must log in as                   |
| `ARENA_OPPONENT` | **our** username — the bot must `accept_challenges(this, n)`    |
| `ARENA_FORMAT`   | battle format, e.g. `gen9ou`                                   |

`run.sh` must:
1. Activate its own venv / container.
2. Connect to `ARENA_SERVER`, log in as `ARENA_USERNAME`.
3. `accept_challenges(ARENA_OPPONENT, n)` in `ARENA_FORMAT` (n large/looping is fine).
4. Print a line matching the config's `ready_regex` once logged in & accepting
   (default matches `logged in|listening|ready`) so the launcher knows it's up.

The local Showdown server must be running (`node pokemon-showdown start
--no-security`, ws://localhost:8000). Prereqs: `git`, plus `uv` (PokeChamp) or a
venv + `huggingface_hub` (Metamon), or `docker` for the container fallback.

## Projects

- `pokechamp/` — sethkarten/pokechamp (ICML 2025), LLM minimax. Bridged.
- `metamon/`   — UT-Austin-RPL/metamon (RLC 2025), offline-RL transformer. Bridged (Docker on macOS).
- `pokellmon/` — git-disl/PokeLLMon (2024). **Reference oracle only** — the agent
  we actually run is the native port `arena.agents.llm.pokellmon` (type `llm_pokellmon`).
