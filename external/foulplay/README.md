# Foul Play bridge

[pmariglia/foul-play](https://github.com/pmariglia/foul-play) — MCTS (DUCT, root-parallel) on the Rust
[poke-engine](https://github.com/pmariglia/poke-engine), with opponent sets inferred from Smogon usage stats and
public sets. PokéAgent Challenge 2025 Gen 9 OU champion; >80% GXE on the official gen9ou ladder (peak 1879).

## Setup (isolated venv; nothing goes into the arena environment)

```bash
FOULPLAY_PYTHON=/path/to/python3.11+ bash external/foulplay/setup.sh
```

`setup.sh` clones the repo, installs a minimal Rust toolchain under `RUST_ROOT` (default
`/data/wangxinqi/rust`) and builds `poke-engine` with the Gen 9 features (terastallization), as in Foul Play's
`requirements.txt`. At run time Foul Play downloads public set data (data.foulplay.cc, Showdown's sets,
smogon.com/stats) into `repo/fp/data/*_cache`.

## Bridge contract (../README.md)

- `challenger: them`: Foul Play challenges `ARENA_OPPONENT` battle after battle and we accept. In its
  accept mode a challenge that arrives while it is still leaving the previous battle is read and dropped.
- `ARENA_TEAM_FILE` is required (its team; mirror teams).
- `bridge_run.py` runs Foul Play's own `run_foul_play` with three adaptations, no change to its search:
  login on the local `--no-security` server with `/trn` (no assertion from play.pokemonshowdown.com); the battle
  timer it turns on at every battle start is dropped (our other opponents do not use one); formes set at team
  validation (Keldeo-Resolute) are linked back to their base species so it finds them in its own team.
- Search settings: `FOULPLAY_SEARCH_TIME_MS` (default 2000 per sampled state), `FOULPLAY_PARALLELISM` (4 sampled
  opponent states searched in parallel), `FOULPLAY_THREADS` (1 per state). Ready line: `Challenging ...`.
