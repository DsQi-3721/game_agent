#!/usr/bin/env bash
# Bridge entrypoint: Foul Play challenges our player (challenger: them). Contract: ../README.md; details:
# bridge_run.py. Setup: setup.sh (Rust toolchain + poke-engine built for gen9, in venv/).
set -euo pipefail
cd "$(dirname "$0")"

: "${ARENA_SERVER:?missing ARENA_SERVER}"
: "${ARENA_USERNAME:?missing ARENA_USERNAME}"
: "${ARENA_OPPONENT:?missing ARENA_OPPONENT}"
: "${ARENA_FORMAT:?missing ARENA_FORMAT}"
: "${ARENA_TEAM_FILE:?missing ARENA_TEAM_FILE (Foul Play needs a team: use teams.mode mirror)}"

echo "[foulplay] login=${ARENA_USERNAME} challenge=${ARENA_OPPONENT} format=${ARENA_FORMAT} search=${FOULPLAY_SEARCH_TIME_MS:-2000}ms x${FOULPLAY_PARALLELISM:-4}"
exec venv/bin/python -u bridge_run.py
