#!/usr/bin/env bash
# Launch the stub external bot using the SystemofML python (it only needs poke-env,
# which SystemofML already has -- no conflicting deps, so no separate venv needed).
set -euo pipefail
cd "$(dirname "$0")"
PY="${ARENA_PY:-/Users/hoi/miniconda3/envs/SystemofML/bin/python}"
exec "$PY" bot.py
