#!/usr/bin/env bash
# Bridge entrypoint: run a pretrained Metamon policy as an opponent that ACCEPTS
# our challenges. Metamon has a first-class challenge mode (no shim needed):
#   --eval_type challenge --role acceptor --username X --opponent_username Y
# It connects to ws://localhost:8000 (poke-env LocalhostServerConfiguration) -- the
# same host as our Showdown server -- so a native venv needs no networking patch.
set -euo pipefail
cd "$(dirname "$0")"

: "${ARENA_USERNAME:?missing ARENA_USERNAME}"
: "${ARENA_OPPONENT:?missing ARENA_OPPONENT}"
: "${ARENA_FORMAT:?missing ARENA_FORMAT}"

# ARENA_FORMAT like "gen9ou" -> gens=9, formats=ou
GEN="$(echo "$ARENA_FORMAT" | sed -E 's/^gen([0-9]+).*/\1/')"
TIER="$(echo "$ARENA_FORMAT" | sed -E 's/^gen[0-9]+//')"
# SmallRLGen9Beta is the gen9-trained checkpoint (plain SmallRL targets gen1-4).
CKPT="${METAMON_CHECKPOINT:-SmallRLGen9Beta}"
N="${ARENA_N_BATTLES:-1000}"

# Metamon caches HF checkpoints/data here (required env var).
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-$(pwd)/cache}"
mkdir -p "$METAMON_CACHE_DIR"

# shellcheck disable=SC1091
source venv/bin/activate

echo "[metamon] agent=${CKPT} gen=${GEN} tier=${TIER} login=${ARENA_USERNAME} accept-from=${ARENA_OPPONENT}"
# acceptor must be online before our side challenges; print a ready line for the launcher.
exec python -m metamon.rl.evaluate \
  --eval_type challenge \
  --agent "${CKPT}" \
  --username "${ARENA_USERNAME}" \
  --opponent_username "${ARENA_OPPONENT}" \
  --role acceptor \
  --gens "${GEN}" \
  --formats "${TIER}" \
  --total_battles "${N}"
