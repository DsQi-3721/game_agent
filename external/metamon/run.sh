#!/usr/bin/env bash
# Bridge entrypoint: Metamon's Kakuna (the current best Metamon policy) accepts our challenges on the local
# Showdown server. kakuna_local.py: local weights ($METAMON_WEIGHTS), flex attention instead of flash-attn,
# our Showdown port, ARENA_TEAM_FILE as its team (mirror teams), no battle timer. Contract: ../README.md.
set -euo pipefail
cd "$(dirname "$0")"

: "${ARENA_SERVER:?missing ARENA_SERVER}"
: "${ARENA_USERNAME:?missing ARENA_USERNAME}"
: "${ARENA_OPPONENT:?missing ARENA_OPPONENT}"
: "${ARENA_FORMAT:?missing ARENA_FORMAT}"

# ws://host:port/showdown/websocket -> http://host:port (kakuna_local.py reads ARENA_SHOWDOWN_URL)
HOSTPORT="$(echo "$ARENA_SERVER" | sed -E 's#^wss?://([^/]+).*#\1#')"
export ARENA_SHOWDOWN_URL="http://${HOSTPORT}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/data/wangxinqi/metamon_cache}"
export METAMON_WEIGHTS="${METAMON_WEIGHTS:-/data/models/metamon}"

echo "[metamon] Kakuna epoch ${METAMON_KAKUNA_EPOCH:-34} login=${ARENA_USERNAME} accept-from=${ARENA_OPPONENT} format=${ARENA_FORMAT}"
exec venv/bin/python -u kakuna_local.py accept --checkpoint "${METAMON_KAKUNA_EPOCH:-34}"
