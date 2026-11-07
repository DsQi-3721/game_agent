#!/usr/bin/env bash
# Bridge entrypoint: PokeChamp's minimax player accepts our challenges via our shim (bridge_accept.py), in its own
# venv (POKECHAMP_VENV, built by setup.sh with uv). LLM: any OpenAI-compatible endpoint -- by default the local
# vLLM serving Qwen3.8-27B on :8102 (no API key, nothing leaves the machine). Contract + env vars: ../README.md.
set -euo pipefail
cd "$(dirname "$0")"

: "${ARENA_SERVER:?missing ARENA_SERVER}"
: "${ARENA_USERNAME:?missing ARENA_USERNAME}"
: "${ARENA_OPPONENT:?missing ARENA_OPPONENT}"
: "${ARENA_FORMAT:?missing ARENA_FORMAT}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8102/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export POKECHAMP_MODEL="${POKECHAMP_MODEL:-Qwen3.8-27B}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost" no_proxy="${no_proxy:-},127.0.0.1,localhost"
VENV="${POKECHAMP_VENV:-/data/wangxinqi/venvs/pokechamp_sys}"     # setup.sh (POKECHAMP_BASE_PYTHON=... on this server)

echo "[pokechamp] starting minimax bot (model=${POKECHAMP_MODEL} via ${OPENAI_BASE_URL})"
exec "$VENV/bin/python" -u "$(pwd)/bridge_accept.py"
