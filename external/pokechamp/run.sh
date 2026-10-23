#!/usr/bin/env bash
# Bridge entrypoint: run PokeChamp's minimax player in accept-challenges mode via
# our shim (bridge_accept.py), inside the repo's uv venv. LLM = DeepSeek (routed
# through GPTPlayer + OPENAI_BASE_URL). Contract + env vars: ../README.md.
set -euo pipefail
cd "$(dirname "$0")"

: "${ARENA_SERVER:?missing ARENA_SERVER}"
: "${ARENA_USERNAME:?missing ARENA_USERNAME}"
: "${ARENA_OPPONENT:?missing ARENA_OPPONENT}"
: "${ARENA_FORMAT:?missing ARENA_FORMAT}"

# Load the arena .env (DEEPSEEK_API_KEY) if present.
ENV_FILE="$(cd ../.. && pwd)/.env"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi

# Route PokeChamp's OpenAI client at DeepSeek's OpenAI-compatible endpoint.
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export POKECHAMP_MODEL="${POKECHAMP_MODEL:-deepseek-chat}"

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "ERROR: no DeepSeek/OpenAI key (set DEEPSEEK_API_KEY in arena .env)" >&2
  exit 1
fi

echo "[pokechamp] starting minimax bot (model=${POKECHAMP_MODEL} via ${OPENAI_BASE_URL})"
exec uv run --project repo python "$(pwd)/bridge_accept.py"
