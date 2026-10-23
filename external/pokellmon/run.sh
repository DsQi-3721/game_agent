#!/usr/bin/env bash
# Bridge entrypoint: run the OFFICIAL PokeLLMon in accept-challenges mode (full
# mode: knowledge + self-consistency + history feedback) via bridge_accept.py,
# inside its venv. LLM = DeepSeek (OPENAI_BASE_URL). Contract: ../README.md.
set -euo pipefail
cd "$(dirname "$0")"

: "${ARENA_USERNAME:?missing ARENA_USERNAME}"
: "${ARENA_OPPONENT:?missing ARENA_OPPONENT}"
: "${ARENA_FORMAT:?missing ARENA_FORMAT}"

# Load the arena .env (DEEPSEEK_API_KEY) if present.
ENV_FILE="$(cd ../.. && pwd)/.env"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export POKELLMON_MODEL="${POKELLMON_MODEL:-deepseek-chat}"

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "ERROR: no DeepSeek/OpenAI key (set DEEPSEEK_API_KEY in arena .env)" >&2
  exit 1
fi

echo "[pokellmon] starting official bot (model=${POKELLMON_MODEL} via ${OPENAI_BASE_URL})"
exec venv/bin/python "$(pwd)/bridge_accept.py"
