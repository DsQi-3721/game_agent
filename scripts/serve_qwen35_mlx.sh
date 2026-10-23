#!/usr/bin/env bash
# Serve the local Qwen3.5-4B as an OpenAI-compatible API with mlx_lm (Apple Silicon).
#
#   bash scripts/serve_qwen35_mlx.sh [model_dir] [port]
#
# Registry entry: qwen3.5-4b-mlx (model id "default_model" = the model loaded here).
# Sampling defaults follow the Qwen3.5 non-thinking recommendation; agents override
# temperature / max_tokens / enable_thinking per request.
set -euo pipefail

MODEL="${1:-/Users/hoi/project/models/Qwen3.5-4B}"
PORT="${2:-8080}"
BIN="${MLX_BIN:-/Users/hoi/miniconda3/envs/SystemofML/bin}"

# local traffic only: never route through a system proxy
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy WS_PROXY ws_proxy
export NO_PROXY="localhost,127.0.0.1,::1"

exec "$BIN/mlx_lm.server" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 2048 \
  --chat-template-args '{"enable_thinking": false}'
