#!/usr/bin/env bash
# Serve Qwen3.5-4B with vLLM on a Linux + NVIDIA box (registry entry: qwen3.5-4b-vllm).
#
#   bash scripts/serve_qwen35_vllm.sh [model_path_or_hf_id] [port]
#
# Flags follow the Qwen3.5 model card: reasoning parser `qwen3`, tool-call parser
# `qwen3_coder` (XML-style tool calls), text-only weights. Install per the card:
#   uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
set -euo pipefail

# On a shared GPU: GPU_MEM_UTIL=0.3 (vLLM's default 0.9 needs 90% of the card free), HOST=127.0.0.1.
# Qwen3.5 is a hybrid (Mamba) model: with a small GPU_MEM_UTIL also lower MAX_NUM_SEQS (e.g. 64) so the
# Mamba cache blocks cover the CUDA-graph batch sizes.
MODEL="${1:-Qwen/Qwen3.5-4B}"
PORT="${2:-8001}"   # 8000 is taken by the local Showdown server
MAX_LEN="${MAX_MODEL_LEN:-65536}"

exec vllm serve "$MODEL" --served-model-name Qwen3.5-4B --port "$PORT" --host "${HOST:-0.0.0.0}" \
  --max-model-len "$MAX_LEN" ${GPU_MEM_UTIL:+--gpu-memory-utilization "$GPU_MEM_UTIL"} \
  ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only
