#!/usr/bin/env bash
# End-to-end EVE-RLCD MicroRTS training on A800 (pinned to GPU 0 by default).
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_train_a800.sh runs/a800-v2-pro
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT="${1:-runs/a800-v2-pro}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAVA_HOME="$ROOT/.runtime/java"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${CONDA_ENV_NAME:-game-agent}"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

DATA_TRAIN="$ROOT/data/expert-v2-pro/train_pro_clean.jsonl.gz"
DATA_VAL="$ROOT/data/expert-v2-pro/val_pro_clean.jsonl.gz"
UNSEEN_MAPS="maps/8x8/basesWorkers8x8C.xml,maps/12x12/basesWorkers12x12B.xml,maps/16x16/basesWorkers16x16B.xml"
EVAL_OPPS="RandomBiasedAI,LightRush,WorkerRush,CoacAI"

mkdir -p "$OUT"
echo "[$(date -Is)] Starting EVE-RLCD MicroRTS pipeline on GPU=$CUDA_VISIBLE_DEVICES -> $OUT"

echo "=== Stage 0: Base Latency Benchmark & Unseen-Map Baseline Eval ==="
"$PY" -u -m microrts_agent.decisions.train bench \
  --device cuda --dtype bf16 --iters 10 | tee "$OUT/bench_base.json"

"$PY" -u -m microrts_agent.decisions.train eval \
  --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$EVAL_OPPS" --players 0,1 --repeats 1 \
  --greedy --out "$OUT/eval-base"

echo "=== Stage 1: SFT Warmup (v2-pro 24,549 train / 6,045 unseen-map val) ==="
"$PY" -u -m microrts_agent.decisions.train sft \
  --data "$DATA_TRAIN" --val-data "$DATA_VAL" \
  --out "$OUT/sft" --device cuda --dtype bf16 \
  --rank 16 --epochs 2 --micro-batch 16 --accum 2 --lr 1e-4 --head-lr-mult 5.0 \
  --no-win-noul --seed 7

"$PY" -u -m microrts_agent.decisions.train eval \
  --checkpoint "$OUT/sft/best.pt" --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$EVAL_OPPS" --players 0,1 --repeats 1 \
  --greedy --out "$OUT/eval-sft"

echo "=== Stage 2a: Exact EVE-RLCD Bandit Training (arm=rlcd, G=4) ==="
"$PY" -u -m microrts_agent.decisions.train bandit-rlcd \
  --checkpoint "$OUT/sft/best.pt" \
  --data "$DATA_TRAIN" --val-data "$DATA_VAL" \
  --out "$OUT/bandit-rlcd" --device cuda --dtype bf16 \
  --arm rlcd --group 4 --epochs 2 --micro-batch 16 --accum 2 --lr 3e-5 --head-lr-mult 5.0 \
  --no-win-noul --seed 7

echo "=== Stage 2a (Ablation): RLVR Bandit Baseline (arm=rlvr, G=4) ==="
"$PY" -u -m microrts_agent.decisions.train bandit-rlcd \
  --checkpoint "$OUT/sft/best.pt" \
  --data "$DATA_TRAIN" --val-data "$DATA_VAL" \
  --out "$OUT/bandit-rlvr" --device cuda --dtype bf16 \
  --arm rlvr --group 4 --epochs 2 --micro-batch 16 --accum 2 --lr 3e-5 --head-lr-mult 5.0 \
  --no-win-noul --seed 7

"$PY" -u -m microrts_agent.decisions.train eval \
  --checkpoint "$OUT/bandit-rlcd/best.pt" --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$EVAL_OPPS" --players 0,1 --repeats 1 \
  --greedy --out "$OUT/eval-bandit-rlcd"

echo "=== Stage 2b: Online MicroRTS PPO + RLCD Win-Calibration ==="
"$PY" -u -m microrts_agent.decisions.train online-rl \
  --checkpoint "$OUT/bandit-rlcd/best.pt" \
  --out "$OUT/online-rl" --device cuda --dtype bf16 \
  --maps "maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml,maps/12x12/basesWorkers12x12A.xml" \
  --opponents "RandomBiasedAI,LightRush,WorkerRush,CoacAI" \
  --players 0,1 --max-ticks 2000 --updates 32 --batch-episodes 4 \
  --ppo-epochs 2 --micro-batch 16 --lr 2e-5 --head-lr-mult 5.0 \
  --arm rlcd --group 4 --calibration-weight 1.0 \
  --eval-every 8 --eval-repeats 1 --eval-greedy --seed 7

echo "=== Stage 3: Final Unseen-Map Evaluation, LoRA Merge Export & Latency Bench ==="
FINAL_CKPT="$OUT/online-rl/best.pt"
[ -f "$FINAL_CKPT" ] || FINAL_CKPT="$OUT/online-rl/last.pt"

"$PY" -u -m microrts_agent.decisions.train eval \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$EVAL_OPPS" --players 0,1 --repeats 2 \
  --greedy --out "$OUT/eval-final"

"$PY" -u -m microrts_agent.decisions.train export \
  --checkpoint "$FINAL_CKPT" --out "$OUT/exported-decision"

"$PY" -u -m microrts_agent.decisions.train bench \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 --iters 10 | tee "$OUT/bench_final.json"

echo "[$(date -Is)] Pipeline completed cleanly! Outputs in $OUT"
