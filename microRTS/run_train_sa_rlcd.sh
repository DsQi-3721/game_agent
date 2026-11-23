#!/usr/bin/env bash
# SA-RLCD (Sequential Advantage-Calibrated RLCD) full training & evaluation pipeline on A800.
# Innovations over naive eve-rlcd reproduction (Round 1 & Round 2):
#   1. Force-summary tactical state enrichment (1-line unit/HP/phase macro readout)
#   2. IS-RLCD: Unbiased importance-sampled exploratory bandit RLCD (eliminates saturation dead-zone)
#   3. RB-TD(lambda)-RLCD: Exact Rao-Blackwellized TD(lambda) win-probability calibration (zero variance, bootstraps timeouts)
#   4. WPC-PPO + Uncertainty-Adaptive Entropy + Anti-Forgetting Calibration Replay + KL Anchor
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

INIT_SFT="${1:-runs/a800-v2-pro/sft/best.pt}"
OUT="${2:-runs/a800-r3-sa-rlcd}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAVA_HOME="$ROOT/.runtime/java"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${CONDA_ENV_NAME:-game-agent}"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

DATA_DIR="data/expert-v2-pro"
TRAIN_MAPS="maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml,maps/8x8/basesWorkers8x8D.xml,maps/12x12/basesWorkers12x12A.xml,maps/open_competition/basesWorkers16x16A.xml"
TRAIN_OPPS="WorkerRush,WorkerRush,CoacAI,Mayari,LightRush"
UNSEEN_MAPS="maps/8x8/basesWorkers8x8C.xml,maps/12x12/basesWorkers12x12B.xml,maps/16x16/basesWorkers16x16B.xml"
VAL_OPPS="WorkerRush,CoacAI,Mayari"
FULL_EVAL_OPPS="RandomBiasedAI,LightRush,WorkerRush,CoacAI,Mayari"

mkdir -p "$OUT"
echo "[$(date -Is)] Starting SA-RLCD pipeline on GPU=$CUDA_VISIBLE_DEVICES -> $OUT (init=$INIT_SFT)"

echo "=== Stage 1: Force-Summary SFT + Balanced NoulQ Warm-Start (1 epoch) ==="
"$PY" -u -m microrts_agent.decisions.sa_rlcd sft \
  --checkpoint "$INIT_SFT" \
  --data "$DATA_DIR/train_pro_clean.jsonl.gz" \
  --val-data "$DATA_DIR/val_pro_clean.jsonl.gz" \
  --win-data "$DATA_DIR/win_pro_train.jsonl.gz" \
  --win-val-data "$DATA_DIR/win_pro_val.jsonl.gz" \
  --out "$OUT/stage1-sft" --device cuda --dtype bf16 \
  --epochs 1 --micro-batch 16 --accum 2 --lr 5e-5 --head-lr-mult 5.0 --seed 101

echo "=== Stage 2a: Offline IS-RLCD (ChoiceQ) + RB-RLCD (NoulQ) (1 epoch) ==="
"$PY" -u -m microrts_agent.decisions.sa_rlcd is-rlcd \
  --checkpoint "$OUT/stage1-sft/best.pt" \
  --data "$DATA_DIR/train_pro_clean.jsonl.gz" \
  --val-data "$DATA_DIR/val_pro_clean.jsonl.gz" \
  --win-data "$DATA_DIR/win_pro_train.jsonl.gz" \
  --win-val-data "$DATA_DIR/win_pro_val.jsonl.gz" \
  --out "$OUT/stage2a-is-rlcd" --device cuda --dtype bf16 \
  --epochs 1 --micro-batch 16 --accum 2 --lr 3e-5 --head-lr-mult 5.0 \
  --group 4 --eps 0.15 --seed 202

echo "=== Stage 2b: Online SA-RLCD (Undiscounted WPC-PBRS + Asymmetric Entropy + Joint Mini-Batch) ==="
"$PY" -u -m microrts_agent.decisions.sa_rlcd online \
  --checkpoint "$OUT/stage2a-is-rlcd/best.pt" \
  --calib-replay "$DATA_DIR/win_pro_train.jsonl.gz" \
  --out "$OUT/stage2b-online-sa-rlcd" --device cuda --dtype bf16 \
  --maps "$TRAIN_MAPS" \
  --opponents "Mayari,Mayari,WorkerRush,CoacAI" \
  --val-maps "$UNSEEN_MAPS" \
  --val-opponents "$VAL_OPPS" \
  --players 0,1 \
  --max-ticks 2500 --skill-ticks 64 --candidate-limit 12 \
  --updates 60 --batch-episodes 4 --ppo-epochs 3 --ppo-batch-size 64 --micro-batch 16 \
  --clip-eps 0.2 --ent-min 0.003 --ent-max 0.025 --kl-coef 0.005 \
  --calibration-weight 1.0 --td-lambda 0.85 --gae-lambda 0.95 --potential-scale 0.15 \
  --replay-win-n 16 --replay-timeout-n 4 \
  --lr 2.5e-5 --head-lr-mult 5.0 \
  --eval-every 10 --eval-repeats 1 --eval-greedy --seed 2029

FINAL_CKPT="$OUT/stage2b-online-sa-rlcd/best.pt"
[ -f "$FINAL_CKPT" ] || FINAL_CKPT="$OUT/stage2b-online-sa-rlcd/last.pt"

echo "=== Stage 3: Apples-to-Apples 60-Game Unseen-Map Benchmark ==="
"$PY" -u -m microrts_agent.decisions.train eval \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$FULL_EVAL_OPPS" --players 0,1 --repeats 2 \
  --max-ticks 2500 --greedy --out "$OUT/eval-final"

echo "=== Stage 4: LoRA Merge Export & Latency Benchmark ==="
"$PY" -u -m microrts_agent.decisions.train export \
  --checkpoint "$FINAL_CKPT" --out "$OUT/exported-decision"
"$PY" -u -m microrts_agent.decisions.train bench \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 --iters 10 | tee "$OUT/bench_final.json"

echo "[$(date -Is)] SA-RLCD pipeline completed cleanly! Outputs in $OUT"
