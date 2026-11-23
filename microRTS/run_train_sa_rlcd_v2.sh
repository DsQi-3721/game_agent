#!/usr/bin/env bash
# SA-RLCD v2 (Fully Self-Consistent Online Training & 60-Game Unseen-Map Evaluation on A800).
# Starts directly from untainted Stage 2a (IS-RLCD + RB-RLCD) checkpoint for a controlled ablation vs R3 v1:
#   1. Strictly undiscounted (gamma=1) WPC-PBRS advantages with conservation-neutral material potential Phi(s) (Phi(s_T)=0)
#   2. Strict NoulQ timeout semantics (Y=0) + online timeout ring-buffer + balanced offline Calib-Replay
#   3. Asymmetric failure-driven exploration entropy beta_ent(s) = ent_min + (ent_max - ent_min) * (1 - V(s)^2)
#   4. Single-objective mini-batch joint optimization (ppo_batch_size=64, ppo_epochs=3, no post-hoc anchor overwrite)
#   5. Focused hard-adversary curriculum: Mayari (50%) + WorkerRush (25%) + CoacAI (25%)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

INIT_CKPT="${1:-runs/a800-r3-sa-rlcd/stage2a-is-rlcd/best.pt}"
OUT="${2:-runs/a800-r4-sa-rlcd-v2}"

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
TRAIN_OPPS="Mayari,Mayari,WorkerRush,CoacAI"
UNSEEN_MAPS="maps/8x8/basesWorkers8x8C.xml,maps/12x12/basesWorkers12x12B.xml,maps/16x16/basesWorkers16x16B.xml"
VAL_OPPS="Mayari,WorkerRush,CoacAI"
FULL_EVAL_OPPS="RandomBiasedAI,LightRush,WorkerRush,CoacAI,Mayari"

mkdir -p "$OUT"
echo "[$(date -Is)] Starting SA-RLCD v2 on GPU=$CUDA_VISIBLE_DEVICES -> $OUT (init=$INIT_CKPT)"

echo "=== Stage 1: Online SA-RLCD v2 (Undiscounted WPC-PBRS + Asymmetric Entropy + Joint Mini-Batch) ==="
"$PY" -u -m microrts_agent.decisions.sa_rlcd online \
  --checkpoint "$INIT_CKPT" \
  --calib-replay "$DATA_DIR/win_pro_train.jsonl.gz" \
  --out "$OUT/online-sa-rlcd-v2" --device cuda --dtype bf16 \
  --maps "$TRAIN_MAPS" \
  --opponents "$TRAIN_OPPS" \
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

FINAL_CKPT="$OUT/online-sa-rlcd-v2/best.pt"
[ -f "$FINAL_CKPT" ] || FINAL_CKPT="$OUT/online-sa-rlcd-v2/last.pt"

echo "=== Stage 2: Apples-to-Apples 60-Game Unseen-Map Benchmark ==="
"$PY" -u -m microrts_agent.decisions.train eval \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$FULL_EVAL_OPPS" --players 0,1 --repeats 2 \
  --max-ticks 2500 --greedy --out "$OUT/eval-final"

echo "=== Stage 3: LoRA Merge Export & Latency Benchmark ==="
"$PY" -u -m microrts_agent.decisions.train export \
  --checkpoint "$FINAL_CKPT" --out "$OUT/exported-decision"
"$PY" -u -m microrts_agent.decisions.train bench \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 --iters 10 | tee "$OUT/bench_final.json"

echo "[$(date -Is)] SA-RLCD v2 pipeline completed cleanly! Outputs in $OUT"
