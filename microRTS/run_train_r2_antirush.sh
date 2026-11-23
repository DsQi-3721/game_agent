#!/usr/bin/env bash
# Round 2 targeted online PPO + RLCD training to solve WorkerRush while keeping CoacAI/LightRush strength.
# Key fixes from Round 1 diagnosis:
#   1. --reward win_focused (+20/-20 terminal win/loss, light shaping, -6.0 stalemate/timeout penalty)
#   2. --gamma 0.9985 per tick (preserves terminal credit assignment across 500-1500 tick matches)
#   3. Heavy anti-rush opponent mix: WorkerRush (50%) + CoacAI (25%) + Mayari (25%) across 5 maps & P0/P1
#   4. Periodic validation on unseen maps (8x8C, 12x12B, 16x16B) vs WorkerRush,CoacAI to select best.pt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

INIT_CKPT="${1:-runs/a800-v2-pro/online-rl/best.pt}"
OUT="${2:-runs/a800-r2-antirush}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAVA_HOME="$ROOT/.runtime/java"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${CONDA_ENV_NAME:-game-agent}"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

TRAIN_MAPS="maps/8x8/basesWorkers8x8A.xml,maps/8x8/basesWorkers8x8B.xml,maps/8x8/basesWorkers8x8D.xml,maps/12x12/basesWorkers12x12A.xml,maps/open_competition/basesWorkers16x16A.xml"
TRAIN_OPPS="WorkerRush,WorkerRush,CoacAI,Mayari"
UNSEEN_MAPS="maps/8x8/basesWorkers8x8C.xml,maps/12x12/basesWorkers12x12B.xml,maps/16x16/basesWorkers16x16B.xml"
VAL_OPPS="WorkerRush,CoacAI"
FULL_EVAL_OPPS="RandomBiasedAI,LightRush,WorkerRush,CoacAI,Mayari"

mkdir -p "$OUT"
echo "[$(date -Is)] Starting Round 2 Anti-WorkerRush Online PPO + RLCD on GPU=$CUDA_VISIBLE_DEVICES -> $OUT (init=$INIT_CKPT)"

"$PY" -u -m microrts_agent.decisions.train online-rl \
  --checkpoint "$INIT_CKPT" \
  --out "$OUT/online-rl" --device cuda --dtype bf16 \
  --maps "$TRAIN_MAPS" \
  --opponents "$TRAIN_OPPS" \
  --val-maps "$UNSEEN_MAPS" \
  --val-opponents "$VAL_OPPS" \
  --players 0,1 \
  --max-ticks 2500 --skill-ticks 64 --candidate-limit 12 \
  --gamma 0.9985 --reward win_focused \
  --updates 80 --batch-episodes 4 --ppo-epochs 2 --micro-batch 16 \
  --lr 1.5e-5 --head-lr-mult 5.0 --entropy 0.015 \
  --arm rlcd --group 4 --calibration-weight 1.0 \
  --eval-every 10 --eval-repeats 1 --eval-greedy --seed 2027

FINAL_CKPT="$OUT/online-rl/best.pt"
[ -f "$FINAL_CKPT" ] || FINAL_CKPT="$OUT/online-rl/last.pt"

echo "=== Round 2 Final Unseen-Map Benchmark (60 games: 3 unseen maps x 5 opps x 2 players x 2 repeats) ==="
"$PY" -u -m microrts_agent.decisions.train eval \
  --checkpoint "$FINAL_CKPT" --device cuda --dtype bf16 \
  --maps "$UNSEEN_MAPS" --opponents "$FULL_EVAL_OPPS" --players 0,1 --repeats 2 \
  --max-ticks 2500 --greedy --out "$OUT/eval-final"

echo "=== Round 2 LoRA Merge Export ==="
"$PY" -u -m microrts_agent.decisions.train export \
  --checkpoint "$FINAL_CKPT" --out "$OUT/exported-decision"

echo "[$(date -Is)] Round 2 completed cleanly! Outputs in $OUT"
