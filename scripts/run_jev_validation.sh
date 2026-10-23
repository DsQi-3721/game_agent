#!/usr/bin/env bash
# One-shot Jev validation (after TYPESAFE_API_KEY is in .env). Cost: well under $1.
#   bash scripts/run_jev_validation.sh            # API probe + offline evals + online System-1 battles
#   bash scripts/run_jev_validation.sh --dual     # also the dual-process real-time battle (needs mlx server)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/Users/hoi/miniconda3/envs/SystemofML/bin/python}
# local Showdown websockets break behind a SOCKS proxy: drop only the SOCKS/WS variables and
# exclude localhost (same as arena.cli); other proxy settings are left as the user has them
unset ALL_PROXY all_proxy WS_PROXY ws_proxy || true
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"

echo "== 1/5 API probe"
$PY scripts/jev_probe.py --models jev-1.13.0,jev-1.14.0
echo "== 2/5 offline decisions on recorded positions"
[ -d data/jev_eval/snapshots ] || $PY scripts/collect_decision_snapshots.py --out data/jev_eval/snapshots
$PY scripts/eval_jev_decisions.py --snapshots data/jev_eval/snapshots --models jev-1.13.0,jev-1.14.0
echo "== 3/5 memory routing"
[ -f data/jev_eval/memory_bench.jsonl ] || $PY scripts/build_memory_bench.py
$PY scripts/eval_jev_memory_router.py --model jev-1.13.0
$PY scripts/eval_jev_memory_router.py --mock > /dev/null
echo "== 4/5 online System-1 battles (needs the Showdown server on :8000)"
lsof -nP -i :8000 > /dev/null || { echo "start Showdown first: cd ../pokemon-showdown && node pokemon-showdown start --no-security"; exit 1; }
$PY -m arena.cli run configs/experiments/jev_fast_vs_baselines.yaml --emit-events
if [ "${1:-}" = "--dual" ]; then
  echo "== 5/5 dual-process real-time battle (needs the mlx server: bash scripts/serve_qwen35_mlx.sh)"
  $PY -m arena.cli run configs/experiments/hymem_dual_latency_randbats.yaml --emit-events
fi
$PY scripts/summarize_jev_validation.py --out reports/jev_validation_results.md
