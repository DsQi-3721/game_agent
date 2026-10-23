#!/usr/bin/env bash
# Reproduce Metamon (UT-Austin-RPL/metamon, RLC 2025) in an isolated venv and
# fetch a pretrained checkpoint. Installs its pinned poke-env fork (0.8.3.3) +
# torch + amago into the venv -- isolated from our framework's poke-env.
#
# *** RUN THIS ON A LINUX MACHINE WITH AN NVIDIA GPU. ***  Metamon's amago
# Transformer requires flash-attention 2 (CUDA-only). It installs/imports on macOS
# but cannot run the model there (no flash-attn / no CUDA). See README.md.
set -euo pipefail
cd "$(dirname "$0")"

IS_LINUX="$([[ "$(uname)" == "Linux" ]] && echo 1 || echo 0)"

REPO="https://github.com/UT-Austin-RPL/metamon.git"
PIN="${METAMON_PIN:-main}"

if [ ! -d repo ]; then
  git clone "$REPO" repo
fi
cd repo
git fetch --all --tags
git checkout "$PIN"

# need python >=3.10; override with METAMON_PYTHON if your GPU box differs
PY="${METAMON_PYTHON:-python3}"
"$PY" -m venv ../venv
# shellcheck disable=SC1091
source ../venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install huggingface_hub

# flash-attention 2 is REQUIRED to run the model (CUDA-only). On a Linux+GPU box,
# install it; elsewhere skip (install will fail / model won't run).
if [[ "$IS_LINUX" == "1" ]]; then
  echo "[setup] installing flash-attn (GPU) ..."
  pip install "amago[flash]" || pip install flash-attn --no-build-isolation || \
    echo "[setup] WARN: flash-attn install failed; ensure CUDA toolkit + GPU are present"
else
  echo "[setup] non-Linux: skipping flash-attn. The model needs a Linux+NVIDIA GPU to run."
fi

# Patch: amago v3.4.0 random_traj_length passes gin-configured FLOAT save_every to
# random.randint (int-only) -> TypeError at eval. Int-cast. Idempotent.
AMAGO_ENV="$(../venv/bin/python -c 'import amago.envs.amago_env as m; print(m.__file__)')"
python3 - "$AMAGO_ENV" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
bad = "return random.randint(*self.save_every) if self.save_every else None"
good = ("return random.randint(int(self.save_every[0]), int(self.save_every[1])) "
        "if self.save_every else None  # arena: gin sets these as float")
if bad in t:
    p.write_text(t.replace(bad, good)); print("[patch] amago random_traj_length int-cast")
PY

# 49 checkpoints live at HF `jakegrigsby/metamon`. SmallRLGen9Beta is the gen9 one.
echo "Metamon setup complete. Set METAMON_CACHE_DIR (run.sh defaults it to ./cache)."
echo "Choose a checkpoint via METAMON_CHECKPOINT (default SmallRLGen9Beta for gen9 OU)."
echo "Smoke test (Linux+GPU, Showdown server running):"
echo "  export METAMON_CACHE_DIR=\$PWD/cache"
echo "  source ../venv/bin/activate && python -m metamon.rl.evaluate --help"
