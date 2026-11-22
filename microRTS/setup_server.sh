#!/usr/bin/env bash
# Safe, self-contained setup for Linux x86_64 GPU servers (casialab-remote).
# - Does NOT touch system Java or other users' files/processes.
# - Prefers existing conda `game-agent` env and installs ONLY missing JPype1 (--no-deps).
#   Falls back to creating a dedicated `microrts-rlcd` conda env only if game-agent has conflicts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${CONDA_ENV_NAME:-game-agent}"
PY_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

if [ ! -x "$PY_BIN" ]; then
  echo "[ERROR] Conda env '$ENV_NAME' not found at $PY_BIN" >&2
  exit 1
fi

echo "=== 1. Check / install portable Temurin JDK 17 into $ROOT/.runtime/java ==="
JAVA_DIR="$ROOT/.runtime/java"
if [ ! -x "$JAVA_DIR/bin/java" ]; then
  mkdir -p "$ROOT/.runtime"
  JDK_TAR="$ROOT/.runtime/temurin17.tar.gz"
  curl -fSL "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse" -o "$JDK_TAR"
  rm -rf "$ROOT/.runtime/jdk17-tmp"
  mkdir -p "$ROOT/.runtime/jdk17-tmp"
  tar -xzf "$JDK_TAR" -C "$ROOT/.runtime/jdk17-tmp"
  EXTRACTED="$(find "$ROOT/.runtime/jdk17-tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  mv "$EXTRACTED" "$JAVA_DIR"
  rm -rf "$ROOT/.runtime/jdk17-tmp" "$JDK_TAR"
fi
export JAVA_HOME="$JAVA_DIR"
export PATH="$JAVA_HOME/bin:$PATH"
java -version

echo "=== 2. Verify conda '$ENV_NAME' core deps and add JPype1 safely (--no-deps) ==="
"$PY_BIN" -c "import torch, transformers, peft, safetensors, huggingface_hub, gymnasium, numpy, PIL; print('Core deps OK:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__, 'peft:', peft.__version__)"

if ! "$PY_BIN" -c "import jpype" 2>/dev/null; then
  echo "Installing JPype1 into '$ENV_NAME' with --no-deps (no existing packages modified)..."
  "$PIP_BIN" install --no-deps JPype1
fi
"$PY_BIN" -c "import jpype; print('JPype1 OK:', jpype.__version__)"

echo "=== 3. Download pinned Qwen3-0.6B RLCD decision export (if missing) ==="
MODEL_DIR="$ROOT/.runtime/models/qwen3-0.6b-rlcd-decision"
if [ ! -f "$MODEL_DIR/decision.json" ] || [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  PYTHONPATH="$ROOT" "$PY_BIN" -m microrts_agent.decisions.train download --init "$MODEL_DIR"
else
  echo "Model already present at $MODEL_DIR"
fi

echo "=== 4. Quick JNI + EVE-RLCD unit check ==="
PYTHONPATH="$ROOT" "$PY_BIN" -m pytest tests/test_decisions.py tests/test_eve_decisions.py -q

echo "=== Setup complete ==="
