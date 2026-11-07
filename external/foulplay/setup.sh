#!/usr/bin/env bash
# Reproduce Foul Play (pmariglia/foul-play) in an isolated venv. poke-engine is compiled from source (Rust) with
# the Gen 9 features (terastallization), as in Foul Play's requirements.txt. The Rust toolchain goes under
# RUST_ROOT (defaults to the local user cache).
set -euo pipefail
cd "$(dirname "$0")"

PIN="${FOULPLAY_PIN:-main}"
RUST_ROOT="${RUST_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/game-agent/rust}"
PY="${FOULPLAY_PYTHON:-python3}"            # >= 3.11

[ -d repo ] || git clone https://github.com/pmariglia/foul-play.git repo
git -C repo fetch --all --tags && git -C repo checkout "$PIN"

export RUSTUP_HOME="$RUST_ROOT/rustup" CARGO_HOME="$RUST_ROOT/cargo" CARGO_TARGET_DIR="$RUST_ROOT/target"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
export PATH="$CARGO_HOME/bin:$PATH"

[ -d venv ] || "$PY" -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install requests==2.33.0 websockets==14.1 python-dateutil==2.8.0
ENGINE="$(grep -oE 'poke-engine==[0-9.]+' repo/requirements.txt)"
venv/bin/pip install -v --no-cache-dir --force-reinstall "$ENGINE" \
  --config-settings="build-args=--features poke-engine/terastallization --no-default-features"
echo "Foul Play setup complete ($(git -C repo log -1 --format=%h))."
