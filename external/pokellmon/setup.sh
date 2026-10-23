#!/usr/bin/env bash
# Reproduce the OFFICIAL PokeLLMon (git-disl/PokeLLMon) in an isolated venv and run
# its REAL code as an external opponent (faithful, unlike the native llm_pokellmon
# variant). It vendors a modified poke-env under src/, so we keep it fully isolated.
set -euo pipefail
cd "$(dirname "$0")"

REPO="https://github.com/git-disl/PokeLLMon.git"
PIN="${POKELLMON_PIN:-main}"

if [ ! -d repo ]; then
  git clone "$REPO" repo
fi
cd repo
git fetch --all --tags
git checkout "$PIN"

# python >=3.10; isolated venv (vendored poke-env never touches ours)
PY="${POKELLMON_PYTHON:-/Users/hoi/miniconda3/envs/SystemofML/bin/python}"
"$PY" -m venv ../venv
# shellcheck disable=SC1091
source ../venv/bin/activate
pip install --upgrade pip
# No requirements.txt upstream; deps inferred from imports (PokeLLMon + its vendored poke-env).
pip install openai orjson omegaconf "hydra-core" numpy tqdm websockets requests tabulate aiofiles gymnasium

# Patches (idempotent):
python3 - <<'PY'
import pathlib

# 1) route the OpenAI client at DeepSeek (honor OPENAI_BASE_URL) so PokeLLMon's
#    real prompts run on our DeepSeek key.
p = pathlib.Path("src/player/gpt_player.py"); t = p.read_text()
bad = "client = OpenAI(api_key=self.api_key)"
good = 'client = OpenAI(api_key=self.api_key, base_url=__import__("os").getenv("OPENAI_BASE_URL") or None)  # arena: DeepSeek route'
if bad in t:
    t = t.replace(bad, good); p.write_text(t); print("[patch] gpt_player base_url override")

# 2) src/player/__init__ eagerly imports LLMPlayer (torch, local-HF variant) which
#    the GPT/API path doesn't need. Make it optional.
p = pathlib.Path("src/player/__init__.py"); t = p.read_text()
bad = "from src.player.llm_player import LLMPlayer"
good = ("try:\n    from src.player.llm_player import LLMPlayer  # needs torch\n"
        "except Exception:  # arena: GPT/API path doesn't need torch\n    LLMPlayer = None")
if bad in t and "except Exception:  # arena" not in t:
    t = t.replace(bad, good); p.write_text(t); print("[patch] optional LLMPlayer import")
PY

echo "PokeLLMon (official) setup complete."
echo "Smoke: source ../venv/bin/activate && python -c 'from src.player import GPTPlayer; print(\"ok\")'  (run from repo/)"
