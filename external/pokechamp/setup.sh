#!/usr/bin/env bash
# Reproduce PokeChamp (sethkarten/pokechamp, ICML 2025) in an isolated venv.
# Its minimax search ships with a *modified* poke_env, so we keep it fully
# separate from our framework's poke-env.
set -euo pipefail
cd "$(dirname "$0")"

REPO="https://github.com/sethkarten/pokechamp.git"
PIN="${POKECHAMP_PIN:-main}"   # set POKECHAMP_PIN=<sha> to lock a known-good commit

if [ ! -d repo ]; then
  git clone "$REPO" repo
fi
cd repo
git fetch --all --tags
git checkout "$PIN"

# PokeChamp uses uv for dependency management.
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
  exit 1
fi

# Apply arena reproduction patches (all idempotent) so a fresh clone works on
# macOS against a current Showdown server and routes its LLM to DeepSeek.
ARENA_IS_DARWIN="$([[ "$(uname)" == "Darwin" ]] && echo 1 || echo 0)" python3 - <<'PY'
import os, re, pathlib

# 1) macOS arm64 has no bitsandbytes wheel (local-quant only; unused for API models).
if os.environ.get("ARENA_IS_DARWIN") == "1":
    p = pathlib.Path("pyproject.toml"); t = p.read_text()
    t2 = re.sub(r'^(\s*)"bitsandbytes",\s*$',
                r'\1"bitsandbytes; sys_platform == \'linux\'",  # arena: no macOS wheel',
                t, flags=re.M)
    if t2 != t: p.write_text(t2); print("[patch] gated bitsandbytes to Linux")

# 2) Old vendored poke_env crashes on modern Showdown's "|player|pX|" (player-left)
#    message. Tolerate it instead of raising.
p = pathlib.Path("poke_env/environment/abstract_battle.py"); t = p.read_text()
bad = 'raise RuntimeError(f"Invalid player message: {split_message}")'
if bad in t:
    t = t.replace(bad, "pass  # arena: tolerate empty/player-left message")
    p.write_text(t); print("[patch] tolerate empty |player| message")

# 3) gpt_player: honor OPENAI_BASE_URL (DeepSeek route) + stop hardcoding gpt-4o
#    in the json branch so a deepseek-chat model id is actually used.
p = pathlib.Path("pokechamp/gpt_player.py"); t = p.read_text()
old_cli = "client = OpenAI(api_key=self.api_key)"
new_cli = 'client = OpenAI(api_key=self.api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)  # arena'
if old_cli in t:
    t = t.replace(old_cli, new_cli); print("[patch] gpt_player base_url override")
t = t.replace("response_format={\"type\": \"json_object\"},\n                    model='gpt-4o',",
              "response_format={\"type\": \"json_object\"},\n                    model=model,  # arena: honor requested model")
p.write_text(t)

# 4) llm_player: the VISUAL_EFFECTS minimax-progress branch references undefined
#    `depth`/`evaluated_nodes` -> NameError every call -> minimax silently falls
#    back to the damage calculator (its headline feature never runs). Drop it.
p = pathlib.Path("pokechamp/llm_player.py"); t = p.read_text()
bad = ("            if VISUAL_EFFECTS:\n"
       "                print(visual.minimax_progress(depth, evaluated_nodes, end_time - start_time))\n"
       "            else:\n"
       "                print(f\"[PERF] Optimized minimax: {end_time - start_time:.2f}s, nodes: {evaluated_nodes}, \"\n"
       "                      f\"Pool reuse: {stats['pool_stats']['reuse_rate']:.2f}, \"\n"
       "                      f\"Cache hit rate: {stats['cache_stats']['hit_rate']:.2f}\")")
good = ("            print(f\"[PERF] Optimized minimax: {end_time - start_time:.2f}s, \"\n"
        "                  f\"Pool reuse: {stats['pool_stats']['reuse_rate']:.2f}, \"\n"
        "                  f\"Cache hit rate: {stats['cache_stats']['hit_rate']:.2f}\")  # arena: drop buggy visual branch")
if bad in t:
    t = t.replace(bad, good); p.write_text(t); print("[patch] llm_player minimax visual NameError")
PY

uv sync

echo "PokeChamp setup complete."
echo "Smoke test (needs a running Showdown server + an LLM key, e.g. OPENROUTER_API_KEY):"
echo "  cd external/pokechamp/repo && uv run python -m pokechamp.play --help"
