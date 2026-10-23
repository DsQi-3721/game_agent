"""PokeChamp accept-challenges bridge shim (runs INSIDE PokeChamp's own venv).

PokeChamp ships no "accept challenges from a username" mode -- only local_1v1
(both players in one process) and ladder play. This shim builds its minimax
`LLMPlayer` via the repo's own `get_llm_player(...)` and parks it on
`accept_challenges(ARENA_OPPONENT, n)` so our framework's player can challenge it.

It imports PokeChamp's vendored poke_env (NOT ours) -- that's why it lives in
external/ and is launched in the repo's venv, never on the `arena` import path.

Handshake env (see ../README.md): ARENA_SERVER / ARENA_USERNAME / ARENA_OPPONENT
/ ARENA_FORMAT, plus POKECHAMP_MODEL + OPENAI_API_KEY/OPENROUTER_API_KEY.
"""

import argparse
import asyncio
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "repo")
sys.path.insert(0, _REPO)  # PokeChamp repo root (common.py, poke_env, ...)
os.chdir(_REPO)            # PokeChamp loads static data (pokedex/move effects) via cwd-relative paths

from common import *  # noqa: F401,F403  (PNUMBER1, prompt_algos, ...)
from poke_env.data import to_id_str
from poke_env.player.team_util import get_llm_player, load_random_team
from pokechamp.gpt_player import GPTPlayer

USERNAME = os.environ["ARENA_USERNAME"]
OPPONENT = os.environ["ARENA_OPPONENT"]
FMT = os.environ["ARENA_FORMAT"]
N = int(os.environ.get("ARENA_N_BATTLES", "1000"))
BACKEND = os.environ.get("POKECHAMP_MODEL", "gpt-4o")
KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

# args namespace get_llm_player expects (.temperature/.log_dir/.device)
# PokeChamp writes a replay here at battle end; ensure it exists or it crashes the
# battle-finish handler (FileNotFoundError) and the battle never finalizes.
_LOG_DIR = os.path.join(_REPO, "battle_log", "bridge")
os.makedirs(_LOG_DIR, exist_ok=True)
_ns = argparse.Namespace(temperature=0.3, log_dir=_LOG_DIR, device=0)


async def main() -> None:
    # Force GPTPlayer as the LLM backend: a model id like "deepseek-chat" matches
    # none of PokeChamp's routing branches (would raise), and GPTPlayer is patched
    # to honor OPENAI_BASE_URL so it hits DeepSeek's OpenAI-compatible endpoint.
    llm_backend = GPTPlayer(KEY)
    # online=False -> server_configuration=None -> poke-env localhost (ws://localhost:8000).
    # PNUMBER1='' (default) so the login name is EXACTLY ARENA_USERNAME (our side
    # challenges that exact id; any suffix would break the handshake).
    player = get_llm_player(_ns, BACKEND, "minimax", "pokechamp", KEY=KEY,
                            battle_format=FMT, USERNAME=USERNAME, online=False,
                            llm_backend=llm_backend)

    # wait until actually logged in before announcing ready (avoid a challenge race)
    try:
        await player.ps_client.logged_in.wait()
    except Exception:  # noqa: BLE001  (older poke_env: best-effort)
        await asyncio.sleep(3)
    print(f"[pokechamp] logged in as {USERNAME}; accepting challenges from {OPPONENT}", flush=True)

    for _ in range(N):
        if "random" not in FMT:
            player.update_team(load_random_team())
        await player.accept_challenges(to_id_str(OPPONENT), 1)


if __name__ == "__main__":
    asyncio.run(main())
