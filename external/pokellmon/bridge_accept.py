"""Official PokeLLMon accept-challenges bridge (runs INSIDE PokeLLMon's own venv).

Constructs the real `GPTPlayer` in FULL PokeLLMon mode and parks it on
`accept_challenges(ARENA_OPPONENT, n)` so our framework's player can challenge it.
This runs PokeLLMon's actual code (its prompts, knowledge injection, self-
consistency, and accumulated text-history feedback) -- not our native reimpl.

Imports PokeLLMon's vendored poke-env (`src.*`), so it lives in external/ and runs
in the repo's venv, never on the `arena` import path. GPTPlayer's OpenAI client is
patched (setup.sh) to honor OPENAI_BASE_URL -> DeepSeek.
"""

import asyncio
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "repo")
sys.path.insert(0, _REPO)
os.chdir(_REPO)  # PokeLLMon loads data/prompt templates via cwd-relative paths

from omegaconf import OmegaConf
from src.client import AccountConfiguration, LocalhostServerConfiguration
from src.data.normalize import to_id_str
from src.player import GPTPlayer

USERNAME = os.environ["ARENA_USERNAME"]
OPPONENT = os.environ["ARENA_OPPONENT"]
FMT = os.environ.get("ARENA_FORMAT", "gen9randombattle")
N = int(os.environ.get("ARENA_N_BATTLES", "1000"))
KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("POKELLMON_MODEL", "deepseek-chat")

# FULL PokeLLMon (the paper's headline agent -- defaults ship all three OFF):
#   knowledge=True  -> (A) knowledge-augmented generation (move effects + type buckets)
#   reason_algo=sc  -> (B) consistent action generation (self-consistency over CoT)
#   T=2             -> (C) in-context feedback: render the last T+1 turns of history
config = OmegaConf.create({
    "battle_format": FMT,
    "knowledge": True,
    "reason_algo": "sc",
    "T": 2,
    "n_shot": 1,
    "temperature": 0.7,
    "max_output_length": 500,
    "seed": 0,
})

os.makedirs("./battle_log/bridge", exist_ok=True)

player = GPTPlayer(
    config=config, api_key=KEY, backend=MODEL, reason_algo="sc",
    save_replay_dir="./battle_log/bridge",
    account_configuration=AccountConfiguration(USERNAME, None),
    server_configuration=LocalhostServerConfiguration,
)
player._dynamax_disable = True


async def main() -> None:
    try:
        await player.ps_client.wait_for_login()
    except Exception:  # noqa: BLE001  (older vendored poke-env: best-effort)
        await asyncio.sleep(3)
    print(f"[pokellmon] logged in as {USERNAME}; accepting challenges from {OPPONENT}", flush=True)
    await player.accept_challenges(to_id_str(OPPONENT), N)


if __name__ == "__main__":
    asyncio.run(main())
