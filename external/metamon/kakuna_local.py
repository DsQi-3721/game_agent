"""Metamon's Kakuna policy on this machine: local weights, plain attention, our Showdown port.

Run with the Metamon venv (external/metamon/venv), never the main environment.

  * weights: the checkpoint already downloaded to $METAMON_WEIGHTS (default /data/models/metamon)
  * attention: amago's SlidingWindowFlexAttention (PyTorch flex attention) instead of FlashAttention (no
    flash-attn wheel for this torch); same 96-timestep causal window, so the same function
  * Showdown: $ARENA_SHOWDOWN_URL (default http://localhost:8100) instead of Metamon's hard-coded :8000

    METAMON_CACHE_DIR=/data/wangxinqi/metamon_cache venv/bin/python kakuna_local.py strength --battles 20
    venv/bin/python kakuna_local.py accept       # arena external-opponent bridge (run.sh; contract: ../README.md)
"""

from __future__ import annotations

import argparse
import json
import os

import gin  # noqa: F401  (imported before metamon so the override below can be registered)

WEIGHTS = os.environ.get("METAMON_WEIGHTS", "/data/models/metamon")


def _patch_server() -> None:
    from poke_env.ps_client.server_configuration import ServerConfiguration

    import metamon.env.wrappers as w
    url = (os.environ.get("ARENA_SHOWDOWN_URL") or "http://localhost:8100").rstrip("/")
    host = url.split("://", 1)[-1]
    w.LocalhostServerConfiguration = ServerConfiguration(f"ws://{host}/showdown/websocket",
                                                         "https://play.pokemonshowdown.com/action.php?")


def kakuna(checkpoint: int = 34):
    from metamon.rl.pretrained import Kakuna

    class LocalKakuna(Kakuna):
        def __init__(self):
            super().__init__()
            # gin files are applied after Python overrides: rewrite the model config itself
            src = open(self.model_gin_config_path).read()
            lines = [ln for ln in src.splitlines() if "FlashAttention.window_size" not in ln]
            text = "\n".join(lines).replace("@transformer.FlashAttention", "@transformer.SlidingWindowFlexAttention")
            text += "\ntransformer.SlidingWindowFlexAttention.window_size = 96"   # == flash window_size (96, 0)
            out = os.path.join(os.environ.get("METAMON_CACHE_DIR", "/tmp"), "superkazam_vanilla.gin")
            with open(out, "w") as f:
                f.write(text + "\n")
            self.model_gin_config_path = out

        def get_path_to_checkpoint(self, ckpt: int) -> str:
            return os.path.join(WEIGHTS, self.model_name, "ckpts", "policy_weights", f"policy_epoch_{ckpt}.pt")

    m = LocalKakuna()
    m.default_checkpoint = checkpoint
    return m


def strength(args) -> None:
    import metamon
    from metamon.rl.evaluate.__main__ import pretrained_vs_baselines

    _patch_server()
    team_set = (metamon.env.TeamSet(args.team_dir, args.format) if args.team_dir   # e.g. only teams legal today
                else metamon.env.get_metamon_teams(args.format, args.team_set))
    res = pretrained_vs_baselines(kakuna(), battle_format=args.format, team_set=team_set,
                                  total_battles=args.battles, parallel_actors_per_baseline=args.parallel,
                                  baselines=args.baselines.split(","))
    print(json.dumps(res, indent=1, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)


def accept(args) -> None:
    """Accept the arena's challenges: ARENA_USERNAME / ARENA_OPPONENT / ARENA_FORMAT, ARENA_TEAM_FILE (mirror
    team; otherwise Metamon's competitive TeamSet), ARENA_N_BATTLES. No battle timer: Metamon starts one by default,
    our other opponents do not. "accepting challenges" is printed once the player is logged in (a challenge sent
    to a user who is not online yet is lost)."""
    import asyncio
    import shutil
    import tempfile

    import metamon
    import metamon.env.wrappers as w
    from metamon.rl.evaluate.__main__ import _pretrained_on_ladder
    from metamon.rl.metamon_to_amago import make_challenge_env

    _patch_server()
    fmt, user = os.environ["ARENA_FORMAT"], os.environ["ARENA_USERNAME"]
    team_file = os.environ.get("ARENA_TEAM_FILE")
    if team_file:
        d = tempfile.mkdtemp(prefix="arena_team_")
        shutil.copy(team_file, os.path.join(d, f"team.{fmt}_team"))
        team_set = metamon.env.TeamSet(d, fmt)
    else:
        team_set = metamon.env.get_metamon_teams(fmt, "competitive")

    init = w.ChallengeByUsername.__init__

    def init_then_ready(self, *a, **kw):
        init(self, *a, **kw)
        from poke_env.concurrency import POKE_LOOP
        asyncio.run_coroutine_threadsafe(self.agent.ps_client.logged_in.wait(), POKE_LOOP).result(timeout=300)
        print(f"[metamon] {user} logged in; accepting challenges", flush=True)

    w.ChallengeByUsername.__init__ = init_then_ready
    model = kakuna(args.checkpoint)
    _pretrained_on_ladder(model, make_ladder=make_challenge_env,
                          total_battles=int(os.environ.get("ARENA_N_BATTLES", "1000")),
                          checkpoint=args.checkpoint, log_to_wandb=False,
                          player_username=user, opponent_username=os.environ["ARENA_OPPONENT"], role="acceptor",
                          player_team_set=team_set, battle_backend=model.battle_backend, battle_format=fmt,
                          start_timer_on_battle_start=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strength", help="Kakuna vs Metamon's heuristic baselines on the local Showdown")
    s.add_argument("--format", default="gen9ou")
    s.add_argument("--team-set", default="competitive")
    s.add_argument("--team-dir", default=None, help="directory of .<format>_team files (overrides --team-set)")
    s.add_argument("--battles", type=int, default=40)
    s.add_argument("--parallel", type=int, default=2)
    s.add_argument("--baselines", default="PokeEnvHeuristic,Grunt")
    s.add_argument("--out", default=None)
    a = sub.add_parser("accept", help="accept the arena's challenges (external-opponent bridge)")
    a.add_argument("--checkpoint", type=int, default=34)
    args = ap.parse_args()
    if args.cmd == "strength":
        strength(args)
    elif args.cmd == "accept":
        accept(args)


if __name__ == "__main__":
    main()
