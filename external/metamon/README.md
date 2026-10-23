# Metamon bridge

[UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon) — RLC 2025.
Offline RL + Transformer; the NeurIPS 2025 PokéAgent Challenge baseline. 40+
pretrained checkpoints on HF [`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon)
(4.7M–200M params). Supports **gen9 OU** (and gen1–4 OU/UU/NU/Ubers).

We run a **pretrained** policy as an opponent — no retraining.

## ⚠️ Requires a Linux machine with an NVIDIA GPU

Metamon's amago Transformer needs **flash-attention 2 (CUDA-only)**. It installs
and imports on macOS/CPU but **cannot run the model there** — it aborts with
`AssertionError: Missing flash attention 2 install`. There is no Docker fix on a
machine without an NVIDIA GPU. Run the whole stack on a Linux+GPU box.

(amago has a `VanillaAttention` CPU fallback that's mathematically equivalent; a
gin override could force it for slow CPU inference, but it's untested here and not
wired up. The supported path is GPU.)

## Run on the GPU box

Everything meets on a local Showdown server, and Metamon hardcodes `localhost:8000`
— so run all three on the GPU machine:

```bash
# 1) Showdown server (sibling checkout) on :8000
cd pokemon-showdown && node pokemon-showdown start --no-security &

# 2) Metamon venv + gpu deps (idempotent; installs flash-attn on Linux)
cd pokemonbot && bash external/metamon/setup.sh

# 3) run the eval from our framework (our agents challenge the Metamon bot)
python -m arena.cli run configs/experiments/external_metamon.yaml
```

`setup.sh` auto-applies the amago `random_traj_length` int-cast patch. Pick a
checkpoint with `METAMON_CHECKPOINT` (default `SmallRLGen9Beta` for gen9 OU; pulled
from HF on first run into `METAMON_CACHE_DIR`, which `run.sh` defaults to `./cache`).
Its pinned poke-env fork installs INTO the venv, isolated from ours.

If the GPU box and your Showdown server are on different hosts, run the server on
the GPU box (Metamon can only reach `localhost`).

## Bridge contract

Metamon has a first-class challenge mode — no shim needed. `run.sh` runs:

```
python -m metamon.rl.evaluate --eval_type challenge --agent $METAMON_CHECKPOINT \
  --username $ARENA_USERNAME --opponent_username $ARENA_OPPONENT --role acceptor \
  --gens 9 --formats ou --total_battles N
```

(`ARENA_FORMAT=gen9ou` → `--gens 9 --formats ou`.) Grey-box: we capture Metamon's
per-turn moves from the public battle log into `traj_<name>.jsonl`; win/Elo are exact.

**Validate during reproduction:** confirm the acceptor's "ready" stdout line and
tune the config's `ready_regex` to match; verify `--agent` checkpoint names against
`python -c "from metamon.rl.pretrained import get_pretrained_model_names; print(get_pretrained_model_names())"`.
