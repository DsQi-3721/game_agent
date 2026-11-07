"""Label Metamon trajectories with Kakuna's action distribution (offline, no battles).

For every battle file (``*.json.lz4``, Metamon format) the whole trajectory goes through Metamon's own
dataset pipeline (observations, legal-action mask, previous action / reward) and one forward pass of
Kakuna's tstep encoder -> trajectory transformer -> actor gives P(action | history) for every step.

Output JSONL, one line per battle: {"file", "probs": [[13 floats] per step], "chosen": [...], "legal": [[...]]}
(Metamon action indices: 0-3 moves and 9-12 tera moves in alphabetical move order, 4-8 switches in
alphabetical species order). Run with the Metamon venv:

``--extra-illegal masks.jsonl`` (``scripts/kakuna_masks.py``): actions the trajectory shows were not
selectable (0 PP, choice lock, ...) are added to Metamon's legal mask, so ``probs`` is Kakuna's own masked
policy over what could really be chosen; the unmasked rows are kept as ``probs_metamon_mask`` with the
probability mass they put on those actions (``extra_mass``). ``--causal-check K``: also labels each
trajectory cut after step K and reports the largest change at steps <= K (must be 0: the policy is causal).

    METAMON_CACHE_DIR=/data/wangxinqi/metamon_cache venv/bin/python kakuna_label.py \
        --files "/data/.../gen9ou/*.json.lz4" --out labels.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from kakuna_local import kakuna


def _loader(model):
    """A MetamonDataset that only loads given files (no dataset-root scan)."""
    import copy

    from metamon.data.parsed_replay_dset import MetamonDataset
    d = MetamonDataset.__new__(MetamonDataset)
    d.observation_space = copy.deepcopy(model.observation_space)
    d.action_space = copy.deepcopy(model.action_space)
    d.reward_function = copy.deepcopy(model.reward_function)
    d.max_seq_len = None
    d.dset_root = "/"
    return d


WINDOW = 96   # Kakuna: FlashAttention.window_size = (96, 0)  ==  q_idx - kv_idx <= 96


def _sdpa_window(module):
    """Whole-sequence causal sliding-window attention with plain SDPA and an explicit mask.

    Same function as the flex-attention path (default 1/sqrt(d) scale, no dropout), but no per-length
    kernel compilation: compiled flex kernels for long battles exceed the A800's shared memory."""
    import torch.nn.functional as F

    def forward(qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, dim=2))      # b h l e
        L = q.shape[2]
        i = torch.arange(L, device=q.device)
        mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] <= WINDOW)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)   # b l h e
    return forward


def _label(policy, wrap, data, device, extra=None):
    """Kakuna's per-step action distribution for one trajectory; ``extra``: {step: [action indices]} to add to
    the illegal-action mask before the actor."""
    from amago.loading import RLData_pad_collate
    rl = wrap._process_data(data)
    batch = RLData_pad_collate([rl]).to(device)
    o = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
    emb, _ = policy.traj_encoder(o, time_idxs=batch.time_idxs, hidden_state=None)
    straight = {k: batch.obs[k].clone() for k in policy.pass_obs_keys_to_actor}
    if extra:
        mask = straight["illegal_actions"]
        for t, idx in extra.items():
            if t < mask.shape[1]:
                for i in idx:
                    mask[0, t, i] = True if mask.dtype == torch.bool else 1
        straight["illegal_actions"] = mask
    dist = policy.actor(emb, straight_from_obs=straight)
    return dist.probs[0, :, -1, :].float().cpu(), None      # [T+1, 13], last gamma = test-time policy


def _cut(data, k):
    """The trajectory up to (and including) decision k, as if the battle had ended there."""
    obs, action_infos, rewards, dones = data
    import numpy as np
    obs_k = {key: v[: k + 2] for key, v in obs.items()}
    infos_k = {key: v[: k + 1] for key, v in action_infos.items()}
    dones_k = np.array(dones[: k + 1]).copy()
    dones_k[-1] = True
    return obs_k, infos_k, np.array(rewards[: k + 1]), dones_k


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="", help="glob of Metamon trajectory files")
    ap.add_argument("--file-list", default=None, help="text file with one trajectory path per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint", type=int, default=34)
    ap.add_argument("--extra-illegal", default=None, help="jsonl from scripts/kakuna_masks.py")
    ap.add_argument("--causal-check", type=int, default=0, help="also label the trajectory cut after this step")
    args = ap.parse_args()
    extra = {}
    if args.extra_illegal:
        for line in open(args.extra_illegal):
            r = json.loads(line)
            if r.get("steps"):
                extra[r["file"]] = {int(t): v for t, v in r["steps"].items()}

    from amago.loading import RLData_pad_collate
    from metamon.rl.metamon_to_amago import MetamonAMAGODataset

    model = kakuna(args.checkpoint)
    exp = model.initialize_agent(checkpoint=args.checkpoint, log=False)
    policy = exp.policy.eval()
    # flex attention runs whole sequences only in "training" mode (the eval path is the step-by-step kv cache);
    # it applies no dropout, so flipping just the attention modules gives the eval-mode function
    from amago.nets.transformer import FlexAttention
    for m in policy.modules():
        if isinstance(m, FlexAttention):
            m.train()
            m.forward = _sdpa_window(m)
    device = next(policy.parameters()).device
    loader = _loader(model)
    wrap = MetamonAMAGODataset.__new__(MetamonAMAGODataset)
    wrap.parsed_replay_dset = loader

    files = sorted(glob.glob(args.files, recursive=True)) if args.files else []
    if args.file_list:
        files += [x.strip() for x in open(args.file_list) if x.strip()]
    files = files[:args.limit] if args.limit else files
    t0, n_steps = time.time(), 0
    with open(args.out, "w") as out:
        for i, f in enumerate(files):
            try:
                obs, action_infos, rewards, dones = loader.load_filename(f)
            except Exception as e:  # noqa: BLE001
                print(f"skip {os.path.basename(f)}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            try:
                probs, emb_inputs = _label(policy, wrap, (obs, action_infos, rewards, dones), device)
                rec = {}
                if f in extra:                       # Kakuna's own mask over what could really be chosen
                    rec["probs_metamon_mask"] = [[round(x, 5) for x in r] for r in probs.tolist()]
                    probs = _label(policy, wrap, (obs, action_infos, rewards, dones), device, extra=extra[f])[0]
                    rec["extra_illegal"] = {str(t): v for t, v in extra[f].items()}
                    rec["extra_mass"] = {str(t): round(sum(rec["probs_metamon_mask"][t][i] for i in v), 5)
                                         for t, v in extra[f].items() if t < len(rec["probs_metamon_mask"])}
                if args.causal_check and len(action_infos["chosen"]) > args.causal_check + 1:
                    k = args.causal_check
                    cut = _cut((obs, action_infos, rewards, dones), k)
                    short = _label(policy, wrap, cut, device, extra=extra.get(f))[0]
                    rec["causal_max_abs_diff"] = round(float((short[:k + 1] - probs[:k + 1]).abs().max()), 7)
            except Exception as e:  # noqa: BLE001  (one bad battle must not stop the run)
                print(f"skip {os.path.basename(f)}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            n = len(action_infos["chosen"])
            out.write(json.dumps({"file": f, "probs": [[round(x, 5) for x in r] for r in probs[:n].tolist()],
                                  "chosen": [int(c) for c in action_infos["chosen"]],
                                  "missing": [bool(m) for m in action_infos["missing"]],
                                  "legal": [sorted(int(a) for a in l) for l in action_infos["legal"]], **rec}) + "\n")
            n_steps += n
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(files)} battles, {n_steps} steps, {n_steps / (time.time() - t0):.1f} steps/s",
                      flush=True)
    print(f"done: {len(files)} battles, {n_steps} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
