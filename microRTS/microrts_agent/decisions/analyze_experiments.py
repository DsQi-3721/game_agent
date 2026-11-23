"""Comprehensive analysis & ablation experiment suite for SA-RLCD vs baselines on MicroRTS.

Runs 4 diagnostic experiments on A800 GPU 0 and writes structured JSON outputs:
  1. 60-game Unseen-Map Progression Ladder (Base -> SFT -> R1 -> R2 -> Stage2a -> R3 -> R4)
     plus External Java Competition Bot Strength Anchors (WorkerRush, LightRush, HeavyRush,
     RangedRush, Droplet, CoacAI, Mayari) on the exact same 60 games.
  2. Zero-Shot Unseen Strong Opponents Stress Test (HeavyRush, RangedRush, Droplet, Tiamat,
     StrategyTactics on unseen maps 8x8C, 12x12B, 16x16B) for SFT, R1, R2, Stage2a, R4.
  3. Within-Game Tick-by-Tick Win-Probability & Potential Trajectories + Tactical Skill Profiles
     + Exact Murphy Brier Decomposition (Reliability / Resolution / Uncertainty) & AUROC.
  4. Merged-LoRA Exported Decision-Only Model vs Adapter SharedPrefixCache Latency Benchmark.
"""

import argparse
import contextlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import jpype
import numpy as np
import torch
from rlcd.export import load_decision_only
from rlcd.schema import Question, render_prefix, render_suffix

from . import eve_policy
from . import state as state_mod
from .eve_policy import DEFAULT_MODEL, EveRLCDPolicy, SharedPrefixCache, resolve_device
from .metrics import aggregate_games, atomic_json, episode_outcome
from .train import EnvPool, evaluate_policy, play_episode

UNSEEN_MAPS = [
    "maps/8x8/basesWorkers8x8C.xml",
    "maps/12x12/basesWorkers12x12B.xml",
    "maps/16x16/basesWorkers16x16B.xml",
]
BENCH_OPPS = ["RandomBiasedAI", "LightRush", "WorkerRush", "CoacAI", "Mayari"]
UNSEEN_OPPS = ["HeavyRush", "RangedRush", "Droplet", "Tiamat", "StrategyTactics"]
EXTERNAL_BOTS = ["WorkerRush", "LightRush", "HeavyRush", "RangedRush", "Droplet", "CoacAI", "Mayari"]


@contextlib.contextmanager
def native_state_format(use_force=True):
    """Evaluate pre-Force checkpoints (Base, SFT, R1, R2) in their exact native prompt format."""
    if use_force:
        yield
        return
    orig_state_fn = state_mod.state_to_text
    orig_policy_fn = eve_policy.state_to_text

    def _no_force_state_to_text(st):
        txt = orig_state_fn(st)
        return "\n".join(ln for ln in txt.splitlines() if not ln.startswith("Force: "))

    state_mod.state_to_text = _no_force_state_to_text
    eve_policy.state_to_text = _no_force_state_to_text
    try:
        yield
    finally:
        state_mod.state_to_text = orig_state_fn
        eve_policy.state_to_text = orig_policy_fn


def run_external_java_bots(out_path, max_ticks=2500):
    """Evaluate pure Java competition bots on the exact same 60 unseen-map games."""
    from microrts_agent.envs.rl_vec_env import MicroRTSRLVecEnv
    from microrts_agent.registries import ai

    from .env import _seed_java_bot

    results = {}
    for bot_name in EXTERNAL_BOTS:
        games = []
        idx = 0
        for m in UNSEEN_MAPS:
            for opp in BENCH_OPPS:
                for player in (0, 1):
                    # MicroRTSRLVecEnv handles JVM init & UTT cleanly; run bot_name vs opp directly in Java
                    vec = MicroRTSRLVecEnv(
                        num_selfplay_envs=0,
                        num_bot_envs=1,
                        partial_obs=False,
                        max_steps=max_ticks,
                        ai2s=[ai.AI_MAPPING[opp]],
                        map_paths=[m],
                        reward_weight=np.array([1.0] + [0.0] * 9),
                        cycle_maps=[],
                        jvm_args=["-Djava.awt.headless=true"],
                        player=player,
                    )
                    try:
                        client = vec.vec_client.clients[0]
                        utt = vec.real_utt
                        ai_me = ai.AI_MAPPING[bot_name](utt)
                        for rep in range(2):
                            seed = 200_000 + idx * 100 + rep
                            client.reset(player)
                            ai_me.reset(utt)
                            jpype.JClass("rts.UnitAction").r.setSeed(seed)
                            _seed_java_bot(ai_me, seed)
                            _seed_java_bot(client.ai2, seed)
                            gs = client.gs
                            while not gs.gameover() and int(gs.getTime()) < max_ticks:
                                pa_me = ai_me.getAction(player, gs)
                                pa_opp = client.ai2.getAction(1 - player, gs)
                                gs.issueSafe(pa_me)
                                gs.issueSafe(pa_opp)
                                gs.cycle()
                            terminated = bool(gs.gameover())
                            truncated = not terminated and int(gs.getTime()) >= max_ticks
                            winner = int(gs.winner()) if terminated else None
                            outcome = episode_outcome(terminated, truncated, winner, player)
                            games.append(
                                {
                                    "map": m,
                                    "opponent": opp,
                                    "player": player,
                                    "seed": seed,
                                    "outcome": outcome,
                                    "ticks": int(gs.getTime()),
                                    "decisions": 0,
                                    "reward": 1.0 if outcome == "win" else (-1.0 if outcome == "loss" else 0.0),
                                    "terminated": terminated,
                                    "truncated": truncated,
                                }
                            )
                    finally:
                        vec.close()
                    idx += 1
        summary = aggregate_games(games)
        by_opp = {}
        for o in BENCH_OPPS:
            rs = [g for g in games if g["opponent"] == o]
            by_opp[o] = {
                "win": sum(g["outcome"] == "win" for g in rs),
                "loss": sum(g["outcome"] == "loss" for g in rs),
                "timeout": sum(g["outcome"] == "timeout" for g in rs),
                "draw": sum(g["outcome"] == "draw" for g in rs),
                "win_rate": sum(g["outcome"] == "win" for g in rs) / len(rs),
            }
        results[bot_name] = {"summary": summary, "by_opponent": by_opp}
        print(
            json.dumps(
                {
                    "external_bot": bot_name,
                    "win_rate": round(summary["win_rate"], 4),
                    "W": summary["win"],
                    "L": summary["loss"],
                    "T": summary["timeout"],
                    "D": summary["draw"],
                }
            ),
            flush=True,
        )
    atomic_json(out_path, results)
    return results


def eval_model_suite(name, ckpt, use_force, pool, out_dir, *, opponents, repeats, seed_base, device):
    out_file = out_dir / f"{name}.json"
    if out_file.exists():
        print(f"[skip] {out_file} already exists", flush=True)
        return json.loads(out_file.read_text())
    t0 = time.monotonic()
    if ckpt is None:
        pol = EveRLCDPolicy(DEFAULT_MODEL, device=device, rank=16, dtype="bf16")
    else:
        pol = EveRLCDPolicy.load(ckpt, device=device, dtype="bf16")
    try:
        with native_state_format(use_force=use_force):
            games, summary = evaluate_policy(
                pol,
                pool,
                maps=UNSEEN_MAPS,
                opponents=opponents,
                players=[0, 1],
                repeats=repeats,
                seed_base=seed_base,
                greedy=True,
            )
    finally:
        del pol
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "name": name,
        "checkpoint": str(ckpt) if ckpt else "BASE_ZERO_SHOT",
        "use_force": use_force,
        "elapsed_sec": time.monotonic() - t0,
        "summary": summary,
        "games": games,
    }
    atomic_json(out_file, payload)
    print(
        json.dumps(
            {
                "eval_done": name,
                "games": summary["games"],
                "win_rate": round(summary["win_rate"], 4),
                "W/L/T": f"{summary['win']}/{summary['loss']}/{summary['timeout']}",
                "brier": round(summary.get("mean_episode_brier", -1), 4),
                "ece": round(summary.get("game_weighted_ece", -1), 4),
                "elapsed_sec": round(payload["elapsed_sec"], 1),
            }
        ),
        flush=True,
    )
    return payload


def capture_trajectories_and_skills(out_path, pool, device):
    """Capture within-game p_win / Phi(s) trajectories + tactical skill profiles."""
    showcase = [
        ("maps/8x8/basesWorkers8x8C.xml", "Mayari", 0),
        ("maps/12x12/basesWorkers12x12B.xml", "Mayari", 0),
        ("maps/16x16/basesWorkers16x16B.xml", "WorkerRush", 1),
        ("maps/12x12/basesWorkers12x12B.xml", "CoacAI", 0),
        ("maps/8x8/basesWorkers8x8C.xml", "RandomBiasedAI", 0),
    ]
    models = [
        ("SFT_v2pro", "runs/a800-v2-pro/sft/best.pt", False),
        ("R1_eve_rlcd", "runs/a800-v2-pro/online-rl/best.pt", False),
        ("R2_antirush", "runs/a800-r2-antirush/online-rl/best.pt", False),
        ("Stage2a_IS_RLCD", "runs/a800-r3-sa-rlcd/stage2a-is-rlcd/best.pt", True),
        ("R3_sa_rlcd_v1", "runs/a800-r3-sa-rlcd/stage2b-online-sa-rlcd/best.pt", True),
        ("R4_sa_rlcd_v2", "runs/a800-r4-sa-rlcd-v2/online-sa-rlcd-v2/best.pt", True),
    ]
    trajectories = defaultdict(dict)
    skill_profiles = {}

    for name, ckpt, use_force in models:
        pol = EveRLCDPolicy.load(ckpt, device=device, dtype="bf16")
        skills = Counter()
        total_dts = []
        try:
            with native_state_format(use_force=use_force):
                for m, opp, p in showcase:
                    env = pool.get(m, opp, p)
                    steps, _, gm = play_episode(env, pol, seed=200_000, greedy=True)
                    tick = 0
                    pts = []
                    for st in steps:
                        pts.append(
                            {
                                "tick": tick,
                                "p_win": round(float(st["win_probability"]), 4),
                                "potential": round(float(st.get("state_potential", 0.0)), 4),
                                "duration": int(st["duration"]),
                            }
                        )
                        tick += int(st["duration"])
                        total_dts.append(int(st["duration"]))
                        for ud in st["unit_decisions"]:
                            cs = ud["chosen_skill"]
                            lbl = cs["skill"] + (":" + cs["unit_type"] if cs.get("unit_type") else "")
                            skills[lbl] += 1
                    key = f"{Path(m).stem.replace('basesWorkers', '')}_P{p}_vs_{opp}"
                    trajectories[name][key] = {
                        "outcome": gm["outcome"],
                        "ticks": gm["ticks"],
                        "decisions": gm["decisions"],
                        "mean_p": gm.get("mean_win_probability"),
                        "brier": gm.get("win_brier"),
                        "points": pts,
                    }
        finally:
            del pol
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        total_s = sum(skills.values()) or 1
        skill_profiles[name] = {
            "total_unit_decisions": total_s,
            "mean_dt": float(np.mean(total_dts)) if total_dts else 0.0,
            "counts": dict(skills),
            "ratios": {k: v / total_s for k, v in skills.items()},
        }
        print(json.dumps({"profile_done": name, "mean_dt": round(skill_profiles[name]["mean_dt"], 2), "top": skills.most_common(5)}), flush=True)

    atomic_json(out_path, {"trajectories": dict(trajectories), "skill_profiles": skill_profiles})


def benchmark_exported_vs_adapter(out_path, device):
    """Benchmark latency of merged decision-only export vs LoRA adapter across Q=1,3,8,16."""
    adapter_ckpt = "runs/a800-r4-sa-rlcd-v2/online-sa-rlcd-v2/best.pt"
    export_dir = "runs/a800-r4-sa-rlcd-v2/exported-decision"
    ctx = (
        "MicroRTS t=220 map=12x12 me=P0 opp=Mayari\n"
        "Res: me=4(avail=4) opp=3(avail=3)\n"
        "Force: me[B=1 Rax=1 W=3 Mil=2 hp=22] vs opp[B=1 Rax=1 W=2 Mil=2 hp=20] phase=mid\n"
        "My units (7):\n"
        "- U6 Base(2,1) hp=10/10 idle\n- U8 Worker(1,2) hp=1/1 act=harvest(4t)\n"
        "- U9 Worker(2,3) hp=1/1 idle\n- U10 Barracks(3,2) hp=4/4 idle\n"
        "- U11 Light(5,5) hp=4/4 idle\n- U12 Ranged(4,5) hp=1/1 idle\n"
        "- U13 Worker(1,3) hp=1/1 act=return(2t)\n"
        "Enemy units (6):\n"
        "- U7 Base(9,10) hp=10/10 idle\n- U14 Barracks(8,9) hp=4/4 idle\n"
        "- U15 Worker(10,9) hp=1/1 idle\n- U16 Worker(9,8) hp=1/1 idle\n"
        "- U17 Light(6,6) hp=4/4 idle\n- U18 Ranged(7,6) hp=1/1 idle"
    )
    base_q = Question(
        "choice",
        ctx,
        "Best skill for Light U11 at (5,5) to win this match?",
        ["U11: attack nearest enemy", "U11: attack Base U7(9,10)", "U11: move to (6,6)", "Wait 4t"],
    ).validate()
    wq = Question(
        "noul",
        ctx,
        "Following the current greedy policy against this opponent, will our player win this match? A draw counts as false.",
        ["true", "false"],
    ).validate()

    def _bench_shared_model(backbone, letter_head, tok, prepend_bos, pad_id, qs, iters=15):
        prefix_ids = ([tok.bos_token_id] if prepend_bos else []) + tok.encode(
            render_prefix(ctx), add_special_tokens=False
        )
        suffix_seqs = [tok.encode(render_suffix(q), add_special_tokens=False) for q in qs]
        p_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        m = len(suffix_seqs)
        width = max(len(s) for s in suffix_seqs)
        s_ids = torch.full((m, width), pad_id, dtype=torch.long, device=device)
        s_mask = torch.zeros((m, width), dtype=torch.long, device=device)
        s_last = torch.zeros(m, dtype=torch.long, device=device)
        for i, s in enumerate(suffix_seqs):
            s_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            s_mask[i, : len(s)] = 1
            s_last[i] = len(s) - 1
        p_len = p_tensor.shape[1]
        attn_mask = torch.cat([torch.ones((m, p_len), dtype=torch.long, device=device), s_mask], dim=1)
        pos_ids = torch.arange(p_len, p_len + width, device=device).unsqueeze(0).expand(m, -1)

        times, pref_ms, suff_ms = [], [], []
        with torch.no_grad():
            for it in range(iters + 3):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                prefill_out = backbone(input_ids=p_tensor, use_cache=True)
                cache = SharedPrefixCache(prefill_out.past_key_values)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = backbone(
                    input_ids=s_ids,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=cache,
                    use_cache=False,
                )
                rows = out[0][torch.arange(m, device=device), s_last]
                _ = letter_head(rows)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                if it >= 3:
                    times.append((t2 - t0) * 1000.0)
                    pref_ms.append((t1 - t0) * 1000.0)
                    suff_ms.append((t2 - t1) * 1000.0)
        return {
            "total_ms": float(np.median(times)),
            "ms_per_q": float(np.median(times) / m),
            "prefill_ms": float(np.median(pref_ms)),
            "suffix_ms": float(np.median(suff_ms)),
        }

    results = {"q_counts": [1, 3, 8, 16], "lora_adapter": {}, "merged_export": {}, "uncached_merged": {}}

    # 1. LoRA adapter
    pol = EveRLCDPolicy.load(adapter_ckpt, device=device, dtype="bf16")
    try:
        for nq in [1, 3, 8, 16]:
            qs = [base_q] * max(0, nq - 1) + [wq]
            ks = [q.k for q in qs]
            rlcd_mod = pol.rlcd
            results["lora_adapter"][str(nq)] = _bench_shared_model(
                pol.backbone,
                lambda rows, _m=rlcd_mod, _ks=ks: _m.logits_from_hidden(rows, _ks),
                pol.rlcd.tok,
                pol.rlcd.prepend_bos,
                pol.rlcd.pad_id,
                qs,
            )
    finally:
        pol = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Merged decision-only export (bf16 on CUDA)
    exp = load_decision_only(export_dir, device=device)
    exp.model.to(dtype=torch.bfloat16)
    try:
        for nq in [1, 3, 8, 16]:
            qs = [base_q] * max(0, nq - 1) + [wq]
            ks = [q.k for q in qs]
            results["merged_export"][str(nq)] = _bench_shared_model(
                exp.model,
                lambda rows, _e=exp, _ks=ks: _e.logits_from_hidden(rows.float(), _ks),
                exp.tok,
                exp.prepend_bos,
                exp.pad_id,
                qs,
            )
            # Uncached full-prompt sequential baseline on merged model
            t_unc = []
            with torch.no_grad():
                for it in range(8):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for q in qs:
                        exp.decision_logits([q], 2048, device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    if it >= 2:
                        t_unc.append((t1 - t0) * 1000.0)
            results["uncached_merged"][str(nq)] = {
                "total_ms": float(np.median(t_unc)),
                "ms_per_q": float(np.median(t_unc) / nq),
            }
    finally:
        exp = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(out_path, results)
    print(json.dumps({"latency_bench": results}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/analysis-suite"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = resolve_device(args.device)
    out = args.out
    (out / "ladder_60g").mkdir(parents=True, exist_ok=True)
    (out / "unseen_opps_30g").mkdir(parents=True, exist_ok=True)

    print("=== EXP 1B: External Java Competition Bot Baselines (60 unseen-map games each) ===", flush=True)
    if not (out / "external_bots_60g.json").exists():
        run_external_java_bots(out / "external_bots_60g.json", max_ticks=2500)

    pool = EnvPool(max_ticks=2500, skill_ticks=64, candidate_limit=12, gamma=1.0, reward="win_focused")
    try:
        print("=== EXP 1A: Full 60-Game Progression Ladder (Base, SFT, R1, Stage2a) ===", flush=True)
        for name, ckpt, use_force in [
            ("Base_ZeroShot", None, False),
            ("SFT_v2pro", "runs/a800-v2-pro/sft/best.pt", False),
            ("R1_eve_rlcd", "runs/a800-v2-pro/online-rl/best.pt", False),
            ("Stage2a_IS_RLCD", "runs/a800-r3-sa-rlcd/stage2a-is-rlcd/best.pt", True),
        ]:
            eval_model_suite(
                name,
                ckpt,
                use_force,
                pool,
                out / "ladder_60g",
                opponents=BENCH_OPPS,
                repeats=2,
                seed_base=200_000,
                device=device,
            )

        print("=== EXP 2: Zero-Shot Unseen Strong Opponents Stress Test (30 games each) ===", flush=True)
        for name, ckpt, use_force in [
            ("SFT_v2pro", "runs/a800-v2-pro/sft/best.pt", False),
            ("R1_eve_rlcd", "runs/a800-v2-pro/online-rl/best.pt", False),
            ("R2_antirush", "runs/a800-r2-antirush/online-rl/best.pt", False),
            ("Stage2a_IS_RLCD", "runs/a800-r3-sa-rlcd/stage2a-is-rlcd/best.pt", True),
            ("R4_sa_rlcd_v2", "runs/a800-r4-sa-rlcd-v2/online-sa-rlcd-v2/best.pt", True),
        ]:
            eval_model_suite(
                name,
                ckpt,
                use_force,
                pool,
                out / "unseen_opps_30g",
                opponents=UNSEEN_OPPS,
                repeats=1,
                seed_base=300_000,
                device=device,
            )

        print("=== EXP 3: Within-Game Trajectories & Skill Profiles ===", flush=True)
        capture_trajectories_and_skills(out / "trajectories_and_skills.json", pool, device)
    finally:
        pool.close()

    print("=== EXP 4: Merged-LoRA Export vs Adapter Latency Benchmark ===", flush=True)
    benchmark_exported_vs_adapter(out / "export_latency_bench.json", device)
    print("=== ALL ANALYSIS EXPERIMENTS COMPLETED ===", flush=True)


if __name__ == "__main__":
    main()
