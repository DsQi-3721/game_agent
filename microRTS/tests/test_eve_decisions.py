"""RLCD estimator and adapter contract; tiny real transformers, no network/weights."""

import itertools
import json
import string

import pytest
import torch

pytest.importorskip("rlcd", reason="Install setup/eve_rlcd.sh for the optional model adapter")
pytest.importorskip("peft")

from microrts_agent.decisions.eve_policy import (  # noqa: E402
    EveRLCDPolicy,
    choice_question,
    forecast_loss,
)
from microrts_agent.decisions.eve_run import forecast_metrics  # noqa: E402

STATE = {
    "time": 0,
    "player": 0,
    "width": 8,
    "height": 8,
    "players": [],
    "terrain": "0" * 64,
    "unit_types": {},
    "units": [],
}
CANDIDATES = [
    {"skill": "wait", "ticks": 8},
    {"skill": "move", "actor": 3, "target": [2, 1]},
    {"skill": "attack", "actor": 3, "target_id": 7},
]


@pytest.mark.parametrize("group", [1, 2])
def test_expected_rlcd_gradient_equals_half_brier(group):
    # Enumerate answers AND stochastic game outcomes, rather than testing the formula
    # against itself. The game outcome is independent of the sampled binary forecast.
    logits = torch.tensor([-0.4, 0.8], dtype=torch.float64, requires_grad=True)
    probs = logits.softmax(-1)
    expected = torch.zeros_like(logits)
    q_win = 0.3
    for won, outcome_mass in [(True, q_win), (False, 1 - q_win)]:
        for answers in itertools.product(range(2), repeat=group):
            mass = probs.detach()[list(answers)].prod() * outcome_mass
            loss = forecast_loss(logits, answers, won, "rlcd")
            expected += mass * torch.autograd.grad(loss, logits)[0]
    target = (
        q_win * 0.5 * (probs - torch.tensor([1.0, 0.0])).square().sum()
        + (1 - q_win) * 0.5 * (probs - torch.tensor([0.0, 1.0])).square().sum()
    )
    exact = torch.autograd.grad(target, logits)[0]
    torch.testing.assert_close(expected, exact, atol=1e-12, rtol=1e-12)


@pytest.fixture()
def tiny_export(tmp_path):
    from rlcd.export import export
    from rlcd.policies import HFDecoderPolicy
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocab = {word: i for i, word in enumerate(["[UNK]", "[PAD]", *string.ascii_uppercase])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    torch.manual_seed(8)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=2048,
            tie_word_embeddings=False,
        )
    )
    source, out = tmp_path / "source", tmp_path / "base"
    HFDecoderPolicy(model, tok).save(str(source), {})
    export(str(source), str(out))
    return out


def test_lora_readout_update_reload_and_prompt_limits(tiny_export, tmp_path):
    policy = EveRLCDPolicy(tiny_export, rank=2)
    question = choice_question(STATE, CANDIDATES)
    logits, value = policy(STATE, CANDIDATES)
    original, _ = policy.rlcd.decision_logits([question], 2048, "cpu")
    torch.testing.assert_close(logits, original[0, :3])
    assert value.ndim == 0 and logits.shape == (3,)
    frozen = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}
    before = policy.trainable_state()
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    loss = -logits.log_softmax(-1)[1] + 0.5 * (value - 1).square()
    loss += forecast_loss(policy.win_logits(STATE, CANDIDATES[1]), [0, 1, 1, 0], False, "rlcd")
    loss.backward()
    optimizer.step()
    after = policy.trainable_state()
    assert any(not torch.equal(before[n], after[n]) for n in before if "lora_" in n)
    assert not torch.equal(before["letter_weight"], after["letter_weight"])
    assert not torch.equal(before["value_head.weight"], after["value_head.weight"])
    assert all(torch.equal(frozen[n], p) for n, p in policy.named_parameters() if n in frozen)

    path = tmp_path / "adapter.pt"
    policy.save(path, {"test": True})
    loaded = EveRLCDPolicy.load(path)
    torch.testing.assert_close(policy(STATE, CANDIDATES)[0], loaded(STATE, CANDIDATES)[0])
    torch.testing.assert_close(
        policy.win_logits(STATE, CANDIDATES[1]), loaded.win_logits(STATE, CANDIDATES[1])
    )
    # Forecasts must not perturb the action-sampling RNG.
    rng_before = torch.get_rng_state()
    policy.forecast(STATE, CANDIDATES[1], 4, torch.Generator().manual_seed(10))
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert policy(STATE, CANDIDATES[:1])[0].softmax(-1).tolist() == [1.0]
    policy.max_len = 8
    with pytest.raises(ValueError, match="never silently cut"):
        policy(STATE, CANDIDATES)

    checkpoint = torch.load(path, weights_only=True)
    checkpoint["base_sha256"]["model.safetensors"] = "wrong base"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="different base export"):
        EveRLCDPolicy.load(path)
    assert json.loads((tiny_export / "decision.json").read_text())["base_model"] is None


def test_timeout_forecasts_are_censored_and_draw_is_not_win():
    transition = {
        "terminated": False,
        "winner": -1,
        "state": {"player": 0},
        "forecast_probs": [0.2, 0.8],
    }
    assert forecast_metrics([transition]) == {"forecast_labeled_decisions": 0}
    result = forecast_metrics([transition | {"terminated": True}])
    assert result["forecast_labeled_decisions"] == 1
    assert result["win_brier"] == pytest.approx(0.08)
    assert result["win_ece"] == pytest.approx(0.2)


def test_optimizer_resume_matches_uninterrupted_update(tiny_export, tmp_path):
    from microrts_agent.decisions.experiment import cpu_tree

    policy = EveRLCDPolicy(tiny_export, rank=2)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.001)

    def update(model, opt):
        opt.zero_grad()
        logits, value = model(STATE, CANDIDATES)
        (-logits.log_softmax(-1)[1] + 0.5 * (value - 1).square()).backward()
        opt.step()

    update(policy, optimizer)
    path = tmp_path / "resume.pt"
    policy.save(path, {}, {"optimizer": cpu_tree(optimizer.state_dict()), "completed": 1})
    restored = EveRLCDPolicy.load(path)
    restored_opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
    saved = torch.load(path, map_location="cpu", weights_only=True)["training_state"]
    restored_opt.load_state_dict(saved["optimizer"])
    update(policy, optimizer)
    update(restored, restored_opt)
    for name, value in policy.trainable_state().items():
        torch.testing.assert_close(value, restored.trainable_state()[name], atol=1e-8, rtol=1e-8)


def test_game_weighting_and_selection_are_independent_of_episode_length():
    from microrts_agent.decisions.experiment_metrics import (
        aggregate,
        forecast_bins,
        selection_score,
    )

    games = []
    for p, n in [(0.8, 1), (0.2, 10)]:
        rows = [
            {
                "terminated": True,
                "winner": 1,
                "state": {"player": 0},
                "forecast_probs": [p, 1 - p],
                "win_probability": p,
            }
        ] * n
        games.append(
            dict(
                outcome="loss",
                reward=-10,
                ticks=n,
                decisions=n,
                terminated=True,
                forecast_bins=forecast_bins(rows),
                **forecast_metrics(rows),
            )
        )
    result = aggregate(games)
    assert result["mean_episode_brier"] == pytest.approx(0.68)
    assert result["game_weighted_ece"] == pytest.approx(0.5)
    assert result["win_rate_ci95"][1] > 0
    assert selection_score({"match_score": 0.5, "mean_reward": -100}) > selection_score(
        {"match_score": 0, "mean_reward": 100}
    )


def test_seeded_java_games_reproduce_unit_ids_and_actions():
    import io

    from microrts_agent.decisions.experiment import SeededSkillEnv
    from microrts_agent.decisions.policy import TinyPolicy
    from microrts_agent.decisions.run import collect_episode

    torch.manual_seed(13)
    policy = TinyPolicy().eval()
    env = SeededSkillEnv(seed=71, opponent="WorkerRush")
    try:
        histories = []
        for _ in range(2):
            torch.manual_seed(71)
            rows, _ = collect_episode(env, policy, 0, io.StringIO())
            histories.append(rows)
        assert histories[0] == histories[1]
    finally:
        env.close()


def test_multi_unit_concurrent_skill_env_and_bot_matching():
    from microrts_agent.decisions.env import SkillEnv
    from microrts_agent.registries.ai import AI_MAPPING

    env = SkillEnv(opponent="PassiveAI", player=0, max_ticks=300, skill_ticks=64)
    try:
        state = env.reset(seed=42)
        idle = env.idle_unit_candidates(state)
        # Both Base and Worker should be offered decisions on tick 0 simultaneously.
        unit_types = {u["type"] for u, _ in idle}
        assert unit_types == {"Base", "Worker"}
        # Assign Base to produce Worker AND Worker to harvest concurrently.
        assignments = {}
        for u, cands in idle:
            if u["type"] == "Base":
                assignments[u["id"]] = next(c for c in cands if c["skill"] == "produce")
            else:
                assignments[u["id"]] = next(c for c in cands if c["skill"] == "harvest")
        res = env.step_units(assignments)
        assert res["duration"] > 0
        # Verify Base and Worker progressed in parallel (worker produced & resource harvested).
        next_units = [u for u in res["next_state"]["units"] if u["owner"] == 0]
        assert len(next_units) >= 3  # Base + initial Worker + newly produced Worker

        # Verify expert bot action matching produces a valid candidate index.
        teacher = AI_MAPPING["WorkerRush"](env.base.real_utt)
        cloned = env.gs.clone()
        pa = teacher.getAction(0, cloned)
        by_id = {int(pair.m_a.getID()): pair.m_b for pair in pa.getActions()}
        for u, cands in env.idle_unit_candidates(res["next_state"]):
            ju = env.gs.getUnit(u["id"])
            cju = cloned.getUnit(u["id"])
            idx = env.skills.match_bot_skill(
                env.gs, ju, by_id.get(u["id"]), cands, abstract_action=teacher.getAbstractAction(cju)
            )
            assert 0 <= idx < len(cands)
    finally:
        env.close()


def test_smdp_gae_and_candidate_shuffling():
    import random

    from microrts_agent.decisions.eve_policy import shuffle_candidates
    from microrts_agent.decisions.metrics import compute_smdp_gae

    steps = [
        {"discounted_reward": 2.0, "discount": 0.5, "value": 1.0, "terminated": False},
        {"discounted_reward": 4.0, "discount": 0.5, "value": 2.0, "terminated": True},
    ]
    advs, rets = compute_smdp_gae(steps, bootstrap=99.0, gae_lambda=1.0)
    assert rets == pytest.approx([4.0, 4.0])
    assert advs == pytest.approx([3.0, 2.0])

    # Truncated episode retains bootstrap.
    steps[1]["terminated"] = False
    advs_t, rets_t = compute_smdp_gae(steps, bootstrap=6.0, gae_lambda=1.0)
    assert rets_t == pytest.approx([5.5, 7.0])
    assert advs_t == pytest.approx([4.5, 5.0])

    cands = [{"skill": "a"}, {"skill": "b"}, {"skill": "c"}, {"skill": "d"}]
    shuffled, perm, new_idx = shuffle_candidates(cands, chosen_idx=2, rng=random.Random(3))
    assert shuffled[new_idx] == cands[2]
    assert len(set(perm)) == 4


def test_offline_sft_bandit_rlcd_ppo_and_decision_export(tiny_export, tmp_path):
    from rlcd.export import load_decision_only

    from microrts_agent.decisions.eve_policy import unit_choice_question, win_question
    from microrts_agent.decisions.train import ppo_rlcd_update, run_offline_stage

    policy = EveRLCDPolicy(tiny_export, device="cpu", rank=2, dtype="fp32")
    rows = [
        {
            "primitive": "choice",
            "context": "MicroRTS t=0 map=8x8 me=P0",
            "question": "Best skill for Base U6?",
            "choices": ["U6: produce Worker at (2,0)", "Wait 4t"],
            "answer": 0,
            "episode": 0,
            "tick": 0,
            "terminated": True,
            "won": True,
        },
        {
            "primitive": "choice",
            "context": "MicroRTS t=10 map=8x8 me=P0",
            "question": "Best skill for Worker U8?",
            "choices": ["U8: harvest R4 -> B6", "U8: attack nearest enemy", "Wait 4t"],
            "answer": 1,
            "episode": 1,
            "tick": 10,
            "terminated": True,
            "won": False,
        },
    ]

    # 1. Stage 1 SFT
    sft_metrics = run_offline_stage(
        policy, rows, tmp_path / "sft", mode="sft", epochs=1, micro_batch=2, accum_steps=1, lr=1e-3
    )
    assert sft_metrics["steps"] >= 1
    assert (tmp_path / "sft" / "policy.pt").is_file()

    # 2. Stage 2a Bandit-RLCD
    before = policy.trainable_state()
    rlcd_metrics = run_offline_stage(
        policy,
        rows,
        tmp_path / "rlcd",
        mode="bandit",
        arm="rlcd",
        epochs=1,
        micro_batch=2,
        accum_steps=1,
        lr=1e-3,
        group=2,
    )
    assert rlcd_metrics["steps"] >= 1
    after = policy.trainable_state()
    assert any(not torch.equal(before[k], after[k]) for k in before)

    # 3. Stage 2b PPO + RLCD update
    u = {"id": 6, "type": "Base", "x": 2, "y": 1}
    cands = [{"skill": "produce", "actor": 6, "unit_type": "Worker", "target": [2, 0]}, {"skill": "wait", "ticks": 4}]
    q = unit_choice_question(STATE, u, cands)
    wq = win_question(STATE)
    with torch.no_grad():
        logits, val = policy.forward_questions([q], return_values=True)
        logp0 = float(logits[0, :2].log_softmax(-1)[0].item())
    step = {
        "unit_decisions": [
            {"unit_id": 6, "question": q, "candidates": cands, "chosen_index": 0, "log_prob": logp0}
        ],
        "win_question": wq,
        "forecast_answers": [0, 1],
        "value": float(val[0].item()),
        "reward": 1.0,
        "discounted_reward": 1.0,
        "duration": 10,
        "discount": 0.9,
        "terminated": True,
        "truncated": False,
        "winner": 0,
    }
    opt = torch.optim.AdamW(policy.param_groups(lr=1e-3))
    stats = ppo_rlcd_update(
        policy,
        opt,
        [([step], STATE, {"outcome": "win", "terminated": True})],
        ppo_epochs=2,
        micro_batch=2,
        arm="rlcd",
    )
    assert stats["ppo_samples"] == 1 and stats["forecast_samples"] == 1
    assert stats["calibration_loss"] is not None

    # 4. In-place adapter swap & LoRA merge decision-only export
    policy.load_adapter(tmp_path / "sft" / "policy.pt")
    export_dir = tmp_path / "exported"
    record = policy.export_decision_only(export_dir, meta={"test": True})
    assert "model.safetensors" in record["sha256"]
    exported = load_decision_only(str(export_dir), device="cpu")
    with torch.no_grad():
        p_logits = policy.forward_questions([q])[0, :2]
        e_logits, _ = exported.decision_logits([q], 2048, "cpu")
    torch.testing.assert_close(p_logits, e_logits[0, :2], atol=1e-5, rtol=1e-5)


def test_audit_data_cleaning_dedup_and_win_noul():
    from microrts_agent.decisions.audit_data import (
        audit_split,
        dedup_and_clean,
        evaluate_gates,
        extract_win_noul_rows,
    )

    base = {
        "primitive": "choice",
        "context": "MicroRTS t=0 map=8x8 me=P0",
        "question": "Best skill for Base U6?",
        "choices": ["Produce Worker", "Wait 16t"],
        "answer": 0,
        "skill": "produce",
        "unit_type": "Base",
        "match_kind": "exact",
        "map": "maps/8x8/basesWorkers8x8A.xml",
        "map_size": "8x8",
        "teacher": "CoacAI",
        "opponent": "WorkerRush",
        "player": 0,
        "episode": 0,
        "tick": 0,
        "won": True,
        "terminated": True,
    }
    rows = [
        base,
        dict(base),  # exact duplicate
        dict(base, match_kind="wait_fallback", skill="wait", answer=1, tick=10),  # fallback noise
        dict(base, context="MicroRTS t=20 map=8x8 me=P1", player=1, episode=1, tick=20, won=False),
    ]
    rep = audit_split(rows, tok=None)
    assert rep["schema_errors"] == 0
    assert rep["wait_fallback_rate"] == pytest.approx(0.25)
    cleaned, stats = dedup_and_clean(rows, drop_wait_fallback=True, shuffle_seed=11)
    assert stats["dropped_dup"] == 1
    assert stats["dropped_fallback"] == 1
    assert stats["kept"] == 2

    win_rows = extract_win_noul_rows(rows)
    assert len(win_rows) == 3
    assert {r["answer"] for r in win_rows} == {0, 1}

    gates = evaluate_gates(audit_split(cleaned, tok=None))
    assert any(g["check"] == "schema_validity" and g["status"] == "PASS" for g in gates)


@pytest.mark.parametrize("group", [1, 2])
def test_is_rlcd_expected_gradient_equals_half_brier_even_when_saturated(group):
    """Exact enumeration proof: IS-RLCD is unbiased for 0.5 * grad Brier even at extreme p."""
    from microrts_agent.decisions.sa_rlcd import importance_sampled_rlcd_loss, is_rlcd_proposal

    # Extreme saturated logits (p ~ [0.99, 0.008, 0.002]) where vanilla RLCD would stall!
    logits = torch.tensor([4.5, -0.3, -1.7], dtype=torch.float64, requires_grad=True)
    probs = logits.softmax(-1).unsqueeze(0)
    q_prop = is_rlcd_proposal(probs, [3], eps=0.2)[0]
    q_true = [0.2, 0.5, 0.3]

    expected = torch.zeros_like(logits)
    for y, y_mass in enumerate(q_true):
        for actions in itertools.product(range(3), repeat=group):
            mass = q_prop[list(actions)].prod() * y_mass
            act_t = torch.tensor([list(actions)], dtype=torch.long)
            out_t = (act_t == y).double()
            loss = importance_sampled_rlcd_loss(logits.unsqueeze(0), act_t, out_t, q_prop.unsqueeze(0))
            expected += mass * torch.autograd.grad(loss, logits)[0]

    target = sum(
        y_mass * 0.5 * (probs[0] - torch.eye(3, dtype=torch.float64)[y]).square().sum()
        for y, y_mass in enumerate(q_true)
    )
    exact = torch.autograd.grad(target, logits)[0]
    torch.testing.assert_close(expected, exact, atol=1e-12, rtol=1e-12)


def test_state_potential_conservation_neutrality_and_zero_sum():
    """Verify Phi(s) is bounded, zero-sum, and conservation-neutral for building/producing."""
    from microrts_agent.decisions.state import state_potential

    base_state = {
        "player": 0,
        "players": [{"id": 0, "resources": 10}, {"id": 1, "resources": 10}],
        "units": [
            {"id": 1, "owner": 0, "type": "Base", "hp": 10, "resources": 0},
            {"id": 2, "owner": 1, "type": "Base", "hp": 10, "resources": 0},
        ],
        "unit_types": {
            "Base": {"cost": 10, "hp": 10},
            "Barracks": {"cost": 5, "hp": 4},
            "Worker": {"cost": 1, "hp": 1},
        },
    }
    # Symmetric start -> exactly 0.0
    assert state_potential(base_state) == pytest.approx(0.0, abs=1e-12)

    # Spending 5 gold to build a full-HP Barracks is strictly conservation-neutral!
    after_build = {
        **base_state,
        "players": [{"id": 0, "resources": 5}, {"id": 1, "resources": 10}],
        "units": base_state["units"] + [{"id": 3, "owner": 0, "type": "Barracks", "hp": 4, "resources": 0}],
    }
    assert state_potential(after_build) == pytest.approx(0.0, abs=1e-12)

    # Net harvesting or dealing damage strictly increases Phi(s), and P0 == -P1 (zero-sum)
    after_damage = {
        **base_state,
        "units": [
            {"id": 1, "owner": 0, "type": "Base", "hp": 10, "resources": 0},
            {"id": 2, "owner": 1, "type": "Base", "hp": 5, "resources": 0},
        ],
    }
    phi_p0 = state_potential(after_damage, scale=0.15)
    phi_p1 = state_potential({**after_damage, "player": 1}, scale=0.15)
    assert 0.0 < phi_p0 < 0.15
    assert phi_p0 == pytest.approx(-phi_p1, abs=1e-12)


def test_wpc_pbrs_exact_telescoping_positive_long_win_and_asymmetric_entropy():
    """Verify the 3 core theoretical invariants of undiscounted WPC-PBRS and asymmetric entropy."""
    from microrts_agent.decisions.sa_rlcd import (
        asymmetric_entropy_weight,
        compute_wpc_advantages,
        td_lambda_win_targets,
    )

    # 1. Exact telescoping when gae_lambda=1.0: A_k == Y - V(s_k) - Phi(s_k) (win, loss, timeout)
    for terminated, won, expected_y in [(True, True, 1.0), (True, False, 0.0), (False, False, 0.0)]:
        steps = [
            {"win_probability": 0.45, "state_potential": 0.10, "terminated": False, "truncated": False, "won": won},
            {"win_probability": 0.60, "state_potential": 0.35, "terminated": False, "truncated": False, "won": won},
            {"win_probability": 0.80, "state_potential": 0.50, "terminated": terminated, "truncated": not terminated, "won": won},
        ]
        advs = compute_wpc_advantages(steps, gae_lambda=1.0, potential_scale=0.15)
        for s, a in zip(steps, advs, strict=True):
            exact = expected_y - s["win_probability"] - 0.15 * s["state_potential"]
            assert a == pytest.approx(exact, abs=1e-12)
        z = td_lambda_win_targets(steps, td_lambda=0.85)
        assert z[-1] == pytest.approx(expected_y, abs=1e-12)

    # 2. Long 60-step winning trajectory (simulating 1200 ticks): mean advantage strictly positive!
    long_win = [
        {
            "win_probability": 0.50 + 0.45 * (i / 59.0),
            "state_potential": 0.0 + 0.6 * (i / 59.0),
            "discount": 0.9985**20,  # ignored by undiscounted WPC-PBRS!
            "terminated": i == 59,
            "truncated": False,
            "won": True,
        }
        for i in range(60)
    ]
    long_advs = compute_wpc_advantages(long_win, gae_lambda=0.95, potential_scale=0.15)
    assert sum(long_advs) / len(long_advs) > 0.05

    # 3. Flat-critic losing trajectory (Mayari-like V~0.10): gaining material potential gives positive step delta!
    flat_loss = [
        {"win_probability": 0.10, "state_potential": 0.00, "terminated": False, "truncated": False, "won": False},
        {"win_probability": 0.10, "state_potential": 0.40, "terminated": False, "truncated": False, "won": False},
        {"win_probability": 0.08, "state_potential": -0.30, "terminated": True, "truncated": False, "won": False},
    ]
    # Pure 1-step delta (gae_lambda=0) isolates immediate PBRS credit
    d_flat = compute_wpc_advantages(flat_loss, gae_lambda=0.0, potential_scale=0.15)
    assert d_flat[0] > 0.0  # improved board material gets immediate positive credit
    assert d_flat[1] < 0.0  # losing board material gets immediate negative credit

    # 4. Asymmetric failure-driven entropy is strictly decreasing in calibrated win probability
    w_lose = asymmetric_entropy_weight(0.05, ent_min=0.003, ent_max=0.025)
    w_even = asymmetric_entropy_weight(0.50, ent_min=0.003, ent_max=0.025)
    w_win = asymmetric_entropy_weight(0.95, ent_min=0.003, ent_max=0.025)
    assert w_lose > w_even > w_win > 0.003


def test_sa_rlcd_force_summary_wpc_advantages_and_update(tiny_export):
    from collections import deque

    from microrts_agent.decisions.eve_policy import unit_choice_question, win_question
    from microrts_agent.decisions.metrics import aggregate_games, forecast_summary
    from microrts_agent.decisions.sa_rlcd import (
        compute_wpc_advantages,
        rb_rlcd_loss,
        sa_rlcd_update,
        td_lambda_win_targets,
    )
    from microrts_agent.decisions.state import ensure_force_summary

    raw_ctx = (
        "MicroRTS t=200 map=8x8 me=P0 opp=WorkerRush\n"
        "Res: me=5(avail=5) opp=5(avail=5)\n"
        "My units (2):\n- U6 Base(2,1) hp=10/10 idle\n- U8 Worker(1,1) hp=1/1 idle\n"
        "Enemy units (1):\n- U7 Base(5,6) hp=10/10 idle"
    )
    enriched = ensure_force_summary(raw_ctx)
    assert "Force: me[B=1 Rax=0 W=1 Mil=0 hp=11] vs opp[B=1 Rax=0 W=0 Mil=0 hp=10] phase=mid" in enriched
    assert ensure_force_summary(enriched) == enriched  # idempotent

    # Timeout forecast_summary tracks mean_win_probability & all_game_win_brier while keeping labeled=0
    t_fs = forecast_summary([0.6, 0.7], terminated=False, won=False)
    assert t_fs["forecast_labeled_decisions"] == 0
    assert t_fs["mean_win_probability"] == pytest.approx(0.65)
    assert t_fs["all_game_win_brier"] > 0.8
    agg = aggregate_games(
        [{"outcome": "timeout", "terminated": False, "reward": -6.0, "ticks": 2500, "decisions": 2, **t_fs}]
    )
    assert agg["timeout_mean_win_probability"] == pytest.approx(0.65)
    assert agg["all_games_brier"] == pytest.approx(t_fs["all_game_win_brier"])

    # TD(lambda) targets & WPC advantages
    steps = [
        {"win_probability": 0.4, "state_potential": 0.0, "terminated": False, "truncated": False, "won": True},
        {"win_probability": 0.7, "state_potential": 0.2, "terminated": True, "truncated": False, "won": True},
    ]
    z = td_lambda_win_targets(steps, td_lambda=0.8)
    assert z[1] == pytest.approx(1.0)
    assert 0.7 < z[0] < 1.0
    advs = compute_wpc_advantages(steps)
    assert advs[0] > 0 and advs[1] > 0

    # RB-RLCD exact Brier check
    wl = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    assert rb_rlcd_loss(wl, [1.0]).item() == pytest.approx(0.25)

    # Full online SA-RLCD v2 single-objective mini-batch update (with timeout ring buffer)
    policy = EveRLCDPolicy(tiny_export, device="cpu", rank=2, dtype="fp32")
    ref = EveRLCDPolicy(tiny_export, device="cpu", rank=2, dtype="fp32")
    ref.load_trainable_state(policy.trainable_state())
    u = {"id": 6, "type": "Base", "x": 2, "y": 1}
    cands = [{"skill": "produce", "actor": 6, "unit_type": "Worker", "target": [2, 0]}, {"skill": "wait", "ticks": 16}]
    q = unit_choice_question(STATE, u, cands)
    wq = win_question(STATE)
    with torch.no_grad():
        lp = float(policy.forward_questions([q])[0, :2].log_softmax(-1)[0].item())
    ep_step = {
        "unit_decisions": [{"unit_id": 6, "question": q, "candidates": cands, "chosen_index": 0, "log_prob": lp}],
        "win_question": wq,
        "win_probability": 0.5,
        "state_potential": 0.1,
        "terminated": True,
        "truncated": False,
    }
    timeout_step = {
        "unit_decisions": [{"unit_id": 6, "question": q, "candidates": cands, "chosen_index": 0, "log_prob": lp}],
        "win_question": wq,
        "win_probability": 0.6,
        "state_potential": 0.0,
        "terminated": False,
        "truncated": True,
    }
    tbuf = deque(maxlen=16)
    opt = torch.optim.AdamW(policy.param_groups(lr=1e-3))
    stats = sa_rlcd_update(
        policy,
        ref,
        opt,
        [
            ([ep_step], STATE, {"outcome": "win", "terminated": True}),
            ([timeout_step], STATE, {"outcome": "timeout", "terminated": False}),
        ],
        timeout_cal_buffer=tbuf,
        ppo_epochs=2,
        ppo_batch_size=1,
        micro_batch=1,
    )
    assert stats["ppo_samples"] == 2
    assert stats["opt_steps"] == 4  # 2 epochs x 2 mini-batches
    assert stats["calibration_loss"] > 0
    assert stats["mean_adv_win"] > 0 and stats["mean_adv_loss"] < 0
    assert len(tbuf) == 1 and tbuf[0]["target"] == 0.0
