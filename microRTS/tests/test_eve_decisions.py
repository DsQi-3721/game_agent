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
