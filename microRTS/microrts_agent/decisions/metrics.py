"""Evaluation, calibration, and SMDP advantage metrics for MicroRTS decision models."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def wilson(wins, total):
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p, scale = wins / total, 1 + z * z / total
    mid = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0.0, mid - radius), min(1.0, mid + radius)]


def compute_smdp_gae(steps, bootstrap=0.0, gae_lambda=0.95):
    """SMDP Generalized Advantage Estimation over variable-duration decision steps.

    Each step dict has: discounted_reward, discount (gamma**duration), value, terminated.
    Returns (advantages, returns) as float lists aligned with `steps`.
    """
    n = len(steps)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    next_value = float(bootstrap)
    for i in range(n - 1, -1, -1):
        s = steps[i]
        mask = 0.0 if s["terminated"] else 1.0
        discount = s["discount"]
        delta = s["discounted_reward"] + discount * mask * next_value - s["value"]
        gae = delta + discount * gae_lambda * mask * gae
        advantages[i] = gae
        returns[i] = gae + s["value"]
        next_value = s["value"]
    return advantages, returns


def episode_outcome(terminated, truncated, winner, player):
    if truncated:
        return "timeout"
    if winner == -1:
        return "draw"
    return "win" if winner == player else "loss"


def forecast_summary(win_probs, terminated, won):
    """Brier (0-2 two-class sum, matching eve-rlcd) and 10-bin ECE for one episode.

    Always records mean_win_probability and strict draw=false Brier (all_game_win_brier)
    so timeout stalemates remain observable, while keeping forecast_labeled_decisions=0 on
    truncated games for apples-to-apples comparison with terminated-only summaries.
    """
    if not win_probs:
        return {"forecast_labeled_decisions": 0}
    p = np.asarray(win_probs, dtype=np.float64)
    y = 1.0 if (terminated and won) else 0.0
    brier_val = float(2.0 * np.mean((p - y) ** 2))
    if not terminated:
        return {
            "forecast_labeled_decisions": 0,
            "mean_win_probability": float(p.mean()),
            "all_game_win_brier": brier_val,
        }
    bins = np.zeros((10, 3), dtype=np.float64)
    for pi in p:
        bins[min(int(pi * 10), 9)] += np.array([1.0, pi, y]) / len(p)
    ece_val = float(np.abs(bins[:, 1] - bins[:, 2]).sum() / max(bins[:, 0].sum(), 1e-12))
    return {
        "forecast_labeled_decisions": len(win_probs),
        "win_brier": brier_val,
        "all_game_win_brier": brier_val,
        "win_ece": ece_val,
        "mean_win_probability": float(p.mean()),
        "forecast_bins": bins.tolist(),
    }


def aggregate_games(games):
    counts = Counter(g["outcome"] for g in games)
    n = len(games)
    labeled = [g for g in games if g.get("terminated") and g.get("forecast_labeled_decisions", 0)]
    all_briers = [g["all_game_win_brier"] for g in games if "all_game_win_brier" in g]
    timeout_ps = [
        g["mean_win_probability"]
        for g in games
        if g.get("outcome") == "timeout" and g.get("mean_win_probability") is not None
    ]
    result = {
        "games": n,
        **{k: counts[k] for k in ("win", "loss", "draw", "timeout")},
        "win_rate": counts["win"] / n if n else 0.0,
        "win_rate_ci95": wilson(counts["win"], n),
        "match_score": (counts["win"] + 0.5 * counts["draw"]) / n if n else 0.0,
        "mean_reward": float(np.mean([g["reward"] for g in games])) if n else 0.0,
        "mean_ticks": float(np.mean([g["ticks"] for g in games])) if n else 0.0,
        "mean_decisions": float(np.mean([g["decisions"] for g in games])) if n else 0.0,
        "forecast_labeled_games": len(labeled),
    }
    if labeled:
        bins = np.array([g["forecast_bins"] for g in labeled]).sum(0)
        result["mean_episode_brier"] = float(np.mean([g["win_brier"] for g in labeled]))
        result["game_weighted_ece"] = float(
            np.abs(bins[:, 1] - bins[:, 2]).sum() / max(bins[:, 0].sum(), 1e-12)
        )
    if all_briers:
        result["all_games_brier"] = float(np.mean(all_briers))
    if timeout_ps:
        result["timeout_mean_win_probability"] = float(np.mean(timeout_ps))
    latencies = [g["ms_per_decision"] for g in games if "ms_per_decision" in g]
    if latencies:
        result["mean_ms_per_decision"] = float(np.mean(latencies))
    return result


def selection_score(summary):
    return [summary["match_score"], summary["mean_reward"]]
