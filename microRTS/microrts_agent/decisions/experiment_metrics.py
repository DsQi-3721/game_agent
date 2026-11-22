"""Game-weighted evaluation and reports for the preregistered training run."""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wilson(wins, total):
    z = 1.959963984540054
    p, scale = wins / total, 1 + z * z / total
    mid = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0, mid - radius), min(1, mid + radius)]


def forecast_bins(transitions):
    bins = np.zeros((10, 3))
    if transitions[-1]["terminated"]:
        won = transitions[-1]["winner"] == transitions[0]["state"]["player"]
        for t in transitions:
            p = t["win_probability"]
            bins[min(int(p * 10), 9)] += np.array([1, p, won]) / len(transitions)
    return bins.tolist()


def aggregate(games):
    counts = Counter(g["outcome"] for g in games)
    labeled = [g for g in games if g["terminated"]]
    result = {
        "games": len(games),
        **{k: counts[k] for k in ("win", "loss", "draw", "timeout")},
        "win_rate": counts["win"] / len(games),
        "win_rate_ci95": wilson(counts["win"], len(games)),
        "match_score": (counts["win"] + 0.5 * counts["draw"]) / len(games),
        "mean_reward": float(np.mean([g["reward"] for g in games])),
        "mean_ticks": float(np.mean([g["ticks"] for g in games])),
        "mean_decisions": float(np.mean([g["decisions"] for g in games])),
        "forecast_labeled_games": len(labeled),
    }
    if labeled:
        bins = np.array([g["forecast_bins"] for g in labeled]).sum(0)
        result.update(
            mean_episode_brier=float(np.mean([g["win_brier"] for g in labeled])),
            game_weighted_ece=float(np.abs(bins[:, 1] - bins[:, 2]).sum() / bins[:, 0].sum()),
        )
    return result


def selection_score(summary):
    # Select a playing policy, never a model merely good at predicting defeats.
    return [summary["match_score"], summary["mean_reward"]]


def comparison(initial, trained):
    assert [g["case"] for g in initial] == [g["case"] for g in trained]
    differences = np.array(
        [
            int(b["outcome"] == "win") - int(a["outcome"] == "win")
            for a, b in zip(initial, trained, strict=True)
        ]
    )
    return {
        "paired_win_rate_difference": float(differences.mean()),
        "new_wins": int((differences == 1).sum()),
        "lost_wins": int((differences == -1).sum()),
        "note": "Paired fixed-seed games; one training seed, not a multi-seed RL benchmark.",
    }


def write_progress(out, state):
    rows = [dict(step=v["step"], **v["summary"]) for v in state["validations"]]
    if not rows:
        return
    fields = [
        "step",
        "win_rate",
        "match_score",
        "mean_reward",
        "mean_ticks",
        "mean_episode_brier",
        "game_weighted_ece",
        "timeout",
    ]
    with (out / "validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, key, title in zip(
        axes.flat,
        ["win_rate", "mean_reward", "mean_episode_brier", "game_weighted_ece"],
        [
            "Validation win rate",
            "Mean game reward",
            "Game-weighted Brier (0-2)",
            "Game-weighted ECE",
        ],
        strict=True,
    ):
        ax.plot([r["step"] for r in rows], [r.get(key, float("nan")) for r in rows], marker="o")
        ax.set(xlabel="Training episodes", title=title)
        if key == "win_rate":
            ax.fill_between(
                [r["step"] for r in rows],
                [r["win_rate_ci95"][0] for r in rows],
                [r["win_rate_ci95"][1] for r in rows],
                alpha=0.15,
            )
            ax.set_ylim(0, 1)
        elif key == "game_weighted_ece":
            ax.set_ylim(0, 1)
        elif key == "mean_episode_brier":
            ax.set_ylim(0, 2)
        ax.grid(alpha=0.2)
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def write_report(out, state, results):
    summaries = {name: aggregate(games) for name, games in results.items()}
    comparisons = {
        name: comparison(results["initial"], games)
        for name, games in results.items()
        if name != "initial"
    }
    per_condition = {}
    for name, games in results.items():
        groups = sorted({(g["map"], g["player"]) for g in games})
        per_condition[name] = [
            dict(
                map=m,
                player=p,
                **aggregate([g for g in games if g["map"] == m and g["player"] == p]),
            )
            for m, p in groups
        ]
    atomic_json(
        out / "results.json",
        {
            "completed_episodes": state["completed"],
            "best_step": state["best_step"],
            "training_seconds": state["training_seconds"],
            "stop_reason": state["stop_reason"],
            "test": summaries,
            "comparisons": comparisons,
            "per_condition": per_condition,
        },
    )
    lines = [
        "# eve-rlcd microRTS 训练与独立测试",
        "",
        f"训练更新：{state['completed']} 局；最佳验证 checkpoint：第 {state['best_step']} 局。",
        f"训练用时：{state['training_seconds'] / 3600:.2f} 小时；结束原因：{state['stop_reason']}。",
        "",
        "| 模型 | 测试局数 | 胜/负/平/超时 | 胜率（Wilson 95% 区间） | 平均奖励 | Brier | ECE |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for name, s in summaries.items():
        low, high = s["win_rate_ci95"]
        lines.append(
            f"| {name} | {s['games']} | {s['win']}/{s['loss']}/{s['draw']}/{s['timeout']} | "
            f"{s['win_rate']:.1%} ({low:.1%}–{high:.1%}) | {s['mean_reward']:.3f} | "
            f"{s.get('mean_episode_brier', float('nan')):.4f} | "
            f"{s.get('game_weighted_ece', float('nan')):.4f} |"
        )
    lines += [
        "",
        "最佳模型只由固定验证集选取；独立测试在选模结束后进行。",
        "预测指标对每局等权，超时不提供胜负标签；所有测试对局均进入对局成绩统计。",
        "一次训练种子及有限地图不能证明普遍提升；Brier/ECE 下降也不能代替实际胜率提升。",
        "",
        "详细分地图、阵营结果见 results.json；训练曲线见 learning_curves.png。",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
