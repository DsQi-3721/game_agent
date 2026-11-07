"""Export policy evaluation figures and their underlying comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root: Path):
    result = json.loads((root / "summary.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    model = Path(protocol["training"]["output"])
    training = json.loads((model / "s1_train_meta.json").read_text())
    out = root / "charts"
    out.mkdir(exist_ok=True)
    arms = (("old", "Original", "#64748b"), ("new", "Retrained", "#2563eb"))
    opponents = ("kakuna", "foulplay")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for offset, (arm, label, color) in zip((-.19, .19), arms):
        rows = [result["arms"][arm][opponent] for opponent in opponents]
        rates = np.array([row["win_rate"] * 100 for row in rows])
        low = np.array([row["win_ci95"][0] * 100 for row in rows])
        high = np.array([row["win_ci95"][1] * 100 for row in rows])
        bars = axes[0, 0].bar(np.arange(2) + offset, rates, width=.36, color=color, label=label,
                             yerr=[rates-low, high-rates], capsize=4)
        axes[0, 0].bar_label(bars, [f"{row['wins']}/{row['games']}" for row in rows], label_type="center", color="white")
    axes[0, 0].set_xticks([0, 1], ["Kakuna", "Foul Play"])
    axes[0, 0].set_ylabel("Win rate (%)")
    axes[0, 0].set_ylim(0, 30)
    axes[0, 0].set_title("480 games: win rate and 95% Wilson interval")
    axes[0, 0].legend()

    for offset, (arm, label, color) in zip((-.19, .19), arms):
        values = [result["latency"][arm][key] * 1000 for key in ("mean_s", "p90_s")]
        bars = axes[0, 1].bar(np.arange(2) + offset, values, width=.36, color=color, label=label)
        axes[0, 1].bar_label(bars, fmt="%.0f", padding=3)
    axes[0, 1].set_xticks([0, 1], ["Mean", "p90"])
    axes[0, 1].set_ylabel("Milliseconds")
    axes[0, 1].set_ylim(0, 520)
    axes[0, 1].set_title("Request-to-action latency")
    axes[0, 1].legend()

    preferences = result["heldout_preferences"]
    for index, (arm, label, color) in enumerate(arms):
        value = preferences["arms"][arm]
        bars = axes[1, 0].bar(index, value["accuracy"] * 100, width=.55, color=color)
        axes[1, 0].bar_label(bars, [f"{value['accuracy']:.1%} ({value['correct']}/{preferences['n']})"], padding=3)
    axes[1, 0].set_xticks([0, 1], [row[1] for row in arms])
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_ylabel("Preference accuracy (%)")
    axes[1, 0].set_title("Held-out search preferences, both option orders")

    history = training["history"]
    axes[1, 1].plot([row["step"] for row in history],
                    [row["valid_by_purpose"]["action"]["loss"] for row in history], "o-", color="#2563eb")
    axes[1, 1].axvline(training["best_step"], color="#15803d", linestyle="--",
                      label=f"Selected checkpoint: {training['best_step']:,}")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Cross entropy")
    axes[1, 1].set_title("Full-action validation objective")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "evaluation_overview.png", dpi=180)
    fig.savefig(out / "evaluation_overview.svg")
    plt.close(fig)

    with (out / "battle_comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["opponent", "model", "wins", "games", "win_rate",
                                                  "ci95_low", "ci95_high", "mean_s", "p90_s"])
        writer.writeheader()
        for opponent in opponents:
            for arm, _, _ in arms:
                row = result["arms"][arm][opponent]
                writer.writerow({"opponent": opponent, "model": arm,
                    **{key: row[key] for key in ("wins", "games", "win_rate")},
                    "ci95_low": row["win_ci95"][0], "ci95_high": row["win_ci95"][1],
                    "mean_s": row["latency"]["mean_s"], "p90_s": row["latency"]["p90_s"]})
    print(out / "evaluation_overview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root.resolve())
