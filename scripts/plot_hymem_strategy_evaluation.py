"""Export the strategic evaluation figure and comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root, result):
    out = root / 'charts'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    arms = [('baseline', 'Policy + search', '#64748b'), ('strategic', 'Policy + search + HyMem', '#2563eb')]
    opponents = ('kakuna', 'foulplay')
    table = []
    for offset, (arm, label, color) in zip((-.19, .19), arms):
        rows = [result['arms'][arm][opponent] for opponent in opponents]
        rates = np.array([row['win_rate'] * 100 for row in rows])
        low = np.array([row['win_ci95'][0] * 100 for row in rows])
        high = np.array([row['win_ci95'][1] * 100 for row in rows])
        bars = axes[0].bar(np.arange(2) + offset, rates, width=.36, color=color, label=label,
                           yerr=[np.maximum(0, rates-low), np.maximum(0, high-rates)], capsize=4)
        axes[0].bar_label(bars, [f"{row['wins']}/{row['games']}" for row in rows], padding=14)
        values = [result['overall'][arm]['latency'][key] for key in ('mean_s', 'p90_s')]
        bars = axes[1].bar(np.arange(2) + offset, values, width=.36, color=color, label=label)
        axes[1].bar_label(bars, fmt='%.3f', padding=3)
        for opponent, row in zip(opponents, rows):
            table.append({'opponent': opponent, 'system': arm,
                **{key: row[key] for key in ('wins', 'games', 'win_rate')},
                'ci95_low': row['win_ci95'][0], 'ci95_high': row['win_ci95'][1],
                'mean_s': row['latency']['mean_s'], 'p90_s': row['latency']['p90_s']})
    axes[0].set_xticks([0, 1], ['Kakuna', 'Foul Play'])
    axes[0].set_ylabel('Win rate (%)')
    axes[0].set_title(f"{result['completed_games']} games; 95% Wilson intervals")
    axes[0].set_ylim(0, min(110, max(20, axes[0].get_ylim()[1] * 1.2)))
    axes[1].set_xticks([0, 1], ['Mean', 'p90'])
    axes[1].set_ylabel('Seconds')
    axes[1].set_title('Request-to-action latency')
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.15)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    fig.savefig(out / 'evaluation_overview.png', dpi=180)
    fig.savefig(out / 'evaluation_overview.svg')
    plt.close(fig)
    with (out / 'battle_comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    if result.get('games'):
        with (out / 'primary_games.csv').open('w', newline='') as file:
            fields = ['battle_tag','arm','opponent','team','won','turns','duration_s','decisions','mean_s','p90_s']
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in result['games']:
                writer.writerow({**{k:row[k] for k in fields if k in row},
                                 'mean_s':row['latency']['mean_s'],'p90_s':row['latency']['p90_s']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    plot(args.root, json.loads((args.root / 'summary.json').read_text()))
