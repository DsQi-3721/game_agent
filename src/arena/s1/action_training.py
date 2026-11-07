"""Action-only supervision, category diagnostics and local experiment records."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time


def permute_action(row, rng):
    row = deepcopy(row)
    order = list(range(len(row['keys'])))
    rng.shuffle(order)
    keys = row['keys']
    row['keys'] = [keys[i] for i in order]
    row['target'] = [row['target'][i] for i in order]
    row['question']['criteria'] = {k: row['question']['criteria'][k] for k in row['keys']}
    return row


def action_category(key):
    return 'switch' if key.startswith('switch_') else 'tera' if key.endswith('_tera') else 'move'


def policy_metrics(examples, predictions):
    import numpy as np
    totals = defaultdict(lambda: defaultdict(float))
    for ex, pred in zip(examples, predictions):
        target = np.asarray(ex['target']); p = np.asarray(pred[:len(target)])
        best = int(target.argmax()); chosen = int(p.argmax())
        category = action_category(ex['keys'][best])
        buckets = ['all', ex.get('family', 'historical'), 'teacher_' + category]
        if target[best] >= .9: buckets.append('teacher_p90')
        for name in buckets:
            m = totals[name]; m['n'] += 1
            m['ce'] += float(-(target * np.log(np.maximum(p, 1e-30))).sum())
            m['top1'] += chosen == best
            m['teacher_mass_selected'] += float(target[chosen])
            m['category_agree'] += action_category(ex['keys'][chosen]) == category
            m['sample_teacher_top1'] += float(p[best])
    return {k: {'n': int(v['n']), **{s: value/v['n'] for s,value in v.items() if s != 'n'}}
            for k,v in totals.items()}


class Journal:
    """JSONL is the durable record; CSV, PNG and W&B use the same scalar events."""
    def __init__(self, root, config, stage, resume=False):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'metrics.jsonl'
        self.events = [json.loads(l) for l in self.path.open()] if resume and self.path.exists() else []
        self.run = None
        setting = config.get('logging', {})
        if setting.get('wandb_mode') != 'disabled':
            import wandb
            self.run = wandb.init(project=setting['wandb_project'], name=f'{self.root.parent.name}-{stage}',
                dir=str(self.root), config=config, mode=setting.get('wandb_mode', 'offline'))

    def log(self, event):
        event = {'time': time.time(), **event}
        self.events.append(event)
        with self.path.open('a') as f: f.write(json.dumps(event, allow_nan=False)+'\n')
        if self.run: self.run.log(event)

    def plot(self):
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = sorted({k for e in self.events for k,v in e.items() if isinstance(v, (int,float))})
        with (self.root / 'metrics.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(self.events)
        plots = [('loss', 'policy_loss', 'valid_ce', 'valid_policy_ce'),
                 ('dev_score', 'dev_kakuna', 'dev_frozen', 'valid_top1'),
                 ('kl', 'entropy', 'clip_fraction', 'value_loss'),
                 ('actor_lr', 'lr', 'decisions_per_s', 'tokens_per_s')]
        fig, axes = plt.subplots(2,2,figsize=(12,8))
        for ax, names in zip(axes.flat, plots):
            right = None
            for name in names:
                rows = [e for e in self.events if name in e]
                xkey = 'env_steps' if any('env_steps' in e for e in rows) else 'step'
                rows.sort(key=lambda e:e.get(xkey,e.get('round',0)))
                target = ax
                if name in ('lr','actor_lr'):
                    right = right or ax.twinx(); target = right; right.set_ylabel('Learning rate')
                if rows: target.plot([e.get(xkey,e.get('round',0)) for e in rows], [e[name] for e in rows],
                                     marker='.',label=name)
            ax.set_xlabel('Environment decisions' if any('env_steps' in e for e in self.events) else 'Optimizer step')
            ax.grid(alpha=.2)
            lines = list(ax.lines) + (list(right.lines) if right else [])
            if lines: ax.legend(lines,[line.get_label() for line in lines])
        fig.tight_layout(); fig.savefig(self.root / 'curves.png', dpi=150); plt.close(fig)

    def close(self):
        self.plot()
        if self.run: self.run.finish()
