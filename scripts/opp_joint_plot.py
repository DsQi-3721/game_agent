"""Update a single research learning-curve figure from the append-only JSON log."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory):
    rows=[json.loads(line) for line in (directory/'learning_curve.jsonl').read_text().splitlines() if line.strip()]
    rows=[r for r in rows if 'epoch' in r]
    validation=[r for r in rows if 'validation' in r]
    fig,axes=plt.subplots(2,3,figsize=(15,8));colors={'human':'#176bb3','controlled':'#c45a20'}
    for i,source in enumerate(('human','controlled')):
        ax=axes[0,i];v=[r for r in validation if source in r['validation']]
        ax.plot([r['epoch'] for r in v],[r['validation'][source]['nll'] for r in v],'.-',label='Fixed validation',color=colors[source])
        t=[r for r in rows if source in r.get('training_interval',{})]
        if t:ax.plot([r['epoch'] for r in t],[r['training_interval'][source]['nll'] for r in t],'--',label='Recent training mean',color='#777777')
        ax.set_title('Human: observed outcome NLL' if source=='human' else 'Controlled: submitted action NLL');ax.set_ylabel('NLL');ax.legend()
        axes[0,2].plot([r['epoch'] for r in v],[100*r['validation'][source]['top1'] for r in v],'.-',label=source,color=colors[source])
    axes[0,2].set_title('Fixed validation: Top-1');axes[0,2].set_ylabel('%');axes[0,2].legend()
    reference=directory/'oracle_reference.json'
    if reference.exists():
        oracle=json.loads(reference.read_text())
        axes[0,1].axhline(oracle['controlled_validation_oracle']['nll'],color='#508044',linestyle=':',label='Oracle: privileged policy')
        axes[0,1].legend()
    lr=[r for r in rows if 'lr' in r and 'gradient_norm' in r];axes[1,0].plot([r['epoch'] for r in lr],[r['lr'] for r in lr]);axes[1,0].set_title('Learning rate');axes[1,0].set_yscale('log')
    for source,ax,fields in [('human',axes[1,1],('query:item','query:tera')),('controlled',axes[1,2],('config_nll','post_observation_proposal_nll','change_ce'))]:
        for field in fields:
            v=[r for r in validation if field in r['validation'].get(source,{}).get('auxiliary_loss',{})]
            if v:ax.plot([r['epoch'] for r in v],[r['validation'][source]['auxiliary_loss'][field] for r in v],'.-',label=field.replace('post_observation_',''))
        ax.set_title(source.title()+': auxiliary validation losses')
        if ax.lines:ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Additional complete epochs');ax.grid(alpha=.2)
    for change in (r for r in rows if r.get('event')=='parameter_adjustment'):
        for ax in axes[0]:ax.axvline(change['epoch'],color='#888888',alpha=.45,linestyle='--',linewidth=.8)
    fig.suptitle('Opponent-model continuation | two GPUs | dashed vertical lines: parameter adjustments')
    fig.tight_layout();fig.savefig(directory/'learning_curve.png',dpi=140);fig.savefig(directory/'learning_curve.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]/'artifacts/opponent_joint/training');plot(p.parse_args().directory)
