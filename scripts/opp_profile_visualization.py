"""Visualize recorded main-experiment choices and standardized player memories."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from arena.opponent.joint.network import JointNetwork
from arena.opponent.joint.strategy import BEHAVIORS

BASE = ROOT / 'artifacts/opponent_joint/training'
OUT = BASE / 'visualizations/player_profiles'
MAIN_NAMES = ['攻击', '变化招式', '主动换人', '强化', '回复']
PROFILE_NAMES = ['受威胁时换人', '高血量时强化', '低血量时回复', '可用时太晶化']
FAMILIES = {'rule':'规则型', 'search':'搜索型', 'learned':'学习型', 'adaptive':'适应型', 'random':'随机型'}
COLORS = {'rule':'#3477a8', 'search':'#dc8f23', 'learned':'#3b9778', 'adaptive':'#aa5c96', 'random':'#8a8e96'}
SCENARIOS = {'known':'已知', 'unseen_team':'未见队伍', 'unseen_set':'未见配置',
             'unseen_family':'未见策略', 'abrupt':'突变', 'gradual':'渐变', 'confounded':'混淆'}


def main_profiles():
    folder = BASE / 'joint_validation'
    protocol = json.loads((folder / 'protocol.json').read_text())
    records = [r for f in sorted((folder / 'joint/parts').glob('*.json'))
               for r in json.loads(f.read_text())['records']]
    ordinary = [r for r in records if r.get('phase') == 'turn']
    valid = [r for r in ordinary if r['valid_prediction']]
    moves = sorted({a.split(':')[1] for r in valid for a in (r['action']['top5'][0], r['action']['target']) if a.startswith('move:')})
    program = """
const fs=require('fs'), {dex}=require('./scripts/node/opp_joint_common.js');
const result={};
for(const id of JSON.parse(fs.readFileSync(0,'utf8'))){
 const m=dex.moves.get(id); if(!m.exists)throw new Error(id);
 result[id]=[Number(m.category!=='Status'),Number(m.category==='Status'),0,
  Number(m.category==='Status'&&!!m.boosts&&Object.values(m.boosts).some(v=>v>0)),
  Number(!!m.heal||['rest','wish','strengthsap'].includes(m.id))];
}
process.stdout.write(JSON.stringify(result));
"""
    move_features = json.loads(subprocess.run(['node', '-e', program], input=json.dumps(moves),
        capture_output=True, text=True, check=True, cwd=ROOT).stdout)
    def vector(action):
        if action.startswith('switch:'): return [0, 0, 1, 0, 0]
        return move_features[action.split(':')[1]]
    players = []
    for entry in protocol['cohort']:
        player = entry['player']
        if any(p['player'] == player for p in players): continue
        selected = [r for r in valid if r['player'] == player]
        predicted = np.array([vector(r['action']['top5'][0]) for r in selected], float)
        actual = np.array([vector(r['action']['target']) for r in selected], float)
        players.append(dict(alias=f'M{len(players)+1:02}', player=player, scenario=entry['scenario'],
            decisions=len(selected), excluded_invalid=sum(r['player']==player and not r['valid_prediction'] for r in ordinary),
            games=list(dict.fromkeys(r['game'] for r in selected)),
            predicted=predicted.mean(0).tolist(), observed=actual.mean(0).tolist()))
    return dict(source='training/joint_validation/joint/parts', players=players,
        dimensions=MAIN_NAMES, ordinary_decisions=len(ordinary), used_decisions=len(valid),
        omitted_forced=sum(r.get('phase')=='forced' for r in records),
        excluded_invalid=len(ordinary)-len(valid),
        definition='各玩家有效普通决策中，模型Top-1行动或实际提交行动属于相应行为的比例。攻击、变化招式、主动换人合计100%；强化和回复为可重叠的附加标签。',
        limitations='局面、队伍和行为机会不同会影响频率；这不是共同局面下的稳定玩家画像。原始行动评分合并了太晶选项，不能还原太晶预测概率。')


@torch.inference_mode()
def memory_profiles():
    folder = BASE / 'experiments/profile_transfer'
    protocol = json.loads((folder / 'protocol.json').read_text())
    model = JointNetwork.load(BASE / 'adaptation/model.pt', 'cuda:0')
    players = []
    for entry in protocol['cohort']:
        root = torch.load(folder / 'memory' / (entry['player'] + '.pt'), map_location='cpu', weights_only=True)['roots'][5]
        weights = root['weights'].double(); weights /= weights.sum()
        profile = model.behavior_profile(root['z'].to('cuda:0'), weights)
        fields = [profile[name] for name in BEHAVIORS]
        players.append(dict(alias=f'P{len(players)+1:02}', player=entry['player'],
            family=entry['family'], scenario=entry['scenario'], histories=5,
            effective_decisions=root['effective_decisions'], profile=profile,
            probabilities=[f['probability'] for f in fields],
            posterior_interval_90=[f['posterior_interval_90'] for f in fields]))
    x = np.array([p['probabilities'] for p in players]); center = x.mean(0)
    u, s, vt = np.linalg.svd(x-center, full_matrices=False)
    for k in range(len(vt)):
        if vt[k,np.argmax(abs(vt[k]))] < 0: vt[k] *= -1
    coordinates = (x-center) @ vt[:2].T
    for p, point in zip(players, coordinates): p['pca'] = point.tolist()
    widths = np.diff(np.array([p['posterior_interval_90'] for p in players]), axis=2)[:,:,0]
    spread = {name:dict(minimum=float(x[:,j].min()), maximum=float(x[:,j].max()),
        mean=float(x[:,j].mean()), between_player_sd=float(x[:,j].std(ddof=1)),
        median_posterior_interval_width=float(np.median(widths[:,j]))) for j,name in enumerate(BEHAVIORS)}
    distance = np.abs(x[:,None]-x[None]).mean(-1)
    query = json.loads((folder / 'results.json').read_text())['player_profiles']
    agreement = {}
    for field in ('switch','setup','recovery','tera'):
        pairs = [(p['behaviors'][field]['predicted_probability'],p['behaviors'][field]['teacher_probability']) for p in query if field in p['behaviors']]
        a = np.array(pairs)
        agreement[field] = dict(players=len(a), mae=float(np.abs(a[:,0]-a[:,1]).mean()))
    return dict(source='training/experiments/profile_transfer/memory',
        model='training/adaptation/model.pt', players=players, dimensions=PROFILE_NAMES,
        reference_situations={k:len(v) for k,v in model.behavior_probes.items()},
        definition='把每位玩家前5局积累的加权策略粒子Z放入完全相同的参考局面和动作空间，计算档案的四项行为概率。网络权重与参考局面固定。',
        limitations='50名玩家来自已完成的画像与换队迁移实验，均为受控留出玩家；并非主实验14名玩家。概率差异表示模型画像不同，不等同于真实人格分类。',
        uncertainty='90%区间为该玩家粒子后验的行为概率分位区间，不是跨玩家均值的置信区间。',
        spread=spread, mean_pairwise_absolute_difference=float(distance[np.triu_indices(len(x),1)].mean()),
        pca=dict(explained_variance_ratio=(s*s/(s@s)).tolist(),
            center=center.tolist(), components=vt.tolist(),
            preprocessing='概率直接中心化，不做逐维标准化；不聚类，控制器类别仅用于着色。'),
        query_probability_agreement=agreement)


def csv_data(main, memory):
    with (OUT/'main_profiles.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['player','alias','scenario','decisions','source',*MAIN_NAMES])
        for p in main['players']:
            for kind in ('predicted','observed'): writer.writerow([p['player'],p['alias'],p['scenario'],p['decisions'],kind,*p[kind]])
    with (OUT/'standardized_profiles.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['player','alias','family','history_games','effective_decisions',*PROFILE_NAMES,'PC1','PC2'])
        for p in memory['players']:writer.writerow([p['player'],p['alias'],p['family'],5,p['effective_decisions'],*p['probabilities'],*p['pca']])


def figures(main, memory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def heat(ax,x,labels,columns):
        im=ax.imshow(x,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set(xticks=range(len(columns)),xticklabels=columns,yticks=range(len(labels)),yticklabels=labels)
        ax.tick_params(axis='y',length=0,labelsize=8)
        for i in range(len(x)):
            for j in range(len(columns)):
                ax.text(j,i,f'{100*x[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if x[i,j]>.55 else '#183047')
        return im
    labels=[f"{p['alias']} {SCENARIOS[p['scenario']]} · n={p['decisions']}" for p in main['players']]
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,kind,title in zip(axes,('predicted','observed'),('模型Top-1预测的行为','对手实际提交的行为')):
        im=heat(ax,np.array([p[kind] for p in main['players']]),labels,MAIN_NAMES);ax.set_title(title,pad=12)
    fig.colorbar(im,ax=axes,label='行为频率',shrink=.8,format=matplotlib.ticker.PercentFormatter(1))
    fig.suptitle(f"主实验：14名玩家、28局、{main['used_decisions']}个有效普通决策\n同一批决策配对展示；强化、回复属于附加标签",fontsize=13)
    save(fig,'main_behavior_profiles')

    x=np.array([p['probabilities'] for p in memory['players']]); xy=np.array([p['pca'] for p in memory['players']])
    ev=np.array(memory['pca']['explained_variance_ratio'])
    labels=[f"{p['alias']} {FAMILIES[p['family']]}" for p in memory['players']]
    fig,axes=plt.subplots(1,2,figsize=(15,13),gridspec_kw={'width_ratios':[1,1.12]},layout='constrained')
    im=heat(axes[0],x,labels,PROFILE_NAMES);axes[0].set_title('共同参考局面下的玩家画像（%）',pad=12)
    axes[0].tick_params(axis='x',labelsize=9)
    fig.colorbar(im,ax=axes[0],label='行为概率',shrink=.48,format=matplotlib.ticker.PercentFormatter(1))
    for family,label in FAMILIES.items():
        indices=[i for i,p in enumerate(memory['players']) if p['family']==family]
        axes[1].scatter(xy[indices,0],xy[indices,1],s=75,color=COLORS[family],label=label,edgecolors='white',linewidth=.7,alpha=.85)
    axes[1].set(xlabel=f'PC1 · 解释{ev[0]:.1%}方差',ylabel=f'PC2 · 解释{ev[1]:.1%}方差',
        title=f'四维行为概率的二维投影\n累计解释{ev[:2].sum():.1%}方差')
    axes[1].set_box_aspect(1);axes[1].grid(alpha=.18);axes[1].legend(title='已知控制器类别（用于着色）',loc='best')
    fig.suptitle('标准化玩家画像：50名留出玩家，每人5局历史，每个维度使用相同的16个参考局面',fontsize=14)
    save(fig,'standardized_player_profiles')

    fig,axes=plt.subplots(2,2,figsize=(14,11),layout='constrained')
    intervals=np.array([p['posterior_interval_90'] for p in memory['players']])
    for j,ax in enumerate(axes.flat):
        for i,p in enumerate(memory['players']):
            ax.plot(intervals[i,j]*100,[i,i],color=COLORS[p['family']],alpha=.3,linewidth=1)
            ax.scatter(x[i,j]*100,i,s=14,color=COLORS[p['family']])
        ax.set(title=PROFILE_NAMES[j],xlim=(-2,102),xlabel='行为概率（%）',
               yticks=np.arange(0,50,5),yticklabels=[memory['players'][i]['alias'] for i in range(0,50,5)])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
    handles=[plt.Line2D([],[],marker='o',linestyle='none',color=COLORS[f],label=label) for f,label in FAMILIES.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('玩家画像的不确定性：点为后验均值，线为90%粒子后验区间',fontsize=14)
    save(fig,'profile_uncertainty')


def interactive(main,memory):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    panels=[]
    fig=make_subplots(rows=1,cols=2,subplot_titles=['模型Top-1预测','实际提交'])
    for col,kind in enumerate(('predicted','observed'),1):
        players=main['players'];x=np.array([p[kind] for p in players])*100
        labels=[p['alias']+' '+SCENARIOS[p['scenario']] for p in players]
        info=[[[p['player'],p['decisions']] for _ in MAIN_NAMES] for p in players]
        fig.add_trace(go.Heatmap(z=x,x=MAIN_NAMES,y=labels,coloraxis='coloraxis',customdata=info,
            hovertemplate='%{customdata[0]}<br>%{x}：%{z:.1f}%<br>普通决策：%{customdata[1]}<extra></extra>'),row=1,col=col)
    fig.update_yaxes(autorange='reversed');fig.update_layout(height=650,title='主实验：预测行为与实际行为',coloraxis=dict(colorscale='YlGnBu',cmin=0,cmax=100,colorbar_title='%'))
    panels.append(fig)
    players=memory['players'];x=np.array([p['probabilities'] for p in players])*100
    info=[[[p['player'],FAMILIES[p['family']],100*interval[0],100*interval[1],p['effective_decisions']] for interval in p['posterior_interval_90']] for p in players]
    fig=go.Figure(go.Heatmap(z=x,x=PROFILE_NAMES,y=[p['alias']+' '+FAMILIES[p['family']] for p in players],
        customdata=info,colorscale='YlGnBu',zmin=0,zmax=100,colorbar_title='%',
        hovertemplate='%{customdata[0]} · %{customdata[1]}<br>%{x}：%{z:.1f}%<br>90%后验区间：%{customdata[2]:.1f}–%{customdata[3]:.1f}%<br>有效历史决策：%{customdata[4]}<extra></extra>'))
    fig.update_layout(height=1250,title='共同参考局面：50名玩家的四维画像');fig.update_yaxes(autorange='reversed');panels.append(fig)
    ev=memory['pca']['explained_variance_ratio'];fig=go.Figure()
    for family,label in FAMILIES.items():
        selected=[p for p in players if p['family']==family]
        info=[[p['alias'],p['player'],*[100*v for v in p['probabilities']]] for p in selected]
        fig.add_trace(go.Scatter(x=[p['pca'][0] for p in selected],y=[p['pca'][1] for p in selected],mode='markers',name=label,
            marker=dict(size=12,color=COLORS[family],line=dict(color='white',width=1)),customdata=info,
            hovertemplate='%{customdata[0]} · %{customdata[1]}<br>受威胁换人：%{customdata[2]:.1f}%<br>高血量强化：%{customdata[3]:.1f}%<br>低血量回复：%{customdata[4]:.1f}%<br>太晶化：%{customdata[5]:.1f}%<extra></extra>'))
    fig.update_layout(height=700,title=f'四维画像PCA · 前两维解释{sum(ev[:2]):.1%}方差',
        xaxis_title=f'PC1 ({ev[0]:.1%})',yaxis_title=f'PC2 ({ev[1]:.1%})',legend_title='控制器类别（仅用于着色）');panels.append(fig)
    html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>对手画像可视化</title>',
        '<style>body{max-width:1280px;margin:32px auto;padding:0 20px;font-family:"Noto Sans CJK SC",sans-serif;color:#183047}p{line-height:1.7}h1{font-size:26px}footer{margin:24px 0;color:#536578}</style>',
        '<h1>对手画像：行为差异与预测分布</h1>',
        f'<p>主实验：14名玩家、28局，保留{main["used_decisions"]}个有效普通决策；排除{main["excluded_invalid"]}个无效预测及强制换人。频率受局面与队伍影响，强化和回复标签可与其他维度重叠。</p>']
    for i,fig in enumerate(panels):
        fig.update_layout(template='plotly_white',font=dict(family='Noto Sans CJK SC, sans-serif',size=13),margin=dict(l=100,r=45,t=85,b=60))
        if i==1:html.append('<p>标准化画像来自已完成的玩家画像与换队迁移实验：50名留出玩家，每人前5局的策略记忆，使用相同网络与相同参考局面。每维16个参考局面。鼠标悬停可查看玩家标识、概率与后验区间。</p>')
        html.append(fig.to_html(full_html=False,include_plotlyjs=i==0,config=dict(displaylogo=False,toImageButtonOptions=dict(format='png',scale=2))))
    html.append('<footer>二维投影采用中心化后的四维行为概率，不使用玩家类别拟合，不强制聚类。点之间的分离表示模型预测的画像存在差异；预测是否准确由独立的预测指标判断。后验区间见同目录 profile_uncertainty.png。</footer></html>')
    (OUT/'player_profiles.html').write_text('\n'.join(html))


if __name__=='__main__':
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    main=main_profiles();memory=memory_profiles()
    result=dict(main_experiment=main,standardized_memory=memory)
    (OUT/'profiles.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    csv_data(main,memory);figures(main,memory);interactive(main,memory)
    print(json.dumps(dict(main_players=len(main['players']),main_decisions=main['used_decisions'],
        memory_players=len(memory['players']),spread=memory['spread'],
        pca_explained_variance=memory['pca']['explained_variance_ratio'],
        mean_pairwise_absolute_difference=memory['mean_pairwise_absolute_difference'],output=str(OUT)),ensure_ascii=False,indent=2))
