"""Encode human/controlled episodes once; keep source labels and chronology separate."""
import argparse,json,pickle,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arena.opponent.joint.tokens import Vocabulary
from arena.opponent.joint.data import games
from arena.opponent.joint.training_data import human_records,controlled_records


def encode_file(job):
    source,split,path,out,catalog=job;vocab=Vocabulary(catalog)
    prefix=Path(path).parent.name+'-' if source=='controlled' else ''
    name=source+'/'+split+'/'+prefix+Path(path).name.replace('.jsonl.gz','.parquet')
    target=Path(out)/name;target.parent.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('payload',pa.binary())]);writer=pq.ParquetWriter(target,schema,compression='zstd')
    buffer=[];index=[];number=0;steps=0
    if source=='human':
        records=(row for batch in pq.ParquetFile(path).iter_batches(batch_size=32) for row in batch.to_pylist())
    else:records=games(path)
    for game in records:
        if source=='controlled' and any(len({m['species'] for m in team})!=6 for team in game['teams'].values()):continue
        generator=human_records(game,vocab) if source=='human' else controlled_records(game,vocab)
        for side,data in generator:
            if len(data['target'])<2:continue
            si=int(side[1])-1;bid=game['battle_id'] if source=='human' else game['id']
            player=game['players'][si]
            index.append(dict(battle_id=bid,side=side,player_id=player,source=source,split=split,file=name,row=number,
                              order=game.get('upload_time') or game.get('month') or game.get('round',0),length=len(data['target'])))
            buffer.append({'payload':pickle.dumps(data,protocol=5)});number+=1;steps+=len(data['target'])
            if len(buffer)==32:writer.write_table(pa.Table.from_pylist(buffer,schema=schema));buffer=[]
    if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=schema))
    writer.close()
    return index,steps


def main(args):
    started=time.time();base=args.directory;out=base/'training/data';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2027);jobs=[]
    for split in ('train','validation','test'):
        jobs.extend(('human',split,str(p),str(out),str(base/'catalog.json')) for p in sorted((base/'human'/split).glob('*.parquet')))
    train=sorted((base/'data/train').glob('*.gz'))+sorted((base/'data/supplement').glob('*.gz'))
    train=[train[i] for i in sorted(rng.choice(len(train),min(args.controlled_train_groups,len(train)),replace=False))]
    validation=sorted((base/'data/calibration').glob('*.gz'))
    validation=[validation[i] for i in sorted(rng.choice(len(validation),min(40,len(validation)),replace=False))]
    validation+=sorted((base/'training/validation_source/data/train').glob('*.gz'))
    for split,paths in [('train',train),('validation',validation),('test',sorted((base/'data/test').glob('*.gz')))]:
        for p in paths:
            if 'validation_source' in str(p):
                # IDs are generated outside all original group ranges.
                jobs.append(('controlled',split,str(p),str(out),str(base/'catalog.json')))
            else:jobs.append(('controlled',split,str(p),str(out),str(base/'catalog.json')))
    indices=[];steps=0
    with ProcessPoolExecutor(args.workers) as pool:
        pending=[pool.submit(encode_file,j) for j in jobs]
        for i,future in enumerate(as_completed(pending),1):
            rows,n=future.result();indices.extend(rows);steps+=n
            if i%20==0 or i==len(jobs):print(json.dumps({'files':i,'total':len(jobs),'player_games':len(indices),'encoded_steps':steps,'seconds':round(time.time()-started,1)}),flush=True)
    frame=pd.DataFrame(indices)
    # Human links were already checked against strict earlier upload/month order.
    links=pq.read_table(base/'human/players.parquet',columns=['battle_id','side','previous_battle_id','previous_side']).to_pandas()
    human=frame[frame.source=='human'].merge(links,on=['battle_id','side'],validate='one_to_one')
    controlled=frame[frame.source=='controlled'].sort_values(['player_id','order'])
    controlled['previous_battle_id']=controlled.groupby('player_id').battle_id.shift()
    controlled['previous_side']=controlled.groupby('player_id').side.shift()
    frame=pd.concat([human,controlled],ignore_index=True)
    frame=frame.sort_values(['source','split','battle_id','side'])
    frame['order']=frame['order'].astype(str)
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),out/'index.parquet',compression='zstd')
    vocab=Vocabulary(base/'catalog.json')
    summary={'player_games':frame.groupby(['source','split']).size().to_dict(),'encoded_steps':steps,
             'vocabulary':len(vocab.tokens),'seconds':time.time()-started,'controlled_train_groups':len(train),
             'human_label':'visible primary action x observed Tera; unobserved outcome is its own class, not a submitted no-op',
             'configuration_label':'full controlled truth; human first-revealed initial item and Tera only',
             'support':'previous available game; current-game encoder resets, strategy state carries across games'}
    summary['player_games']={':'.join(k):v for k,v in summary['player_games'].items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'artifacts/opponent_joint');p.add_argument('--workers',type=int,default=8);p.add_argument('--controlled-train-groups',type=int,default=600);main(p.parse_args())
