"""Frozen Qwen3.8-27B baselines on the shared complete-game opponent cases."""
import argparse
import asyncio
import json
import pickle
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
from opp_baseline_topk import Scores

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/opponent_joint/training/comparison/baselines'
MODEL='/data/models/Qwen3.8-27B'
DEX=json.loads((ROOT/'external/pokechamp/repo/poke_env/data/static/pokedex/gen9pokedex.json').read_text())
SYSTEM="""You predict an opponent in Pokemon Showdown gen9ou. Use your Pokemon knowledge and the supplied causal observations to infer the opponent's original hidden configuration and next submitted action. Predict the opponent, not our action. Our private information can explain observations but is not automatically known to the opponent. Previous encounters can inform behavior; their configurations are not confirmed facts in the current game.
Return one JSON object with this structure:
{"team":[{"slot":1,"moves":["move_id",...],"item":["item_id",...],"ability":["ability_id",...],"tera":["type",...]},...],"actions":[{"move":"move_id"},{"switch":"species_name"},...]}
Include all six opponent roster slots in preview order. Give EXACTLY FIVE DISTINCT ranked candidates for moves, item and tera, most likely first, including lower-confidence alternatives. High confidence in your first answer does not remove the requirement for alternatives. For ability, rank all legal species abilities; the schema gives their public count. For moves, rank still-unrevealed moves by their probability of being in the original moveset; this is a candidate list, not a five-move configuration. Predict original items and abilities, distinguishing them from replacements or consumptions. Already revealed fields are not scored. Use English Showdown identifiers; use "none" for no held item.
Provide FIVE ranked next submitted action hypotheses, including lower-confidence alternatives. Each action has exactly one of "move" or "switch". Merge Terastallization variants of the same action. A forced switch permits only switching. Use null only for remaining positions when fewer than five distinct actions are possible given public facts. During team preview return an empty actions list. Return compact JSON. All names must be lowercase alphanumeric Showdown IDs, without spaces or explanations. Do not include probability numbers."""


def norm(value):
    return ''.join(c for c in value.lower() if c.isalnum())


def action_key(action):
    if not isinstance(action,dict):return None
    for kind in ('move','switch'):
        if isinstance(action.get(kind),str):
            name=norm(action[kind])
            if kind=='switch':name=norm(DEX.get(name,{}).get('baseSpecies',name))
            return kind+':'+name
    return None


def unique(values):
    return list(dict.fromkeys(v for v in values if v))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def output_schema(preview,phase):
    identifier=dict(type='string',pattern='^[a-z0-9]+$',maxLength=64)
    def ranked(count):return dict(type='array',items=identifier,minItems=count,maxItems=count)
    team=[]
    for slot,species in enumerate(preview,1):
        count=len(set(DEX[norm(species)]['abilities'].values()))
        fields=dict(slot=dict(type='integer',const=slot),moves=ranked(5),item=ranked(5),ability=ranked(count),tera=ranked(5))
        team.append(dict(type='object',properties=fields,required=list(fields),additionalProperties=False))
    action={'anyOf':[dict(type='object',properties={kind:identifier},required=[kind],additionalProperties=False)
                     for kind in ('move','switch')]+[{'type':'null'}]}
    fields=dict(team=dict(type='array',prefixItems=team,minItems=6,maxItems=6),
                actions=dict(type='array',items=action,minItems=0 if phase=='preview' else 5,maxItems=0 if phase=='preview' else 5))
    return dict(type='json_schema',json_schema=dict(name='opponent_predictions',strict=True,
                schema=dict(type='object',properties=fields,required=list(fields),additionalProperties=False)))


def pure_cases():
    protocol=json.loads((OUT/'cases.json').read_text())
    players={(r['game'],r['side']):r['player'] for r in protocol['cohort']}
    past={};result=[]
    for path in sorted((OUT/'cases').glob('*.pkl')):
        for game in pickle.loads(path.read_bytes()):
            for side,case in game['cases'].items():
                observer='p2' if side=='p1' else 'p1';player=players[game['id'],side]
                preview=[l.split('|')[3].split(',')[0] for l in game['public'] if l.startswith('|poke|'+side+'|')]
                decisions=[d for d in game['decisions'] if d['side']==side]
                for t,(end,decision) in enumerate(zip(case['cutoffs'],decisions)):
                    request=decision['request']
                    phase='preview' if request.get('teamPreview') else 'forced_switch' if request.get('forceSwitch') else 'turn'
                    data=dict(format='gen9ou',observer_side=observer,opponent_side=side,
                        opponent_preview=preview,observer_initial_configuration=game['teams'][observer],
                        previous_encounters=past.get(player,[]),
                        observer_visible_history='\n'.join(game['private'][observer][:end]),phase=phase)
                    action=decision['selected'];label={action['kind']:action.get('move',action.get('species'))}
                    result.append(dict(game=game['id'],side=side,decision=t,phase=phase,system=SYSTEM,
                        prompt=json.dumps(data,ensure_ascii=False,separators=(',',':')),
                        response_format=output_schema(preview,phase),
                        label=label,truth=case['truth'],hidden=case['hidden'][t].tolist()))
                past.setdefault(player,[]).append(dict(opponent_side=side,public_history='\n'.join(game['public'])))
    return result


def hidden_predictions(answer):
    predicted=[{} for _ in range(6)]
    for mon in answer.get('team',[]):
        if not isinstance(mon,dict) or mon.get('slot') not in range(1,7):continue
        for field in ('moves','item','ability','tera'):
            values=mon.get(field,[])
            if isinstance(values,list):
                ranked=unique(norm(v) for v in values if isinstance(v,str))
                predicted[mon['slot']-1][field]={v:1/(i+1) for i,v in enumerate(ranked[:5])}
    return predicted


async def evaluate(args):
    native=args.method=='pokechamp';stem='pokechamp_qwen38_27b' if native else 'qwen38_27b'
    cases=read_rows(OUT/'pokechamp_action_cases.jsonl') if native else pure_cases()
    path=OUT/(stem+'_predictions.jsonl');records=read_rows(path)
    done={(r['game'],r['side'],r['decision']) for r in records}
    clients=[AsyncOpenAI(base_url=url,api_key='local',timeout=600,max_retries=0) for url in args.url]
    concurrency=args.concurrency*len(clients) if len(args.concurrency)==1 else args.concurrency
    started=time.monotonic()
    pending=iter((i,row) for i,row in enumerate(cases) if (row['game'],row['side'],row['decision']) not in done)
    async def predict(index,row,endpoint):
        tick=time.monotonic()
        response=await clients[endpoint].chat.completions.create(model='Qwen3.8-27B',
            messages=[dict(role='system',content=row['system']),dict(role='user',content=row['prompt'])],
            response_format={'type':'json_object'} if native else row['response_format'],temperature=.3 if native else 0,
            n=32 if native else 1,max_tokens=64 if native else 2048,seed=2027+index,
            extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        texts=[c.message.content or '' for c in response.choices];answers=[];invalid=0
        for text in texts:
            try:
                parsed=json.loads(text)
                if not isinstance(parsed,dict):raise ValueError('Expected JSON object')
                answers.append(parsed)
            except ValueError:answers.append({});invalid+=1
        record={k:row[k] for k in ('game','side','decision','phase')}
        if native:
            counts=Counter(k for answer in answers if (k:=action_key(answer)))
            invalid=len(answers)-sum(counts.values())
            order=sorted(counts,key=lambda k:(-counts[k],k))
            record.update(counts=dict(counts),candidate_covered=action_key(row['label']) in {action_key(a) for a in row['candidates']})
        else:
            answer=answers[0];actions=answer.get('actions',[])
            order=unique(action_key(a) for a in actions) if isinstance(actions,list) else []
            record['prediction']=answer
        target=action_key(row['label']) if row['phase']!='preview' else None
        record.update(target=target,top5=order[:5],hits=[int(target in order[:k]) for k in (1,3,5)] if target else None,
            responses=texts,invalid_generations=invalid,finish_reasons=[c.finish_reason for c in response.choices],
            usage=response.usage.model_dump(),seconds=time.monotonic()-tick)
        return record
    async def worker(endpoint,stream):
        for index,case in pending:
            row=await predict(index,case,endpoint)
            records.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            if len(records)%16==0 or len(records)==1:
                print(json.dumps(dict(method=stem,points=len(records),total=len(cases),seconds=round(time.monotonic()-started,1),invalid_generations=sum(r['invalid_generations'] for r in records))),flush=True)
    try:
        with path.open('a') as stream:
            await asyncio.gather(*(worker(i,stream) for i,n in enumerate(concurrency) for _ in range(n)))
    finally:await asyncio.gather(*(client.close() for client in clients))
    actions=[r for r in records if r['phase']!='preview'];hits=[sum(r['hits'][i] for r in actions) for i in range(3)]
    metrics=dict(moves=None,item=None,ability=None,tera=None,
        action=dict(count=len(actions),hits=hits,**{f'top{k}':h/len(actions) for k,h in zip((1,3,5),hits)}))
    if not native:
        by_case={(r['game'],r['side'],r['decision']):r for r in records};scores=Scores()
        for row in cases:scores.add(hidden_predictions(by_case[row['game'],row['side'],row['decision']]['prediction']),row['truth'],row['hidden'])
        metrics.update(scores.result())
        assert {k:metrics[k]['count'] for k in ('moves','item','ability','tera')}==json.loads((OUT/'cases.json').read_text())['counts']
    assert len(records)==len(cases) and len(actions)==len(read_rows(OUT/'pokechamp_action_cases.jsonl'))
    result=dict(method=('PokéChamp opponent prompt + Qwen3.8-27B' if native else 'Pure Qwen3.8-27B'),category='LLM',
        backbone=MODEL,quantization='bitsandbytes 4-bit',thinking=False,
        player_games=len({(r['game'],r['side']) for r in records}),decision_points=len(records),metrics=metrics,
        temperature=.3 if native else 0,samples=32 if native else 1,max_tokens=64 if native else 2048,
        invalid_generations=sum(r['invalid_generations'] for r in records),
        truncated_generations=sum(f=='length' for r in records for f in r['finish_reasons']),
        mean_request_seconds=sum(r['seconds'] for r in records)/len(records),
        input=('Frozen native PokéChamp opponent prompts and candidates from the shared causal cases.' if native else
               'Complete observer-visible history, own initial configuration, opponent public preview and previous selected encounter public history. No opponent hidden labels or true legal-action list.'),
        ranking=('Empirical frequencies of 32 native-prompt JSON generations per decision; ties use action name.' if native else
                 'One schema-constrained JSON ranking per decision: five moves/items/Tera candidates, all species abilities, five action positions with null allowed for insufficient public action support. No trained opponent module, usage prior or search.'),
        scoring='Same locked cohort and unrevealed masks. Moves use micro recall@k. Actions merge Tera and exclude preview. Invalid or missing predictions count as misses.')
    if native:result['candidate_coverage']=sum(r['candidate_covered'] for r in records)/len(records)
    (OUT/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=['pokechamp','pure'])
    parser.add_argument('--url',nargs='+',default=['http://127.0.0.1:8102/v1'])
    parser.add_argument('--concurrency',type=int,nargs='+',default=[4])
    asyncio.run(evaluate(parser.parse_args()))
