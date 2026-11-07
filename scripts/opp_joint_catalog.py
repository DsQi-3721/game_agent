"""Build legal complete sets and team compositions for joint-model research."""
import argparse, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/opponent_joint'

def main():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',type=Path,required=True,help='Showdown complete-set JSON with a dex mapping')
    parser.add_argument('--teams',type=Path,required=True,help='Foul Play team-composition JSON')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.sets.read_text())
    usage=json.load(open(ROOT/'data/knowledge/gen9ou_sets.json'))['species']
    from poke_env.data import to_id_str
    stats=['hp','atk','def','spa','spd','spe'];sets=[]
    for species,entries in source['dex'].items():
        for title,original in entries.items():
            row=dict(original,species=species,name=species,level=100)
            row['moves']=[m[0] if isinstance(m,list) else m for m in row['moves']]
            row['evs']={s:row.get('evs',{}).get(s,0) for s in stats}
            row['ivs']={s:row.get('ivs',{}).get(s,31) for s in stats}
            sets.append(row)
            sid=to_id_str(species)
            # Legal variations are simulated configurations, never asserted human truth.
            for j,spread in enumerate(usage.get(sid,{}).get('spreads',[])[:3]):
                variant=dict(row,evs=dict(zip(stats,spread['evs'])),ivs=dict(zip(stats,spread.get('ivs',[31]*6))),nature=spread['nature'].title())
                variants=usage.get(sid,{}).get('tera',[])
                if variants:variant['teraType']=variants[j%len(variants)]['id']
                sets.append(variant)
    teams=json.loads(args.teams.read_text())
    compositions=[key.split('|') for key in teams]
    (args.output/'catalog-source.json').write_text(json.dumps({'sets':sets,'compositions':compositions}))
    print({'candidate_sets':len(sets),'team_compositions':len(compositions)})
if __name__=='__main__':main()
