// Lossless battle reconstruction: initial teams + seed + submitted commands.
// Public events and player-local requests are stored separately from truth.
const fs=require('fs'),zlib=require('zlib'),path=require('path');
const {Battle,dex,extractChannelMessages,id,clone,rng,sample,softmax,actions,PublicView,features,currentState}=require('./opp_joint_common');
const [directory,role,groupText,countText,teamMode='vary']=process.argv.slice(2),group=Number(groupText),gameCount=Number(countText||100);
const catalog=JSON.parse(fs.readFileSync(directory+'/catalog.json'));
const learned=JSON.parse(fs.readFileSync(directory+'/provider.json'));
const random=rng(17+group*9301+['train','calibration','test','supplement'].indexOf(role)*10000019);
const families=['rule','search','learned','random','adaptive'];
const testGroup=group%7;
const accounts=Array.from({length:20},(_,i)=>({id:`${role}-${group}-${i}`,family:families[i%5],
 changeTurn:8+Math.floor(random()*25),changeDuration:10+Math.floor(random()*20),
 temperature:.25+random()*.7,preference:[random()*2-1,random()*2-1,random()*2-1,random()*2-1],counts:{attack:1,switch:1,status:1},games:0}));
if(role==='test'&&testGroup===3)for(const account of accounts)account.family='heldout';
const policy=require('./opp_joint_policy').makePolicy(learned,role,group);
const playerTeams=new Map();
function chooseTeam(round,player){
 if(teamMode==='same'&&playerTeams.has(player.id))return clone(playerTeams.get(player.id));
 const heldout=role==='test'&&testGroup===1;
 const split=Math.floor(catalog.compositions.length*.8);
 const range=heldout?[split,catalog.compositions.length]:[0,split];
 let compositionIndex=range[0]+Math.floor(random()*(range[1]-range[0]));while(new Set(catalog.compositions[compositionIndex]).size!==6)compositionIndex=range[0]+Math.floor(random()*(range[1]-range[0]));const composition=catalog.compositions[compositionIndex];
 // Reusable complete teams cross player-policy assignments. A shared team
 // theme correlates set choices instead of independently assembling marginals.
 const theme=Math.floor(random()*4),teamRandom=rng(5519+compositionIndex*7919+theme*104729);
 const team=composition.map((species,slot)=>{
  const indices=catalog.bySpecies[species],n=indices.length;
  const reserved=n>2?indices[n-1]:indices[0];
  const available=indices.slice(0,Math.max(1,n-1));
  const values=available.map(j=>{const s=catalog.sets[j],moves=s.moves.map(m=>dex.moves.get(m));const offense=(s.evs.atk+s.evs.spa+s.evs.spe)/508,bulk=(s.evs.hp+s.evs.def+s.evs.spd)/508;const support=moves.filter(m=>m.category==='Status').length/4,pivot=moves.some(m=>m.selfSwitch)?1:0;return [offense*2,bulk*2,support*2,pivot*2+offense][theme];});
  let index=role==='test'&&testGroup===2?reserved:available[sample(softmax(values,.6),teamRandom)];
  if(role==='test'&&testGroup===6&&slot===0){const locked=Number(player.id.split('-').at(-1))%2===0;const choices=indices.filter(j=>catalog.sets[j].item.startsWith('choice')===locked);if(choices.length)index=choices[Math.floor(random()*choices.length)];}
  const set=clone(catalog.sets[index]);
  const speciesData=dex.species.get(species);set.gender=speciesData.gender||(random()<(speciesData.genderRatio?.M??.5)?'M':'F');
  if(role==='supplement'&&random()<.2){const freed=set.evs.spe;set.ivs.spe=0;set.evs.spe=0;set.evs.hp=Math.min(252,set.evs.hp+freed);}
  set.name=set.species;return set;
 });
 if(teamMode==='same')playerTeams.set(player.id,clone(team));
 return team;
}
const games=[],ring=[...accounts];
for(let round=0;games.length<gameCount;round++){
 for(let pair=0;pair<10&&games.length<gameCount;pair++){
  const players=[ring[pair],ring[19-pair]],teams={p1:chooseTeam(round,players[0]),p2:chooseTeam(round,players[1])};
  const seed=Array.from({length:4},()=>Math.floor(random()*65536));
  const game={id:`${role}-${group}-${games.length}`,role,group,round,players:players.map(p=>p.id),teams,seed,provider_revision:'legal_views_memory_resource_holdout',team_sampling:'shared_theme_reusable_full_teams',
    controllers:players.map(p=>clone(p)),team_mode:teamMode,
    families:players.map(p=>p.family),test_group:role==='test'?['known','unseen_team','unseen_set','unseen_family','abrupt','gradual','confounded'][testGroup]:null,public:[],private:{p1:[],p2:[]},decisions:[],commands:[]};
  const view=new PublicView(),pending={};let battle;
  const send=(type,data)=>{const text=Array.isArray(data)?data.join('\n'):data;
   if(type==='update'){const c=extractChannelMessages(text,[0,1,2]);game.public.push(...c[0]);game.private.p1.push(...c[1]);game.private.p2.push(...c[2]);view.update(c[0]);}
   if(type==='sideupdate'){const i=text.indexOf('\n'),side=text.slice(0,i),line=text.slice(i+1);if(line.startsWith('|request|')){const r=JSON.parse(line.slice(9));if(!r.wait)pending[side]=r;}}
  };
  battle=new Battle({formatid:'gen9ou',seed,send});
  for(let i=0;i<2;i++){const side='p'+(i+1);battle.setPlayer(side,{name:players[i].id,team:clone(teams[side])});battle.getSide(side).pokemon.forEach((p,j)=>p.jointSlot=j+1);}
  battle.sendUpdates();
  while(!battle.ended){
   if(battle.turn>100){battle.forceWin(null);battle.sendUpdates();game.capped=true;break;}
   const side=['p1','p2'].find(s=>pending[s]);if(!side)throw Error('No request');
   const request=pending[side];delete pending[side];const index=Number(side[1])-1,account=players[index];
   const slots=battle.getSide(side).pokemon.map(p=>p.jointSlot),choices=actions(request,slots);
   const x=features(request,choices,view,side,slots),teacher=policy(account,request,choices,x,view,side,round),chosen=sample(teacher.probabilities,random);
   const copy=clone(request);delete copy.side.name;
   game.decisions.push({side,turn:battle.turn,cutoff:game.public.length,private_cutoff:game.private[side].length,
    request:copy,slots,actions:choices.map(({command,...a})=>a),features:x,target:chosen,distribution:teacher.probabilities,
    truth:{family:teacher.family,change:teacher.change,mix:teacher.mix,memory:clone(account.counts),state:currentState(battle,side),other_state:currentState(battle,side==='p1'?'p2':'p1')}});
   game.commands.push([side,choices[chosen].command]);
   if(!battle.choose(side,choices[chosen].command))throw Error('Rejected action '+choices[chosen].command);
   battle.sendUpdates();
   // Adaptive providers see only the other side's resolved public behavior.
   for(const observed of game.public.slice(game.decisions.at(-1).cutoff)){
    const p=observed.split('|'),actor=p[2]?.slice(0,2);if(!['p1','p2'].includes(actor))continue;
    const watcher=players[actor==='p1'?1:0];
    if(p[1]==='switch')watcher.counts.switch++;
    if(p[1]==='move'&&!p.slice(4).some(s=>s.startsWith('[from]')))watcher.counts[dex.moves.get(p[3]).category==='Status'?'status':'attack']++;
   }
  }
  game.inputLog=battle.inputLog;game.turns=battle.turn;game.winner=battle.winner;games.push(game);for(const p of players)p.games++;
 }
 ring.splice(1,0,ring.pop());
}
const out=path.join(directory,'data',role);fs.mkdirSync(out,{recursive:true});
const body=games.map(g=>JSON.stringify(g)).join('\n')+'\n',filename=path.join(out,`group-${String(group).padStart(5,'0')}.jsonl.gz`);
fs.writeFileSync(filename+'.tmp',zlib.gzipSync(body,{level:3}));fs.renameSync(filename+'.tmp',filename);
console.log(JSON.stringify({role,group,games:games.length,decisions:games.reduce((n,g)=>n+g.decisions.length,0),bytes:body.length}));
