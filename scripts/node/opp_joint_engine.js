// Showdown observation likelihoods. A tape records equivalent random outcomes;
// each outcome carries the summed probability of its underlying integer rolls.
const readline=require('readline');
const {Worker,isMainThread,parentPort}=require('worker_threads');
const {Battle,Dex,dex,TeamValidator,extractChannelMessages,clone,actions,features,PublicView,currentState,canonical,rng,sample,id}=require('./opp_joint_common');
const snapshots=new Map(),stateKeys=new Map(),stateIndex=new Map();let nextId=1;
const continuations=new Map();let nextContinuation=1;
function saveContinuation(state){const handle=nextContinuation++;continuations.set(handle,state);return handle;}
const battles=new Map();
function put(state){
 // Protocol history belongs to the Python journal. At a request boundary the
 // engine only needs the most recent move line (retargeting annotations).
 const cut=state.lastMoveLine>=0?state.lastMoveLine:state.log.length;
 state.log=state.log.slice(cut);state.sentLogPos=Math.max(0,state.sentLogPos-cut);if(state.lastMoveLine>=0)state.lastMoveLine-=cut;state.inputLog=[];
 const key=JSON.stringify(state);if(stateIndex.has(key))return stateIndex.get(key);const n=nextId++;snapshots.set(n,state);stateKeys.set(n,key);stateIndex.set(key,n);return n;
}
function mergeBranches(branches){const map=new Map();for(const b of branches){const key=JSON.stringify(b.tape);if(map.has(key))map.get(key).weight+=b.weight;else map.set(key,{...b});}return [...map.values()];}
function create(args){
 let output=[];const b=new Battle({formatid:args.format||'gen9ou',seed:[17,23,31,41],send:(type,data)=>{if(type==='update')output.push(...extractChannelMessages(Array.isArray(data)?data.join('\n'):data,[0])[0]);}});
 for(const side of ['p1','p2']){b.setPlayer(side,{name:side,team:clone(args.teams[side])});b.getSide(side).pokemon.forEach((p,i)=>p.jointSlot=i+1);}
 b.sendUpdates();return {handle:put(b.toJSON()),events:output};
}
// The cached serialization still parses into an independent battle per branch.
function restore(handle){const b=Battle.fromJSON(stateKeys.get(handle));b.restart(()=>{});return b;}
function readBattle(handle){if(!battles.has(handle))battles.set(handle,restore(handle));return battles.get(handle);}
const sharedTypes=new Set([Dex.Condition,Dex.Ability,Dex.Item,Dex.Move,Dex.Species]);
function battleFactory(battle){
 const nodes=[],links=[],indices=new Map(),shared=new Set([battle.dex,battle.ruleTable,battle.format]);
 function visit(x){
  if(x===null||typeof x!=='object'||shared.has(x)||(sharedTypes.has(x.constructor)&&!Object.hasOwn(x,'hit')))return;
  if(indices.has(x))return indices.get(x);
  const i=nodes.length;indices.set(x,i);nodes.push(x);
  // Battle's Sets contain hints and integer switch slots. RuleTable is shared.
  if(x instanceof Set)return i;
  for(const key of Object.keys(x)){const child=visit(x[key]);if(child!==undefined)links.push([i,key,child]);}
  return i;
 }
 visit(battle);
 return ()=>{
 const copies=nodes.map(x=>Array.isArray(x)?x.slice():x instanceof Set?new Set(x):Object.assign(Object.create(Object.getPrototypeOf(x)),x));
 for(const [i,key,child] of links)copies[i][key]=copies[child];
 const result=copies[0];
 // These two Showdown callbacks capture their Pokemon rather than using this.
 for(const side of result.sides)for(const p of side.pokemon){
  p.getHealth=()=>{
   if(!p.hp)return {side:p.side.id,secret:'0 fnt',shared:'0 fnt'};
   const status=p.status?' '+p.status:'',secret=`${p.hp}/${p.maxhp}${status}`;
   let percent=Math.ceil(100*p.hp/p.maxhp);if(percent===100&&p.hp<p.maxhp)percent=99;
   return {side:p.side.id,secret,shared:p.battle.reportExactHP?secret:`${percent}/100${status}`};
  };
  p.getFullDetails=()=>{
   const health=p.getHealth();let details=p.details;
   if(p.illusion)details=p.illusion.getUpdatedDetails(p.battle.ruleTable.has('illusionlevelmod')?p.illusion.level:p.level);
   if(p.terastallized)details+=`, tera:${p.terastallized}`;
   return {side:health.side,secret:`${details}|${health.secret}`,shared:`${details}|${health.shared}`};
  };
 }
 return result;
 };
}
// Gen9 secondary rolls only test an integer roll against a chance threshold.
// Summing the equivalent rolls preserves both the event and state distribution.
function secondaries(targets,source,move,moveData,isSelf){
 if(!moveData.secondaries)return;
 for(const target of targets){
  if(target===false)continue;
  const effects=this.battle.runEvent('ModifySecondaries',target,source,moveData,moveData.secondaries.slice());
  for(const effect of effects){
   if(effect.chance===undefined||this.battle.randomChance(Math.ceil(effect.chance),100))this.moveHit(target,source,move,effect,true,isSelf);
  }
 }
}
function selfDrops(targets,source,move,moveData,isSecondary){
 for(const target of targets){
  if(target===false||!moveData.self||move.selfDropped)continue;
  if(!isSecondary&&moveData.self.boosts){
   if(moveData.self.chance===undefined||this.battle.randomChance(Math.ceil(moveData.self.chance),100))this.moveHit(source,source,move,moveData.self,isSecondary,true);
   if(!move.multihit)move.selfDropped=true;
  }else this.moveHit(source,source,move,moveData.self,isSecondary,true);
 }
}
function view(handle,side,events=[],publicView=null){const b=readBattle(handle),s=b.getSide(side),request=s.activeRequest||{wait:true},slots=s.pokemon.map(p=>p.jointSlot),legal=request.wait?[]:actions(request,slots);if(!publicView){publicView=new PublicView();publicView.update(events);}return {request,slots,actions:legal,features:legal.length?features(request,legal,publicView,side,slots):[],state:currentState(b,side),turn:b.turn,ended:b.ended};}
function views(args){if(args.handles.length>=16)return transitionGroups({...args,jobs:args.handles.map(handle=>({handle,view_only:true}))});const publicView=new PublicView();publicView.update(args.events);return args.handles.map(h=>view(h,args.side,[],publicView));}
class Draw{constructor(options){this.options=options;}}
class Contradiction{}
function transitionFactory(handle){
 const initial=readBattle(handle);initial.actions.secondaries=secondaries;initial.actions.selfDrops=selfDrops;
 return battleFactory(initial);
}
function transition(args,fork=transitionFactory(args.handle)){
 const observer=args.observer||'p1',channel=Number(observer[1]);
 const expected=args.events===undefined?null:canonical(args.events);
 const saved=args.continuation||{};
 const capacity=args.branches===0?Infinity:args.branches||64,random=rng(saved.random_state??(args.seed||83)),completed=saved.completed||[];
 let frontier=saved.frontier||[{tape:[],weight:1}],attempts=saved.attempts||0,unresolved=0,resamplingSteps=saved.resampling_steps||0,firstMismatch=saved.first_mismatch||null;
 let depth=saved.depth||0,cursor=saved.cursor||0,pending=saved.pending||[];
 const prefixMatches=(actual,partial=false)=>{
  if(!expected)return true;
  // Showdown edits move lines after random draws (Fickle Beam's animation,
  // retargeting, etc.). Check those lines when resolution is complete.
  const i=actual.findIndex((line,i)=>{
   if(partial&&line.startsWith('|move|')&&expected[i]?.startsWith('|move|'))return line.split('|').slice(0,4).join('|')!==expected[i].split('|').slice(0,4).join('|');
   return i>=expected.length||line!==expected[i];
  });
  if(i<0)return true;
  firstMismatch??={index:i,expected:expected[i],actual:actual[i]};return false;
 };
 function run(branch){
  const b=fork();let cursor=0,logProbability=0;
  const start=b.log.length;
  const observed=()=>canonical(extractChannelMessages(b.log.slice(start).join('\n'),[args.private?channel:0])[args.private?channel:0]);
  if(expected){
   const add=b.add,addSplit=b.addSplit;let splitting=false;
   const check=()=>{if(!prefixMatches(observed(),true))throw new Contradiction();};
   b.add=function(...parts){add.apply(this,parts);if(!splitting)check();};
   b.addSplit=function(...parts){splitting=true;addSplit.apply(this,parts);splitting=false;check();};
   const modifyDamage=b.actions.modifyDamage;
   b.actions.modifyDamage=function(base,source,target,move,suppressMessages=false){
    if(!suppressMessages){
     const remaining=expected.slice(observed().length),actor=target.toString().split(': ')[0];
     const stop=remaining.findIndex(line=>/^\|(move|switch|drag|turn|-hitcount)\|/.test(line));
     const damage=remaining.findIndex(line=>line.startsWith(`|-damage|${actor}|`));
     if(damage>=0&&(stop<0||damage<stop)){
      const crit=remaining.slice(0,damage).includes(`|-crit|${actor}`);
      if(!!target.getMoveHitData(move).crit!==crit)throw new Contradiction();
     }
    }
    return modifyDamage.call(this,base,source,target,move,suppressMessages);
   };
  }
  const choose=(options,continuous=false)=>{
   if(!prefixMatches(observed(),true))throw new Contradiction();
   if(cursor>=branch.tape.length)throw new Draw(options);
   const value=branch.tape[cursor++];
   if(continuous){if(typeof value!=='number'||value<0||value>=1)throw new Contradiction();}
   else {const selected=options.find(o=>o.value===value);if(!selected)throw new Contradiction();logProbability+=Math.log(selected.p);}
   return value;
  };
  b.prng.randomChance=(n,d)=>{const p=Math.min(1,Math.max(0,Math.ceil(n)/Math.floor(d)));if(p===0)return false;if(p===1)return true;return choose([{value:false,p:1-p},{value:true,p}]);};
  b.prng.random=(a,z)=>{
   if(a===undefined){return choose(Array.from({length:256},()=>({value:random(),p:1/256})),true);}
   const lo=z===undefined?0:Math.floor(a),hi=z===undefined?Math.floor(a):Math.floor(z);
   if(hi-lo===1)return lo;
   return choose(Array.from({length:hi-lo},(_,i)=>({value:lo+i,p:1/(hi-lo)})));
  };
  b.prng.sample=items=>{
   const groups=new Map();
   for(let i=0;i<items.length;i++){
    if(!groups.has(items[i]))groups.set(items[i],{value:i,p:0});
    groups.get(items[i]).p+=1/items.length;
   }
   const options=[...groups.values()];
   return items[options.length===1?options[0].value:choose(options)];
  };
  b.randomizer=baseDamage=>{
   const weights=new Map();
   for(let roll=0;roll<16;roll++){
    const damage=b.trunc(b.trunc(baseDamage*(100-roll))/100);
    weights.set(damage,(weights.get(damage)||0)+1/16);
   }
   const options=[...weights].map(([value,p])=>({value,p}));
   return options.length===1?options[0].value:choose(options);
  };
  try{
   for(const [side,command] of args.commands){if(b.getSide(side).activeRequest?.wait)continue;if(!b.choose(side,command))return {invalid:true};}
   b.sendUpdates();const events=observed();
   if(!prefixMatches(events)||(expected&&events.length!==expected.length)||cursor!==branch.tape.length)return {invalid:true};
   return {state:b.toJSON(),events,weight:branch.weight,tape:branch.tape,logProbability};
  }catch(e){if(e instanceof Draw)return {options:e.options};if(e instanceof Contradiction)return {invalid:true};throw e;}
 }
 // A fixed random path has a joint density, not a marginal event likelihood.
 // Retained histories use this density when replaying their random paths.
 if(args.tape!==undefined){
  const result=run({tape:args.tape,weight:1});
  return result.state?{matches:true,log_probability:result.logProbability,handle:put(result.state),events:result.events}:{matches:false,log_probability:null};
 }
 for(;frontier.length&&depth<256;depth++){
  for(;cursor<frontier.length;cursor++){
   if(Date.now()>=args.deadline)return {timed_out:true,likelihood:null,likelihood_is_exact:false,attempts,outcomes:[],
    continuation:{frontier,cursor,pending,completed,depth,attempts,resampling_steps:resamplingSteps,first_mismatch:firstMismatch,random_state:random.state()}};
   const branch=frontier[cursor];
   attempts++;const result=run(branch);
   if(result.state)completed.push(result);
   else if(result.options)pending.push({...branch,options:result.options});
  }
  // Condition on the observations emitted by a draw before resampling it.
  // Resampling prior draws first repeatedly loses rare but observed critical
  // hits and multi-hit damage rolls, even when the configuration is correct.
  if(expected&&pending.length>capacity){resamplingSteps++;const total=pending.reduce((s,b)=>s+b.weight,0),probs=pending.map(b=>b.weight/total);pending=Array.from({length:capacity},()=>({...pending[sample(probs,random)],weight:total/capacity}));}
  const expanded=pending.flatMap(b=>b.options.filter(o=>o.p>0).map(o=>({tape:[...b.tape,o.value],weight:b.weight*o.p})));
  const unique=mergeBranches(expanded);
  if(!expected&&unique.length>capacity){resamplingSteps++;const total=unique.reduce((s,b)=>s+b.weight,0),probs=unique.map(b=>b.weight/total);frontier=mergeBranches(Array.from({length:capacity},()=>{const b=unique[sample(probs,random)];return {tape:b.tape,weight:total/capacity};}));}
  else frontier=unique;
  cursor=0;pending=[];
 }
 unresolved=frontier.reduce((s,b)=>s+b.weight,0);
 const likelihood=completed.reduce((s,b)=>s+b.weight,0);
 let selected=completed;
 if(completed.length>(args.keep||4)){
  selected=Array.from({length:args.keep||4},()=>completed[sample(completed.map(c=>c.weight/likelihood),random)]);
  selected=selected.map(s=>({...s,weight:likelihood/selected.length}));
 }
 return {likelihood,likelihood_is_exact:resamplingSteps===0&&unresolved===0,unresolved_mass:unresolved,resampling_steps:resamplingSteps,attempts,method:'engine_random_branch_smc',...(args.diagnostic?{first_mismatch:firstMismatch}:{}),
  outcomes:selected.map(s=>({handle:put(s.state),weight:s.weight,probability:s.weight/likelihood,events:s.events,random_tape:s.tape,random_log_probability:s.logProbability}))};
}
function transitions(args){
 const fork=transitionFactory(args.handle);
 return args.commands.map((commands,i)=>transition({...args,commands,continuation:args.continuations?.[i]},fork));
}
function historyPath(args){
 const saved=args.continuation;
 let handle=saved?put(saved.state):args.handle,history=saved?.history||[...args.history],logMass=saved?.log_mass||0,attempts=saved?.attempts||0,exact=saved?.exact??true;
 const rows=saved?.rows||[],steps=saved?.steps||[],random=rng(saved?.random_state??args.seed);
 let results=saved?.results?.map(r=>r&&({...r,outcomes:r.outcomes.map(({state,...o})=>({...o,handle:put(state)}))})),windowSeed=saved?.window_seed;
 const pause=()=>({complete:false,timed_out:true,attempts:attempts+(results||[]).reduce((s,r)=>s+(r?.attempts||0),0),
  continuation:{state:snapshots.get(handle),history,rows,steps,log_mass:logMass,attempts,exact,random_state:random.state(),window_seed:windowSeed,
   results:results?.map(r=>r&&({...r,outcomes:r.outcomes.map(({handle,...o})=>({...o,state:snapshots.get(handle)}))}))}});
 for(let t=steps.length;t<args.records.length;t++){
  const record=args.records[t];
  if(Date.now()>=args.deadline)return pause();
  const before=view(handle,args.side,history),legal=before.actions.length?before.actions:[{key:'waiting',kind:'waiting'}];
  const commands=legal.map(action=>{
   const pair=record.our_command==null?[]:[[args.observer,record.our_command]];
   if(action.command)pair.push([args.side,action.command]);return pair;
  });
  results??=Array(commands.length).fill(null);windowSeed??=Math.floor(random()*0x7fffffff);
  const missing=commands.map((_,i)=>i).filter(i=>!results[i]||results[i].timed_out);
  const updates=transitions({...args,handle,commands:missing.map(i=>commands[i]),continuations:missing.map(i=>results[i]?.continuation),events:record.private_events??record.events,
   private:record.private_events!=null,keep:1,seed:windowSeed});
  missing.forEach((i,j)=>results[i]=updates[j]);const result=results;
  if(result.some(r=>r.timed_out))return pause();
  attempts+=result.reduce((sum,r)=>sum+r.attempts,0);
  exact&&=result.every(r=>r.likelihood_is_exact);
  const mass=result.map(r=>r.likelihood),sum=mass.reduce((a,b)=>a+b,0);
  if(!sum)return {complete:false,failed_window:steps.length,likelihood_is_exact:exact,attempts};
  const j=sample(mass.map(p=>p/sum),random),action=legal[j],outcome=result[j].outcomes[0];
  logMass+=Math.log(sum);
  if(before.request.teamPreview){
   const indices=Array.from({length:6},(_,i)=>before.actions.findIndex(a=>a.order[0]===i+1));
   before.actions=indices.map(i=>before.actions[i]);before.features=indices.map(i=>before.features[i]);
  }
  rows.push(before);steps.push({action:action.key,command:action.command||null,random_tape:outcome.random_tape,random_log_probability:outcome.random_log_probability});
  const state=snapshots.get(outcome.handle);
  snapshots.clear();stateKeys.clear();stateIndex.clear();battles.clear();handle=put(state);
  history.push(...record.events);
  results=null;windowSeed=null;
 }
 return {complete:true,handle,rows,steps,log_mechanics_mass:logMass,likelihood_is_exact:exact,attempts};
}
let workers;
async function transitionGroups(args){
 workers??=[];
 while(workers.length<Math.min(16,args.jobs.length))workers.push(new Worker(__filename));
 let cursor=0;const results=Array(args.jobs.length),deadline=args.budget_ms==null?Infinity:Date.now()+args.budget_ms;
 await Promise.all(workers.map(async worker=>{
  while(cursor<args.jobs.length){
   if(Date.now()>=deadline)break;
   const i=cursor++,job=args.jobs[i];
   const answer=await new Promise((resolve,reject)=>{
    worker.once('message',resolve);worker.once('error',reject);
    worker.postMessage({...args,...job,jobs:undefined,deadline,state:snapshots.get(job.handle),
     continuation:job.continuation==null?undefined:continuations.get(job.continuation),
     continuations:job.continuations?.map(h=>h==null?null:continuations.get(h))});
   });
   worker.removeAllListeners('error');
   if(answer.error)throw Error(answer.error);
   for(const h of job.continuations||[job.continuation])continuations.delete(h);
   results[i]=answer;
  }
 }));
 // Register states in input order so worker completion order cannot change
 // handle ordering and the random streams used by the next filtering step.
 return Array.from(results,answer=>{
  if(!answer)return null;
  if(answer.view)return answer.view;
  const ids=new Map(answer.states.map(([h,state])=>[h,put(state)]));
  if(answer.trajectory){
   if(answer.trajectory.complete)answer.trajectory.handle=ids.get(answer.trajectory.handle);
   if(answer.trajectory.timed_out)answer.trajectory.continuation=saveContinuation(answer.trajectory.continuation);
   return answer.trajectory;
  }
  for(const r of answer.results){
   for(const o of r.outcomes)o.handle=ids.get(o.handle);
   if(r.timed_out)r.continuation=saveContinuation(r.continuation);
  }
  return answer.results;
 });
}
function replay(args){
 let output=[];const b=new Battle({formatid:args.format||'gen9ou',seed:args.seed,send:(type,data)=>{if(type==='update')output.push(...extractChannelMessages(Array.isArray(data)?data.join('\n'):data,[0])[0]);}});
 for(const side of ['p1','p2']){b.setPlayer(side,{name:side,team:clone(args.teams[side])});b.getSide(side).pokemon.forEach((p,i)=>p.jointSlot=i+1);}b.sendUpdates();
 for(const [side,command] of args.commands){if(!b.choose(side,command))throw Error('Invalid replay command');b.sendUpdates();}
 return {handle:put(b.toJSON()),events:output};
}
function legal(args){const validator=new TeamValidator(args.format||'gen9ou');return args.teams?args.teams.map(team=>({errors:validator.validateTeam(clone(team))})):args.sets.map(set=>({set,errors:validator.validateSet(clone(set))}));}
function options(args){
 const species=dex.species.get(args.species),moves=new Set(),seen=new Set(),pending=[species];
 while(pending.length){const s=pending.pop();if(!s?.exists||seen.has(s.id))continue;seen.add(s.id);const ls=dex.species.getLearnsetData(s.id).learnset||{};for(const [m,sources] of Object.entries(ls))if(sources.some(x=>x.startsWith('9')))moves.add(m);if(s.prevo)pending.push(dex.species.get(s.prevo));if(s.forme)pending.push(dex.species.get(s.changesFrom||s.baseSpecies));}
 return {species:species.name,baseStats:species.baseStats,natureEffects:Object.fromEntries(dex.natures.all().map(n=>[n.id,[n.plus||null,n.minus||null]])),gender:species.gender,genderRatio:species.genderRatio,moves:[...moves].filter(m=>!dex.moves.get(m).isNonstandard),abilities:Object.values(species.abilities).map(id),items:dex.items.all().filter(x=>!x.isNonstandard).map(x=>x.id),natures:dex.natures.all().map(x=>x.id),types:dex.types.all().map(x=>id(x.name))};
}
function publicOptions(args){
 if(args.phase==='preview')return args.preview.map((s,i)=>'lead:'+(i+1));
 const view=new PublicView();view.update(args.events);const active=view.active(args.side),slot=args.preview.findIndex(s=>id(s)===id(active?.species));
 const moves=new Set(active?options({species:active.species}).moves:[]);moves.add('struggle');for(const m of active?.moves||[])moves.add(m);
 const out=[];if(args.phase!=='forced')for(const move of moves)for(const tera of view.sides[args.side].tera?[false]:[false,true])out.push(`move:${move}:tera=${tera}`);
 for(let i=0;i<args.preview.length;i++)if(i!==slot)out.push('switch:'+(i+1));return out;
}
const handlers={create,view:a=>view(a.handle,a.side,a.events),views,transition,transitions,transition_groups:transitionGroups,replay,legal,options,public_options:publicOptions,version:()=>global.__version.head,
 snapshot_continuations:a=>Object.fromEntries(a.handles.map(h=>[h,continuations.get(h)])),
 load_continuations:a=>Object.fromEntries(Object.entries(a.states).map(([h,state])=>[h,saveContinuation(state)])),
 release_continuations:a=>{for(const h of a.handles)continuations.delete(h);return {};},
 batch:a=>a.jobs.map(job=>handlers[job.method](job.args)),
 snapshot:a=>snapshots.get(a.handle),load:a=>({handle:put(a.state)}),
 release:a=>{for(const h of a.handles){stateIndex.delete(stateKeys.get(h));stateKeys.delete(h);snapshots.delete(h);battles.delete(h);}return {};},
 facts:a=>{const b=readBattle(a.handle);return {p1:currentState(b,'p1'),p2:currentState(b,'p2')};}};
if(!isMainThread){
 parentPort.on('message',args=>{
  try{
   args.handle=put(args.state);
   if(args.view_only){parentPort.postMessage({view:view(args.handle,args.side,args.events)});}
   else if(args.records){
    const trajectory=historyPath(args);
    parentPort.postMessage({trajectory,states:trajectory.complete?[[trajectory.handle,snapshots.get(trajectory.handle)]]:[]});
   }else{
    const results=transitions(args),handles=new Set(results.flatMap(r=>r.outcomes.map(o=>o.handle)));
    parentPort.postMessage({results,states:[...handles].map(h=>[h,snapshots.get(h)])});
   }
  }catch(e){parentPort.postMessage({error:String(e)});}
  snapshots.clear();stateKeys.clear();stateIndex.clear();battles.clear();
 });
}else if(require.main===module){
 const input=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
 input.on('line',async line=>{try{const q=JSON.parse(line);const result=await handlers[q.method](q.args||{});process.stdout.write(JSON.stringify({result})+'\n');}catch(e){process.stdout.write(JSON.stringify({error:String(e),stack:e.stack})+'\n');}});
 input.on('close',()=>{for(const worker of workers||[])worker.terminate();});
}
module.exports={...handlers,restore,battleFactory};
