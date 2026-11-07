const path=require('path');
global.__version={head:require('child_process').execFileSync('git',['-C',path.resolve(__dirname,'../../../pokemon-showdown'),'rev-parse','HEAD'],{encoding:'utf8'}).trim()};
const {Battle,Dex,Teams,TeamValidator}=require(path.resolve(__dirname,'../../../pokemon-showdown/dist/sim'));
const {extractChannelMessages}=require(path.resolve(__dirname,'../../../pokemon-showdown/dist/sim/battle'));
const dex=Dex.mod('gen9');
const id=s=>String(s||'').toLowerCase().replace(/[^a-z0-9]/g,'');
const clone=x=>JSON.parse(JSON.stringify(x));
const stats=['hp','atk','def','spa','spd','spe'];
const hp=s=>{if(!s||String(s).includes('fnt'))return 0;const [a,b]=String(s).split(' ')[0].split('/').map(Number);return b?a/b:1;};
function rng(seed){let a=seed>>>0;const random=()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};random.state=()=>a;return random;}
function sample(p,random){let u=random();for(let i=0;i<p.length;i++){u-=p[i];if(u<=0)return i;}return p.length-1;}
function softmax(x,temp=1){const max=Math.max(...x);const p=x.map(v=>Math.exp((v-max)/temp)),sum=p.reduce((a,b)=>a+b,0);return p.map(v=>v/sum);}
function permute(xs){return xs.length?xs.flatMap((x,i)=>permute(xs.filter((_,j)=>i!==j)).map(p=>[x,...p])):[[]];}
function actions(request,slots){
 if(request.wait)return [];
 if(request.teamPreview)return permute(slots).map(order=>({key:'team:'+order.join(''),kind:'team',order,command:'team '+order.join('')}));
 const out=[],mons=request.side.pokemon;
 if(!request.forceSwitch?.[0])for(const [i,m] of (request.active?.[0]?.moves||[]).entries())if(!m.disabled&&m.pp!==0){
  for(const tera of (request.active[0].canTerastallize?[false,true]:[false]))out.push({key:`move:${m.id}:tera=${tera}`,kind:'move',move:m.id,tera,command:`move ${i+1}${tera?' terastallize':''}`});
 }
 if(request.forceSwitch?.[0]||!request.active?.[0]?.trapped)for(let i=0;i<mons.length;i++)if(!mons[i].active&&(request.side.pokemon.some(m=>m.reviving)?!!mons[i].reviving:!mons[i].condition.endsWith(' fnt')))out.push({key:'switch:'+slots[i],kind:'switch',slot:slots[i],species:mons[i].details.split(',')[0],command:'switch '+(i+1)});
 return out;
}
function effectiveness(type,types){return dex.getImmunity(type,types)?2**dex.getEffectiveness(type,types):0;}
class PublicView{
 constructor(){this.sides={p1:{mons:{},active:null,tera:false},p2:{mons:{},active:null,tera:false}};this.turn=0;this.weather='';this.fields={};this.events=[];}
 update(lines){for(const line of lines){const p=line.split('|'),kind=p[1],side=p[2]?.slice(0,2),s=this.sides[side];
  if(kind==='turn')this.turn=Number(p[2]);if(kind==='-weather')this.weather=p[2];
  if(!s)continue;const name=p[2]?.split(': ')[1];
  if(['switch','drag','replace'].includes(kind)){const species=p[3].split(',')[0];s.active=name;s.mons[name]??={species,moves:[],boosts:{},hp:1,status:'',item:null,ability:null,tera:null};Object.assign(s.mons[name],{species,hp:hp(p[4]),status:p[4]?.split(' ')[1]||'',boosts:{}});}
  const m=s.mons[name];if(!m)continue;
  if(kind==='move'&&!p.slice(4).some(x=>x.startsWith('[from]'))){m.lastMove=id(p[3]);if(!m.moves.includes(id(p[3])))m.moves.push(id(p[3]));}
  if(['-damage','-heal'].includes(kind)){m.hp=hp(p[3]);m.status=p[3].split(' ')[1]||'';}
  if(kind==='faint')m.hp=0;
  if(kind==='-status')m.status=p[3];if(kind==='-curestatus')m.status='';
  if(kind==='-item')m.item=id(p[3]);if(kind==='-enditem')m.item='';if(kind==='-ability')m.ability=id(p[3]);
  if(kind==='-terastallize'){s.tera=true;m.tera=p[3];}
  if(kind==='-boost'||kind==='-unboost')m.boosts[p[3]]=(m.boosts[p[3]]||0)+(kind==='-boost'?1:-1)*Number(p[4]);
 }this.events.push(...lines);}
 active(side){const s=this.sides[side];return s.mons[s.active]||null;}
}
const featureNames=['attack','status','switch','tera','power','effectiveness','accuracy','priority','stab','hp','foe_hp','heal','setup','pivot','protect','hazard','hazard_remove','status_inflict','switch_hp','switch_matchup','switch_speed','damage','ko','threat','low_hp_heal','setup_hp','tera_stab','repeat','force','preview','physical','special'];
function features(request,acts,view,side,slots){
 const me=request.side.pokemon.find(m=>m.active)||request.side.pokemon[0],opp=view.active(side==='p1'?'p2':'p1');
 const ownSpecies=dex.species.get(me.details.split(',')[0]),enemy=opp?dex.species.get(opp.species):ownSpecies;
 const ownTypes=me.terastallized?[me.terastallized]:ownSpecies.types,enemyTypes=opp?.tera?[opp.tera]:enemy.types;
 const myHP=hp(me.condition),oppHP=opp?.hp??1,threat=Math.max(...enemyTypes.map(t=>effectiveness(t,ownTypes)),1);
 const ownPublic=view.active(side),repeat=ownPublic?.lastMove;
 return acts.map(a=>{
  const x=Array(32).fill(0);x[9]=myHP;x[10]=oppHP;x[23]=Math.log2(Math.max(.125,threat))/3;
  if(a.kind==='team'){const species=dex.species.get(request.side.pokemon[slots.indexOf(a.order[0])].details.split(',')[0]);x[29]=1;x[4]=Math.max(species.baseStats.atk,species.baseStats.spa)/150;x[20]=species.baseStats.spe/150;return x;}
  if(a.kind==='switch'){const incoming=request.side.pokemon[slots.indexOf(a.slot)],s=dex.species.get(a.species);x[2]=1;x[18]=hp(incoming.condition);x[19]=-Math.log2(Math.max(.125,...enemyTypes.map(t=>effectiveness(t,s.types))))/3;x[20]=s.baseStats.spe/150;x[28]=Number(!!request.forceSwitch);return x;}
  const m=dex.moves.get(a.move),atk=m.category!=='Status',eff=effectiveness(m.type,enemyTypes),stab=ownTypes.includes(m.type)?1.5:1;
  const setup=m.category==='Status'&&m.boosts&&Object.values(m.boosts).some(v=>v>0),heal=!!m.heal||['rest','wish','strengthsap'].includes(m.id);
  const power=(m.basePower|| (atk?60:0)),offense=me.stats?.[m.category==='Physical'?'atk':'spa']||250;
  const defense=2*enemy.baseStats[m.category==='Physical'?'def':'spd']+36;
  const damage=atk?((42*power*offense/Math.max(1,defense)/50+2)*stab*eff*.925)/(2*enemy.baseStats.hp+141):0;
  x[0]=Number(atk);x[1]=Number(!atk);x[3]=Number(!!a.tera);x[4]=Math.min(power,180)/150;x[5]=Math.log(Math.max(.1,eff));
  x[6]=m.accuracy===true?1:m.accuracy/100;x[7]=m.priority/3;x[8]=Number(stab>1);x[11]=Number(heal);x[12]=Number(!!setup);x[13]=Number(!!m.selfSwitch);x[14]=Number(['protect','detect'].includes(m.id));
  x[15]=Number(!!m.sideCondition);x[16]=Number(['rapidspin','defog','mortalspin','tidyup'].includes(m.id));x[17]=Number(!!m.status);x[21]=Math.min(2,damage);x[22]=Number(damage>=oppHP);x[24]=heal*(1-myHP);x[25]=setup?myHP:0;x[26]=a.tera&&id(me.teraType)===id(m.type)?1:0;x[27]=Number(m.id===repeat);x[30]=Number(m.category==='Physical');x[31]=Number(m.category==='Special');return x;
 });
}
function currentState(battle,side){return battle.getSide(side).pokemon.map(m=>({slot:m.jointSlot,species:m.species.name,active:m.isActive,hp:m.hp,maxhp:m.maxhp,status:m.status,item:m.item,ability:m.ability,tera:m.terastallized||null,boosts:{...m.boosts},locked_move:m.volatiles.choicelock?.move||m.volatiles.lockedmove?.move||null,moves:m.moveSlots.map(x=>({id:x.id,pp:x.pp,disabled:x.disabled})),volatiles:Object.keys(m.volatiles)}));}
const ignored=new Set(['','t:','timestamp','upkeep','debug','hint','-hint','message','-message','sentchoice','request','player','win','tie']);
function canonical(lines){return lines.filter(l=>l.startsWith('|')&&!ignored.has(l.split('|')[1])).map(line=>line.split('|').map((x,i)=>{if(i>1&&/^p[12][a-z]?: /.test(x))return x.split(': ')[0];return x;}).filter(x=>!['[still]','[miss]','[notarget]'].includes(x)).join('|'));}
module.exports={Battle,Dex,Teams,TeamValidator,dex,extractChannelMessages,id,clone,stats,hp,rng,sample,softmax,actions,PublicView,features,featureNames,currentState,canonical};
