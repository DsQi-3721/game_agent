// Controlled data providers; private parameters are used only for labels.
const {dex,softmax}=require('./opp_joint_common');
function makePolicy(learned,role='calibration',group=0){
const families=['rule','search','learned','random','adaptive'],testGroup=group%7;
function mlp(x){const layer=(w,b,input,activate)=>w.map((row,i)=>{const v=row.reduce((s,a,j)=>s+a*input[j],b[i]);return activate?Math.tanh(v):v;});return layer(learned.w2,learned.b2,layer(learned.w1,learned.b1,x,true),false)[0];}
return function policy(account,request,acts,x,view,side,round){
 let family=account.family,change='keep',mix=0;
 if((role==='test'&&testGroup===4)||(role==='train'&&group%6===0)){
  if(view.turn>=account.changeTurn){family=families[(families.indexOf(family)+2)%5];change=view.turn===account.changeTurn?'abrupt':'keep';}
 }
 if((role==='test'&&testGroup===5)||(role==='train'&&group%6===1)){mix=Math.max(0,Math.min(1,(view.turn-account.changeTurn)/account.changeDuration));change=mix>0&&mix<1?'gradual':'keep';}
 const unknown=role==='test'&&testGroup===3;
 const scores=x.map((f,i)=>{
  if(role==='test'&&testGroup===6&&acts[i].kind==='team')return acts[i].order[0]===1?8:0;
  if(family==='random')return 0;
  if(acts[i].kind==='team')return f[4]*(family==='rule'?1:2)+f[20]*account.preference[0];
  let score;
  const attack=3*f[21]+2*f[22]+.2*f[7],support=1.2*f[24]+.6*f[25]+.35*f[15]-.8*f[11]*f[9];
  const switching=f[2]*(1.3*f[19]+.7*f[23]+.35*f[18]-.5);
  if(family==='heldout'){
   // Unseen resource-preserving controller: PP, remaining team and observed
   // reply frequencies govern the attack/risk tradeoff, absent from training.
   const move=request.active?.[0]?.moves.find(m=>m.id===acts[i].move),reserve=move?.pp!==undefined?move.pp/Math.max(1,move.maxpp):1;
   const alive=request.side.pokemon.filter(m=>!m.condition.includes('fnt')).length;
   const observed=(account.counts.attack+1)/(account.counts.attack+account.counts.status+account.counts.switch+3);
   score=2*f[21]*reserve+f[22]*3-2*(1-f[6])*f[0]+f[14]*observed+f[2]*(alive>2?1:-1)*f[19]+f[24]*2-f[3]*(alive>2?2:-1);
  }
  else if(family==='learned')score=mlp(f);
  else if(family==='search'){
   // A two-step abstract minimax provider. It searches own action / possible
   // reply branches using only own request and public enemy types/moves.
   const incomingThreat=f[2]?-f[19]:f[23];
   const branches=[.25,.55,.9].map(enemyDamage=>{
    const remaining=f[9]-enemyDamage*(1+incomingThreat)*(f[22]?.25:1);
    const immediate=f[21]+f[24]*.6+(f[2]?f[19]*.35:0);
    const followup=Math.max(f[12]&&remaining>0?f[21]*1.5+.35:0,remaining>0?f[21]:-.7,f[13]?.15:0);
    return immediate+.7*followup-Math.max(0,-remaining)-enemyDamage*.25;
   });score=unknown?branches.reduce((a,b)=>a+b,0)/3:Math.min(...branches);score+=support*.5+switching;
  }else if(family==='adaptive'){
   const switchRate=account.counts.switch/(account.counts.attack+account.counts.status+account.counts.switch);
   score=attack*(1-switchRate)+support*(.5+switchRate)+switching+f[13]*switchRate;
   if(unknown)score+=2*f[7]*(f[9]<.35?1:0)-2*(1-f[6])*f[0];
  }else score=attack+support+switching;
  score+=account.preference[0]*f[0]+account.preference[1]*f[2]+account.preference[2]*f[12]+account.preference[3]*f[11];
  score+=f[3]*(.6*f[22]+.35*f[26]-.6)-.15*f[27];
  if(mix)score=(1-mix)*score+mix*(mlp(f)+1.2*f[2]-f[0]);
  if(role==='test'&&testGroup===6)score+=f[0]*3;
  return score;
 });return {probabilities:softmax(scores,account.temperature),family,change,mix};
}
}
module.exports={makePolicy};
