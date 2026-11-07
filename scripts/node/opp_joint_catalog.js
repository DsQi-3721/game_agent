const fs=require('fs'),{dex,TeamValidator,id,clone}=require('./opp_joint_common');
const dir=process.argv[2],source=JSON.parse(fs.readFileSync(dir+'/catalog-source.json'));
const validator=new TeamValidator('gen9ou'),sets=[],bySpecies={};
for(let set of source.sets){set=clone(set);if(validator.validateSet(set))continue;set.moves=set.moves.map(id);set.item=id(set.item);set.ability=id(set.ability);set.nature=id(set.nature);set.teraType=id(set.teraType);const key=JSON.stringify(set);if(sets.some(s=>JSON.stringify(s)===key))continue;const i=sets.length;sets.push(set);(bySpecies[id(set.species)]??=[]).push(i);}
const compositions=source.compositions.filter(team=>team.length===6&&new Set(team).size===6&&team.every(s=>bySpecies[s])).slice(0,6000);
const tokens=['<pad>','<unknown>','<config>','<event>','<decision>','<game>'];
for(const s of dex.species.all())tokens.push('species:'+s.id);
for(const m of dex.moves.all())tokens.push('move:'+m.id);
for(const i of dex.items.all())tokens.push('item:'+i.id);
for(const a of dex.abilities.all())tokens.push('ability:'+a.id);
for(const n of dex.natures.all())tokens.push('nature:'+n.id);
for(const t of dex.types.all())tokens.push('type:'+id(t.name));
for(let v=0;v<=252;v++)tokens.push('ev:'+v);
for(let v=0;v<=31;v++)tokens.push('iv:'+v);
tokens.push('item:none');
const keys=['move','switch','drag','faint','-damage','-heal','-status','-curestatus','-boost','-unboost','-setboost','-clearboost','-clearallboost','-item','-enditem','-ability','-activate','-start','-end','-weather','-fieldstart','-fieldend','-sidestart','-sideend','-terastallize','-crit','-miss','cant','turn','poke','-immune','-fail','detailschange','-formechange','-transform','-singleturn','-singlemove','-prepare'];
for(const e of keys)tokens.push('event:'+e);
const catalog={format:'gen9ou',sets,bySpecies,compositions,tokens:[...new Set(tokens)]};
fs.writeFileSync(dir+'/catalog.json',JSON.stringify(catalog));console.log(JSON.stringify({sets:sets.length,species:Object.keys(bySpecies).length,compositions:compositions.length,tokens:catalog.tokens.length}));
