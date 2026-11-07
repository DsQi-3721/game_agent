// Resolve a Showdown format's full rule table (nested rulesets expanded) at the running commit.
//   node scripts/node/showdown_rules.js <showdown_dir> gen9ou gen9randombattle > out.json
const path = require('path');
const [dir, ...formats] = process.argv.slice(2);
const { Dex } = require(path.join(path.resolve(dir), 'dist/sim'));
const out = {};
for (const id of formats) {
  const f = Dex.formats.get(id, true);
  const rt = Dex.formats.getRuleTable(f);
  const rules = [], bans = [], unbans = [];
  for (const [k, v] of rt) {
    if (k.startsWith('-')) bans.push(k.slice(1));
    else if (k.startsWith('+')) unbans.push(k.slice(1));
    else rules.push(k);
  }
  out[id] = {
    name: f.name, mod: f.mod, gameType: f.gameType || 'singles', team: f.team || null,
    ruleset: f.ruleset, banlist: f.banlist, rules: rules.sort(), bans: bans.sort(), unbans: unbans.sort(),
    teamPreview: rt.has('teampreview'), hpPercentage: rt.has('hppercentagemod'),
    pickedTeamSize: rt.pickedTeamSize || null, maxTeamSize: rt.maxTeamSize, minTeamSize: rt.minTeamSize,
    adjustLevel: rt.adjustLevel || null, defaultLevel: rt.defaultLevel,
  };
}
console.log(JSON.stringify(out, null, 1));
