// Scripted mechanics battles for the independent benchmark (I/O audit §25): situations that are rare or absent
// in the strict tier's natural battles (double screens, Trick Room with weather and terrain, Skill Swap,
// volatiles cleared by switching, Counter / Mirror Coat, recharge, Transform, Revival Blessing, first-turn-only
// moves, Endeavor).
//   node scripts/node/scenarios.js <showdown_dir> <out_dir>
//
// Each scenario is a "[Gen 9] Custom Game" battle (a mechanics test environment: no team validation, so the
// sets are chosen for the mechanic, not for OU legality) with fixed teams, a fixed seed and a queue of choices
// per side, consumed whenever that side has a request. The battle is driven through the simulator's Battle
// object exactly as BattleStream does, and written as a server-style log ({roomid, format, seed, inputLog,
// log, scenario, refused}) so the strict-tier checks (replay_battle.js, truth_states.js, probe_choices.js,
// calc_truth.js and the Python verifiers) run on it unchanged. A scripted choice the simulator refuses is
// recorded under ``refused`` and the side falls back to its default choice, so a log never hides a mistake.
const fs = require('fs');
const path = require('path');

const [dir, outDir] = process.argv.slice(2);
const root = path.resolve(dir);
const { Battle } = require(path.join(root, 'dist/sim/battle'));

const mon = (species, ability, item, moves, extra = {}) => ({
	name: species, species, ability, item, moves, nature: 'Hardy', evs: { hp: 84, atk: 84, def: 84, spa: 84, spd: 84, spe: 84 },
	ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 100, teraType: extra.teraType || 'Normal', ...extra,
});

const SCENARIOS = [
	{
		name: 'double-screens',
		what: 'Reflect and Light Screen up together, attacks into them, Defog removes them',
		p1: [mon('Grimmsnarl', 'Prankster', 'Light Clay', ['Reflect', 'Light Screen', 'Spirit Break', 'Thunder Wave']),
			mon('Corviknight', 'Pressure', 'Leftovers', ['Brave Bird', 'Roost', 'U-turn', 'Defog'])],
		p2: [mon('Garchomp', 'Rough Skin', 'Leftovers', ['Earthquake', 'Dragon Claw', 'Swords Dance', 'Stealth Rock']),
			mon('Corviknight', 'Pressure', 'Leftovers', ['Brave Bird', 'Roost', 'Iron Head', 'Defog'])],
		q1: ['team 12', 'move 1', 'move 2', 'move 3', 'move 3', 'move 3', 'switch 2', 'move 2', 'move 2'],
		q2: ['team 12', 'move 4', 'move 3', 'move 1', 'switch 2', 'move 4', 'move 1', 'move 3', 'move 3'],
	},
	{
		name: 'rain-terrain-trickroom',
		what: 'Grassy Terrain from Grassy Surge, Rain Dance, then Trick Room on top; a U-turn mid-turn switch',
		p1: [mon('Porygon2', 'Download', 'Eviolite', ['Trick Room', 'Rain Dance', 'Recover', 'Ice Beam']),
			mon('Kingdra', 'Swift Swim', 'Life Orb', ['Hydro Pump', 'Draco Meteor', 'Rain Dance', 'Iron Head'])],
		p2: [mon('Rillaboom', 'Grassy Surge', 'Choice Band', ['Grassy Glide', 'Wood Hammer', 'U-turn', 'Knock Off']),
			mon('Torkoal', 'Drought', 'Heat Rock', ['Lava Plume', 'Stealth Rock', 'Rapid Spin', 'Yawn'])],
		q1: ['team 12', 'move 2', 'move 1', 'move 4', 'move 4', 'switch 2', 'move 1', 'move 1', 'move 1'],
		q2: ['team 12', 'move 3', 'switch 2', 'move 2', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
	},
	{
		name: 'skill-swap-taunt-switch',
		what: 'Skill Swap trades abilities; a Taunt ends silently when the taunted Pokemon switches out',
		p1: [mon('Slowking', 'Regenerator', 'Leftovers', ['Skill Swap', 'Psychic', 'Slack Off', 'Thunder Wave']),
			mon('Blissey', 'Natural Cure', 'Heavy-Duty Boots', ['Seismic Toss', 'Soft-Boiled', 'Toxic', 'Teleport'])],
		p2: [mon('Tornadus', 'Prankster', 'Leftovers', ['Taunt', 'Bleakwind Storm', 'Nasty Plot', 'U-turn']),
			mon('Gholdengo', 'Good as Gold', 'Leftovers', ['Make It Rain', 'Shadow Ball', 'Nasty Plot', 'Recover'])],
		q1: ['team 12', 'move 1', 'move 2', 'switch 2', 'switch 2', 'move 4', 'move 2', 'move 2'],
		q2: ['team 12', 'move 3', 'move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
	},
	{
		name: 'counter-mirror-coat',
		what: 'Counter and Mirror Coat return the damage taken this turn',
		p1: [mon('Wobbuffet', 'Telepathy', 'Sitrus Berry', ['Counter', 'Mirror Coat', 'Encore', 'Destiny Bond']),
			mon('Blissey', 'Natural Cure', 'Leftovers', ['Seismic Toss', 'Soft-Boiled', 'Toxic', 'Teleport'])],
		p2: [mon('Volcarona', 'Flame Body', 'Leftovers', ['Fiery Dance', 'Bug Buzz', 'Quiver Dance', 'Roost']),
			mon('Kingambit', 'Supreme Overlord', 'Leftovers', ['Kowtow Cleave', 'Iron Head', 'Sucker Punch', 'Swords Dance'])],
		q1: ['team 12', 'move 2', 'switch 2', 'switch 2', 'move 1', 'move 3', 'move 1'],
		q2: ['team 12', 'move 1', 'switch 2', 'move 1', 'move 1', 'move 1', 'move 1'],
	},
	{
		name: 'recharge-transform',
		what: 'Hyper Beam forces a recharge turn; Ditto transforms on entry (Imposter)',
		p1: [mon('Porygon-Z', 'Adaptability', 'Choice Specs', ['Hyper Beam', 'Shadow Ball', 'Ice Beam', 'Thunderbolt']),
			mon('Ditto', 'Imposter', 'Choice Scarf', ['Transform'])],
		p2: [mon('Hippowdon', 'Sand Stream', 'Leftovers', ['Earthquake', 'Slack Off', 'Stealth Rock', 'Whirlwind']),
			mon('Toxapex', 'Regenerator', 'Black Sludge', ['Surf', 'Recover', 'Toxic', 'Haze'])],
		q1: ['team 12', 'move 1', 'move 1', 'switch 2', 'move 1', 'move 1', 'move 1', 'move 1'],
		q2: ['team 12', 'move 3', 'move 2', 'move 1', 'move 1', 'move 2', 'move 2', 'move 2'],
	},
	{
		name: 'revival-blessing',
		what: 'A teammate faints, Pawmot uses Revival Blessing (revival request: choose a fainted ally)',
		p1: [mon('Shuckle', 'Sturdy', 'Focus Sash', ['Sticky Web', 'Toxic', 'Rest', 'Infestation'], { level: 1 }),
			mon('Pawmot', 'Volt Absorb', 'Leftovers', ['Revival Blessing', 'Double Shock', 'Close Combat', 'Ice Punch'])],
		p2: [mon('Dragonite', 'Multiscale', 'Leftovers', ['Extreme Speed', 'Earthquake', 'Dragon Dance', 'Roost']),
			mon('Clefable', 'Magic Guard', 'Leftovers', ['Moonblast', 'Soft-Boiled', 'Calm Mind', 'Flamethrower'])],
		q1: ['team 12', 'move 1', 'move 1', 'switch 2', 'move 1', 'switch 1', 'move 3', 'move 3', 'move 3'],
		q2: ['team 12', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
	},
	{
		name: 'first-turn-only-endeavor',
		what: 'First Impression works on the first turn only; Endeavor sets the target to the user\'s HP',
		p1: [mon('Lokix', 'Tinted Lens', 'Choice Band', ['First Impression', 'Sucker Punch', 'U-turn', 'Knock Off']),
			mon('Dudunsparce', 'Serene Grace', 'Leftovers', ['Endeavor', 'Glare', 'Headbutt', 'Roost'], { hp: 1 })],
		p2: [mon('Gliscor', 'Poison Heal', 'Toxic Orb', ['Earthquake', 'Protect', 'Knock Off', 'Toxic']),
			mon('Garchomp', 'Rough Skin', 'Leftovers', ['Earthquake', 'Dragon Claw', 'Swords Dance', 'Stealth Rock'])],
		q1: ['team 12', 'move 1', 'move 1', 'switch 2', 'move 1', 'move 1', 'move 1', 'move 2'],
		q2: ['team 12', 'move 4', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
	},
];

function run(sc, i) {
	const seed = 'sodium,' + (0x5cea0000 + i).toString(16).padStart(32, '0');
	const battle = new Battle({ formatid: 'gen9customgame', seed, send: () => {} });
	battle.setPlayer('p1', { name: 'scenario-p1', team: sc.p1 });
	battle.setPlayer('p2', { name: 'scenario-p2', team: sc.p2 });
	const queues = { p1: [...sc.q1], p2: [...sc.q2] };
	const refused = { p1: [], p2: [] };
	for (let guard = 0; guard < 200 && !battle.ended; guard++) {
		let moved = false;
		for (const sideId of ['p1', 'p2']) {
			const side = battle.getSide(sideId);
			const req = side.activeRequest;
			if (!req || req.wait || side.isChoiceDone()) continue;
			const choice = queues[sideId].shift();
			if (choice === undefined) {
				battle.choose(sideId, 'default');
			} else if (!battle.choose(sideId, choice)) {
				refused[sideId].push({ choice, error: String(side.choice.error || '') });
				side.clearChoice();
				battle.choose(sideId, 'default');
			}
			moved = true;
		}
		if (!moved) break;
		if (!queues.p1.length && !queues.p2.length) break;
	}
	return {
		roomid: `battle-gen9customgame-scenario${i + 1}`, format: 'gen9customgame', seed, scenario: sc.name,
		what: sc.what, inputLog: battle.inputLog, log: battle.log, refused, ended: battle.ended, turns: battle.turn,
	};
}

fs.mkdirSync(outDir, { recursive: true });
SCENARIOS.forEach((sc, i) => {
	const res = run(sc, i);
	fs.writeFileSync(path.join(outDir, `${res.roomid.slice(7)}.log.json`), JSON.stringify(res));
	console.log(`${sc.name}: ${res.turns} turns, refused p1 ${res.refused.p1.length} p2 ${res.refused.p2.length}`);
});
