// The simulator's true state at every decision of a logged battle (ground truth for observation checks).
//   node scripts/node/truth_states.js <showdown_dir> [--rejected rejected.json] <a.log.json> [...] > truth.jsonl
//
// The battle is rebuilt from its server log exactly as probe_choices.js does (refused choices from the client
// capture can be passed with --rejected, as for replay_battle.js, so rqids match the live stream). Before
// every accepted choice of a side, one line: {id, side, rqid, turn, field, sides: {p1, p2}} with, per side,
// every Pokemon (species, hp, status, item, ability, boosts, volatiles with durations, tera, move PP),
// side conditions with layers / durations, slot conditions, and whether the side can still Terastallize.
// This is everything the simulator knows; checks compare it with what a player's observation claims.
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const dir = args.shift();
let rejected = {};
if (args[0] === '--rejected') {
	args.shift();
	rejected = JSON.parse(fs.readFileSync(args.shift(), 'utf8'));
}
const root = path.resolve(dir);
const { Battle } = require(path.join(root, 'dist/sim/battle'));

function splitFirst(s, sep) {
	const i = s.indexOf(sep);
	return i < 0 ? [s, ''] : [s.slice(0, i), s.slice(i + sep.length)];
}

function effectState(state) {
	const out = {};
	for (const k of ['duration', 'layers', 'turns', 'startTime', 'time', 'stage', 'counter']) {
		if (state && state[k] !== undefined) out[k] = state[k];
	}
	if (state && state.move) out.move = typeof state.move === 'string' ? state.move : state.move.id;
	return out;
}

function mon(p) {
	const volatiles = {};
	for (const [id, st] of Object.entries(p.volatiles)) volatiles[id] = effectState(st);
	return {
		ident: p.fullname, species: p.species.name, baseSpecies: p.baseSpecies.name, details: p.details,
		hp: p.hp, maxhp: p.maxhp, status: p.status || '', statusState: effectState(p.statusState),
		fainted: p.fainted, active: p.isActive, position: p.position,
		item: p.item || '', lastItem: p.lastItem || '', ability: p.ability, baseAbility: p.baseAbility,
		types: p.getTypes(), boosts: { ...p.boosts }, volatiles,
		terastallized: p.terastallized || '', teraType: p.teraType, canTerastallize: p.canTerastallize || null,
		moves: p.moveSlots.map(m => ({ id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled })),
		lastMove: p.lastMove ? p.lastMove.id : null, stats: { ...p.storedStats },
	};
}

function sideTruth(side) {
	const conditions = {};
	for (const [id, st] of Object.entries(side.sideConditions)) conditions[id] = effectState(st);
	const slots = side.slotConditions.map(sc => {
		const o = {};
		for (const [id, st] of Object.entries(sc)) o[id] = effectState(st);
		return o;
	});
	return {
		pokemon: side.pokemon.map(mon), sideConditions: conditions, slotConditions: slots,
		canTerastallize: side.pokemon.some(p => p.canTerastallize), pokemonLeft: side.pokemonLeft,
	};
}

function fieldTruth(field) {
	const pseudo = {};
	for (const [id, st] of Object.entries(field.pseudoWeather)) pseudo[id] = effectState(st);
	return {
		weather: field.weather || '', weatherState: effectState(field.weatherState),
		terrain: field.terrain || '', terrainState: effectState(field.terrainState), pseudoWeather: pseudo,
	};
}

function run(file) {
	const log = JSON.parse(fs.readFileSync(file, 'utf8'));
	const input = Array.isArray(log.inputLog) ? log.inputLog : String(log.inputLog).split('\n');
	const id = log.roomid || path.basename(file, '.log.json');
	const refused = rejected[id] || {};
	const out = [];
	let battle = null;
	let rqid = 1;
	const lastRqid = {};
	const send = (type, data) => {
		if (type !== 'sideupdate') return;
		const [side, line] = splitFirst(Array.isArray(data) ? data.join('\n') : data, '\n');
		if (line.startsWith('|request|')) lastRqid[side] = ++rqid;
	};
	const accepted = { p1: 0, p2: 0, p3: 0, p4: 0 };
	for (const line of input) {
		if (!line.startsWith('>')) continue;
		const [type, message] = splitFirst(line.slice(1), ' ');
		if (type === 'start') {
			battle = new Battle({ ...JSON.parse(message), send });
		} else if (type === 'player') {
			const [slot, opts] = splitFirst(message, ' ');
			battle.setPlayer(slot, JSON.parse(opts));
		} else if (/^p[1-4]$/.test(type)) {
			const k = accepted[type]++;
			for (const [at, choice] of (refused[type] || [])) {
				if (at === k) {
					battle.choose(type, choice);
					battle.sendUpdates();
				}
			}
			out.push({
				id, side: type, rqid: lastRqid[type] ?? null, turn: battle.turn, field: fieldTruth(battle.field),
				sides: { p1: sideTruth(battle.sides[0]), p2: sideTruth(battle.sides[1]) },
			});
			battle.choose(type, message);
		} else if (type === 'forcewin' || type === 'forcetie') {
			battle.win(type === 'forcewin' ? message : null);
		} else if (type === 'forcelose') {
			battle.lose(message);
		}
		if (battle) battle.sendUpdates();
	}
	return out;
}

for (const file of args) {
	for (const rec of run(file)) process.stdout.write(JSON.stringify(rec) + '\n');
}
