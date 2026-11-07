// True damage ranges and speeds at every decision of a logged battle (ground truth for the calc tools).
//   node scripts/node/calc_truth.js <showdown_dir> <a.log.json> [...] > calc.jsonl
//
// The battle is rebuilt from its server log as in truth_states.js. Before every accepted choice of a side,
// on a clone: for each move of that side's active Pokemon against the opposing active, the simulator's own
// damage (battle.actions.getDamage, with the hidden real sets) at the lowest (85) and highest (100) roll and
// no critical hit; and both actives' effective Speed (getStat('spe'), with boosts and every modifier) and
// action order speed (getActionSpeed, Trick Room applied).
// As in useMoveInner, the move first goes through the ModifyType / ModifyMove events (-ate abilities, Weather
// Ball, Tera Blast, Mold Breaker's ignoreAbility ...), and the checks a hit passes before damage that do not
// depend on this turn's choices: the TryHit event (Water Absorb, Flash Fire, Levitate-like abilities, Sap
// Sipper, Bulletproof ...), type immunity (incl. Air Balloon / Levitate grounding), TryImmunity, and the
// move's own Try for moves whose failure is fixed by the state (Poltergeist without an item, Dream Eater on
// an awake target ...). A blocked move is {min: 0, max: 0, blocked: reason}.
// Output: {id, side, rqid, turn, attacker, defender, defender_hp, defender_maxhp, moves: {id: {min, max}},
//          speed: {own, opp}, action_speed: {own, opp}}; min/max are null for status moves, false if immune.
const fs = require('fs');
const path = require('path');

const [dir, ...files] = process.argv.slice(2);
const root = path.resolve(dir);
const { Battle } = require(path.join(root, 'dist/sim/battle'));

function splitFirst(s, sep) {
	const i = s.indexOf(sep);
	return i < 0 ? [s, ''] : [s.slice(0, i), s.slice(i + sep.length)];
}

// moves whose failure the state alone decides (not the opponent's choice this turn)
const STATE_TRY = new Set(['poltergeist', 'dreameater', 'steelroller', 'burnup', 'doubleshock', 'synchronoise',
	'lastresort', 'belch', 'fakeout', 'firstimpression']);

function blockedBy(clone, s, t, move) {
	if (STATE_TRY.has(move.id)) {
		s.activeMoveActions++;          // runMove counts the action before the move's Try runs (Fake Out, First Impression)
		if (clone.singleEvent('Try', move, null, s, t, move) === false) return 'move fails';
	}
	const tryHit = clone.runEvent('TryHit', t, s, move);
	if (tryHit === null || tryHit === false || tryHit === '') return 'blocked (TryHit)';
	if (move.ignoreImmunity === undefined) move.ignoreImmunity = (move.category === 'Status');
	if (!t.runImmunity(move)) return 'immune (type)';
	if (!clone.singleEvent('TryImmunity', move, {}, t, s, move)) return 'immune (move)';
	return null;
}

function calc(battle, sideId) {
	const foeId = sideId === 'p1' ? 'p2' : 'p1';
	const src = battle.getSide(sideId).active[0];
	const tgt = battle.getSide(foeId).active[0];
	if (!src || !tgt || src.fainted || tgt.fainted) return null;
	const moves = {};
	for (const slot of src.moveSlots) {
		const out = {};
		for (const [label, roll] of [['min', 85], ['max', 100]]) {
			const clone = Battle.fromJSON(battle.toJSON());
			clone.send = () => {};
			clone.randomizer = d => clone.trunc(clone.trunc(d * roll) / 100);
			const s = clone.getSide(sideId).active[0];
			const t = clone.getSide(foeId).active[0];
			let move = clone.dex.getActiveMove(slot.id);
			let d;
			try {
				clone.setActiveMove(move, s, t);
				clone.singleEvent('ModifyType', move, null, s, t, move, move);
				clone.singleEvent('ModifyMove', move, null, s, t, move, move);
				move = clone.runEvent('ModifyType', s, t, move, move) || move;
				move = clone.runEvent('ModifyMove', s, t, move, move) || move;
				move.willCrit = false;
				const blocked = move.category === 'Status' ? null : blockedBy(clone, s, t, move);
				if (blocked) {
					out.blocked = blocked;
					d = 0;
				} else {
					d = clone.actions.getDamage(s, t, move, true);
				}
			} catch (e) {
				d = 'error';
			}
			out[label] = d === undefined ? null : d;
		}
		// getDamage is one hit: scale by the hits this attacker will land (Skill Link 5, Loaded Dice 4-5)
		const mv = battle.dex.moves.get(slot.id);
		if (mv.multiaccuracy) {
			out.min = out.max = 'multiaccuracy';         // each hit has its own power and accuracy roll
		} else if (mv.multihit && typeof out.min === 'number' && typeof out.max === 'number') {
			let lo = Array.isArray(mv.multihit) ? mv.multihit[0] : mv.multihit;
			let hi = Array.isArray(mv.multihit) ? mv.multihit[1] : mv.multihit;
			if (Array.isArray(mv.multihit) && src.hasAbility('skilllink')) lo = hi;
			else if (Array.isArray(mv.multihit) && src.hasItem('loadeddice')) lo = Math.max(lo, 4);
			out.min *= lo;
			out.max *= hi;
		}
		moves[slot.id] = out;
	}
	return {
		attacker: src.fullname, defender: tgt.fullname, defender_hp: tgt.hp, defender_maxhp: tgt.maxhp,
		moves, speed: { own: src.getStat('spe'), opp: tgt.getStat('spe') },
		action_speed: { own: src.getActionSpeed(), opp: tgt.getActionSpeed() },
		trick_room: !!battle.field.getPseudoWeather('trickroom'),
	};
}

function run(file) {
	const log = JSON.parse(fs.readFileSync(file, 'utf8'));
	const input = Array.isArray(log.inputLog) ? log.inputLog : String(log.inputLog).split('\n');
	const id = log.roomid || path.basename(file, '.log.json');
	const out = [];
	let battle = null;
	let rqid = 1;
	const lastRqid = {};
	const send = (type, data) => {
		if (type !== 'sideupdate') return;
		const [side, line] = splitFirst(Array.isArray(data) ? data.join('\n') : data, '\n');
		if (line.startsWith('|request|')) lastRqid[side] = ++rqid;
	};
	for (const line of input) {
		if (!line.startsWith('>')) continue;
		const [type, message] = splitFirst(line.slice(1), ' ');
		if (type === 'start') {
			battle = new Battle({ ...JSON.parse(message), send });
		} else if (type === 'player') {
			const [slot, opts] = splitFirst(message, ' ');
			battle.setPlayer(slot, JSON.parse(opts));
		} else if (/^p[1-4]$/.test(type)) {
			const c = battle.getSide(type).activeRequest && !battle.getSide(type).activeRequest.teamPreview ?
				calc(battle, type) : null;
			if (c) out.push({ id, side: type, rqid: lastRqid[type] ?? null, turn: battle.turn, ...c });
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

for (const file of files) {
	for (const rec of run(file)) process.stdout.write(JSON.stringify(rec) + '\n');
}
