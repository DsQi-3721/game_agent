// Ask the simulator which choices it accepts at every decision of a logged battle.
//   node scripts/node/probe_choices.js <showdown_dir> <a.log.json> [b.log.json ...] > probes.jsonl
//
// The battle is rebuilt from its server log (seed, teams, inputLog) exactly as BattleStream does it (one
// input line at a time, sendUpdates after each). Before every accepted choice of a side, the battle is
// cloned (Battle.toJSON/fromJSON) and every syntactically possible singles choice is tried on the clone with
// side.choose(): move 1-4 with and without terastallize, switch 1-6, pass; team preview requests try one
// order. The clone keeps the probes from changing the real battle ([Unavailable choice] updates requests).
//
// Output, one line per decision: {file, id, side, rqid, choice, accepted: {choice: committed}, rejected:
// {choice: "[type] msg"}}. ``committed`` is side.getChoice() after the choice -- what the simulator would
// actually run (e.g. "move 1 terastallize" while locked into Outrage commits "move outrage": tera dropped).
// rqid is numbered like the server (first request is 2), so it joins with replay_battle.js / the capture.
const fs = require('fs');
const path = require('path');

const [dir, ...files] = process.argv.slice(2);
const root = path.resolve(dir);
const { Battle } = require(path.join(root, 'dist/sim/battle'));

function splitFirst(s, sep) {
	const i = s.indexOf(sep);
	return i < 0 ? [s, ''] : [s.slice(0, i), s.slice(i + sep.length)];
}

const CANDIDATES = [];
for (let i = 1; i <= 4; i++) CANDIDATES.push(`move ${i}`, `move ${i} terastallize`);
for (let i = 1; i <= 6; i++) CANDIDATES.push(`switch ${i}`);
CANDIDATES.push('pass');

function probe(battle, sideId) {
	const accepted = {};
	const rejected = {};
	const req = battle.getSide(sideId).activeRequest;
	const tries = req && req.teamPreview ? ['team 123456'] : CANDIDATES;
	for (const c of tries) {
		const clone = Battle.fromJSON(battle.toJSON());
		let error = '';
		clone.send = (type, data) => {
			if (type === 'sideupdate' && data.includes('|error|')) error = data.slice(data.indexOf('|error|') + 7);
		};
		const side = clone.getSide(sideId);
		if (side.choose(c)) accepted[c] = side.getChoice();
		else rejected[c] = error;
	}
	return { accepted, rejected };
}

function run(file) {
	const log = JSON.parse(fs.readFileSync(file, 'utf8'));
	const input = Array.isArray(log.inputLog) ? log.inputLog : String(log.inputLog).split('\n');
	const id = log.roomid || path.basename(file, '.log.json');
	const out = [];
	let battle = null;
	let rqid = 1; // as server/room-battle.ts
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
			out.push({ file, id, side: type, rqid: lastRqid[type] ?? null, choice: message, ...probe(battle, type) });
			if (message === 'undo') battle.undoChoice(type);
			else battle.choose(type, message);
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
