// Replay Showdown server battle logs through the simulator and emit what each player received.
//   node scripts/node/replay_battle.js <showdown_dir> [--rejected rejected.json] <a.log.json> [...] > replays.jsonl
//
// Input: logs written by the server with Config.logchallenges (seed, teams and the inputLog of every
// accepted choice). The inputLog is written one line at a time, as the server does, so requests are emitted
// at the same points. Requests get the server's rqid numbering (server/room-battle.ts: one counter per
// battle starting at 1, incremented before every request sent to either side).
//
// The inputLog only holds accepted choices. Choices the server refused ([Invalid choice] / [Unavailable
// choice], which can make it re-send an updated request and so shift rqids) come from the client capture:
// --rejected maps battle id -> {p1: [[k, choice], ...], p2: [...]}, each written just before that player's
// k-th accepted choice, as the server received them.
//
// Output: one JSON line per log: {file, id, format, version, p1, p2, spectator, omniscient, end}, where p1/p2
// are the lines that player received (its private |split| lines, |request|<json>, sim |error| lines) and
// spectator/omniscient are the two other channels.
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const dir = args.shift();
let rejected = {};
if (args[0] === '--rejected') {
	args.shift();
	rejected = JSON.parse(fs.readFileSync(args.shift(), 'utf8'));
}
const files = args;
const root = path.resolve(dir);
const { BattleStream } = require(path.join(root, 'dist/sim/battle-stream'));
const { extractChannelMessages } = require(path.join(root, 'dist/sim/battle'));

function splitFirst(s, sep) {
	const i = s.indexOf(sep);
	return i < 0 ? [s, ''] : [s.slice(0, i), s.slice(i + sep.length)];
}

async function replay(log, refused) {
	const stream = new BattleStream();
	const out = { p1: [], p2: [], spectator: [], omniscient: [], end: null };
	let rqid = 1; // RoomBattle starts its counter at 1, so the first request is rqid 2
	const reader = (async () => {
		for await (const chunk of stream) {
			const [type, data] = splitFirst(chunk, '\n');
			if (type === 'update') {
				const ch = extractChannelMessages(data, [-1, 0, 1, 2]);
				out.omniscient.push(...ch[-1]);
				out.spectator.push(...ch[0]);
				out.p1.push(...ch[1]);
				out.p2.push(...ch[2]);
			} else if (type === 'sideupdate') {
				const [side, line] = splitFirst(data, '\n');
				if (line.startsWith('|request|')) {
					const req = JSON.parse(line.slice(9));
					req.rqid = ++rqid;
					out[side].push('|request|' + JSON.stringify(req));
				} else {
					out[side].push(line);
				}
			} else if (type === 'end') {
				out.end = JSON.parse(data);
			}
		}
	})();
	const input = Array.isArray(log.inputLog) ? log.inputLog : String(log.inputLog).split('\n');
	const accepted = { p1: 0, p2: 0, p3: 0, p4: 0 };
	for (const line of input) {
		const side = /^>(p[1-4]) /.exec(line);
		if (side) {
			const k = accepted[side[1]]++;
			for (const [at, choice] of (refused[side[1]] || [])) {
				if (at === k) await stream.write(`>${side[1]} ${choice}`);
			}
		}
		await stream.write(line);
	}
	await stream.writeEnd();
	await reader;
	return out;
}

(async () => {
	for (const file of files) {
		const log = JSON.parse(fs.readFileSync(file, 'utf8'));
		const input = Array.isArray(log.inputLog) ? log.inputLog : String(log.inputLog).split('\n');
		const version = input.filter(l => l.startsWith('>version')).map(l => l.slice(1));
		const id = log.roomid || path.basename(file, '.log.json');
		const res = await replay(log, rejected[id] || {});
		process.stdout.write(JSON.stringify({
			file, id, format: log.format, version, ...res,
		}) + '\n');
	}
})().catch(err => {
	console.error(err);
	process.exit(1);
});
