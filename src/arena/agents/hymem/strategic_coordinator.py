"""One background strategic worker per battle; publication only at decision boundaries."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import time

from arena.agents.hymem.session import DecisionStats, stats_scope
from arena.agents.hymem.messages import Entry
from arena.agents.hymem.strategic_state import StrategicHistory
from arena.agents.hymem.strategic_plan import parse_plan, planner_messages, plan_schema, reflection_messages
from arena.agents.hymem.strategic_execution import compile_execution
from arena.agents.hymem.strategic_memory import experience


class StrategicCoordinator:
    def __init__(self, agent, sess, memory, player_side):
        self.agent, self.sess, self.memory = agent, sess, memory
        self.history = StrategicHistory(sess.tag, player_side)
        self.state = None; self.plan = None; self.ready = None
        self.worker = None; self.finalizer = None; self.closed = False
        self.pending = set(); self.sequence = 0; self.seen_facts = set()
        self.experiences = []; self.experience_sequence = 0
        self.search = {}; self.card = None; self.deviation = 0
        self.trace = []; self.last_packet = None; self.experience_after = {}

    def log(self, kind, **data):
        event = {'kind': kind, 'decision': self.state['decision'], 'time': time.time(), **data}
        self.trace.append(event)
        self.agent.log_event(self.sess, 'strategy_' + kind, data)

    def _experience(self, stage, result):
        actions = [a for a in self.history.decisions if a['plan_id'] == self.plan.id and a['phase'] == stage['id']
                   and a['decision'] > self.experience_after.get((self.plan.id, stage['id']), -1)]
        self.experience_sequence += 1
        row = experience(self.plan, stage, actions, self.state, result, self.history.events, self.experience_sequence)
        if row:
            self.experiences.append(row)
            self.experience_after[(self.plan.id, stage['id'])] = self.state['decision']

    def boundary(self, state, prefix, card=None):
        self.state, self.card = deepcopy(state), deepcopy(card)
        added = self.history.ingest(prefix)
        if self.history.decisions:
            action = self.history.decisions[-1]
            if 'actual' not in action and (action['decision'] < state['decision'] or self.sess.closed):
                before = action['before']
                action['actual'] = {'decision': state['decision'],
                    'hp_change': {side: {name: round(mon['hp'] - before[side][name]['hp'], 4)
                                        for name, mon in state[side].items() if mon.get('hp_knowledge') == 'current'
                                        and before[side].get(name, {}).get('hp_knowledge') == 'current'
                                        and mon['hp'] is not None and before[side][name]['hp'] is not None}
                                  for side in ('own', 'opponent')},
                    'events': [e['id'] for e in added]}
        triggers = set()
        if self.plan:
            for change in self.plan.advance(state):
                self.log('progress', plan_id=self.plan.id, change=change)
                triggers.add(change['kind'])
                if change['kind'] in ('completed', 'invalidated'):
                    stage = next((s for s in self.plan.definition['stages'] if s['id'] == change.get('stage')), self.plan.stage)
                    if stage:
                        self._experience(stage, 'achieved' if change['kind'] == 'completed' else 'invalidated')
        if self.ready:
            candidate, source, experience_ids = self.ready; self.ready = None
            if self.plan and self.plan.id == candidate.id:
                candidate = deepcopy(self.plan)
                candidate.last_review_decision = source
            candidate.advance(state)
            if candidate.status == 'active':
                previous = self.plan
                if previous and previous.id != candidate.id and previous.stage:
                    self._experience(previous.stage, 'revised')
                self.plan = candidate
                self.pending.difference_update({'opening', 'route_review'})
                self.log('published', plan_id=candidate.id, source_decision=source,
                         phase=candidate.stage['id'], prior_experiences=experience_ids,
                         continued=previous is not None and previous.id == candidate.id)
            else:
                triggers.add('plan_invalid_on_arrival')
                self.log('not_adopted', plan_id=candidate.id, status=candidate.status)
        for event in added:
            parts = event['protocol'].split('|')
            if event['kind'] == 'faint':
                triggers.add('member_lost')
            if event['kind'] in ('-item', '-enditem', '-ability', '-terastallize', '-immune', '-fail') or (
                    event['kind'] == 'move' and event['side'] == 'opponent'):
                fact = (event['kind'], event['subject'], parts[3] if len(parts) > 3 else '')
                if fact not in self.seen_facts:
                    self.seen_facts.add(fact); triggers.add('new_information')
        if not self.plan or self.plan.status != 'active':
            triggers.add('opening' if self.sequence == 0 else 'route_review')
        if self.deviation >= 3:
            triggers.add('execution_deviation'); self.deviation = 0
        if not self.closed and not self.sess.closed:
            self.pending.update(triggers)
            self._start()
        self.sess.stats.extra.update(strategy_worker_running=bool(self.worker and not self.worker.done()),
                                     strategy_plan_status=self.plan.status if self.plan else 'pending',
                                     strategy_archive_events=len(self.history.events))

    def _start(self):
        if self.pending and self.ready is None and (self.worker is None or self.worker.done()):
            self.worker = asyncio.create_task(self._run())
            self.agent._strategy_tasks.add(self.worker)
            self.worker.add_done_callback(self.agent._strategy_tasks.discard)

    async def _flush_memory(self):
        rows, self.experiences = self.experiences, []
        await asyncio.to_thread(self.memory.append, rows)
        if rows:
            self.agent.metrics['strategy_experiences_written'] += len(rows)
            self.log('experience_written', ids=[r['id'] for r in rows])

    async def _run(self):
        self.sess.in_flight += 1
        started = time.perf_counter()
        reasons = sorted(self.pending); self.pending.clear()
        state, previous = deepcopy(self.state), deepcopy(self.plan)
        history, card = deepcopy(self.history), deepcopy(self.card)
        entry_cut = len(self.sess.entries)
        self.sequence += 1
        try:
            await self._flush_memory()
            memories = await asyncio.to_thread(self.memory.retrieve, state)
            packet = history.package(state, plan=previous.to_dict() if previous else None,
                                     card=card, memories=memories,
                                     hymem={'episode': deepcopy(self.sess.store.episode)})
            packet['recent_execution'] = [a for a in history.decisions if a['decision'] < state['decision']][-4:]
            packet['recent_execution'] = [{k: a.get(k) for k in ('decision', 'action', 'phase', 'actual')}
                                          for a in packet['recent_execution']]
            packet['search_feedback'] = [{'decision': v['decision'], 'plan_progress': v['plan_progress'],
                'policy': dict(sorted(v['policy'].items(), key=lambda item: -(item[1].get('score') or 0))[:4])}
                for k, v in sorted(self.search.items()) if k < state['decision']][-2:]
            self.last_packet = packet
            self.agent.metrics['strategy_experiences_retrieved'] += len(memories)
            self.log('analysis_started', source_decision=state['decision'], triggers=reasons,
                     prior_experiences=[r['id'] for r in memories])
            with stats_scope(DecisionStats(t_start=time.time())):
                reflection = await self.agent.call('planner', reflection_messages(packet), sess=self.sess)
                messages = planner_messages(packet) + [{'role': 'assistant', 'content': reflection.text},
                    {'role': 'user', 'content': 'Compile the viable winning route into the plan schema. '
                     'win_condition explains how preparation enables the finisher. Cite applicable prior experiences.'}]
                answer = await self.agent.call('executor', messages, sess=self.sess,
                                                extra_body={'response_format': plan_schema(packet)})
                plan, operation = parse_plan(answer.text, packet, previous, self.sequence)
                self.sess.store.summary = json.dumps({'plan': plan.to_dict(),
                    'evidence': [e['id'] for e in packet['evidence']],
                    'observed_events': len(history.events)}, ensure_ascii=False)
                self.sess.store.episode = {'key_events': [e['protocol'] for e in packet['evidence'][-12:]],
                                           'verified_facts': [e['protocol'] for e in packet['evidence']
                                                              if e['kind'] in ('-item', '-ability', 'faint')][-16:]}
                self.sess.store.working = {'immediate_goal': plan.definition['win_condition'],
                    'active_hypotheses': [plan.definition.get('strategic_hypothesis', '')],
                    'next_actions': [s['objective'] for s in plan.definition['stages']]}
                self.sess.store.tool = {'effective_strategies': [r for r in memories if r['result']['status'] == 'achieved'],
                                        'dead_ends': [r for r in memories if r['result']['status'] == 'invalidated']}
                self.sess.keep_recent(12, cut=entry_cut, decision=state['decision'])
                self.sess.add(Entry('note', f"Strategic plan {plan.id}: {plan.definition['win_condition']}",
                                    state['decision'], state['turn']))
            if not self.closed:
                self.ready = (plan, state['decision'], [r['id'] for r in memories])
                self.log('ready', plan_id=plan.id, source_decision=state['decision'], operation=operation['operation'],
                         experience_refs=plan.definition.get('experience_refs', []))
                self.agent.metrics['strategy_plans_ready'] += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.agent.metrics['strategy_errors'] += 1
            self.log('error', source_decision=state['decision'], error=f'{type(error).__name__}: {error}')
            self.pending.add('analysis_retry')
        finally:
            self.agent.metrics['strategy_background_s_sum'] += time.perf_counter() - started
            self.agent.sessions.release(self.sess)

    def search_result(self, decision, result):
        self.search[decision] = {'decision': decision, 'plan_progress': deepcopy(result.get('plan_progress', {})),
                                 'policy': deepcopy(result.get('policy', {}))}
        for action in reversed(self.history.decisions):
            if action['decision'] == decision:
                action['expected'] = deepcopy(self.search[decision]); break
        for old in sorted(self.search)[:-4]:
            self.search.pop(old)

    def action(self, action):
        execution = compile_execution(self.plan, self.state, [action])
        active = self.plan is not None and self.plan.status == 'active'
        phase = self.plan.stage['id'] if active and self.plan.stage else None
        self.history.record_action(self.state, action, self.plan.id if active else None, phase,
                                   self.search.get(self.state['decision']))
        if execution:
            self.deviation = 0 if execution['candidate_actions'] else self.deviation + 1
        self.log('action', action=action, plan_id=self.plan.id if active else None, phase=phase,
                 worker_running=bool(self.worker and not self.worker.done()))
        if active:
            self.agent.metrics['strategy_actions'] += 1

    def close(self, won):
        self.closed = True
        if self.plan and self.plan.stage:
            self._experience(self.plan.stage, 'unresolved')
        for row in self.experiences:
            row['result']['battle_won'] = won
        async def finish():
            if self.worker:
                await asyncio.gather(self.worker, return_exceptions=True)
            await self._flush_memory()
            archive = self.memory.path.parent / 'battles' / f'{self.sess.tag}.json'
            archive.parent.mkdir(parents=True, exist_ok=True)
            content = {'history': self.history.to_dict(), 'plan': self.plan.to_dict() if self.plan else None,
                       'trace': self.trace, 'won': won}
            await asyncio.to_thread(archive.write_text, json.dumps(content, ensure_ascii=False))
        self.finalizer = asyncio.create_task(finish())
        self.agent._strategy_tasks.add(self.finalizer)
        self.finalizer.add_done_callback(self.agent._strategy_tasks.discard)
