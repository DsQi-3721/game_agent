"""Full-battle outcomes, foreground timing and the persistent strategic loop."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from s1_policy_evaluation_results import records, latency, stream_metrics, win_difference
from arena.eval.metrics import wilson_interval

ARMS = ('baseline', 'strategic')
OPPONENTS = ('kakuna', 'foulplay')


def equal_weight_difference(comparisons):
    if any(row is None for row in comparisons.values()):
        return None
    n = len(comparisons)
    difference = sum(row['difference'] for row in comparisons.values()) / n
    lower = math.sqrt(sum((row['difference'] - row['ci95'][0]) ** 2 for row in comparisons.values())) / n
    upper = math.sqrt(sum((row['ci95'][1] - row['difference']) ** 2 for row in comparisons.values())) / n
    return {'difference': difference, 'ci95': [difference - lower, difference + upper],
            'method': 'equal-weight stratified Newcombe score interval'}


def batch(directory, arm, primary_games=None):
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    decisions = records(directory / f'traj_{arm}.jsonl')
    experience = records(directory / f's1exp_{arm}.jsonl')
    calls = records(directory / f'calls_{arm}.jsonl')
    streams = [rows for path in (directory / 'streams').glob('*.jsonl')
               if (rows := records(path)) and rows[0].get('client') == 'poke-env']
    def terminal_line(line):
        return line.startswith(('|win|', '|tie|')) or line == '|tie'
    verified_tags = {rows[0]['battle_tag'] for rows in streams if any(
        terminal_line(line) for row in rows if row['k'] == 'in' for line in row['data'].splitlines())}
    outcomes = sorted((r for r in experience if r['type'] == 'outcome'), key=lambda r: r['ts'])
    if report is None:
        verified = [r for r in outcomes if r['battle_tag'] in verified_tags]
        result = {'games': len(verified), 'wins': sum(r['won'] is True for r in verified)}
        ties = sum(r['won'] is None for r in verified)
    else:
        result = report['agents'][arm]
        ties = sum(match['ties'] for match in report['matches'])
    selected = None
    games, wins = result['games'], result['wins']
    if primary_games is not None or report is None:
        if primary_games is not None:
            outcomes = outcomes[:primary_games]
        if report is None:
            outcomes = [r for r in outcomes if r['battle_tag'] in verified_tags]
        selected = {row['battle_tag'] for row in outcomes}
        games, wins = len(outcomes), sum(row['won'] is True for row in outcomes)
        ties = sum(row['won'] is None for row in outcomes)
        decisions = [row for row in decisions if row['battle_tag'] in selected]
        experience = [row for row in experience if row['battle_tag'] in selected]
        calls = [row for row in calls if row['battle_tag'] in selected]
    counts = Counter(games=games, wins=wins, ties=ties, decisions=len(decisions),
        fallbacks=sum(bool(d['fallback']) for d in decisions),
        foreground_llm_calls=sum(d['extra'].get('llm_calls', 0) for d in decisions),
        model_errors=sum(r['type'] == 's1_error' for r in experience),
        strategy_errors=sum(r.get('space') == 'event:strategy_error' for r in calls),
        actions_with_plan=sum(bool(d['extra'].get('strategy_plan_id')) for d in decisions),
        search_value_actions=sum(d['extra'].get('route') == 'search_value' for d in decisions),
        initial_snapshot_actions=sum(d['extra'].get('la_stage') == 'initial' for d in decisions),
        student_requests=sum(r['type'] == 's1' for r in experience))
    search = [r for r in experience if r['type'] == 'lookahead']
    counts['search_results'] = len(search)
    counts['guided_searches'] = sum(bool(r.get('strategic_plan')) for r in search)
    counts['student_guided_searches'] = sum(any(t.get('prior_iterations', 0) > 0 for t in r.get('tree') or []) for r in search)
    counts['unrecorded_history_actions'] = sum('played unrecorded;' in event for r in experience if r['type'] == 's1'
                                             for event in r['state'].get('recent_decisions', []))
    background = [r for r in calls if r['space'] in ('planner', 'executor')]
    counts['planner_calls'] = sum(r['space'] == 'planner' for r in background)
    counts['compiler_calls'] = sum(r['space'] == 'executor' for r in background)
    counts['background_prompt_tokens'] = sum(r.get('prompt_tokens', 0) for r in background)
    counts['background_completion_tokens'] = sum(r.get('completion_tokens', 0) for r in background)
    counts['background_seconds'] = sum(r.get('latency_s', 0) for r in background)
    intervals = defaultdict(list)
    for call in background:
        intervals[call['battle_tag']].append((call['t_start'], call['t_start'] + call['latency_s']))
    times, terminal_tags, durations, per_game = [], set(), [], []
    for stream in streams:
        tag = stream[0]['battle_tag']
        if selected is not None and tag not in selected:
            continue
        metrics = stream_metrics(stream)
        times.extend(metrics['latencies_s'])
        counts['rejected_orders'] += metrics['rejected_orders']
        counts['unanswered_requests'] += metrics['unanswered_requests']
        terminal = [row['t'] for row in stream if row['k'] == 'in' for line in row['data'].splitlines()
                    if terminal_line(line)]
        if terminal:
            terminal_tags.add(tag)
            started = [row['t'] for row in stream if row['k'] == 'in'
                       and '|start' in row['data'].splitlines()]
            duration = terminal[0] - started[0] if started else None
            if duration is not None:
                durations.append(duration)
            outcome = next((r for r in experience if r['type'] == 'outcome' and r['battle_tag'] == tag), {})
            per_game.append({'battle_tag': tag, 'won': outcome.get('won'), 'turns': outcome.get('turns'),
                             'duration_s': duration, 'decisions': len(metrics['latencies_s']),
                             'latency': latency(metrics['latencies_s'])})
        for row in stream:
            if row['k'] == 'out' and row['data'].startswith('/choose '):
                counts['submitted_actions_during_llm'] += any(start <= row['t'] <= end for start, end in intervals[tag])
    counts['terminal_games'] = len(terminal_tags)
    return {'counts': counts, 'latencies': times, 'calls': calls, 'decisions': decisions,
            'aborted': report.get('aborted') if report else None, 'agent_metrics': result.get('extra', {}),
            'report_available': report is not None, 'game_durations': durations, 'per_game': per_game,
            'recorded_games': result['games'], 'recorded_wins': result['wins'],
            'battle_tags': sorted(selected) if selected is not None else sorted({r['battle_tag'] for r in decisions})}


def preflight_quality(root, protocol):
    rows = []
    for item in protocol['preflight']:
        data = batch(root / item['directory'], item['arm']); c = data['counts']
        passed = (c['games'] == c['terminal_games'] == 1 and not data['aborted']
            and c['student_requests'] > 0 and c['search_value_actions'] > 0
            and not any(c[k] for k in ('foreground_llm_calls', 'model_errors', 'fallbacks',
                                      'unanswered_requests', 'unrecorded_history_actions')))
        if item['arm'] == 'strategic':
            passed = passed and c['actions_with_plan'] > 0 and c['guided_searches'] > 0 and c['strategy_errors'] == 0
        rows.append({**item, **c, 'latency': latency(data['latencies']), 'passed': bool(passed)})
    return {'passed': all(r['passed'] for r in rows), 'games': rows}


def strategic_loop(root, calls, decisions, battle_tags=None):
    experiences, archives = [], []
    for opponent in OPPONENTS:
        memory = root / 'memory' / opponent
        experiences.extend(records(memory / 'experiences.jsonl'))
        archives.extend(json.loads(path.read_text()) for path in (memory / 'battles').glob('*.json'))
    by_id = {e['id']: e for e in experiences}
    store_rows = len(experiences)
    if battle_tags is not None:
        archives = [row for row in archives if row['history']['battle'] in battle_tags]
        experiences = [row for row in experiences if row['battle'] in battle_tags]
    counts, progress, plan_uses = Counter(), Counter(), defaultdict(list)
    used = {(r['battle_tag'], r['extra']['decision']) for r in decisions if r['extra'].get('strategy_plan_id')}
    for archive in archives:
        counts['archived_games'] += 1
        for action in archive['history']['decisions']:
            counts['archived_actions'] += 1
            counts['actions_with_actual_feedback'] += 'actual' in action
            counts['actions_with_search_prediction'] += 'expected' in action
            if action['plan_id'] and (archive['history']['battle'], action['decision']) in used:
                plan_uses[(archive['history']['battle'], action['plan_id'])].append(action)
        for event in archive['trace']:
            if event['kind'] == 'published':
                counts['published_plans'] += 1
                counts['continued_publications'] += bool(event.get('continued'))
            elif event['kind'] == 'progress':
                progress[event['change']['kind']] += 1
    counts['plans_used_multiple_decisions'] = sum(len(actions) >= 2 for actions in plan_uses.values())
    counts['plans_surviving_own_switch'] = sum(len({tuple(name for name, mon in action['before']['own'].items()
        if mon['active']) for action in actions}) > 1 for actions in plan_uses.values())
    references = [(call['battle_tag'], ident) for call in calls if call['space'] == 'event:strategy_ready'
                  for ident in call.get('extra', {}).get('experience_refs', [])]
    counts['cross_game_experience_citations'] = sum(ident in by_id and by_id[ident]['battle'] != tag for tag, ident in references)
    counts['experiences_written'] = len(experiences)
    counts['experience_store_rows'] = store_rows
    return {**counts, 'progress': dict(progress),
            'experience_results': dict(Counter(e['result']['status'] for e in experiences))}


def summarize(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    schedule = json.loads((root / 'schedule.json').read_text())
    totals = {arm: {opp: Counter() for opp in OPPONENTS} for arm in ARMS}
    times = {arm: {opp: [] for opp in OPPONENTS} for arm in ARMS}
    durations = {arm: {opp: [] for opp in OPPONENTS} for arm in ARMS}
    per_team = {arm: {opp: [] for opp in OPPONENTS} for arm in ARMS}
    missing, unfinalized, calls, decisions, strategic_tags, game_rows = [], [], [], [], set(), []
    recorded = {arm: {opp: Counter(games=0, wins=0) for opp in OPPONENTS} for arm in ARMS}
    for item in schedule:
        directory = root / item['directory']
        if not (directory / 'report.json').exists() and not (directory / f"s1exp_{item['arm']}.jsonl").exists():
            missing.append(item['directory']); continue
        data = batch(directory, item['arm'], item.get('primary_games')); c = data['counts']
        arm, opp = item['arm'], item['opponent']
        totals[arm][opp].update(c); times[arm][opp].extend(data['latencies'])
        durations[arm][opp].extend(data['game_durations'])
        game_rows.extend({'arm':arm, 'opponent':opp, 'team':item['team'], **row} for row in data['per_game'])
        if not data['report_available']:
            unfinalized.append(item['directory'])
        recorded[arm][opp].update(games=data['recorded_games'], wins=data['recorded_wins'])
        per_team[arm][opp].append({'team': protocol['teams'][item['team']], 'games': c['games'], 'wins': c['wins']})
        expected = item.get('primary_games', item['games'])
        if (not data['report_available'] or data['aborted'] or c['games'] != expected or c['terminal_games'] != expected
                or data['recorded_games'] != item['games']):
            missing.append(item['directory'])
        if arm == 'strategic':
            calls.extend(data['calls']); decisions.extend(data['decisions'])
            strategic_tags.update(data['battle_tags'])
    arms = {arm: {opp: {**totals[arm][opp],
        'win_rate': totals[arm][opp]['wins'] / totals[arm][opp]['games'] if totals[arm][opp]['games'] else None,
        'win_ci95': wilson_interval(totals[arm][opp]['wins'], totals[arm][opp]['games']),
        'latency': latency(times[arm][opp]), 'game_duration': latency(durations[arm][opp]),
        'per_team': per_team[arm][opp]}
        for opp in OPPONENTS} for arm in ARMS}
    overall = {}
    for arm in ARMS:
        c = Counter(games=0, wins=0, decisions=0)
        for opponent_counts in totals[arm].values():
            c.update(opponent_counts)
        overall[arm] = {**c, 'win_rate': c['wins'] / c['games'] if c['games'] else None,
            'win_ci95': wilson_interval(c['wins'], c['games']),
            'latency': latency(sum(times[arm].values(), [])), 'game_duration': latency(sum(durations[arm].values(), []))}
    comparisons = {opp: win_difference(totals['strategic'][opp]['wins'], totals['strategic'][opp]['games'],
                                       totals['baseline'][opp]['wins'], totals['baseline'][opp]['games']) for opp in OPPONENTS}
    primary = equal_weight_difference(comparisons)
    ratios = {key: overall['strategic']['latency'][key] / overall['baseline']['latency'][key]
              if overall['strategic']['latency'][key] is not None and overall['baseline']['latency'][key] else None
              for key in ('mean_s', 'p90_s')}
    loop = strategic_loop(root, calls, decisions, strategic_tags)
    error_types = Counter(call.get('extra', {}).get('error', 'unknown').split(':', 1)[0]
                          for call in calls if call.get('space') == 'event:strategy_error')
    complete = not missing and all(totals[arm][opp]['games'] == protocol['games_per_arm_opponent']
                                  for arm in ARMS for opp in OPPONENTS)
    loop_passed = (loop.get('archived_games', 0) == overall['strategic']['games'] > 0
        and loop.get('actions_with_actual_feedback', 0) == overall['strategic']['decisions'] > 0
        and all(loop.get(k, 0) > 0 for k in ('published_plans', 'plans_used_multiple_decisions',
            'plans_surviving_own_switch', 'cross_game_experience_citations', 'experiences_written'))
        and sum(loop['progress'].values()) > 0 and overall['strategic'].get('guided_searches', 0) > 0
        and overall['strategic'].get('submitted_actions_during_llm', 0) > 0
        and all(overall[arm].get('foreground_llm_calls', 0) == 0 for arm in ARMS))
    gain_passed = bool(primary and primary['difference'] > 0 and primary['ci95'][0] > 0)
    speed_passed = all(value is not None and value <= 1.10 for value in ratios.values())
    status_path = root / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    return {'root': str(root), 'complete': complete, 'completed_games': sum(r['games'] for r in overall.values()),
        'stopped_by_user': status.get('state') == 'stopped_by_user', 'stopped_at': status.get('stopped_at'),
        'unfinalized_batches': unfinalized, 'games': game_rows,
        'background_error_types': dict(error_types),
        'completed_recorded_games': sum(row['games'] for arm in recorded.values() for row in arm.values()),
        'recorded_arms': recorded,
        'expected_games': protocol['expected_games'], 'missing_batches': missing, 'arms': arms, 'overall': overall,
        'comparison': comparisons, 'primary': primary, 'latency_ratio': ratios, 'strategic_loop': loop,
        'gain_passed': gain_passed, 'speed_passed': speed_passed, 'loop_passed': bool(loop_passed),
        'passed': bool(gain_passed and speed_passed and loop_passed) if complete else None}


def write_report(path, result):
    protocol = json.loads((Path(result['root']) / 'protocol.json').read_text())
    label = ('已按要求停止' if result.get('stopped_by_user') else '通过' if result['passed'] else
             '未达到通过标准' if result['complete'] else '进行中')
    lines = ['# HyMem 持续战略规划完整对局评价', '',
        f"工作五：**{label}**。主评测完成 {result['completed_games']} / {result['expected_games']} 局。已完成对局记录共 {result['completed_recorded_games']} 局。", '',
        '| 对手 | 系统 | 胜局 / 对局 | 胜率 | 平均时延 / 秒 | p90 / 秒 |', '|---|---|---:|---:|---:|---:|']
    for opponent in OPPONENTS:
        for arm, name in (('baseline', 'S1＋搜索'), ('strategic', 'S1＋搜索＋HyMem')):
            row = result['arms'][arm][opponent]; t = row['latency']
            if row.get('games') and t['n']:
                lines.append(f"| {opponent} | {name} | {row['wins']}/{row['games']} | {row['win_rate']:.2%} | {t['mean_s']:.3f} | {t['p90_s']:.3f} |")
    lines.append('')
    extra = result['completed_recorded_games'] - result['completed_games']
    if extra:
        lines.extend([f"主评测按各队伍、对手和系统的完成顺序选取前 {protocol['games_per_team']} 局；额外 {extra} 局保留为补充记录。经验库在全部已完成战略对局之间持续积累。", ''])
    if result['primary']:
        p = result['primary']; lo, hi = p['ci95']
        lines.extend([f"两个对手等权的胜率增量为 {p['difference']*100:+.2f} 个百分点，95% 区间 [{lo*100:+.2f}，{hi*100:+.2f}] 个百分点。", ''])
    if all(v is not None for v in result['latency_ratio'].values()):
        lines.extend([f"战略组 / 对照组的平均时延比为 {result['latency_ratio']['mean_s']:.3f}，p90 比为 {result['latency_ratio']['p90_s']:.3f}；上限均为 1.10。", ''])
    lines.extend(['## 已完成样本的判断', '',
        f"胜率增益标准：{'满足' if result['gain_passed'] else '未满足'}；"
        f"时延标准：{'满足' if result['speed_passed'] else '未满足'}；"
        f"战略闭环：{'完整' if result['loop_passed'] else '未完整验证'}。", ''])
    for arm, name in (('baseline','对照组'),('strategic','战略组')):
        duration = result['overall'][arm]['game_duration']
        if duration['n']:
            lines.append(f"{name}平均每局 {duration['mean_s']/60:.2f} 分钟，p90 {duration['p90_s']/60:.2f} 分钟（开战至终局，不含批次收尾等待）。")
    lines.append('')
    loop = result['strategic_loop']; total = result['overall']['strategic']
    lines.extend(['## 战略闭环', '', '| 指标 | 结果 |', '|---|---:|'])
    for name, value in [('决策采用计划', total.get('actions_with_plan', 0)), ('边界发布计划', loop.get('published_plans', 0)),
        ('多次决策使用同一计划', loop.get('plans_used_multiple_decisions', 0)),
        ('跨换人保留计划', loop.get('plans_surviving_own_switch', 0)),
        ('计划引导搜索', total.get('guided_searches', 0)), ('实际后果记录', loop.get('actions_with_actual_feedback', 0)),
        ('新经验', loop.get('experiences_written', 0)), ('跨局经验引用', loop.get('cross_game_experience_citations', 0)),
        ('LLM 运行期间提交动作', total.get('submitted_actions_during_llm', 0)),
        ('前台 LLM 调用', total.get('foreground_llm_calls', 0)), ('后台规划 / 编译调用', f"{total.get('planner_calls', 0)} / {total.get('compiler_calls', 0)}"),
        ('后台输入 / 输出 token', f"{total.get('background_prompt_tokens', 0)} / {total.get('background_completion_tokens', 0)}"),
        ('规划运行错误', total.get('strategy_errors', 0))]:
        lines.append(f'| {name} | {value} |')
    lines.extend(['', '## 评测条件', '',
        '两组使用同一第 8,000 步 Qwen3.5-2B 决策模型、完整动作输入及八次已完成决策历史。每次搜索采用八个配置假设，每棵持续搜索树 700 ms；最高平均评价选招要求全部配置覆盖及每配置至少 16 次访问。', '',
        f"原计划 12 套镜像队伍，每组对 Kakuna、Foul Play 各 {protocol['games_per_arm_opponent']} 局；"
        f"当前样本覆盖 {len({g['team'] for g in result['games']})} 套队伍。两组按相同队伍同期交错运行，各自使用八个工作线程的独立搜索进程。Kakuna 使用第 34 轮权重，Foul Play 每假设搜索 2 秒、四假设并行。", '',
        '战略组后台使用 Qwen3.5-9B。两类对手各自从空经验库开始，经验跨局持续积累。预检经验单独保存。双方使用相同公开信息及配置先验，个性化资料卡未注入。', '',
        '通过标准为两个对手等权胜率差及其 95% 区间下界大于零，平均时延和 p90 均不超过对照的 110%，且战略计划、执行反馈和跨局记忆闭环完整运行。', '',
        f"完整统计及逐局记录：[评测目录]({result['root']})。"
        f"[胜率与时延对照图]({result['root']}/charts/evaluation_overview.png)。", ''])
    if result.get('background_error_types'):
        lines.extend(['## 已知问题', '',
            '后台错误：' + '；'.join(f'{name} {count} 次' for name,count in result['background_error_types'].items()) + '。',
            '本次主评测覆盖原计划前七套队伍，胜率区间用于描述这批已完成对局。', ''])
    if result.get('unfinalized_batches'):
        lines.extend(['## 记录状态', '',
            '存在未生成批次报告的已结束对局；本报告按对应玩家输入流中的终局消息与 outcome 记录核对后纳入统计。',
            '', *[f'- `{name}`' for name in result['unfinalized_batches']], ''])
    path.write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); args = p.parse_args()
    result = summarize(args.root)
    (args.root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    write_report(args.root / 'results.md', result)
    print(json.dumps({k: result[k] for k in ('complete', 'completed_games', 'passed', 'primary', 'latency_ratio', 'loop_passed')}, indent=2))


if __name__ == '__main__':
    main()
