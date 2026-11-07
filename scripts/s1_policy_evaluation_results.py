"""Battle outcomes, latency and held-out action preferences for policy retraining."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from arena.eval.metrics import wilson_interval
from arena.observation.capture import choice_attempts


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def win_difference(new_wins, new_n, old_wins, old_n):
    """Newcombe score interval for independent binomial proportions."""
    if not new_n or not old_n:
        return None
    new, old = new_wins / new_n, old_wins / old_n
    nl, nu = wilson_interval(new_wins, new_n)
    ol, ou = wilson_interval(old_wins, old_n)
    difference = new - old
    return {"difference": difference,
            "ci95": [difference - math.hypot(new - nl, ou - old),
                     difference + math.hypot(nu - new, old - ol)]}


def latency(values):
    return {"n": len(values), "mean_s": float(np.mean(values)) if values else None,
            "p90_s": float(np.quantile(values, .9)) if values else None}


def stream_metrics(rows):
    requests, sends = {}, {}
    for row in rows:
        if row["k"] == "in":
            for line in row["data"].splitlines():
                if line.startswith("|request|") and line[9:]:
                    request = json.loads(line[9:])
                    if not request.get("wait") and not request.get("teamPreview"):
                        requests.setdefault(request["rqid"], row["t"])
        elif row["k"] == "out" and row["data"].startswith("/choose "):
            sends[row["rqid"]] = row["t"]
    attempts = choice_attempts(rows)
    accepted = {row["rqid"] for row in attempts if row["error"] is None}
    return {"latencies_s": [sends[key] - start for key, start in requests.items()
                            if key in sends and key in accepted],
            "rejected_orders": sum(row["error"] is not None for row in attempts),
            "unanswered_requests": len(requests.keys() - accepted)}


def summarize(root):
    protocol = json.loads((root / "protocol.json").read_text())
    schedule = json.loads((root / "schedule.json").read_text())
    results, missing = {}, []
    all_times = {arm: [] for arm in ("old", "new")}
    for arm in ("old", "new"):
        results[arm] = {}
        for opponent in ("kakuna", "foulplay"):
            counts = Counter()
            times, decision_times, per_team = [], [], []
            for item in (s for s in schedule if s["arm"] == arm and s["opponent"] == opponent):
                directory = root / item["directory"]
                report_path = directory / "report.json"
                if not report_path.exists():
                    missing.append(item["directory"])
                    continue
                report = json.loads(report_path.read_text())
                agent = report["agents"][arm]
                n, wins = agent["games"], agent["wins"]
                ties = sum(match["ties"] for match in report["matches"])
                counts.update(games=n, wins=wins, ties=ties)
                per_team.append({"team": protocol["teams"][item["team"]], "games": n, "wins": wins,
                                 "ties": ties})
                if n != item["games"] or report.get("aborted"):
                    missing.append(item["directory"])
                turns = records(directory / f"traj_{arm}.jsonl")
                counts["decisions"] += len(turns)
                counts["fallbacks"] += sum(bool(row["fallback"]) for row in turns)
                counts["recalls"] += sum(row["extra"].get("note") == "recall" for row in turns)
                decision_times.extend(row["extra"]["latency_s"] for row in turns)
                experience = records(directory / f"s1exp_{arm}.jsonl")
                counts["model_errors"] += sum(row["type"] == "s1_error" for row in experience)
                counts["model_requests"] += sum(row["type"] == "s1" for row in experience)
                counts["assisted_requests"] += sum(row["type"] == "lookahead" or
                    (row["type"] == "s1" and row["role"] != "student") for row in experience)
                counts["llm_calls"] += len(records(directory / f"calls_{arm}.jsonl"))
                for path in sorted((directory / "streams").glob("*.jsonl")):
                    rows = records(path)
                    if rows and rows[0].get("client") == "poke-env":
                        metrics = stream_metrics(rows)
                        times.extend(metrics["latencies_s"])
                        counts["rejected_orders"] += metrics["rejected_orders"]
                        counts["unanswered_requests"] += metrics["unanswered_requests"]
            results[arm][opponent] = {**counts, "win_rate": counts["wins"] / counts["games"] if counts["games"] else None,
                "win_ci95": wilson_interval(counts["wins"], counts["games"]) if counts["games"] else None,
                "latency": latency(times), "agent_latency": latency(decision_times), "per_team": per_team}
            all_times[arm].extend(times)
    comparisons = {}
    for opponent in ("kakuna", "foulplay"):
        new, old = results["new"][opponent], results["old"][opponent]
        comparisons[opponent] = win_difference(new.get("wins", 0), new.get("games", 0),
                                               old.get("wins", 0), old.get("games", 0))
    timing = {arm: latency(values) for arm, values in all_times.items()}
    ratios = {key: timing["new"][key] / timing["old"][key]
              if timing["new"][key] is not None and timing["old"][key] else None
              for key in ("mean_s", "p90_s")}
    complete = not missing and all(results[a][o].get("games", 0) == protocol["games_per_arm_opponent"]
                                  for a in results for o in results[a])
    comparable = complete and all(results[a][o]["latency"]["n"] > 0 and
        not any(results[a][o].get(key) for key in
                ("assisted_requests", "llm_calls", "model_errors", "fallbacks"))
        for a in results for o in results[a])
    primary = comparisons["kakuna"]
    passed = (comparable and primary["difference"] > 0 and primary["ci95"][0] > 0
              and all(value is not None and value <= 1.10 for value in ratios.values()))
    preferences_path = root / "heldout_preferences.json"
    preference_results = json.loads(preferences_path.read_text()) if preferences_path.exists() else None
    return {"root": str(root), "complete": complete, "comparable": comparable,
            "passed": bool(passed) if complete else None, "expected_games": protocol["expected_games"],
            "completed_games": sum(r.get("games", 0) for arm in results.values() for r in arm.values()),
            "missing_batches": missing, "arms": results, "comparison": comparisons,
            "latency": timing, "latency_ratio": ratios,
            "heldout_preferences": {key: value for key, value in preference_results.items() if key != "cases"}
                                    if preference_results else None}


def write_report(path, result):
    def number(value, percent=False):
        if value is None:
            return "待完成"
        return f"{value:.1%}" if percent else f"{value:.3f}"

    conclusion = "通过" if result["passed"] else "未达到通过标准" if result["complete"] else "进行中"
    lines = ["# 决策模型独立博弈评价", "", f"工作五：**{conclusion}**。已完成 {result['completed_games']} / {result['expected_games']} 局。", "",
        "| 对手 | 模型 | 胜局 / 对局 | 胜率 | 平均时延 / 秒 | p90 / 秒 | 回退 | 请求错误 | 拒绝动作 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for opponent in ("kakuna", "foulplay"):
        for arm, label in (("old", "旧模型"), ("new", "再训练模型")):
            row = result["arms"][arm][opponent]
            lines.append(f"| {opponent} | {label} | {row.get('wins', 0)}/{row.get('games', 0)} | "
                f"{number(row['win_rate'], True)} | {number(row['latency']['mean_s'])} | "
                f"{number(row['latency']['p90_s'])} | {row.get('fallbacks', 0)} | "
                f"{row.get('model_errors', 0)} | {row.get('rejected_orders', 0)} |")
    lines.append("")
    for opponent, comparison in result["comparison"].items():
        if comparison:
            lo, hi = comparison["ci95"]
            lines.append(f"{opponent} 胜率增量 {comparison['difference']*100:+.2f} 个百分点，"
                         f"95% 置信区间 [{lo*100:+.2f}, {hi*100:+.2f}] 个百分点。")
            lines.append("")
    lines.extend([f"新模型 / 旧模型的平均时延比为 {number(result['latency_ratio']['mean_s'])}，"
                  f"p90 比为 {number(result['latency_ratio']['p90_s'])}，通过上限均为 1.10。", ""])
    preferences = result.get("heldout_preferences")
    if preferences:
        lines.append(f"留出集 {preferences['n']} 组推演动作偏好的排序正确率："
            f"旧模型 {preferences['arms']['old']['accuracy']:.1%}，"
            f"再训练模型 {preferences['arms']['new']['accuracy']:.1%}。该指标衡量与离线推演偏好的一致性。")
        lines.append("")
    lines.extend(["## 评测条件", "",
        "12 套固定镜像队伍，每模型、每对手、每队伍 10 局。新旧模型同期运行，采用全动作联合概率选招及相同服务参数。"
        "旧模型沿用已训练的分支问题与当前状态输入；新模型使用完整动作问题和八次已完成决策历史。", "",
        "学生独立选招，动作正反顺序分布取平均后选择最大值。Kakuna 使用第 34 轮权重；Foul Play 每个假设搜索 2 秒、4 个假设并行。"
        "时延从动作请求收到计至最终动作发送，包含输入构建、模型推理及重选，排除队伍预览。", "",
        "## 局限", "",
        "本次评价首轮再训练模型。镜像队伍与对手预算固定；结果代表这些对战条件下的能力。完整动作接口和历史输入属于此次策略优化的一部分，"
        "留出集相同输入比较用于辅助识别权重训练带来的变化。", "",
        f"逐局记录与完整统计：[评测目录]({result['root']})。", ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_report(args.root / "results.md", result)
    print(json.dumps({key: result[key] for key in ("complete", "passed", "completed_games", "comparison", "latency_ratio")}, indent=2))


if __name__ == "__main__":
    main()
