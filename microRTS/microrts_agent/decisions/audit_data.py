"""Comprehensive data-quality audit for collected MicroRTS EVE-RLCD decision datasets.

Usage:
    python -m microrts_agent.decisions.audit_data \
        --train runs/dataset/train.jsonl \
        --val runs/dataset/val.jsonl \
        --out runs/dataset/audit
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rlcd.schema import Question, render_prompt

from .eve_policy import DEFAULT_MODEL
from .metrics import atomic_json


def _load_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{idx + 1}: {exc}") from exc
    return rows


def _prompt_key(row):
    """Canonical identity of a decision prompt (order-insensitive choices would hide shuffling)."""
    h = hashlib.sha256()
    h.update(row["context"].strip().encode("utf-8"))
    h.update(b"\x00")
    h.update(row["question"].strip().encode("utf-8"))
    h.update(b"\x00")
    for c in row["choices"]:
        h.update(c.strip().encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _ctx_key(row):
    return hashlib.sha256(row["context"].strip().encode("utf-8")).hexdigest()


def _load_tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def audit_split(rows, tok=None):
    """Audit a single JSONL split across schema, tokens, actions, match quality, balance, and conflicts."""
    schema_errors = []
    token_lengths = []
    tokens_by_map = defaultdict(list)

    skill_counts = Counter()
    unit_type_counts = Counter()
    unit_skill_counts = Counter()
    match_kind_counts = Counter()
    map_counts = Counter()
    map_size_counts = Counter()
    teacher_counts = Counter()
    opponent_counts = Counter()
    player_counts = Counter()
    outcome_counts = Counter()
    won_counts = Counter()
    terminated_counts = Counter()
    k_counts = Counter()
    answer_idx_counts = Counter()
    tick_buckets = Counter()
    episodes = set()

    # Exact prompt -> Counter of chosen option strings (order-invariant label conflict check!)
    prompt_answers = defaultdict(Counter)
    prompt_samples = {}
    ctx_set = set()
    prompt_set = set()

    for i, r in enumerate(rows):
        try:
            q = Question(
                r.get("primitive", "choice"),
                r["context"],
                r["question"],
                list(r["choices"]),
                answer=int(r["answer"]),
            ).validate()
        except Exception as exc:
            schema_errors.append({"row": i, "error": f"{type(exc).__name__}: {exc}"})
            continue

        ans = int(r["answer"])
        choices = r["choices"]
        k_counts[len(choices)] += 1
        answer_idx_counts[ans] += 1
        skill_counts[r.get("skill", "unknown")] += 1
        ut = r.get("unit_type", "unknown")
        unit_type_counts[ut] += 1
        unit_skill_counts[f"{ut}:{r.get('skill', 'unknown')}"] += 1
        match_kind_counts[r.get("match_kind", "unknown")] += 1
        m = r.get("map", "unknown")
        map_counts[m] += 1
        map_size_counts[r.get("map_size", "unknown")] += 1
        teacher_counts[r.get("teacher", "unknown")] += 1
        opponent_counts[r.get("opponent", "unknown")] += 1
        player_counts[str(r.get("player", "unknown"))] += 1
        outcome_counts[r.get("outcome", "unknown")] += 1
        won_counts[str(bool(r.get("won")))] += 1
        terminated_counts[str(bool(r.get("terminated")))] += 1
        episodes.add((m, r.get("teacher"), r.get("opponent"), r.get("player"), r.get("episode")))

        t = int(r.get("tick", 0))
        bucket = "0-99" if t < 100 else "100-299" if t < 300 else "300-699" if t < 700 else "700+"
        tick_buckets[bucket] += 1

        ck = _ctx_key(r)
        pk = _prompt_key(r)
        ctx_set.add(ck)
        prompt_set.add(pk)
        chosen_text = choices[ans]
        prompt_answers[pk][chosen_text] += 1
        if pk not in prompt_samples:
            prompt_samples[pk] = {
                "map": m,
                "tick": t,
                "question": r["question"],
                "choices": choices,
            }

        if tok is not None:
            n_tok = len(tok.encode(render_prompt(q), add_special_tokens=False))
            token_lengths.append(n_tok)
            tokens_by_map[m].append(n_tok)

    n = len(rows)
    dup_prompt_groups = sum(1 for c in prompt_answers.values() if sum(c.values()) > 1)
    dup_extra_rows = sum(sum(c.values()) - 1 for c in prompt_answers.values() if sum(c.values()) > 1)
    conflict_groups = []
    for pk, ans_counter in prompt_answers.items():
        if len(ans_counter) > 1:
            conflict_groups.append(
                {
                    **prompt_samples[pk],
                    "answers": dict(ans_counter),
                }
            )
    conflict_groups.sort(key=lambda x: -sum(x["answers"].values()))

    token_stats = {}
    if token_lengths:
        arr = np.asarray(token_lengths, dtype=np.int64)
        token_stats = {
            "min": int(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "gt_512": int((arr > 512).sum()),
            "gt_1024": int((arr > 1024).sum()),
            "gt_2048": int((arr > 2048).sum()),
            "by_map": {
                m: {
                    "n": len(vals),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                    "max": int(max(vals)),
                    "gt_512": int(sum(v > 512 for v in vals)),
                    "gt_1024": int(sum(v > 1024 for v in vals)),
                }
                for m, vals in sorted(tokens_by_map.items())
            },
        }

    wait_rate = skill_counts["wait"] / max(n, 1)
    fallback_rate = match_kind_counts["wait_fallback"] / max(n, 1)
    ans0_rate = answer_idx_counts[0] / max(n, 1)

    return {
        "rows": n,
        "episodes": len(episodes),
        "unique_contexts": len(ctx_set),
        "unique_prompts": len(prompt_set),
        "schema_errors": len(schema_errors),
        "schema_error_samples": schema_errors[:5],
        "tokens": token_stats,
        "k_stats": {
            "min": min(k_counts) if k_counts else 0,
            "max": max(k_counts) if k_counts else 0,
            "mean": sum(k * c for k, c in k_counts.items()) / max(n, 1),
            "distribution": dict(sorted(k_counts.items())),
        },
        "answer_index_0_rate": ans0_rate,
        "wait_rate": wait_rate,
        "wait_fallback_rate": fallback_rate,
        "skills": dict(skill_counts.most_common()),
        "unit_types": dict(unit_type_counts.most_common()),
        "unit_skills_top20": dict(unit_skill_counts.most_common(20)),
        "match_kinds": dict(match_kind_counts.most_common()),
        "maps": dict(map_counts.most_common()),
        "map_sizes": dict(map_size_counts.most_common()),
        "teachers": dict(teacher_counts.most_common()),
        "opponents": dict(opponent_counts.most_common()),
        "players": dict(player_counts.most_common()),
        "outcomes": dict(outcome_counts.most_common()),
        "won": dict(won_counts.most_common()),
        "terminated": dict(terminated_counts.most_common()),
        "tick_buckets": dict(tick_buckets),
        "duplicates": {
            "duplicate_prompt_groups": dup_prompt_groups,
            "extra_duplicate_rows": dup_extra_rows,
            "duplicate_row_fraction": dup_extra_rows / max(n, 1),
            "conflicting_prompt_groups": len(conflict_groups),
            "conflict_samples": conflict_groups[:8],
        },
        "_sets": {"ctx": ctx_set, "prompt": prompt_set, "maps": set(map_counts)},
    }


def evaluate_gates(train_rep, val_rep=None, leakage=None):
    """Evaluate concrete data-quality gates (PASS / WARN / FAIL)."""
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    # 1. Schema
    err = train_rep["schema_errors"] + (val_rep["schema_errors"] if val_rep else 0)
    add("schema_validity", "PASS" if err == 0 else "FAIL", f"schema_errors={err}")

    # 2. Token budget
    if train_rep["tokens"]:
        gt2048 = train_rep["tokens"]["gt_2048"] + (val_rep["tokens"]["gt_2048"] if val_rep else 0)
        p95 = train_rep["tokens"]["p95"]
        status = "PASS" if gt2048 == 0 and p95 <= 1024 else "WARN" if gt2048 == 0 else "FAIL"
        add(
            "token_length_budget",
            status,
            f"train p50={train_rep['tokens']['p50']:.0f}, p95={p95:.0f}, max={train_rep['tokens']['max']}, gt_2048={gt2048}",
        )

    # 3. Bot skill matching fidelity
    fb = train_rep["wait_fallback_rate"]
    add(
        "bot_skill_match_fidelity",
        "PASS" if fb <= 0.05 else "WARN" if fb <= 0.12 else "FAIL",
        f"wait_fallback_rate={fb:.2%}, wait_rate={train_rep['wait_rate']:.2%}",
    )

    # 4. Strategic & unit diversity
    has_barracks = train_rep["unit_types"].get("Barracks", 0) > 0
    has_combat = sum(train_rep["unit_types"].get(k, 0) for k in ("Light", "Heavy", "Ranged")) > 0
    skill_n = len(train_rep["skills"])
    add(
        "tactical_diversity",
        "PASS" if (has_barracks and has_combat and skill_n >= 6) else "WARN",
        f"skills={skill_n} ({list(train_rep['skills'])[:7]}), barracks={has_barracks}, combat_units={has_combat}",
    )

    # 5. Player symmetry balance
    p0 = train_rep["players"].get("0", 0)
    p1 = train_rep["players"].get("1", 0)
    p0_frac = p0 / max(p0 + p1, 1)
    add(
        "player_symmetry",
        "PASS" if 0.40 <= p0_frac <= 0.60 else "WARN",
        f"P0={p0} ({p0_frac:.1%}), P1={p1} ({1 - p0_frac:.1%})",
    )

    # 6. Duplicate rate
    dup_f = train_rep["duplicates"]["duplicate_row_fraction"]
    add(
        "duplicate_control",
        "PASS" if dup_f <= 0.25 else "WARN",
        f"dup_row_fraction={dup_f:.1%}, conflict_groups={train_rep['duplicates']['conflicting_prompt_groups']}",
    )

    # 7. Split leakage (when val provided)
    if leakage is not None:
        map_ov = len(leakage["shared_maps"])
        p_ov = leakage["shared_prompts"]
        status = "PASS" if map_ov == 0 and p_ov == 0 else "WARN" if p_ov == 0 else "FAIL"
        add(
            "train_val_isolation",
            status,
            f"shared_maps={map_ov}, shared_contexts={leakage['shared_contexts']}, shared_prompts={p_ov}",
        )

    return checks


def write_markdown_report(out_md, train_rep, val_rep, leakage, checks, all_rep=None):
    lines = ["# MicroRTS 专家轨迹数据质量审计报告", ""]
    lines += [
        "## 1. 质量门禁总览",
        "",
        "| 检查项 | 状态 | 详情 |",
        "| --- | :---: | --- |",
    ]
    for c in checks:
        lines.append(f"| `{c['check']}` | **{c['status']}** | {c['detail']} |")

    lines += ["", "## 2. 数据集规模与切分隔离", ""]
    lines += [
        "| 指标 | 训练集 (train) | 验证集 (val, 未见地图) |" + (" 全量含败局 (all) |" if all_rep else ""),
        "| --- | ---: | ---: |" + (" ---: |" if all_rep else ""),
    ]

    def row_line(label, fn):
        vals = [fn(train_rep), fn(val_rep) if val_rep else "-"]
        if all_rep:
            vals.append(fn(all_rep))
        return f"| {label} | " + " | ".join(str(v) for v in vals) + " |"

    lines.append(row_line("决策行数", lambda r: r["rows"]))
    lines.append(row_line("对局数", lambda r: r["episodes"]))
    lines.append(row_line("唯一上下文数", lambda r: r["unique_contexts"]))
    lines.append(row_line("唯一完整题面数", lambda r: r["unique_prompts"]))
    lines.append(row_line("Schema 校验错误", lambda r: r["schema_errors"]))
    lines.append(row_line("平均候选数 K", lambda r: f"{r['k_stats']['mean']:.2f} ({r['k_stats']['min']}–{r['k_stats']['max']})"))
    lines.append(row_line("Wait 占比", lambda r: f"{r['wait_rate']:.2%}"))
    lines.append(row_line("Wait 回退失配率", lambda r: f"{r['wait_fallback_rate']:.2%}"))
    lines.append(row_line("选项0位置占比 (打乱前)", lambda r: f"{r['answer_index_0_rate']:.1%}"))
    lines.append(row_line("重复题面冗余率", lambda r: f"{r['duplicates']['duplicate_row_fraction']:.1%}"))
    lines.append(row_line("同题不同标冲突组", lambda r: r["duplicates"]["conflicting_prompt_groups"]))
    if train_rep["tokens"]:
        lines.append(
            row_line(
                "Token P50 / P95 / Max",
                lambda r: (
                    f"{r['tokens']['p50']:.0f} / {r['tokens']['p95']:.0f} / {r['tokens']['max']}"
                    if r["tokens"]
                    else "-"
                ),
            )
        )
        lines.append(
            row_line(
                "Token >512 / >1024 / >2048",
                lambda r: (
                    f"{r['tokens']['gt_512']} / {r['tokens']['gt_1024']} / {r['tokens']['gt_2048']}"
                    if r["tokens"]
                    else "-"
                ),
            )
        )

    if leakage:
        lines += [
            "",
            f"- **Train / Val 隔离**：共享地图 `{len(leakage['shared_maps'])}` 张，共享上下文 `{leakage['shared_contexts']}` 个，共享完整题面 `{leakage['shared_prompts']}` 个。",
        ]

    lines += ["", "## 3. 技能、兵种与匹配质量分布（训练集）", ""]
    lines.append("**技能分布**：`" + json.dumps(train_rep["skills"]) + "`  ")
    lines.append("**兵种分布**：`" + json.dumps(train_rep["unit_types"]) + "`  ")
    lines.append("**动作匹配方式**：`" + json.dumps(train_rep["match_kinds"]) + "`  ")
    lines.append("**教师分布**：`" + json.dumps(train_rep["teachers"]) + "`  ")
    lines.append("**对手分布**：`" + json.dumps(train_rep["opponents"]) + "`  ")
    lines.append("**阵营分布**：`" + json.dumps(train_rep["players"]) + "`  ")
    lines.append("**时间阶段 (tick)**：`" + json.dumps(train_rep["tick_buckets"]) + "`")

    if train_rep["tokens"] and train_rep["tokens"].get("by_map"):
        lines += [
            "",
            "## 4. 分地图规模与 Token 长度（训练集）",
            "",
            "| 地图 | 行数 | P50 tok | P95 tok | Max tok | >512 | >1024 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for m, st in train_rep["tokens"]["by_map"].items():
            short = Path(m).name
            lines.append(
                f"| `{short}` | {st['n']} | {st['p50']:.0f} | {st['p95']:.0f} | {st['max']} | {st['gt_512']} | {st['gt_1024']} |"
            )

    if train_rep["duplicates"]["conflict_samples"]:
        lines += ["", "## 5. 典型同题多标签冲突样例（策略多样性/歧义）", ""]
        for sample in train_rep["duplicates"]["conflict_samples"][:5]:
            lines.append(
                f"- `{Path(sample['map']).name}` t={sample['tick']} — {sample['question']} → `{sample['answers']}`"
            )

    lines.append("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def dedup_and_clean(
    rows,
    *,
    max_exact_repeats=1,
    max_tokens=None,
    tok=None,
    drop_wait_fallback=True,
    shuffle_seed=None,
):
    """Produce a deduplicated, noise-filtered, option-shuffled clean ChoiceQ dataset."""
    import random

    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    seen = Counter()
    cleaned = []
    dropped_dup = 0
    dropped_len = 0
    dropped_fallback = 0
    for r in rows:
        if drop_wait_fallback and r.get("match_kind") == "wait_fallback":
            dropped_fallback += 1
            continue
        key = (_prompt_key(r), int(r["answer"]))
        if seen[key] >= max_exact_repeats:
            dropped_dup += 1
            continue
        q = Question(
            r.get("primitive", "choice"),
            r["context"],
            r["question"],
            list(r["choices"]),
            answer=int(r["answer"]),
        ).validate()
        if (
            max_tokens is not None
            and tok is not None
            and len(tok.encode(render_prompt(q), add_special_tokens=False)) > max_tokens
        ):
            dropped_len += 1
            continue
        seen[key] += 1
        out_r = dict(r)
        if rng is not None and len(out_r["choices"]) > 1:
            n_c = len(out_r["choices"])
            perm = list(range(n_c))
            rng.shuffle(perm)
            out_r["choices"] = [out_r["choices"][i] for i in perm]
            out_r["answer"] = perm.index(int(out_r["answer"]))
        cleaned.append(out_r)
    return cleaned, {
        "kept": len(cleaned),
        "dropped_dup": dropped_dup,
        "dropped_fallback": dropped_fallback,
        "dropped_len": dropped_len,
    }


def extract_win_noul_rows(all_rows):
    """Extract 1 unique state-level NoulQ win-forecast row per (episode, tick) from terminated games."""
    seen = set()
    out = []
    for r in all_rows:
        if not r.get("terminated"):
            continue
        key = (r.get("map"), r.get("teacher"), r.get("opponent"), r.get("player"), r.get("episode"), r.get("tick"))
        if key in seen:
            continue
        seen.add(key)
        won = bool(r.get("won"))
        out.append(
            {
                "primitive": "noul",
                "context": r["context"],
                "question": (
                    "Following the current sampled policy against this opponent, "
                    "will our player win this match? A draw counts as false."
                ),
                "choices": ["true", "false"],
                "answer": 0 if won else 1,
                "won": won,
                "terminated": True,
                "map": r.get("map"),
                "map_size": r.get("map_size"),
                "teacher": r.get("teacher"),
                "opponent": r.get("opponent"),
                "player": r.get("player"),
                "episode": r.get("episode"),
                "tick": r.get("tick"),
            }
        )
    return out


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path)
    ap.add_argument("--all-train", type=Path, help="Optional wins+losses train file for balance check")
    ap.add_argument("--all-val", type=Path, help="Optional wins+losses val file for NoulQ export")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, required=True, help="Output directory for audit report")
    ap.add_argument("--clean-out", type=Path, help="Output path for cleaned, shuffled train JSONL")
    ap.add_argument("--val-clean-out", type=Path, help="Output path for cleaned, shuffled val JSONL")
    ap.add_argument("--win-train-out", type=Path, help="Output path for state-level NoulQ win train JSONL")
    ap.add_argument("--win-val-out", type=Path, help="Output path for state-level NoulQ win val JSONL")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--shuffle-seed", type=int, default=42)
    args = ap.parse_args()

    tok = _load_tokenizer(args.tokenizer) if args.tokenizer and args.tokenizer.exists() else None
    train_rows = _load_jsonl(args.train)
    train_rep = audit_split(train_rows, tok=tok)

    val_rows = None
    val_rep = None
    leakage = None
    if args.val and args.val.exists():
        val_rows = _load_jsonl(args.val)
        val_rep = audit_split(val_rows, tok=tok)
        leakage = {
            "shared_maps": sorted(train_rep["_sets"]["maps"] & val_rep["_sets"]["maps"]),
            "shared_contexts": len(train_rep["_sets"]["ctx"] & val_rep["_sets"]["ctx"]),
            "shared_prompts": len(train_rep["_sets"]["prompt"] & val_rep["_sets"]["prompt"]),
        }

    all_train_rows = _load_jsonl(args.all_train) if (args.all_train and args.all_train.exists()) else None
    all_rep = audit_split(all_train_rows, tok=None) if all_train_rows is not None else None

    checks = evaluate_gates(train_rep, val_rep=val_rep, leakage=leakage)

    clean_stats = {}
    if args.clean_out:
        cleaned_train, st = dedup_and_clean(
            train_rows,
            max_exact_repeats=1,
            max_tokens=args.max_tokens,
            tok=tok,
            drop_wait_fallback=True,
            shuffle_seed=args.shuffle_seed,
        )
        _write_jsonl(args.clean_out, cleaned_train)
        clean_stats["train"] = st
        clean_stats["train_ans0_rate_after_shuffle"] = (
            sum(r["answer"] == 0 for r in cleaned_train) / max(len(cleaned_train), 1)
        )

    if args.val_clean_out and val_rows is not None:
        cleaned_val, st_v = dedup_and_clean(
            val_rows,
            max_exact_repeats=1,
            max_tokens=args.max_tokens,
            tok=tok,
            drop_wait_fallback=True,
            shuffle_seed=args.shuffle_seed + 1,
        )
        _write_jsonl(args.val_clean_out, cleaned_val)
        clean_stats["val"] = st_v
        clean_stats["val_ans0_rate_after_shuffle"] = (
            sum(r["answer"] == 0 for r in cleaned_val) / max(len(cleaned_val), 1)
        )

    if args.win_train_out and all_train_rows is not None:
        win_tr = extract_win_noul_rows(all_train_rows)
        _write_jsonl(args.win_train_out, win_tr)
        clean_stats["win_train"] = {
            "rows": len(win_tr),
            "win_rate": sum(r["won"] for r in win_tr) / max(len(win_tr), 1),
        }

    if args.win_val_out and args.all_val and args.all_val.exists():
        win_va = extract_win_noul_rows(_load_jsonl(args.all_val))
        _write_jsonl(args.win_val_out, win_va)
        clean_stats["win_val"] = {
            "rows": len(win_va),
            "win_rate": sum(r["won"] for r in win_va) / max(len(win_va), 1),
        }

    # Strip internal sets before JSON serialization.
    def strip(rep):
        return None if rep is None else {k: v for k, v in rep.items() if k != "_sets"}

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "checks": checks,
        "train": strip(train_rep),
        "val": strip(val_rep),
        "all_train": strip(all_rep),
        "leakage": leakage,
        "clean_stats": clean_stats,
    }
    atomic_json(args.out / "audit_report.json", report)
    write_markdown_report(args.out / "AUDIT.md", train_rep, val_rep, leakage, checks, all_rep=all_rep)
    print(json.dumps({"checks": checks, "clean_stats": clean_stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
