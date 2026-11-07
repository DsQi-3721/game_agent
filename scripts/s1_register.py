"""Register an evaluated student and apply the promotion rule (``arena.s1.registry``).

    python scripts/s1_register.py --registry models/s1/registry.json --id s1-v2-2b \
        --base /data/models/Qwen3.5-2B --adapter runs/s1_models/s1-v2-2b --dataset data/s1/s1-v2 \
        --valid runs/s1_eval/s1-v2-2b-validall.json --base-valid runs/s1_eval/base-2B-validall.json \
        [--champion-valid runs/s1_eval/<champion>-validall.json] [--online runs/s1_cascade_r1/summary.json]

For students trained and scored separately (``s1_train.py`` / ``s1_eval.py``); ``s1_loop.py`` does the
same in one go. With a champion in the registry, ``--champion-valid`` (the champion scored on the SAME
held-out set) is required -- its stored metrics come from an older split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arena.s1.registry import Registry, decide_promotion


def load(path):
    return json.loads(Path(path).read_text()) if path else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="runs/s1_models/registry.json")
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--valid", required=True, help="s1_eval.py output of the candidate")
    ap.add_argument("--base-valid", default=None, help="s1_eval.py output of the untrained base (same split)")
    ap.add_argument("--champion-valid", default=None, help="s1_eval.py output of the champion (same split)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-promote", action="store_true", help="record the verdict only")
    args = ap.parse_args()
    reg = Registry(args.registry)
    champ = reg.champion()
    if champ is not None and not args.champion_valid:
        raise SystemExit(f"champion {champ['id']} exists: pass --champion-valid (scored on the same split)")
    meta_path = Path(args.adapter) / "s1_train_meta.json"
    tmeta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics = {"valid": load(args.valid)}
    if args.base_valid:
        metrics["base_valid"] = load(args.base_valid)
    if args.champion_valid:
        metrics["champion_valid"] = load(args.champion_valid)
    entry = reg.add({"id": args.id, "base_model": args.base, "adapter": args.adapter, "backend": args.backend,
                     "parent": champ["id"] if champ else None, "dataset": args.dataset,
                     "n_train": tmeta.get("n_train"),
                     "train": {k: v for k, v in tmeta.items() if k not in ("history", "loss_curve")},
                     "train_history": tmeta.get("history"), "metrics": metrics, "notes": args.notes})
    ok, why = decide_promotion(entry, champ, champion_valid=metrics.get("champion_valid"))
    reg.update(args.id, promotion_check=why, status="candidate" if ok and args.no_promote else
               ("champion" if ok else "rejected"))
    if ok and not args.no_promote:
        reg.promote(args.id, why)
    print(json.dumps({"id": args.id, "promoted": ok and not args.no_promote, "eligible": ok, "reason": why,
                      "champion_before": champ["id"] if champ else None,
                      "valid": metrics["valid"].get("overall"),
                      "base": (metrics.get("base_valid") or {}).get("overall"),
                      "champion": (metrics.get("champion_valid") or {}).get("overall")}, indent=1))


if __name__ == "__main__":
    main()
