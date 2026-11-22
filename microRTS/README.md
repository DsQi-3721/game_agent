# MicroRTS EVE-RLCD Fast Decision Model

Self-contained MicroRTS fast decision-model training and inference package reproducing **EVE-RLCD** (`r = c - stopgrad(p_a)` with leave-one-out group baseline, 26-letter readout head, zero text generation, and `SharedPrefixCache` multi-unit batched inference) on top of `anthonym21/qwen3-0.6b-rlcd-decision`.

## Directory Layout

- `rlcd/` — Self-contained `eve-rlcd` core (`schema`, `rewards`, `policies`, `decide`, `export`, `loop`, `metrics`).
- `microrts_agent/` — Multi-unit concurrent `SkillEnv`, compact state encoding, deduplicated tactical skills, `EveRLCDPolicy`, unified CLI `train.py`, data quality auditor `audit_data.py`, and MicroRTS JNI engine/JARs/maps.
- `data/expert-v2-pro/` — Audited Route-B top competition bot head-to-head dataset (`train_pro_clean.jsonl.gz`: 24,549 rows across 12 maps; `val_pro_clean.jsonl.gz`: 6,045 rows across 4 unseen maps; `win_pro_*.jsonl.gz`: balanced state-level `NoulQ` win-forecast splits; see `AUDIT.md`).
- `setup_server.sh` — Safe Linux x86_64 GPU server setup (installs portable JDK 17 locally in `.runtime/java`, adds only `JPype1 --no-deps` to conda `game-agent`, downloads pinned Qwen3-0.6B RLCD export).
- `run_train_a800.sh` — Full A800 GPU training pipeline (Base Eval → SFT Warmup → Bandit-RLCD & RLVR ablation → Online PPO + RLCD → Unseen-Map Eval → LoRA Merge Export & Latency Bench).

## Quick Start on Server

```bash
cd /data/wangxinqi/microRTS
bash setup_server.sh
CUDA_VISIBLE_DEVICES=0 nohup bash run_train_a800.sh runs/a800-v2-pro > runs/a800-v2-pro.log 2>&1 &
```
