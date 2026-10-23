# 交接文档：HyMem 博弈智能体 + System 1（Jev）深度融合 + 学习闭环

> 交接日期：2026-10-23 · 来源：本机（Apple M4 32GB，无 NVIDIA GPU）开发与验证 → 服务器（GPU）继续。
> 读者：在服务器上接手的 AI / 开发者。先读本文件，再看 `reports/jev_system1_design.md`（调研与实测）和 `reports/s1_learning_loop.md`（学习闭环）。

## 0. 一句话现状

- **项目**：`arena` 是 Pokemon Showdown 上的智能体评测平台。用户自己的智能体是 **HyMem**（用户论文 arXiv 2608.15703 的分层上下文/记忆方法，实现为 `llm_hymem`），骨干是本地 Qwen3.5-4B。
- **问题**：HyMem 在本机每个决策平均 139.5 s，无法实时。
- **已完成**：
  - 引入 System One 决策模型 **Jev**（TypeSafe API）作为"快系统"，并深度融合进 HyMem：前置快路径、内部控制、记忆路由（智能体 `llm_hymem_dual`）。
  - 建立**学习闭环** `arena.s1`：记录 → 数据集 → 蒸馏训练本地 student → 评估 → 注册/晋升 → TypeSafe 兼容服务 → 级联部署（student 先答，不确定时问 teacher）→ 再记录。
- **需要在服务器上做**：GPU 上的完整 LoRA 训练（本机只能训最后一个注意力块）、HyMem 在 vLLM 上的大规模消融、用更强 student 迭代。

## 1. 仓库结构（只列与本工作相关的）

```
src/arena/
  agents/hymem/            HyMem 智能体（论文方法）。新增两个扩展点：_decide()、memory_text()、answer_now()；fold_mode 新增 typed / typed_light
  agents/system1/
    facts.py               局面 → 代码计算的事实（伤害/击倒判定/先后手/钉子/换人风险，带"本回合/下回合"语义）
    decider.py             FastDecider：split 布局（stay Noul + best_move + best_switch，正倒序去偏）+ 门控 + Tier 0 规则
    memory_router.py       记忆路由：kind / valid / useful 三问 → 分发到 planner / iso / executor / 快路径
    controller.py          深度融合：shared facts、step control、S1Router（任务路由）、S1Executor（System One 规划工具调用）
    agents.py              jev_fast（纯 System 1，支持 student→teacher 级联）、llm_hymem_dual（双系统 + 深度融合）
  llm/typesafe.py          TypeSafe 客户端（httpx）；TYPESAFE_BASE_URL 可指向任意兼容服务；显式走系统代理（见 §6）
  s1/                      学习闭环：context / recorder / prompts / scorer(mlx,hf) / server / dataset / train_mlx / train_hf / evaluate / registry
scripts/
  s1_loop.py s1_dataset.py s1_train.py s1_eval.py s1_serve.py      学习闭环 CLI
  eval_jev_decisions.py (--record)  collect_decision_snapshots.py  build_memory_bench.py  eval_jev_memory_router.py
  jev_probe.py  summarize_jev_validation.py  run_jev_validation.sh
configs/experiments/
  s1_collect_randbats.yaml          teacher 对战并记录（数据采集）
  s1_cascade_randbats.yaml          student→teacher 级联对战并记录（部署 + 采集）
  hymem_deep_latency_randbats.yaml  深度融合单局实时性（Mac）
  hymem_s1_ablation_gpu_randbats.yaml  GPU 消融：hymem / hymem_dual / hymem_deep / hymem_deep_st
  jev_fast_vs_baselines.yaml  hymem_dual_latency_randbats.yaml  ...
data/
  s1/experience/*.jsonl    已采集的 teacher 经验（Jev 标签）——服务器上训练的输入，务必保留
  s1/s1-v1/                第一轮数据集（train/valid/meta）
  jev_eval/                399 个真实局面快照、记忆基准（17 次折叠 808 条）、120 条人工标注
models/s1/                 已注册 student 的 LoRA adapter（小文件）
reports/                   jev_system1_design.md、jev_validation_results.md、s1_learning_loop.md、hymem_realtime_report.md
tests/                     163 个离线测试（不联网、不读 .env 的 key）
```

`runs/` 不入库（体积大，可再生）。关键数字都已写进 `reports/`；teacher 经验数据复制到了 `data/s1/experience/`。

## 2. 服务器环境搭建

```bash
git clone https://github.com/DsQi-3721/game_agent.git pokemonbot && cd pokemonbot
conda create -n arena python=3.12 -y && conda activate arena
pip install -e ".[dev,s1,s1-gpu]"            # torch/transformers/peft；vLLM 另装（见 scripts/serve_qwen35_vllm.sh）
python -m pytest -q                          # 期望全部通过（离线）

# Showdown 服务器（与 pokemonbot 同级目录；需要 Node >= 18）
git clone https://github.com/smogon/pokemon-showdown ../pokemon-showdown
cd ../pokemon-showdown && npm install && node pokemon-showdown start --no-security   # ws://localhost:8000
```

- **模型**：Qwen3.5-0.8B / 2B / 4B（HF 或 ModelScope 下载），路径作为 `--base` / `--model` 参数传入。HyMem 的骨干用 vLLM 服务：`bash scripts/serve_qwen35_vllm.sh /path/to/Qwen3.5-4B`，默认端口 **8001**（8000 被 Showdown 占用；registry 中 `qwen3.5-4b-vllm` 已改为 8001）。
- **密钥**：复制 `.env.example` 为 `.env`，按需填写 `TYPESAFE_API_KEY`（Jev）。**不要提交 `.env`**。
- **代理**：本机（中国大陆）直连 TypeSafe 返回 451（地区不可用），测试经用户的 Clash 代理进行；这与 TypeSafe 协议冲突，用户知情后决定使用（§6）。服务器若同样受限，可以不依赖 Jev：用已采集的 teacher 数据训练 student，并以 `teacher: false` 只用 student（§4.4）。

## 3. 已验证的结果（本机，详见 reports/）

| 项目 | 结果 |
|---|---|
| Jev 决策请求延迟（在线） | p50 0.45 s，p90 1.4 s；每决策约 $0.00017 |
| 纯 Jev vs SimpleHeuristics | 6–4、7–3、6–2（三批，合计 19–9） |
| 纯 Jev vs random / max-power | 9–1 / 9–1、10–0 / 9–1、8–0 / 8–0 |
| 离线 399 局面 | 客观失误 0（SimpleHeuristics 1.8%）；拆分布局把非强制换人率从 53% 降到 32%（1.13）/ 19%（1.14） |
| 记忆路由 | 真假事实 AUROC 0.98；矛盾、他局、过期、空话 100% 拒绝；类别准确率 0.73–0.75（规则 0.59）；planner 记忆 −74% |
| 双系统（Mac） | 46–65% 的回合 ≤ 1.3 s；平均 42–61 s（纯 HyMem 139.5 s）；慢路径在 120 s 内完成 0/9、3/20 |
| 学习闭环第 1 轮 | s1-v1：KL −41%、ECE 减半，但一致率未提高 → 拒绝晋升；级联中门控放行 0 次（弱 student 不影响出手） |
| 深度融合单局 | 慢路径 9/10 在上限内完成（之前 0/9、3/20），每个慢回合 1–3 次 LLM 调用，折叠约 0.5 s，平均延迟 42 s |

## 4. 学习闭环怎么用（服务器上的主要工作）

### 4.1 数据格式

`s1exp_<agent>.jsonl`，每行一条：
- `type=s1`：一次 System One 请求。包括 `state`、`questions`、`answers`（完整概率）、`backend`（`typesafe:*` 为 teacher，`local:*` 为 student）、`role`、`purpose`（action / memory / step_control / route_task / tool_plan）、`decision_id`、`latency_s`；
- `type=decision`：最终动作 `final_key`、路由 `route`（tier0 / fast_teacher / fast_student / slow / slow_timeout_fast …），以及 System 2 完成时它自己的选择 `s2_key`；
- `type=outcome`：胜负。

### 4.2 一轮迭代

```bash
# (a) 采集：teacher 对战（有 Jev 时）或 cascade 对战（有 champion 时）
python -m arena.cli run configs/experiments/s1_collect_randbats.yaml --emit-events
# (b)-(e) 数据集 → 训练 → 评估（与未训练 base / 现任 champion 对比）→ 注册 → 晋升
python scripts/s1_loop.py --id s1-v2 --backend hf --base /models/Qwen3.5-4B \
    --experience "data/s1/experience/*.jsonl" --experience "runs/**/s1exp_*.jsonl" \
    --max-train 20000 --epochs 2 --eval-limit 300
# (f) 服务 champion（TypeSafe 兼容）
python scripts/s1_serve.py --backend hf --model /models/Qwen3.5-4B --adapter runs/s1_models/s1-v2 --port 8090
# (g) 级联部署并继续采集：student 先答，不确定才问 teacher → 正好补齐 student 不会的样本
python -m arena.cli run configs/experiments/s1_cascade_randbats.yaml --emit-events
```

### 4.3 训练要点

- 损失：对选项字母位置的软标签交叉熵（teacher 的完整分布）。System 2 完成的决策按 `s2_weight` 混入（慢系统纠正快系统），胜负作为样本权重。
- **mlx（Mac）**：Qwen3.5 是 Gated DeltaNet 与注意力的混合结构，mlx_lm 的 fused kernel 没有反向传播。
  - `--mode head`：只训最后一个注意力块，约 1 s/样本；
  - `--mode full`：走参考实现，约 17 s/样本，不实用。
- **GPU（HF/peft）**：`all-linear` LoRA；transformers 的 Qwen3.5 实现带可反传的 linear attention（CUDA 上建议安装 flash-linear-attention）。**`train_hf.py` 只做过代码层面的检查，没有在 CUDA 上跑过**，第一次运行请先 `--max-train 50` 冒烟。
- 评估指标：与 teacher 的 top-1 一致率、KL、Brier、ECE、延迟，按问题类型（stay / best_move / best_switch / hard …）分组。
- 晋升规则（`arena/s1/registry.py`）：一致率至少高于 champion 0.01，ECE 不变差超过 0.02，在线胜率（如有记录）不低于 champion。

### 4.4 没有 Jev 时如何继续进化

- 用 `data/s1/experience/` 里已有的 teacher 数据训练 student；
- 部署时用 `teacher: false`、`student_url` 指向 student；
- 之后的标签来源改为 System 2（`s2_key`）和对局胜负：把 `s1_dataset.py` 的 `--teachers` 设为空，或扩展 `dataset.build`，从 decision/outcome 生成标签。这一步是**自我进化**，代码已留接口（`BuildConfig.teachers`、`s2_weight`），尚未实现"无 teacher 时以 S2 为主标签"的分支，属于待办 §7。

## 5. 本次在本机完成的闭环验证（结果）

（由 `runs/_s1_loop_v1.log`、`runs/s1_models/registry.json` 汇总，见 `reports/s1_learning_loop.md`）

- **数据**：1,391 条记录（399 个离线局面 + 24 局 teacher 对战）→ 4,758 道题（train 3,753 / valid 1,005，按对局划分）。级联对战又多了 128 对 student/teacher 答案。三份经验都在 `data/s1/experience/`。
- **s1-v1**：Qwen3.5-0.8B + head LoRA（0.2M 参数），训练 1,500 道题，约 39 分钟。
  - 与 teacher 的一致率 0.369（base 0.397）；KL 1.00 → 0.59；ECE 0.34 → 0.17。
  - 晋升判定：**拒绝**（没超过 base）。adapter 在 `models/s1/s1-v1/`，注册表在 `models/s1/registry.json`。
- **级联（6 局）**：student 被问 128 次，门控放行 0 次，全部由 teacher 出手；对 random 3–0、对 heuristic 1–2。
- **深度融合单局**：慢路径 9/10 在 120 s 内完成，每个慢回合 1–3 次 LLM 调用，平均延迟 42 s，折叠约 0.5 s（详见 `reports/s1_learning_loop.md` §3.4）。
- **HF 训练路径**：在 Mac 上 MPS 会触发 Qwen3.5 linear attention 的形状错误；CPU 可以完成前向和评估（before: valid_ce 1.16），但反传极慢（30 分钟仍未完成 3 步）。**在 CUDA 上第一次运行请先用 `--max-train 50` 冒烟**。

## 6. 注意事项 / 坑

1. **地区限制（451）**：直连 TypeSafe 返回 `Typesafe is not available in your region`；只有经代理（美国出口）才可用。协议禁止"绕过访问限制"，用户知情后自行决定。长期方案是本地 student，即本闭环的目标。
2. **`NO_PROXY` 会让 Python 不读 macOS 系统代理**：arena 为本地 websocket 设置了 `NO_PROXY`，导致 Jev 请求直连而得到 451。已修复：`typesafe.system_proxy()` 显式读取系统代理。Linux 服务器上用 `HTTPS_PROXY` 环境变量即可。
3. **离线测试不得读取 `.env` 或联网**：`tests/test_system1_offline.py` 和 `tests/test_s1_loop_offline.py` 都有 guard。新增测试请沿用。
4. **折叠不能在有时间上限的决策中同步执行**：会被取消，然后反复重来。已改为后台执行（`_DualPlanner`）。在单个本地模型服务上，后台折叠会和慢路径抢算力；GPU/vLLM 上影响小。
5. **天梯计时器**：固定 120 s 的慢路径上限在回放中第 3–4 回合判负；上限 ≤ 20 s 才安全。TODO：按剩余时间库动态设上限。
6. **"难不难"（hard）Noul 没有信息量**：门控里已关闭（`max_hard: 1.0`）。
7. **Jev 字面理解**：事实措辞必须写明"本回合/下回合"，否则换人选项会被当成击倒（实测导致 53% 换人率）。
8. **端口**：Showdown 8000，vLLM 8001，mlx 8080，student 8090。
9. **第三方复现**（`external/pokechamp`、`external/pokellmon` 等）的 repo/venv 不入库，用各自的 `setup.sh` 重建，且必须放在独立的环境里。

## 7. 下一步（按优先级）

1. **GPU 上训练更强的 student**：Qwen3.5-2B/4B，全层 LoRA，用全部 teacher 数据，2–3 个 epoch。目标是与 teacher 一致率 ≥ 0.6，单次决策 < 0.3 s（vLLM 或批量 HF）。
2. **GPU 消融**：`configs/experiments/hymem_s1_ablation_gpu_randbats.yaml`，每组 n ≥ 30；同样做一份 gen9ou。指标是胜率、每决策延迟、路由分布、慢路径完成率，以及**记忆使用率**（planner 的分析/答案是否引用了路由后的记忆）。
3. **级联迭代 2–3 轮**：每轮记录 student 的接受率和 teacher 调用比例（应持续下降），以及胜率。
4. **无 teacher 的自我进化**（§4.4）：以 System 2 决策和胜负为标签，实现 `dataset` 的分支；可以再考虑用对局结果做偏好学习（DPO 类方法）。
5. **按计时器动态设定慢路径预算**；类型化折叠的进一步消融（typed vs typed_light vs two_stage）。
6. 本机 `train_hf.py` 未在 CUDA 上验证；`HFScorer` 只做过代码层面的检查，第一次使用请先跑 `scripts/s1_eval.py --backend hf --limit 5`。
